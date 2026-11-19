"""Rede de densidade com caracteristicas do detector e saida de contagem, sem caixas."""
from pathlib import Path
import copy
import torch
from torch import nn
from torch.nn import functional as F

LARGURA,ALTURA,ESCALA=640,384,100.


class ContadorDensidade(nn.Module):
    def __init__(self,pesos):
        super().__init__()
        from ultralytics import YOLO
        origem=YOLO(str(pesos)).model.model
        self.backbone=nn.ModuleList([copy.deepcopy(b) for b in origem[:7]])
        self.head=nn.Sequential(nn.Conv2d(512,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,1,1))
        nn.init.normal_(self.head[-1].weight,std=.001);nn.init.constant_(self.head[-1].bias,-2.)

    def forward(self,x):
        f8=None
        for i,b in enumerate(self.backbone):
            x=b(x)
            if i==4:f8=x
        x=torch.cat([f8,F.interpolate(x,size=f8.shape[-2:],mode='bilinear',align_corners=False)],1)
        return F.softplus(self.head(x))

    def congelar(self,valor):
        for p in self.backbone.parameters():p.requires_grad=not valor


def carregar(pasta,device='cpu'):
    pasta=Path(pasta);root=Path(__file__).resolve().parent
    m=ContadorDensidade(root/'data/processed/modelo_gado.pt')
    m.load_state_dict(torch.load(pasta/'densidade.pt',map_location='cpu',weights_only=True))
    return m.to(device).eval()


def prever(modelo,img,device='cpu'):
    import cv2
    import numpy as np
    im=cv2.cvtColor(cv2.resize(img,(LARGURA,ALTURA)),cv2.COLOR_BGR2RGB)
    x=torch.from_numpy(im.transpose(2,0,1).copy()).unsqueeze(0).to(device).float()/255
    with torch.inference_mode(): mapa=modelo(x).float()[0,0].cpu().numpy()/ESCALA
    return max(0,float(mapa.sum())),mapa


def prever_adaptativo(modelo,img,detector,alvo=48,device='cuda',deteccoes=None):
    """Dimensiona animais pela mediana das caixas, sem usar o gabarito.

    O recorte vem das deteccoes com margem; poucas deteccoes usam a imagem toda.
    A contagem continua sendo uma estimativa de densidade, nao caixas novas.
    """
    import cv2
    import numpy as np
    import contagem as ct
    if deteccoes is None:
        deteccoes=ct.detectar_inteira(detector,img,imgsz=1280,conf_min=.05,iou_nms=.5,device=device)
    b,s=deteccoes;bb=b[s>=.15];h,w=img.shape[:2]
    if len(bb)<8:return prever(modelo,img,device)
    alturas=bb[:,3]-bb[:,1];mediana=max(8,float(np.median(alturas)))
    margem=max(32,mediana*.75)
    x1=max(0,int(bb[:,0].min()-margem));x2=min(w,int(bb[:,2].max()+margem))
    y1=max(0,int(bb[:,1].min()-margem));y2=min(h,int(bb[:,3].max()+margem))
    crop=img[y1:y2,x1:x2];ch,cw=crop.shape[:2]
    escala=float(alvo)/mediana
    escala=min(escala,3072/max(cw,ch),np.sqrt(1250000/(cw*ch)))
    escala=max(escala,64/min(cw,ch))
    nw=max(32,int(np.ceil(cw*escala/16))*16);nh=max(32,int(np.ceil(ch*escala/16))*16)
    rgb=cv2.cvtColor(cv2.resize(crop,(nw,nh)),cv2.COLOR_BGR2RGB)
    x=torch.from_numpy(rgb.transpose(2,0,1).copy()).unsqueeze(0).to(device).float()/255
    with torch.inference_mode():m=modelo(x).float()[0,0].cpu().numpy()/ESCALA
    n=max(0,float(m.sum()));zoom=cv2.resize(m,(cw,ch));zoom*=n/max(float(zoom.sum()),1e-9)
    mapa=np.zeros((h,w),np.float32);mapa[y1:y2,x1:x2]=zoom
    return n,mapa
