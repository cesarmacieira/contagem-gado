"""Estimativa supervisionada de contagem: pontos publicados, sem inventar caixas."""
from pathlib import Path
import numpy as np
import contagem as ct

CONFIANCAS=(.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.60,.70,.80)
ESTRATEGIAS=('inteira1024','inteira1536','recortes768')
VERSAO=1


def extrair(modelo,img,device=0):
    h,w=img.shape[:2];features={};contagens={};detalhes={}
    for nome in ESTRATEGIAS:
        if nome=='recortes768':
            b,s=ct.detectar(modelo,img,tile=768,overlap=.25,conf_min=.025,iou_local=.7,iou_nms=.5,device=device,max_det=1000)
        else:
            b,s=ct.detectar_inteira(modelo,img,imgsz=1024 if nome=='inteira1024' else 1536,conf_min=.025,iou_nms=.5,device=device,max_det=1000)
        detalhes[nome]=(b,s)
        for c in CONFIANCAS:
            chave=f'{nome}_c{c:.2f}';features[chave]=int((s>=c).sum());contagens[chave]=features[chave]
        scores=s[s>=.05]
        bb=b[s>=.05]
        area=(bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1])/(h*w) if len(bb) else np.zeros(0)
        for q in (.1,.25,.5,.75,.9):
            features[f'{nome}_score{q}']=float(np.quantile(scores,q)) if len(scores) else 0.
            features[f'{nome}_area{q}']=float(np.quantile(area,q)) if len(area) else 0.
        features[f'{nome}_soma_scores']=float(scores.sum())
        features[f'{nome}_area_total']=float(area.sum())
        features[f'{nome}_std_x']=float(((bb[:,0]+bb[:,2])/2/w).std()) if len(bb) else 0.
        features[f'{nome}_std_y']=float(((bb[:,1]+bb[:,3])/2/h).std()) if len(bb) else 0.
    features['aspecto']=w/h
    return features,contagens,detalhes


def estimar(pacote,features):
    x=np.array([[features[c] for c in pacote['colunas']]],float)
    return max(0,float(pacote['regressor'].predict(x)[0]))
