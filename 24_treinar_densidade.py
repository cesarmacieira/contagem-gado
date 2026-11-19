"""Treino com pontos publicados e centros das caixas brasileiras, sem olhar teste."""
from pathlib import Path
import importlib.util,json,random,time,argparse
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import densidade as de
import contagem as ct
from avaliacao import metricas_contagem

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/denso_cattle';OUT=ROOT/'data/experimentos/denso_densidade_v1'


class Dados(Dataset):
    def __init__(self,df,pontos,treino=False,mosaico=False):
        self.dados=[];self.treino=treino;self.mosaico=mosaico;self.brasil_denso=[]
        for r in df.itertuples():
            im=cv2.imread(r.img);assert im is not None
            h,w=im.shape[:2]
            if r.fonte=='cattle_pontos':pt=np.array(pontos[r.stem],float).reshape(-1,2)
            else:
                rot=ROOT/'data/brasil_dataset_v2/labels'/(Path(r.arquivo).stem+'.txt')
                b=ct.yolo_para_pixels(rot.read_text().splitlines(),w,h);pt=(b[:,:2]+b[:,2:])/2
            pt=pt*np.array([de.LARGURA/w,de.ALTURA/h])
            im=cv2.cvtColor(cv2.resize(im,(de.LARGURA,de.ALTURA)),cv2.COLOR_BGR2RGB)
            registro=(im,pt,r.stem,r.grupo,int(r.n_bovinos))
            inicio=len(self.dados)
            self.dados.extend([registro]*(4 if treino and r.fonte!='cattle_pontos' else 1))
            if r.fonte!='cattle_pontos' and r.n_bovinos>=5:self.brasil_denso.extend(range(inicio,len(self.dados)))

    def __len__(self):return len(self.dados)

    def __getitem__(self,i):
        im,p,stem,g,n=self.dados[i];p=p.copy();im=im.copy()
        if self.treino and self.mosaico and random.random()<.55:
            indices=random.choices(self.brasil_denso,k=3)+[random.randrange(len(self.dados))]
            random.shuffle(indices);canvas=np.zeros_like(im);partes=[]
            for slot,j in enumerate(indices):
                imagem,pontos,*_=self.dados[j];x=(slot%2)*(de.LARGURA//2);y=(slot//2)*(de.ALTURA//2)
                canvas[y:y+de.ALTURA//2,x:x+de.LARGURA//2]=cv2.resize(imagem,(de.LARGURA//2,de.ALTURA//2))
                partes.append(pontos*.5+[x,y])
            im=canvas;p=np.concatenate(partes,axis=0)
        if self.treino:
            if random.random()<.7:
                w=int(de.LARGURA*random.uniform(.65,1));h=int(de.ALTURA*random.uniform(.65,1))
                x=random.randint(0,de.LARGURA-w);y=random.randint(0,de.ALTURA-h)
                p=p[(p[:,0]>=x)&(p[:,0]<x+w)&(p[:,1]>=y)&(p[:,1]<y+h)]
                p=(p-[x,y])*[de.LARGURA/w,de.ALTURA/h]
                im=cv2.resize(im[y:y+h,x:x+w],(de.LARGURA,de.ALTURA))
            if random.random()<.5:im=im[:,::-1].copy();p[:,0]=de.LARGURA-1-p[:,0]
            if self.mosaico and random.random()<.35:
                fator=random.uniform(.45,.8);w=int(de.LARGURA*fator);h=int(de.ALTURA*fator)
                x=random.randint(0,de.LARGURA-w);y=random.randint(0,de.ALTURA-h)
                fundo=np.median(im.reshape(-1,3)[::100],axis=0).astype(np.uint8)
                canvas=np.full_like(im,fundo);canvas[y:y+h,x:x+w]=cv2.resize(im,(w,h))
                im=canvas;p=p*[w/de.LARGURA,h/de.ALTURA]+[x,y]
            im=np.clip(im.astype(float)*random.uniform(.8,1.2),0,255).astype(np.uint8)
        mapa=np.zeros((de.ALTURA//8,de.LARGURA//8),np.float32)
        if len(p):
            ix=np.rint(p[:,0]/8).astype(int).clip(0,mapa.shape[1]-1);iy=np.rint(p[:,1]/8).astype(int).clip(0,mapa.shape[0]-1)
            np.add.at(mapa,(iy,ix),1)
            mapa=gaussian_filter(mapa,1.2,mode='constant')
            mapa*=len(p)*de.ESCALA/max(float(mapa.sum()),1e-8)
        return torch.from_numpy(im.transpose(2,0,1).copy()),torch.from_numpy(mapa[None]),len(p),stem,g


def avaliar(modelo,loader):
    ys=[];ps=[];groups=[]
    modelo.eval()
    with torch.inference_mode():
        for im,den,n,stem,g in loader:
            pred=modelo(im.cuda().float()/255).float().sum((1,2,3))/de.ESCALA
            ys.extend(n.tolist());ps.extend(pred.cpu().tolist());groups.extend(g)
    p=np.maximum(0,np.rint(ps));m=metricas_contagem(ys,p)
    erros=pd.DataFrame(dict(g=groups,e=np.abs(p-np.array(ys))));m['MAE_macro_grupos']=float(erros.groupby('g').e.mean().mean())
    m['score_validacao']=(m['MAE']+m['MAE_macro_grupos'])/2
    return m,ps


def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experimento',default='denso_densidade_v1')
    ap.add_argument('--mosaico',action='store_true')
    ap.add_argument('--iniciar',type=Path)
    a=ap.parse_args();assert Path(a.experimento).name==a.experimento
    OUT=ROOT/'data/experimentos'/a.experimento
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.backends.cudnn.benchmark=True
    assert torch.cuda.is_available()
    OUT.mkdir(exist_ok=True,parents=True)
    assert not (OUT/'densidade.pt').exists(),'Preserve experimento concluido'
    df=pd.read_csv(ROOT/'data/experimentos/denso_v1/catalogo_congelado.csv')
    train=df[df.conjunto=='train'];val=df[df.conjunto=='val']
    pontos=json.loads((BASE/'pontos.json').read_text())
    treino=Dados(train,pontos,True,a.mosaico);valid=Dados(val,pontos)
    lt=DataLoader(treino,batch_size=12,shuffle=True,num_workers=0,pin_memory=True)
    lv=DataLoader(valid,batch_size=12,shuffle=False,num_workers=0,pin_memory=True)
    modelo=de.ContadorDensidade(ROOT/'data/processed/modelo_gado.pt').cuda()
    if a.iniciar:modelo.load_state_dict(torch.load(a.iniciar,map_location='cuda',weights_only=True))
    opt=torch.optim.AdamW([dict(params=modelo.backbone.parameters(),lr=.00002),dict(params=modelo.head.parameters(),lr=.0002 if a.iniciar else .0005)],weight_decay=.0001)
    amp=torch.amp.GradScaler('cuda');melhor=None;historico=[];inicio=time.time()
    for ep in range(1,41):
        modelo.train();modelo.backbone.eval();modelo.congelar(ep<=3);losses=[]
        for im,den,n,stem,g in lt:
            im=im.cuda(non_blocking=True).float()/255;den=den.cuda(non_blocking=True);n=n.cuda().float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):pred=modelo(im)
            pred=pred.float();count=pred.sum((1,2,3))/de.ESCALA
            loss=(pred-den).square().mean()+.1*((count-n)/(n.sqrt()+3)).square().mean()
            amp.scale(loss).backward();amp.unscale_(opt);nn.utils.clip_grad_norm_(modelo.parameters(),5.)
            amp.step(opt);amp.update();losses.append(float(loss.detach()))
        met,ps=avaliar(modelo,lv);historico.append(dict(epoca=ep,tempo=time.time()-inicio,loss=float(np.mean(losses)),**met))
        pd.DataFrame(historico).to_csv(OUT/'curvas.csv',index=False)
        print(ep,round(float(np.mean(losses)),4),met['MAE'],met['MAE_macro_grupos'],flush=True)
        if melhor is None or met['score_validacao']<melhor['score_validacao']:
            melhor=dict(epoca=ep,**met)
            torch.save({k:v.cpu() for k,v in modelo.state_dict().items()},OUT/'densidade.pt')
            (OUT/'config_congelada.json').write_text(json.dumps(dict(escolha=melhor,train_imagens=len(train),train_exposicoes_por_epoca=len(treino),val_imagens=len(val),teste_executado=False,mosaico=a.mosaico,inicializado_em=str(a.iniciar) if a.iniciar else None,rotulos='Pontos publicados e centros das caixas revisadas',arquitetura='Backbone YOLO treinado, niveis stride 8 e 16, cabeca de densidade positiva; soma da densidade estima quantidade'),indent=2),encoding='utf-8')
        if ep>=18 and ep-melhor['epoca']>=12:break
    print('Final',melhor,flush=True)


if __name__=='__main__':main()
