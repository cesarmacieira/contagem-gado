"""Compara escalas na validacao; o teste nao participa desta escolha."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import densidade,contagem as ct
from avaliacao import metricas_contagem

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/denso_escala_v1'


def met(df,p):
    p=np.maximum(0,np.rint(p));m=metricas_contagem(df.n_bovinos,p)
    e=pd.DataFrame(dict(g=df.grupo,e=np.abs(df.n_bovinos.to_numpy()-p)))
    m['MAE_macro_grupos']=float(e.groupby('g').e.mean().mean());m['score_validacao']=(m['MAE']+m['MAE_macro_grupos'])/2
    return m


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'config_congelada.json').exists():print('Escolha ja congelada');return
    df=pd.read_csv(ROOT/'data/experimentos/denso_v1/catalogo_congelado.csv');val=df[df.conjunto=='val'].copy()
    detector=YOLO(str(ROOT/'data/processed/modelo_gado.pt'));det={};pequenos={}
    for r in val.itertuples():
        im=cv2.imread(r.img);det[r.stem]=ct.detectar_inteira(detector,im,imgsz=1280,conf_min=.05,iou_nms=.5,device=0)
        b,s=det[r.stem];bb=b[s>=.15]
        pequenos[r.stem]=len(bb)>=8 and float(np.median(bb[:,3]-bb[:,1]))/im.shape[0]<.06
    resultados=[];preds=[]
    for variante in ('denso_densidade_v1','denso_densidade_v2'):
        rede=densidade.carregar(ROOT/'data/experimentos'/variante,'cuda')
        ppor={}
        for alvo in (0,32,64):
            p=[]
            for r in val.itertuples():
                im=cv2.imread(r.img)
                n,_=densidade.prever(rede,im,'cuda') if alvo==0 else densidade.prever_adaptativo(rede,im,detector,alvo,'cuda',det[r.stem])
                p.append(n);preds.append(dict(rede=variante,alvo=alvo,stem=r.stem,real=r.n_bovinos,previsto=n,fonte=r.fonte))
            ppor[alvo]=np.array(p)
            m=met(val,p);resultados.append(dict(rede=variante,alvo=alvo,somente_pequenos=False,**m));print(variante,alvo,m['MAE'],m['MAE_macro_grupos'],flush=True)
        for alvo in (32,64):
            p=np.where([pequenos[s] for s in val.stem],ppor[alvo],ppor[0])
            m=met(val,p);resultados.append(dict(rede=variante,alvo=alvo,somente_pequenos=True,**m));print(variante,alvo,'pequenos',m['MAE'],m['MAE_macro_grupos'],flush=True)
    resultados.sort(key=lambda r:r['score_validacao'])
    (OUT/'config_congelada.json').write_text(json.dumps(dict(escolha=resultados[0],candidatas=resultados,teste_nao_usado_na_escolha=True,criterio='MAE e MAE macro por grupo na mesma validacao; escala adaptativa usa apenas caixas do detector'),indent=2),encoding='utf-8')
    pd.DataFrame(preds).to_csv(OUT/'previsoes_validacao.csv',index=False)
    print('Escolhida:',resultados[0],flush=True)


if __name__=='__main__':main()
