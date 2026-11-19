"""Diagnostico visual das caixas extras; reutiliza cache, nao executa detector."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/fontes_pesquisa/nellore_inspecao/avaliacao_recorte_25'
OUT=ROOT/'data/experimentos/auditoria_extras_curral_v1'


def main():
    estado=BASE/'ESTADO_GABARITO.json'
    if estado.exists() and not json.loads(estado.read_text())['valido_para_contagem_exata']:
        raise ValueError('Pontos incompletos: nao classificar caixas sem ponto como extras')
    OUT.mkdir(parents=True,exist_ok=True)
    d=np.load(BASE/'curral_topo.npz');b,s=d['caixas'],d['scores']
    pts=np.array(json.loads((BASE/'gabarito_antes_inferencia.json').read_text())['pontos_xy'])
    inside=((pts[:,None,:]>=b[None,:,:2])&(pts[:,None,:]<=b[None,:,2:])).all(axis=2)
    centers=(b[:,:2]+b[:,2:])/2
    cost=np.linalg.norm((pts[:,None,:]-centers[None,:,:])/np.maximum(b[None,:,2:]-b[None,:,:2],1),axis=2)
    cost=np.where(inside,cost,1e6)
    ip,ib=linear_sum_assignment(cost)
    matches={int(j):int(i) for i,j in zip(ip,ib) if inside[i,j]}
    assert len(matches)==25
    extras=[i for i in range(len(b)) if i not in matches]
    im=cv2.imread(str(BASE/'recorte.png'));h,w=im.shape[:2]
    area=np.prod(b[:,2:]-b[:,:2],axis=1);rows=[];panels=[]
    for i in extras:
        inter=np.maximum(0,np.minimum(b[i,2:],b[:,2:])-np.maximum(b[i,:2],b[:,:2])).prod(axis=1)
        containment=inter/np.maximum(np.minimum(area[i],area),1e-9)
        neighbors=[j for j in matches if inter[j]>0]
        j=max(neighbors,key=lambda j:containment[j]) if neighbors else None
        rows.append(dict(caixa=i,score=float(s[i]),xyxy=b[i].tolist(),vizinha=j,
            contencao=float(containment[j]) if j is not None else 0,
            razao_area=float(area[i]/area[j]) if j is not None else 0,
            pontos_dentro=np.flatnonzero(inside[:,i]).tolist(),classificacao='pendente_revisao_visual'))
        x1,y1,x2,y2=b[i];pad=150
        left,top=max(0,int(x1)-pad),max(0,int(y1)-pad)
        right,bottom=min(w,int(x2)+pad),min(h,int(y2)+pad)
        fig=im[top:bottom,left:right].copy()
        for k in neighbors:
            a=np.round(b[k]-[left,top,left,top]).astype(int)
            cv2.rectangle(fig,tuple(a[:2]),tuple(a[2:]),(255,180,0),2)
        a=np.round(b[i]-[left,top,left,top]).astype(int)
        cv2.rectangle(fig,tuple(a[:2]),tuple(a[2:]),(0,0,255),4)
        for k,(x,y) in enumerate(pts):
            if left<=x<right and top<=y<bottom:
                cv2.circle(fig,(round(x)-left,round(y)-top),7,(0,255,0),-1)
        fig=cv2.resize(fig,(390,340))
        cv2.rectangle(fig,(0,0),(390,32),(255,255,255),-1)
        cv2.putText(fig,f'Caixa {i} / conf {s[i]:.3f}',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
        panels.append(fig)
    atlas=np.full((3*340,3*390,3),255,np.uint8)
    for k,p in enumerate(panels):atlas[(k//3)*340:(k//3+1)*340,(k%3)*390:(k%3+1)*390]=p
    cv2.imwrite(str(OUT/'oito_extras.jpg'),atlas)
    pd.DataFrame(rows).to_csv(OUT/'extras.csv',index=False)
    (OUT/'correspondencias.json').write_text(json.dumps(dict(caixa_para_ponto=matches,
        criterio='Correspondencia um-a-um por ponto contido; menor distancia normalizada ao centro. '
        'Identifica candidatos a extra, nao substitui revisao visual.'),indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
