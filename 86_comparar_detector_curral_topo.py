"""Compara especialista atual e COCO por visao, sem promover automaticamente."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from avaliacao import comparar_caixas,metricas_contagem,taxas_deteccao
import contagem as ct

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/curral_topo_yolo11x_v1'
NOVA=ROOT/'data/fontes_pesquisa/nellore_inspecao/avaliacao_recorte_25'


def main():
    estado=NOVA/'ESTADO_GABARITO.json'
    if estado.exists() and not json.loads(estado.read_text())['valido_para_contagem_exata']:
        raise ValueError('Cena brasileira com gabarito incompleto; avaliacao bloqueada')
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    pesos={'referencia':(ROOT/'data/processed/modelo_gado.pt',[0]),
           'coco11x':(ROOT/'data/modelos_candidatos/yolo11x.pt',[19])}
    images=sorted((ROOT/'data/yolo/images/test').glob('colo_topo__*.jpg'))
    assert len(images)>0
    protocol=dict(pesos={k:dict(arquivo=str(p),classes=c,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,(p,c) in pesos.items()},
        imgsz=1024,conf=.2,iou=.7,augment=False,imagens_historicas=[p.name for p in images],
        nota='Regressao historica. A cena brasileira nova foi vista no experimento 85 e agora tambem e desenvolvimento.',
        promover=False)
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    spec=spec_from_file_location('pontos',ROOT/'67_verificar_especies_no_denso.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    pontos=json.loads((NOVA/'gabarito_antes_inferencia.json').read_text())['pontos_xy']
    rows=[];new=[]
    for nome,(peso,classes) in pesos.items():
        model=YOLO(str(peso))
        if nome=='coco11x':assert model.names[19]=='cow'
        for p in images:
            im=cv2.imread(str(p));h,w=im.shape[:2]
            rot=ROOT/'data/yolo/labels/test'/(p.stem+'.txt')
            ref=ct.yolo_para_pixels(rot.read_text(encoding='utf-8-sig').splitlines(),w,h)
            b,s=ct.detectar_inteira(model,im,imgsz=1024,conf_min=.2,iou_nms=.7,device=0,classes=classes,augment=False)
            rows.append(dict(modelo=nome,arquivo=p.name,real=len(ref),previsto=len(b),**comparar_caixas(ref,b,s)))
        im=cv2.imread(str(NOVA/'recorte.png'))
        b,s=ct.detectar_inteira(model,im,imgsz=1024,conf_min=.2,iou_nms=.7,device=0,classes=classes,augment=False)
        new.append(dict(modelo=nome,real=25,previsto=len(b),pontos_cobertos=mod.correspondencias_pontos(pontos,b)))
        np.savez(OUT/(nome+'_brasil25.npz'),caixas=b,scores=s)
        cv2.imwrite(str(OUT/(nome+'_brasil25.jpg')),ct.desenhar(im,b))
        print(nome,'historicas',len(images),'cena_brasileira',new[-1],flush=True)
    t=pd.DataFrame(rows);t.to_csv(OUT/'por_imagem.csv',index=False)
    metricas={}
    for nome,g in t.groupby('modelo'):
        tp,fp,fn=(int(g[k].sum()) for k in ('TP','FP','FN'))
        metricas[nome]=dict(**metricas_contagem(g.real,g.previsto),TP=tp,FP=fp,FN=fn,**taxas_deteccao(tp,fp,fn))
    result=dict(historico=metricas,cena_brasileira=new,promovido=False,limite=protocol['nota'])
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
