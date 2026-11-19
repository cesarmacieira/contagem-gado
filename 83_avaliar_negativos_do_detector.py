"""Compara detector antes/depois em negativos reservados, ja conhecidos."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/especialista_solo_negativos_v3'


def main():
    torch.set_num_threads(4)
    assert (OUT/'resultado.json').exists(), 'Treino e regressao devem terminar primeiro'
    negativos=[]
    base=ROOT/'data/negativos_solo_v1'
    for r in pd.read_csv(base/'manifesto.csv').to_dict('records'):
        if r['conjunto'] in ('val','test') and r['status']=='aprovado_sem_bovinos':
            negativos.append((r['arquivo'],r['conjunto'],'foto_integral',cv2.imread(str(base/'images'/r['arquivo']))))
    base=ROOT/'data/confundidores_especies_v1'
    for r in json.loads((base/'revisao_visual.json').read_text()):
        if r['conjunto'] not in ('val','test') or not r['revisado']:continue
        im=cv2.imread(str(base/'images'/r['arquivo']));h,w=im.shape[:2]
        x1,y1,x2,y2=np.round(np.asarray(r['bbox_rel'])*[w,h,w,h]).astype(int)
        negativos.append((r['arquivo'],r['conjunto'],'recorte_outra_especie',im[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]))
    rows=[]
    for nome,peso in [('referencia',ROOT/'data/processed/modelo_gado.pt'),('candidato',OUT/'treino/weights/best.pt')]:
        model=YOLO(str(peso))
        for arquivo,split,origem,im in negativos:
            b,s=ct.detectar_inteira(model,im,imgsz=1280,conf_min=.15,iou_nms=.7,device=0,classes=[0],augment=False)
            rows.append(dict(modelo=nome,arquivo=arquivo,conjunto=split,origem=origem,real=0,previsto=len(b)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'negativos_reservados.csv',index=False)
    resumo={}
    for (origem,nome),g in df.groupby(['origem','modelo']):
        resumo.setdefault(origem,{})[nome]=dict(n_imagens=len(g),falsas_caixas=int(g.previsto.sum()),
            imagens_com_falso_positivo=int((g.previsto>0).sum()))
    resultado=dict(resumo=resumo,limite='Negativos reservados do treino, mas ja conhecidos por rodadas anteriores. '
        'Fotos integrais e recortes de especies avaliados separadamente. Nao e teste brasileiro denso independente.')
    (OUT/'negativos_reservados.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    p=OUT/'RELATORIO.md';texto=p.read_text(encoding='utf-8')
    marcador='\n## Negativos reservados\n'
    texto=texto.split(marcador)[0]+marcador+'\n'+resultado['limite']+'\n\n'
    for origem,modelos in resumo.items():
        for modelo,m in modelos.items():
            texto+=f'- {origem}, {modelo}: {m["falsas_caixas"]} falsas caixas em {m["n_imagens"]} imagens/recortes.\n'
    p.write_text(texto,encoding='utf-8')
    print(json.dumps(resultado,indent=2))


if __name__=='__main__':main()
