"""Compara o preset denso preexistente, sem busca no teste ja examinado."""
from pathlib import Path
import json
import cv2
import pandas as pd
from ultralytics import YOLO
import contagem as ct
from avaliacao import metricas_contagem,comparar_caixas,taxas_deteccao

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'data/experimentos/detector_denso_v3'


def main():
    out=EXP/'avaliacao/comparacao_preset_existente.json'
    if out.exists():print(out.read_text());return
    df=pd.read_csv(EXP/'teste_congelado.csv');modelo=YOLO(str(ROOT/'data/processed/modelo_gado.pt'));linhas=[]
    for i,r in enumerate(df.itertuples()):
        im=cv2.imread(r.img);h,w=im.shape[:2]
        ref=ct.yolo_para_pixels(Path(r.rot).read_text().splitlines(),w,h)
        b,s=ct.detectar_inteira(modelo,im,imgsz=1280,iou_nms=.7,device=0);b,s=ct.filtrar(b,s,.15)
        linhas.append(dict(stem=r.stem,real=len(ref),previsto=len(b),**comparar_caixas(ref,b,s)))
        if (i+1)%100==0:print('Preset existente:',i+1,flush=True)
    def met(rows):
        tp=sum(r['TP'] for r in rows);fp=sum(r['FP'] for r in rows);fn=sum(r['FN'] for r in rows)
        return dict(**metricas_contagem([r['real'] for r in rows],[r['previsto'] for r in rows]),**taxas_deteccao(tp,fp,fn))
    res=dict(config=dict(imgsz=1280,iou=.7,conf=.15),teste=met(linhas),teste_8_ou_mais=met([r for r in linhas if r['real']>=8]),protocolo='Preset existente congelado antes do treino; comparacao adicional apos teste, sem busca de novos parametros e sem novo teste independente')
    out.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8');pd.DataFrame(linhas).to_csv(out.with_suffix('.csv'),index=False)
    print(json.dumps(res,indent=2))


if __name__=='__main__':main()
