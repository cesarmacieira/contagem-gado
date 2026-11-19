"""Diagnostico de transferencia, explicitamente exploratorio e sem retreino."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch
import contagem as ct
import densidade

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/denso_avaliacao_v1'


def main():
    torch.set_num_threads(4)
    resultado=json.loads((OUT/'resultados.json').read_text(encoding='utf-8'))
    assert resultado['escolha_congelada']['escolha']=='densidade'
    rede=densidade.carregar(ROOT/'data/experimentos/denso_densidade_v1','cuda')
    detector=YOLO(str(ROOT/'data/processed/modelo_gado.pt'))
    gab=pd.read_csv(ROOT/'data/teste_brasil/gabarito.csv');linhas=[]
    (OUT/'brasil_diagnostico').mkdir(exist_ok=True)
    for r in gab.itertuples():
        img=cv2.imread(str(ROOT/'data/teste_brasil'/r.arquivo));n,mapa=densidade.prever(rede,img,'cuda');n=max(0,int(np.rint(n)))
        b,s=ct.detectar_inteira(detector,img,imgsz=1280,iou_nms=.7,device=0)
        anterior=len(ct.filtrar(b,s,.15)[0])
        erro=lambda p:max(r.minimo-p,0,p-r.maximo)
        linhas.append(dict(arquivo=r.arquivo,minimo=r.minimo,maximo=r.maximo,detector_experimental=anterior,densidade=n,erro_detector=erro(anterior),erro_densidade=erro(n)))
        calor=cv2.resize(mapa,img.shape[1::-1]);calor=cv2.applyColorMap(np.uint8(255*calor/max(calor.max(),1e-8)),cv2.COLORMAP_TURBO)
        vis=cv2.addWeighted(img,.65,calor,.35,0);cv2.imwrite(str(OUT/'brasil_diagnostico'/r.arquivo),vis)
    df=pd.DataFrame(linhas);df.to_csv(OUT/'brasil_exploratorio.csv',index=False)
    resumo=dict(exploratorio=True,usado_para_treinar_ou_calibrar=False,detector=dict(dentro=int((df.erro_detector==0).sum()),MAE_intervalo=float(df.erro_detector.mean())),densidade=dict(dentro=int((df.erro_densidade==0).sum()),MAE_intervalo=float(df.erro_densidade.mean())),limite='As fotos e os intervalos ja foram usados em avaliacoes anteriores. Resultado exploratorio, nao teste brasileiro independente.')
    (OUT/'transferencia_brasil.json').write_text(json.dumps(resumo,ensure_ascii=False,indent=2),encoding='utf-8')
    print(df.to_string(index=False));print(json.dumps(resumo,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
