"""Compara veto de cavalo/ovelha; toda avaliacao e regressao conhecida."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas
from conferencia_especies import manter_bovinos
from verificador_visual import carregar_binario,confirmar,mascara_confirmacao

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/conferencia_especies_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():
        raise ValueError('Resultado existente preservado')
    peso=ROOT/'data/modelos_candidatos/yolo11x.pt'
    protocolo=dict(classes_coco=[17,18],classes=['horse','sheep'],conf=.5,imgsz=1280,
        iou_veto=.6,peso_sha256=hashlib.sha256(peso.read_bytes()).hexdigest(),
        avaliacao='Somente regressao conhecida; sem teste independente novo',promover=False)
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    torch.set_num_threads(4)
    detector=YOLO(str(ROOT/'data/processed/modelo_gado.pt'))
    geral=YOLO(str(peso))
    assert geral.names[17]=='horse' and geral.names[18]=='sheep'
    cfg=json.loads((ROOT/'data/modelos/verificador_solo_v1/config.json').read_text())
    ver=carregar_binario(ROOT/cfg['extrator'],ROOT/cfg['classificador'],cfg['classificador_sha256'])
    amostras=[]
    d=pd.read_csv(ROOT/'data/brasil_dataset_v2/manifesto.csv')
    for r in d[d.conjunto=='test'].itertuples():
        amostras.append((ROOT/'data/brasil_dataset_v2/images'/r.arquivo,
            ROOT/'data/brasil_dataset_v2/labels'/(Path(r.arquivo).stem+'.txt'),'solo_poucos'))
    for nome in ('bdn_19.jpg','bdn_21.jpg'):
        base=ROOT/'data/experimentos/solo_piloto_separado_v1/teste'
        amostras.append((base/'images'/nome,base/'labels'/(Path(nome).stem+'.txt'),'solo_denso'))
    base=ROOT/'data/experimentos/solo_denso_validacao_v2/validacao'
    for nome in ('bdn_08_c1.jpg','bdn_08_c2.jpg'):
        amostras.append((base/'images'/nome,base/'labels'/(Path(nome).stem+'.txt'),'solo_denso'))
    amostras.append((ROOT/'data/busca_teste_denso_independente/bernard_28059238281.jpg',None,'solo_denso'))
    rows=[]
    for imagem,rotulo,rota in amostras:
        im=cv2.imread(str(imagem));h,w=im.shape[:2]
        dense=rota=='solo_denso'
        b,s=ct.detectar_inteira(detector,im,imgsz=1280 if dense else 1024,
            iou_nms=.7 if dense else .5,classes=[0],augment=False,device=0)
        b,s=ct.filtrar(b,s,.15 if dense else .30)
        if dense:
            v=confirmar(ver,im,b)
            mask,_=mascara_confirmacao(b,v,cfg['limiar'],cfg.get('min_lado_px',0))
            b,s=b[mask],s[mask]
        r=geral.predict(im,imgsz=1280,conf=.5,classes=[17,18],augment=False,device=0,verbose=False)[0]
        outras=r.boxes.xyxy.cpu().numpy()
        mask=manter_bovinos(b,outras,.6)
        ref=ct.yolo_para_pixels(rotulo.read_text().splitlines(),w,h) if rotulo else None
        for veto,bb,ss in ((False,b,s),(True,b[mask],s[mask])):
            row=dict(arquivo=imagem.name,rota=rota,veto=veto,previsto=len(bb),outras_especies=len(outras))
            if ref is not None:
                row.update(real=len(ref),erro=abs(len(bb)-len(ref)),**comparar_caixas(ref,bb,ss))
            rows.append(row)
        if not rotulo:
            np.savez(OUT/'bernard_caixas.npz',antes=b,scores=s,outras_especies=outras,manter=mask)
            vis=ct.desenhar(im,b[mask]);cv2.imwrite(str(OUT/'bernard_depois.jpg'),vis)
        print(imagem.name,len(b),'->',int(mask.sum()),flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'por_imagem.csv',index=False)
    rot=d[d.real.notna()]
    a=rot[~rot.veto].set_index('arquivo');b=rot[rot.veto].set_index('arquivo')
    resumo=dict(perdeu_tp=bool((b.TP<a.TP).any()),piorou_erro=bool((b.erro>a.erro).any()),
        FP_antes=int(a.FP.sum()),FP_depois=int(b.FP.sum()),
        bernard=d[d.arquivo=='bernard_28059238281.jpg'].astype(object).where(pd.notnull(d),None).to_dict('records'),
        promovido=False,limite='Regressao conhecida e rotulos incompletos no caso Bernard; nao demonstra contagem exata densa.')
    (OUT/'resultado.json').write_text(json.dumps(resumo,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(resumo,indent=2),flush=True)


if __name__=='__main__':
    main()
