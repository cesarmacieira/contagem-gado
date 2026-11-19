"""Minera candidatos somente de treino; nenhum candidato vira rotulo automaticamente."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/candidatos_negativos_curral_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'candidatos.csv').exists():raise ValueError('Lote existente preservado')
    files=sorted((ROOT/'data/yolo/images/train').glob('colo_topo__*.jpg'))
    rng=np.random.default_rng(1121);ids=rng.permutation(len(files))[:32]
    images=[files[i] for i in ids]
    (OUT/'protocolo_antes_mineracao.json').write_text(json.dumps(dict(
        split='train',imagens=[p.name for p in images],imgsz=1024,conf=.05,iou=.7,
        limite_candidatos=24,criterio='IoU e intersecao/minarea abaixo de .1/.2 contra rotulos existentes; '
        'isso NAO comprova ausencia de bovino: revisao visual obrigatoria.',
        exclusoes='Nenhuma imagem de val/test nem da Fazenda Campanario'),indent=2),encoding='utf-8')
    torch.set_num_threads(4);model=YOLO(str(ROOT/'data/processed/modelo_gado.pt'));candidates=[]
    for p in images:
        im=cv2.imread(str(p));h,w=im.shape[:2]
        label=ROOT/'data/yolo/labels/train'/(p.stem+'.txt')
        ref=ct.yolo_para_pixels(label.read_text().splitlines(),w,h)
        b,s=ct.detectar_inteira(model,im,imgsz=1024,conf_min=.05,iou_nms=.7,device=0,classes=[0],augment=False)
        for j,(box,score) in enumerate(zip(b,s)):
            if min(box[2:]-box[:2])<16:continue
            inter=np.maximum(0,np.minimum(box[2:],ref[:,2:])-np.maximum(box[:2],ref[:,:2])).prod(axis=1)
            area=np.prod(box[2:]-box[:2]);areas=np.prod(ref[:,2:]-ref[:,:2],axis=1)
            io=inter/(area+areas-inter+1e-9);ios=inter/(np.minimum(area,areas)+1e-9)
            if len(ref) and (io.max()>=.1 or ios.max()>=.2):continue
            candidates.append(dict(origem=str(p.relative_to(ROOT)),rotulo=str(label.relative_to(ROOT)),
                indice=j,score=float(score),xyxy=box.tolist(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    # Diversidade de imagens primeiro; no maximo dois candidatos por foto.
    chosen=[];seen={}
    for r in sorted(candidates,key=lambda r:-r['score']):
        if seen.get(r['origem'],0)>=2:continue
        chosen.append(r);seen[r['origem']]=seen.get(r['origem'],0)+1
        if len(chosen)==24:break
    (OUT/'recortes').mkdir(exist_ok=True)
    for j,r in enumerate(chosen):
        im=cv2.imread(str(ROOT/r['origem']));h,w=im.shape[:2];x1,y1,x2,y2=np.round(r['xyxy']).astype(int)
        patch=im[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
        r.update(id=f'C{j+1:02d}',arquivo=f'recortes/C{j+1:02d}.jpg',conjunto='train',
                 classe_revisada='',aprovado=False,nota='Pendente: sombra, parte de animal, bovino real ou outro objeto?')
        cv2.imwrite(str(OUT/r['arquivo']),patch)
    cols=['id','arquivo','origem','rotulo','indice','score','xyxy','sha256','conjunto','classe_revisada','aprovado','nota']
    pd.DataFrame(chosen,columns=cols).to_csv(OUT/'candidatos.csv',index=False)
    print(json.dumps(dict(imagens_mineradas=len(images),candidatos=len(chosen),aprovados=0)))


if __name__=='__main__':main()
