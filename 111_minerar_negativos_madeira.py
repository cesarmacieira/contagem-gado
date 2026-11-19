"""Busca exemplos dificeis somente em treino; nao rotula candidatos automaticamente."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib,json
import cv2,numpy as np,pandas as pd,torch
from PIL import Image,ImageDraw,ImageOps
from ultralytics import YOLO
import contagem as ct
from recuperacao_multiescala import sobreposicoes

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/candidatos_madeira_treino_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'candidatos.csv').exists():raise ValueError('Candidatos preservados')
    for folder in ['nucleos','contextos']:(OUT/folder).mkdir(exist_ok=True)
    sp=spec_from_file_location('fontes',ROOT/'48_especializar_drone.py');m=module_from_spec(sp);sp.loader.exec_module(m)
    df=m.catalogo();train=df[(df.conjunto=='train')&df.arquivo.str.startswith('Jalogny__')]
    selected=train.sample(frac=1,random_state=1120).groupby('grupo',sort=True).head(4)
    assert set(selected.grupo).isdisjoint(df[df.conjunto!='train'].grupo)
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(dict(fonte='Somente train/Jalogny, ate 4 fotos por voo',
        treino=False,usa_coordenadas_val=False,rotulagem_automatica=False,exige_revisao_visual=True),indent=2),encoding='utf-8')
    torch.set_num_threads(4);model=YOLO(str(ROOT/'data/processed/modelo_gado.pt'));pool=[]
    for r in selected.itertuples():
        im=cv2.imread(r.img);h,w=im.shape[:2];ref=ct.yolo_para_pixels(Path(r.rot).read_text().splitlines(),w,h)
        b,s=ct.detectar(model,im,conf_min=.25,iou_local=.7,iou_nms=.5,device=0,classes=[0])
        choices=[]
        for i,box in enumerate(b):
            _,ios=sobreposicoes(box,ref)
            if (ios>.03).any() or (box[2:]-box[:2]).min()<12:continue
            choices.append((float(s[i]),box))
        for score,box in sorted(choices,key=lambda a:-a[0])[:3]:pool.append(dict(origem=r.img,grupo=r.grupo,score=score,box=box.tolist()))
        print('Triagem treino',r.arquivo,'candidatos',len(choices),flush=True)
    pool=sorted(pool,key=lambda a:-a['score'])[:36];rows=[]
    canvas=Image.new('RGB',(1200,((len(pool)+5)//6)*180),'white');draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(pool):
        ident=f'M{i+1:02d}';source=Path(r['origem']);im=Image.open(source).convert('RGB');b=r['box'];x1,y1,x2,y2=[int(round(v)) for v in b]
        crop=im.crop((x1,y1,x2,y2));crop.save(OUT/'nucleos'/(ident+'.png'))
        pad=max(96,int(max(x2-x1,y2-y1)*1.5));context=im.crop((max(0,x1-pad),max(0,y1-pad),min(im.width,x2+pad),min(im.height,y2+pad)))
        context.save(OUT/'contextos'/(ident+'.png'))
        thumb=ImageOps.contain(crop,(190,145));x,y=(i%6)*200,(i//6)*180;canvas.paste(thumb,(x+(200-thumb.width)//2,y+25));draw.text((x+5,y+5),f'{ident} s={r["score"]:.2f}',fill='black')
        rows.append(dict(id=ident,origem=str(source),grupo=r['grupo'],conjunto='train',sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            xyxy=json.dumps([x1,y1,x2,y2]),score=r['score'],nucleo=str(OUT/'nucleos'/(ident+'.png')),contexto=str(OUT/'contextos'/(ident+'.png')),revisado=False))
    pd.DataFrame(rows).to_csv(OUT/'candidatos.csv',index=False);canvas.save(OUT/'triagem.jpg');print('Candidatos',len(rows),flush=True)


if __name__=='__main__':main()
