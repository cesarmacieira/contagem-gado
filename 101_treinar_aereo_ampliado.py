"""Novo especialista com cobertura de treino ampliada, sem consultar teste."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib,json,shutil
import cv2,numpy as np,pandas as pd,torch
from ultralytics import YOLO
import contagem as ct

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/aereo_ampliado_v2'


def preparar():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocolo_antes_treino.json').exists():return
    sp=spec_from_file_location('dados',ROOT/'48_especializar_drone.py');m=module_from_spec(sp);sp.loader.exec_module(m)
    df=m.catalogo();train=df[df.conjunto=='train'];valgr=set(df[df.conjunto!='train'].grupo)
    assert not set(train.grupo)&valgr
    candidates=[]
    for r in train.itertuples():
        im=cv2.imread(r.img);h,w=im.shape[:2];bb=ct.yolo_para_pixels(Path(r.rot).read_text().splitlines(),w,h)
        positive=[];negative=[]
        for x,y in ct.grade_tiles(w,h,1024,.2):
            tw,th=min(1024,w-x),min(1024,h-y)
            if ct.tile_tem_fragmento_sem_rotulo(bb,x,y,tw,th):continue
            lines=ct.rotulos_do_tile(bb,x,y,tw,th)
            item=dict(original=r.img,grupo=r.grupo,x=x,y=y,linhas=lines,n=len(lines),stem=r.arquivo)
            (positive if lines else negative).append(item)
        # Cover every original before adding further dense crops.
        positive.sort(key=lambda q:len(q['linhas']),reverse=True)
        for rank,item in enumerate(positive[:6]):item['rank']=rank;candidates.append(item)
        if negative:
            item=negative[len(negative)//2];item['rank']=6;candidates.append(item)
    # Deterministic round-robin by original, preventing a few flights dominating.
    rng=np.random.default_rng(1101);rng.shuffle(candidates);candidates.sort(key=lambda q:q['rank'])
    chosen=candidates[:1200]
    for folder in ['images/train','labels/train','images/val','labels/val']:(OUT/'dataset'/folder).mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,r in enumerate(chosen):
        im=cv2.imread(r['original']);x,y=r['x'],r['y'];name=f'train_{i:04d}'
        cv2.imwrite(str(OUT/'dataset/images/train'/(name+'.jpg')),im[y:y+1024,x:x+1024],[cv2.IMWRITE_JPEG_QUALITY,95])
        (OUT/'dataset/labels/train'/(name+'.txt')).write_text('\n'.join(r['linhas'])+'\n',encoding='utf-8')
        rows.append({k:v for k,v in r.items() if k!='linhas'}|dict(id=name))
    previous=ROOT/'data/experimentos/especialista_drone_v1/dataset'
    for typ in ['images','labels']:
        for f in (previous/typ/'val').iterdir():shutil.copy2(f,OUT/'dataset'/typ/'val'/f.name)
    pd.DataFrame(rows).to_csv(OUT/'treino_manifesto.csv',index=False)
    (OUT/'dataset.yaml').write_text(f'path: {OUT.as_posix()}/dataset\ntrain: images/train\nval: images/val\nnames:\n  0: vaca\n',encoding='utf-8')
    protocol=dict(train_tiles=len(rows),train_originais=len(set(r['original'] for r in rows)),train_grupos=len(set(r['grupo'] for r in rows)),
        bovinos_treino=sum(r['n'] for r in rows),val_tiles=96,epochs=20,imgsz=1024,lr0=.0002,seed=1102,
        pesos_iniciais_sha256=hashlib.sha256((ROOT/'data/processed/modelo_gado.pt').read_bytes()).hexdigest(),
        campanario_usado=False,teste_usado=False,criterio='Best e last por validacao: reduzir FP+FN e MAE; exigir recall e precisao nao inferiores a referencia atual para promover. Comparar cenas brasileiras conhecidas somente depois da escolha. Nao promove automaticamente.')
    (OUT/'protocolo_antes_treino.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');print(json.dumps(protocol,indent=2),flush=True)


def main():
    torch.set_num_threads(4);assert torch.cuda.is_available();preparar()
    if (OUT/'treino/weights/last.pt').exists():
        progress=pd.read_csv(OUT/'treino/results.csv')
        if int(progress.epoch.iloc[-1])>=20:return
        YOLO(str(OUT/'treino/weights/last.pt')).train(resume=True,device=0,workers=0,batch=16,cache="ram")
    else:
        YOLO(str(ROOT/'data/processed/modelo_gado.pt')).train(data=str(OUT/'dataset.yaml'),project=str(OUT),name='treino',
            epochs=20,imgsz=1024,batch=8,device=0,workers=0,optimizer='AdamW',lr0=.0002,lrf=.1,
            warmup_epochs=2,warmup_bias_lr=0,patience=20,mosaic=.2,close_mosaic=5,scale=.3,translate=.08,
            fliplr=.5,flipud=.5,hsv_v=.5,seed=1102,deterministic=True,cache=False,amp=True,
            save_period=5,plots=False,verbose=False)


if __name__=='__main__':main()
