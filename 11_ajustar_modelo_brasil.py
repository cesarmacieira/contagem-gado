"""Ajuste isolado com novos rotulos e replay; nunca sobrescreve o modelo do app."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent


def copiar(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists(): return
    shutil.copy2(src,dst)  # Snapshot independente: revisar a fonte nao altera um treino passado.


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epocas',type=int,default=20)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--experimento',default='brasil_lote_01_v2')
    ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--freeze',type=int,default=10)
    ap.add_argument('--warmup-bias-lr',type=float,default=.1)
    ap.add_argument('--complemento',action='store_true')
    a=ap.parse_args()
    import torch
    from ultralytics import YOLO
    assert torch.cuda.is_available(), 'Este experimento requer CUDA'
    base=ROOT/'data/brasil_revisado'
    if Path(a.experimento).name!=a.experimento: ap.error('Nome de experimento sem caminho')
    out=ROOT/'data/experimentos'/a.experimento
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'treino/weights/best.pt').exists(), 'Experimento ja executado; preserve os resultados'
    df=pd.read_csv(base/'manifesto.csv')
    df['base']=str(base)
    if a.complemento:
        extra=pd.read_csv(ROOT/'data/brasil_lote_02/revisado/manifesto.csv')
        df=pd.concat([df,extra[extra.conjunto=='train']],ignore_index=True)
    assert df.groupby('grupo').conjunto.nunique().max()==1
    rng=np.random.default_rng(42)
    antigos=list((ROOT/'data/yolo/images/train').glob('*.jpg'))
    # Replay balanceado entre fontes; somente o split antigo de treino.
    dominios={p.name.split('__')[0] for p in antigos}
    replay=[]
    for d in sorted(dominios):
        cand=[p for p in antigos if p.name.split('__')[0]==d]
        replay.extend(rng.choice(cand,min(8,len(cand)),replace=False).tolist())
    registros=[]
    for i,p in enumerate(replay):
        label=ROOT/'data/yolo/labels/train'/(p.stem+'.txt')
        if not label.exists(): continue
        nome=f'replay_{i:04d}.jpg'
        copiar(p,out/'dataset/images/train'/nome)
        copiar(label,out/'dataset/labels/train'/(Path(nome).stem+'.txt'))
        registros.append(dict(arquivo=nome,origem=str(p),conjunto='train',tipo='replay'))
    for r in df.itertuples():
        if r.conjunto=='test': continue  # Nenhuma imagem de teste acessada pelo treino.
        vezes=4 if r.conjunto=='train' else 1
        for k in range(vezes):
            nome=f'novo_{k}_{r.arquivo}'
            origem=Path(r.base)
            copiar(origem/'images'/r.arquivo,out/'dataset/images'/r.conjunto/nome)
            copiar(origem/'labels'/(Path(r.arquivo).stem+'.txt'),out/'dataset/labels'/r.conjunto/(Path(nome).stem+'.txt'))
            registros.append(dict(arquivo=nome,origem=str(origem/'images'/r.arquivo),conjunto=r.conjunto,tipo='novo_repetido_no_train' if vezes>1 else 'novo'))
    pd.DataFrame(registros).to_csv(out/'proveniencia_treino.csv',index=False)
    cfg=out/'dataset.yaml'
    cfg.write_text(f'path: {out.as_posix()}/dataset\ntrain: images/train\nval: images/val\nnames:\n  0: bovino\n',encoding='utf-8')
    pesos=ROOT/'data/processed/modelo_gado.pt'
    (out/'protocolo.json').write_text(json.dumps(dict(modelo_original_sha256=hashlib.sha256(pesos.read_bytes()).hexdigest(),replay=len(replay),novas_train=int((df.conjunto=='train').sum()),oversampling_train=4,val=int((df.conjunto=='val').sum()),test_intocado=True,epocas=a.epocas,criterio_checkpoint='mAP50-95 somente validacao nova',limitacao='Amostra pequena, primeira revisao visual do assistente'),indent=2),encoding='utf-8')
    print('Treino GPU:',torch.cuda.get_device_name(0),'replay:',len(replay),'novas:',len(df),flush=True)
    YOLO(str(pesos)).train(data=str(cfg),project=str(out),name='treino',exist_ok=False,
        epochs=a.epocas,imgsz=1024,batch=a.batch,device=0,workers=2,optimizer='AdamW',freeze=a.freeze,
        lr0=a.lr,lrf=.1,patience=7,warmup_epochs=2,warmup_bias_lr=a.warmup_bias_lr,weight_decay=.0005,
        mosaic=.25,close_mosaic=5,mixup=0,copy_paste=0,flipud=0,fliplr=.5,
        scale=.3,translate=.05,degrees=0,hsv_s=.4,seed=42,deterministic=True,
        plots=True,save=True,cache=False,amp=True,verbose=False)


if __name__=='__main__':
    main()
