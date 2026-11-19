"""Treino supervisionado com caixas publicadas auditadas e replay, em pasta isolada."""
from pathlib import Path
import json,hashlib,shutil,re
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/detectors_cows'
OUT=ROOT/'data/experimentos/detector_denso_v3'


def copiar(img,rot,nome,split):
    for pasta in ('images','labels'):(OUT/'dataset'/pasta/split).mkdir(parents=True,exist_ok=True)
    shutil.copy2(img,OUT/'dataset/images'/split/(nome+'.jpg'))
    shutil.copy2(rot,OUT/'dataset/labels'/split/(nome+'.txt'))


def selecionar(df,max_denso,max_outros,seed):
    rng=np.random.default_rng(seed)
    df=df.copy();df['sorteio']=rng.random(len(df))
    # Limitar sequencias, priorizando variedade antes de quantidade.
    candidatos=df.sort_values('sorteio').groupby('grupo',group_keys=False).head(12)
    a=candidatos[candidatos.n_bovinos>=5].sort_values(['n_bovinos','sorteio'],ascending=[False,True]).head(max_denso)
    b=candidatos[(candidatos.n_bovinos<5)&(~candidatos.stem.isin(a.stem))].sort_values('sorteio').head(max_outros)
    return pd.concat([a,b]).copy()


def main():
    torch.set_num_threads(4);assert torch.cuda.is_available()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'treino/weights/best.pt').exists(),'Preserve o experimento anterior'
    src=pd.read_csv(BASE/'divisao_linhagens.csv')
    assert src.groupby('grupo').conjunto.nunique().max()==1
    novo_train=selecionar(src[src.conjunto=='train'],600,240,42)
    novo_val=selecionar(src[src.conjunto=='val'],120,80,43)
    # Todo o teste novo e conservado, nunca copiado para a pasta de treino.
    registros=[]
    for split,df in [('train',novo_train),('val',novo_val)]:
        for r in df.itertuples():
            nome='pub_'+r.stem;copiar(r.img,r.rot,nome,split)
            registros.append(dict(arquivo=nome+'.jpg',img=r.img,rot=r.rot,grupo=r.grupo,conjunto=split,n_bovinos=int(r.n_bovinos),fonte='caixas_publicadas'))
    brasil=pd.read_csv(ROOT/'data/brasil_dataset_v2/manifesto.csv')
    for r in brasil[brasil.conjunto.isin(['train','val'])].itertuples():
        im=ROOT/'data/brasil_dataset_v2/images'/r.arquivo;rot=ROOT/'data/brasil_dataset_v2/labels'/(Path(r.arquivo).stem+'.txt')
        for k in range(6 if r.conjunto=='train' else 1):
            nome=f'brasil_{k}_'+Path(r.arquivo).stem;copiar(im,rot,nome,r.conjunto)
        registros.append(dict(arquivo=nome+'.jpg',img=str(im),rot=str(rot),grupo=r.grupo,conjunto=r.conjunto,n_bovinos=int(r.n_bovinos),fonte='brasil_revisado'))
    rng=np.random.default_rng(44);antigos=list((ROOT/'data/yolo/images/train').glob('*.jpg'))
    dominios=sorted({p.name.split('__')[0] for p in antigos});replay=[]
    for dom in dominios:
        subset=[p for p in antigos if p.name.split('__')[0]==dom];replay.extend(rng.choice(subset,min(32,len(subset)),replace=False).tolist())
    for i,p in enumerate(replay):
        rot=ROOT/'data/yolo/labels/train'/(p.stem+'.txt');nome=f'replay_{i:04d}';copiar(p,rot,nome,'train')
    pd.DataFrame(registros).to_csv(OUT/'catalogo_treino_validacao.csv',index=False)
    teste=src[src.conjunto=='test'];teste.to_csv(OUT/'teste_congelado.csv',index=False)
    config=OUT/'dataset.yaml';config.write_text(f'path: {OUT.as_posix()}/dataset\ntrain: images/train\nval: images/val\nnames:\n  0: bovino\n',encoding='utf-8')
    pesos=ROOT/'data/processed/modelo_gado.pt'
    protocolo=dict(pesos_iniciais_sha256=hashlib.sha256(pesos.read_bytes()).hexdigest(),imagens_publicadas_train=len(novo_train),caixas_publicadas_train=int(novo_train.n_bovinos.sum()),imagens_publicadas_val=len(novo_val),teste_imagens=len(teste),teste_maximo=int(teste.n_bovinos.max()),brasil_train=22,repeticoes_brasil_train=6,brasil_val=9,replay=len(replay),criterio='Checkpoints e confianca escolhidos na validacao; teste apenas depois',fontes='Caixas reais publicadas e revisao visual brasileira, sem usar os pontos como caixas')
    (OUT/'protocolo.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8');print(json.dumps(protocolo,indent=2),flush=True)
    YOLO(str(pesos)).train(data=str(config),project=str(OUT),name='treino',exist_ok=False,
        epochs=20,imgsz=1280,batch=8,device=0,workers=2,optimizer='AdamW',lr0=.00012,lrf=.1,
        warmup_epochs=2,warmup_bias_lr=0,patience=10,weight_decay=.0005,
        mosaic=.5,close_mosaic=5,scale=.4,translate=.1,fliplr=.5,flipud=0,degrees=0,
        mixup=0,copy_paste=0,seed=42,deterministic=True,cache=False,amp=True,
        save_period=5,max_det=1000,plots=True,verbose=False)


if __name__=='__main__':main()
