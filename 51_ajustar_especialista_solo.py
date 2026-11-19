"""Ajuste de solo com caixas brasileiras novas; apenas validacao, sem teste.

Preserva os modelos de drone/curral e nao promove pesos no app. A validacao
tem poucos grupos e nao substitui um teste brasileiro denso independente.
"""
from pathlib import Path
import hashlib
import json
import random
import re
import shutil
import unicodedata

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao
from integracao_denso import conferir, ler_csv

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'data/brasil_denso_novo'
EXP = ROOT / 'data/experimentos/especialista_solo_v1'
VAL = ROOT / 'data/experimentos/solo_denso_validacao_v1'
PESO = ROOT / 'data/processed/modelo_gado.pt'
CONFS = (.10,.15,.20,.25,.30,.40,.50,.60)


def normalizar(s):
    s = re.sub('<[^>]+>','',str(s))
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower())


def preparar():
    status = conferir(BASE)
    if not status['pronto']:
        raise ValueError(f'Lote inconsistente: {status.get("erros")}')
    manifest = ler_csv(BASE / 'manifesto.csv')
    aprovados = {r['arquivo']:r for r in status['registros']}
    autores_reservados = {normalizar(r['autor']) for r in manifest if r['conjunto'] in ('val','test')}
    records, listas = [], {'train':[], 'val':[]}
    for pasta in ('images','labels'):
        for split in listas:
            (EXP / 'dataset' / pasta / split).mkdir(parents=True,exist_ok=True)

    def incluir(img, rot, nome, split, repeticoes, grupo, fonte, sha=None):
        destino = EXP / 'dataset/images' / split / (nome+Path(img).suffix.lower())
        label = EXP / 'dataset/labels' / split / (nome+'.txt')
        shutil.copy2(img,destino)
        if sha is not None and hashlib.sha256(destino.read_bytes()).hexdigest() != sha:
            raise ValueError('Imagem alterada durante congelamento')
        linhas = ct.linhas_yolo_unicas(Path(rot).read_text(encoding='utf-8-sig').splitlines())
        im = cv2.imread(str(destino))
        h,w = im.shape[:2]
        caixas = ct.yolo_para_pixels(linhas,w,h)
        if len(caixas) and ((caixas < -.01).any() or (caixas[:,[0,2]] > w+.01).any() or (caixas[:,[1,3]] > h+.01).any()):
            raise ValueError(f'Caixa fora da imagem: {nome}')
        label.write_text('\n'.join(linhas)+ ('\n' if linhas else ''),encoding='utf-8')
        listas[split].extend([destino.as_posix()]*repeticoes)
        records.append(dict(nome=nome,original=str(img),grupo=grupo,conjunto=split,repeticoes=repeticoes,
            n_bovinos=len(linhas),fonte=fonte,sha256=hashlib.sha256(destino.read_bytes()).hexdigest()))

    for r in manifest:
        if r['arquivo'] not in aprovados or r['conjunto'] != 'train':
            continue
        a = aprovados[r['arquivo']]
        incluir(BASE / 'images' / r['arquivo'],BASE / 'labels' / (Path(r['arquivo']).stem+'.txt'),
            'novo__'+Path(r['arquivo']).stem,'train',6,r['grupo'],'brasil_denso_novo',a['sha256'])
    antigos = ROOT / 'data/brasil_dataset_v2'
    for r in ler_csv(antigos / 'manifesto.csv'):
        if r['conjunto'] != 'train' or normalizar(r['autor']) in autores_reservados:
            continue
        incluir(antigos / 'images' / r['arquivo'],antigos / 'labels' / (Path(r['arquivo']).stem+'.txt'),
            'br_antigo__'+Path(r['arquivo']).stem,'train',1,'antigo/'+r['grupo'],'brasil_antigo_train')
    # Replay apenas de fotos de solo do treino antigo; fonte externa com falha
    # de rotulagem identificada (detectors_cows) nao e utilizada.
    rng = random.Random(1121)
    opcoes = []
    for p in sorted((ROOT / 'data/yolo/images/train').glob('foto_openimages__*.jpg')):
        rot = ROOT / 'data/yolo/labels/train' / (p.stem+'.txt')
        n = len(ct.linhas_yolo_unicas(rot.read_text(encoding='utf-8-sig').splitlines()))
        opcoes.append((p,rot,n))
    densas = [r for r in opcoes if r[2] >= 5]
    poucas = [r for r in opcoes if r[2] < 5]
    rng.shuffle(densas)
    rng.shuffle(poucas)
    for p,rot,n in densas[:96]+poucas[:32]:
        incluir(p,rot,'replay__'+p.stem,'train',1,'oi/'+p.stem,'openimages_train')
    val = json.loads((VAL / 'catalogo_val.json').read_text(encoding='utf-8'))
    for r in val:
        incluir(VAL / 'validacao/images' / r['arquivo'],VAL / 'validacao/labels' / (Path(r['arquivo']).stem+'.txt'),
            'val__'+Path(r['arquivo']).stem,'val',1,r['grupo'],'nova_validacao',r['sha256'])
    df = pd.DataFrame(records)
    assert df.groupby('grupo').conjunto.nunique().max() == 1
    assert df.groupby('sha256').conjunto.nunique().max() == 1
    for split,paths in listas.items():
        (EXP / 'dataset' / (split+'.txt')).write_text('\n'.join(paths)+'\n',encoding='utf-8')
    (EXP / 'dataset.yaml').write_text(f'path: {EXP.as_posix()}/dataset\ntrain: train.txt\nval: val.txt\nnames:\n  0: vaca\n',encoding='utf-8')
    df.to_csv(EXP / 'catalogo.csv',index=False)
    protocolo = dict(visao='solo_denso',pesos_iniciais_sha256=hashlib.sha256(PESO.read_bytes()).hexdigest(),
        epochs=12,imgsz=1280,flipud=0,repeticoes_novas=6,confiancas=CONFS,
        n_imagens_train=int((df.conjunto=='train').sum()),n_amostras_train_com_repeticao=len(listas['train']),
        caixas_novas_unicas=int(df[df.fonte=='brasil_denso_novo'].n_bovinos.sum()),
        grupos_validacao=int(df[df.conjunto=='val'].grupo.nunique()),
        fingerprint_fonte=status['snapshot_sha256'],consulta_inferencia_teste=False,promove_app=False,
        exclusoes='Nenhuma imagem de val/test usada no treino. Excluidos autores reservados das brasileiras antigas. Fonte detectors_cows excluida.',
        limite='Validacao pequena de um grupo. Melhora nela nao comprova generalizacao. Fonte Open Images nao tem auditoria visual integral.')
    (EXP / 'protocolo.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    print(json.dumps(protocolo,indent=2),flush=True)


def avaliar(peso, cfg):
    modelo = YOLO(str(peso))
    val = json.loads((VAL / 'catalogo_val.json').read_text(encoding='utf-8'))
    linhas = []
    for r in val:
        im = cv2.imread(str(VAL / 'validacao/images' / r['arquivo']))
        h,w = im.shape[:2]
        ref = ct.yolo_para_pixels((VAL / 'validacao/labels' / (Path(r['arquivo']).stem+'.txt')).read_text().splitlines(),w,h)
        b,s = ct.detectar_inteira(modelo,im,imgsz=cfg['imgsz'],iou_nms=.7,device=0,classes=[0],augment=cfg['augment'])
        for conf in CONFS:
            bb,ss = ct.filtrar(b,s,conf)
            linhas.append(dict(peso=str(peso),imgsz=cfg['imgsz'],augment=cfg['augment'],conf=conf,arquivo=r['arquivo'],
                real=len(ref),previsto=len(bb),**comparar_caixas(ref,bb,ss)))
    d = pd.DataFrame(linhas)
    resumo = []
    for conf,g in d.groupby('conf'):
        tp,fp,fn = (int(g[k].sum()) for k in ('TP','FP','FN'))
        resumo.append(dict(peso=str(peso),imgsz=cfg['imgsz'],augment=cfg['augment'],conf=float(conf),
            **metricas_contagem(g.real,g.previsto),**taxas_deteccao(tp,fp,fn),TP=tp,FP=fp,FN=fn))
    return resumo,linhas


def main():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    EXP.mkdir(parents=True,exist_ok=True)
    if (EXP / 'resultado_validacao.json').exists():
        print('Experimento ja concluido; preservado',flush=True)
        return
    if not (EXP / 'protocolo.json').exists():
        preparar()
    last = EXP / 'treino/weights/last.pt'
    if not last.exists():
        YOLO(str(PESO)).train(data=str(EXP / 'dataset.yaml'),project=str(EXP),name='treino',exist_ok=False,
            epochs=12,imgsz=1280,batch=8,device=0,workers=2,optimizer='AdamW',lr0=.00008,lrf=.2,
            warmup_epochs=1,warmup_bias_lr=0,patience=12,mosaic=.3,close_mosaic=2,scale=.3,translate=.05,
            fliplr=.5,flipud=0,degrees=0,hsv_s=.3,hsv_v=.2,mixup=0,copy_paste=0,
            seed=1121,deterministic=True,cache=False,amp=True,save_period=4,plots=False,verbose=False)
    elif int(pd.read_csv(EXP / 'treino/results.csv').epoch.iloc[-1]) < 12:
        YOLO(str(last)).train(resume=True,device=0,workers=2)
    resultados,linhas = [],[]
    for peso in [PESO,EXP / 'treino/weights/best.pt',last]:
        for cfg in [dict(imgsz=1280,augment=False),dict(imgsz=1920,augment=False),dict(imgsz=1280,augment=True)]:
            m,rows = avaliar(peso,cfg)
            resultados.extend(m)
            linhas.extend(rows)
        print('Validado',peso,flush=True)
    tabela = pd.DataFrame(resultados)
    tabela['erro_instancias'] = (tabela.FP+tabela.FN) / tabela.n_imagens
    tabela['objetivo'] = tabela.MAE+tabela.erro_instancias
    tabela = tabela.sort_values(['objetivo','MAE','F1_caixas'],ascending=[True,True,False])
    tabela.to_csv(EXP / 'resumo_validacao.csv',index=False)
    pd.DataFrame(linhas).to_csv(EXP / 'por_imagem_validacao.csv',index=False)
    base = tabela[(tabela.peso==str(PESO)) & (tabela.imgsz==1280) & (~tabela.augment) & (tabela.conf==.15)].iloc[0]
    novos = tabela[tabela.peso!=str(PESO)]
    candidato = novos.iloc[0]
    resultado = dict(visao='solo_denso',referencia=base.to_dict(),candidato=candidato.to_dict(),
        melhor_geral=tabela.iloc[0].to_dict(),melhora_MAE_e_F1_na_validacao=bool(candidato.MAE<base.MAE and candidato.F1_caixas>=base.F1_caixas),
        promovido=False,teste_executado=False,grupos_validacao=1,
        limite='Comparacao exploratoria em duas imagens da mesma origem. Ainda exige novas cenas e teste positivo separado.')
    (EXP / 'resultado_validacao.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    print(json.dumps(resultado,indent=2),flush=True)


if __name__ == '__main__':
    main()
