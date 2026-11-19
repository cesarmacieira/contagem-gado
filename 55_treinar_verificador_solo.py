"""Verificador binario com bovinos anotados e negativos reais de novos autores.

Usa caracteristicas congeladas EfficientNet; nao transforma previsoes em
positivos. Propostas em fotos revisadas sem bovinos sao negativos verdadeiros.
"""
from pathlib import Path
import hashlib
import json
import joblib

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from ultralytics import YOLO

import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao
from verificador_visual import carregar_verificador,recorte_quadrado
from integracao_denso import ler_csv
from importlib.util import spec_from_file_location,module_from_spec

ROOT = Path(__file__).resolve().parent
NEG = ROOT / 'data/negativos_solo_v1'
POS = ROOT / 'data/experimentos/especialista_solo_v1'
VAL = ROOT / 'data/experimentos/solo_denso_validacao_v2'
OLD = ROOT / 'data/experimentos/solo_piloto_separado_v1'
OUT = ROOT / 'data/experimentos/verificador_binario_v1'
PESO = ROOT / 'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth'


def features(extrator,im,b,blur=False):
    if not len(b):
        return np.empty((0,2560),dtype=np.float32)
    modelo,transform,_ = extrator
    pil = Image.fromarray(im[:,:,::-1])
    patches = []
    for box in b:
        for fator in (1.3,2.0):
            crop = recorte_quadrado(pil,box,fator)
            if blur:
                crop = crop.resize((24,24),Image.Resampling.BICUBIC)
            patches.append(transform(crop))
    encoded = []
    with torch.inference_mode():
        for i in range(0,len(patches),32):
            x = torch.stack(patches[i:i+32]).to('cuda')
            x = modelo.avgpool(modelo.features(x)).flatten(1)
            x = torch.nn.functional.normalize(x,dim=1)
            encoded.append(x.cpu().numpy())
    return np.concatenate(encoded).reshape(len(b),2560)


def propostas(modelo,im):
    b,s = ct.detectar_inteira(modelo,im,imgsz=1280,iou_nms=.7,classes=[0],augment=True,device=0)
    return b,s


def main(negativos_extra=None,contextos=(1.3,2.0)):
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT / 'resultado.json').exists():
        print('Experimento ja concluido',flush=True)
        return
    negativos = [r for r in ler_csv(NEG / 'manifesto.csv') if r['revisado'].lower()=='true']
    assert len(negativos)>=12
    assert pd.DataFrame(negativos).groupby('grupo').conjunto.nunique().max()==1
    assert len({r['sha256'] for r in negativos})==len(negativos)
    extractor = carregar_verificador(PESO)
    detector = YOLO(str(ROOT / 'data/processed/modelo_gado.pt'))
    xs,ys,origens = [],[],[]
    positivos = pd.read_csv(POS / 'catalogo.csv')
    positivos = positivos[(positivos.conjunto=='train') & positivos.fonte.isin(['brasil_denso_novo','brasil_antigo_train'])]
    for r in positivos.itertuples():
        caminhos = list((POS / 'dataset/images/train').glob(r.nome+'.*'))
        assert len(caminhos)==1
        im = cv2.imread(str(caminhos[0]))
        h,w = im.shape[:2]
        b = ct.yolo_para_pixels((POS / 'dataset/labels/train' / (r.nome+'.txt')).read_text().splitlines(),w,h)
        if not len(b):
            continue
        for blur in (False,True):
            x = features(extractor,im,b,blur)
            xs.append(x);ys.extend([1]*len(x));origens.extend([r.nome]*len(x))
    print('Positivos de treino:',sum(ys),flush=True)
    # Usa somente train. Val/test negativos nao participam das caracteristicas
    # de treino nem de uma busca de hiperparametros.
    for r in negativos:
        if r['conjunto']!='train':
            continue
        im = cv2.imread(str(NEG / 'images' / r['arquivo']))
        h,w = im.shape[:2]
        b,s = propostas(detector,im)
        b = b[np.argsort(-s)[:12]]
        grade = np.asarray([[0,0,w,h],[0,0,w/2,h/2],[w/2,0,w,h/2],[0,h/2,w/2,h],[w/2,h/2,w,h]],float)
        b = np.concatenate([b,grade])
        for blur in (False,True):
            x = features(extractor,im,b,blur)
            xs.append(x);ys.extend([0]*len(x));origens.extend([r['arquivo']]*len(x))
    for r in negativos_extra or []:
        if r['conjunto']!='train' or not r.get('revisado',False):
            continue
        im=cv2.imread(r['caminho_absoluto'])
        if im is None:
            raise ValueError('Imagem adicional ilegivel')
        h,w=im.shape[:2]
        rel=np.asarray(r['bbox_rel'],dtype=float)
        b=(rel*np.asarray([w,h,w,h])).reshape(1,4)
        for blur in (False,True):
            x=features(extractor,im,b,blur)
            xs.append(x);ys.extend([0]*len(x));origens.extend(['especie__'+r['id']]*len(x))
    x,y = np.concatenate(xs),np.asarray(ys)
    clf = LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=991)
    clf.fit(x,y)
    clf.contextos_=tuple(contextos)
    joblib.dump(clf,OUT / 'classificador.joblib')
    pd.DataFrame(dict(origem=origens,classe=y)).to_csv(OUT / 'origens_treino.csv',index=False)
    print('Classificador ajustado:',len(y),'amostras, positivos',int(y.sum()),flush=True)
    spec = spec_from_file_location('calib_solo',ROOT / '53_verificar_falsos_positivos.py')
    calib = module_from_spec(spec);spec.loader.exec_module(calib)
    val = json.loads((VAL / 'catalogo_val.json').read_text(encoding='utf-8'))
    cache,anchors = {},[]
    for r in val:
        im = cv2.imread(str(VAL / 'validacao/images' / r['arquivo']))
        d = np.load(VAL / 'original/tta1280' / (Path(r['arquivo']).stem+'.npz'))
        b,s = ct.filtrar(d['caixas'],d['scores'],.20)
        ref = d['ref']
        v = clf.predict_proba(features(extractor,im,b))[:,1] if len(b) else np.empty(0)
        anchors.extend(v[calib.indices_correspondentes(ref,b,s)].tolist())
        cache[r['arquivo']] = (im,ref,b,s,v)
    assert anchors
    limiar = float(min(anchors)*.99)
    protocolo = dict(limiar=limiar,C=1.0,classes=[0,1],contextos=list(contextos),n_amostras_train=len(y),
        extrator_sha256=hashlib.sha256(PESO.read_bytes()).hexdigest(),
        classificador_sha256=hashlib.sha256((OUT / 'classificador.joblib').read_bytes()).hexdigest(),
        detector=dict(pesos='data/processed/modelo_gado.pt',imgsz=1280,iou=.7,augment=True,conf=.20),
        negativos_manifesto_sha256=hashlib.sha256((NEG / 'manifesto.csv').read_bytes()).hexdigest(),
        criterio='Limiar conservador, 99% do menor score dos TPs da validacao positiva. C fixo. Nenhum negativo test participa do ajuste.',
        teste_positivo_novo=False,promove_app=False,
        limites='Bovinos positivos de teste bdn_21 ja conhecidos; somente os quatro negativos test novos ainda nao foram inferidos. Sem prova de rebanho denso novo.')
    (OUT / 'escolha_antes_teste.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    rows = []
    def registrar(origem,nome,im,ref,b,s,v):
        for verificar in (False,True):
            mask = v>=limiar if verificar else np.ones(len(b),dtype=bool)
            bb,ss = b[mask],s[mask]
            rows.append(dict(origem=origem,arquivo=nome,verificado=verificar,real=len(ref),previsto=len(bb),**comparar_caixas(ref,bb,ss)))
            cv2.imwrite(str(OUT / (origem+'_'+str(int(verificar))+'_'+nome)),ct.desenhar(im,bb,ref=ref))
    for nome,args in cache.items():
        registrar('val_positiva',nome,*args)
    for r in negativos:
        if r['conjunto']=='train':
            continue
        im = cv2.imread(str(NEG / 'images' / r['arquivo']))
        b,s = propostas(detector,im)
        b,s = ct.filtrar(b,s,.20)
        v = clf.predict_proba(features(extractor,im,b))[:,1] if len(b) else np.empty(0)
        registrar('neg_'+r['conjunto'],r['arquivo'],im,np.empty((0,4)),b,s,v)
    antigos = json.loads((OLD / 'protocolo_antes_teste.json').read_text())['imagens_teste']
    for r in antigos:
        im = cv2.imread(str(OLD / 'teste/images' / r['arquivo']))
        h,w = im.shape[:2]
        ref = ct.yolo_para_pixels((OLD / 'teste/labels' / (Path(r['arquivo']).stem+'.txt')).read_text().splitlines(),w,h)
        b,s = propostas(detector,im)
        b,s = ct.filtrar(b,s,.20)
        v = clf.predict_proba(features(extractor,im,b))[:,1] if len(b) else np.empty(0)
        registrar('regressao_conhecida',r['arquivo'],im,ref,b,s,v)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'por_imagem.csv',index=False)
    resumo = []
    for (origem,verificado),g in d.groupby(['origem','verificado']):
        tp,fp,fn = (int(g[k].sum()) for k in ('TP','FP','FN'))
        resumo.append(dict(origem=origem,verificado=bool(verificado),**metricas_contagem(g.real,g.previsto),
                          **taxas_deteccao(tp,fp,fn),TP=tp,FP=fp,FN=fn))
    (OUT / 'resultado.json').write_text(json.dumps(dict(limiar=limiar,resultados=resumo,promovido=False,
        limite=protocolo['limites']),indent=2),encoding='utf-8')
    print(d.to_string(index=False),flush=True)
    print(pd.DataFrame(resumo)[['origem','verificado','MAE','F1_caixas','TP','FP','FN']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
