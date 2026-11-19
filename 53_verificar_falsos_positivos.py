"""Verificador visual calibrado na validacao; regressao ja conhecida, sem promocao."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao
from verificador_visual import carregar_verificador,pontuar

ROOT = Path(__file__).resolve().parent
VAL = ROOT / 'data/experimentos/solo_denso_validacao_v2'
PILOTO = ROOT / 'data/experimentos/solo_piloto_separado_v1'
OUT = ROOT / 'data/experimentos/verificador_visual_v1'
PESO = ROOT / 'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth'


def indices_correspondentes(ref,b,s):
    ordem = np.argsort(-np.asarray(s),kind='stable')
    positivos,anterior = [],0
    for k,i in enumerate(ordem,1):
        met = comparar_caixas(ref,b[ordem[:k]],s[ordem[:k]])
        if met['TP'] > anterior:
            positivos.append(int(i))
        anterior = met['TP']
    return positivos


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT / 'resultado.json').exists():
        print('Resultado preservado',flush=True)
        return
    verificador = carregar_verificador(PESO)
    val = json.loads((VAL / 'catalogo_val.json').read_text(encoding='utf-8'))
    calibracao,cache = [],{}
    # Detector e limiar ja escolhidos na validacao; nao sao ajustados aqui.
    for r in val:
        im = cv2.imread(str(VAL / 'validacao/images' / r['arquivo']))
        dados = np.load(VAL / 'original/tta1280' / (Path(r['arquivo']).stem+'.npz'))
        b,s = ct.filtrar(dados['caixas'],dados['scores'],.20)
        ref = dados['ref']
        v = pontuar(verificador,im,b)
        positivos = indices_correspondentes(ref,b,s)
        calibracao.extend(v[positivos].tolist())
        cache[r['arquivo']] = (im,ref,b,s,v)
    assert calibracao
    # Margem conservadora: preservar todas as correspondencias positivas da
    # validacao. Nenhum score dos cupinzeiros decide este limiar.
    limiar = float(min(calibracao)*.99)
    protocolo = dict(peso_verificador=str(PESO),sha256=hashlib.sha256(PESO.read_bytes()).hexdigest(),
        classes_imagenet=['ox','water buffalo','bison'],contextos=[1.3,2.0],limiar=limiar,
        criterio='99% do menor score das caixas com correspondencia IoU>=0.5 na validacao',
        positivos_calibracao=len(calibracao),detector=dict(pesos='data/processed/modelo_gado.pt',imgsz=1280,iou=.7,conf=.20,augment=True),
        teste_independente=False,promove_app=False,
        limite='O piloto com cupinzeiros ja foi visto. Esta e regressao de desenvolvimento, nao novo teste independente. Scores ImageNet nao sao probabilidades calibradas de bovino.')
    (OUT / 'escolha_antes_regressao.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    rows = []
    def registrar(origem,nome,im,ref,b,s,v):
        for filtrar in (False,True):
            mask = v>=limiar if filtrar else np.ones(len(b),dtype=bool)
            bb,ss = b[mask],s[mask]
            rows.append(dict(origem=origem,arquivo=nome,verificado=filtrar,real=len(ref),previsto=len(bb),
                **comparar_caixas(ref,bb,ss)))
            cv2.imwrite(str(OUT / (origem+'_'+('verificado_' if filtrar else 'original_')+nome)),ct.desenhar(im,bb,ref=ref))
        pd.DataFrame(dict(x1=b[:,0],y1=b[:,1],x2=b[:,2],y2=b[:,3],conf_detector=s,score_verificador=v,
            mantido=v>=limiar)).to_csv(OUT / (origem+'_'+Path(nome).stem+'.csv'),index=False)
    for nome,args in cache.items():
        registrar('val',nome,*args)
    modelo = YOLO(str(ROOT / 'data/processed/modelo_gado.pt'))
    antigo = json.loads((PILOTO / 'protocolo_antes_teste.json').read_text(encoding='utf-8'))
    for r in antigo['imagens_teste']:
        im = cv2.imread(str(PILOTO / 'teste/images' / r['arquivo']))
        h,w = im.shape[:2]
        ref = ct.yolo_para_pixels((PILOTO / 'teste/labels' / (Path(r['arquivo']).stem+'.txt')).read_text().splitlines(),w,h)
        b,s = ct.detectar_inteira(modelo,im,imgsz=1280,iou_nms=.7,augment=True,classes=[0],device=0)
        b,s = ct.filtrar(b,s,.20)
        v = pontuar(verificador,im,b)
        registrar('regressao_conhecida',r['arquivo'],im,ref,b,s,v)
    dados = pd.DataFrame(rows)
    dados.to_csv(OUT / 'por_imagem.csv',index=False)
    resumo = []
    for (origem,filtrar),d in dados.groupby(['origem','verificado']):
        tp,fp,fn = (int(d[k].sum()) for k in ('TP','FP','FN'))
        resumo.append(dict(origem=origem,verificado=bool(filtrar),**metricas_contagem(d.real,d.previsto),
            **taxas_deteccao(tp,fp,fn),TP=tp,FP=fp,FN=fn))
    (OUT / 'resultado.json').write_text(json.dumps(dict(limiar=limiar,resultados=resumo,promovido=False,
        teste_independente=False,limite=protocolo['limite']),indent=2),encoding='utf-8')
    print(dados.to_string(index=False),flush=True)
    print(pd.DataFrame(resumo)[['origem','verificado','MAE','F1_caixas','TP','FP','FN']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
