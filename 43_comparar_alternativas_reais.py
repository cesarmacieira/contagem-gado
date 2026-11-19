"""Compara familias locais e TTA em desenvolvimento/validacao ja conhecidos.

O lote novo permanece fora da inferencia. Nenhuma escolha e promovida ao app.
"""
from pathlib import Path
import hashlib
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO, RTDETR

import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/experimentos/alternativas_reais_v1'
CONFS = (.10, .15, .20, .25, .30, .35, .45)


def executar(rtdetr=False, maior=False):
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    pesos = {
        'original': ROOT / 'data/processed/modelo_gado.pt',
        'ajustado': ROOT / 'data/experimentos/detector_denso_v3/treino/weights/best.pt',
        'geral_yolo11s': ROOT / 'yolo11s.pt',
        'geral_yolo26n': ROOT / 'yolo26n.pt',
    }
    if rtdetr:
        pesos = {'rtdetr_l': ROOT / 'data/modelos_candidatos/rtdetr-l.pt'}
    if maior:
        pesos = {'geral_yolo11x': ROOT / 'data/modelos_candidatos/yolo11x.pt'}
    candidatos = [(nome, False) for nome in pesos] + [(nome, True) for nome in pesos if nome not in ('geral_yolo26n', 'rtdetr_l')]
    val = pd.read_csv(ROOT / 'data/brasil_dataset_v2/manifesto.csv')
    val = val[val.conjunto == 'val']
    catalogo = [dict(arquivo=r.arquivo, img=ROOT / 'data/brasil_dataset_v2/images' / r.arquivo,
                    rot=ROOT / 'data/brasil_dataset_v2/labels' / (Path(r.arquivo).stem + '.txt'),
                    origem='validacao_brasil', grupo=r.grupo)
                for r in val.itertuples()]
    gab = pd.read_csv(ROOT / 'data/teste_brasil/gabarito.csv')
    catalogo += [dict(arquivo=r.arquivo, img=ROOT / 'data/teste_brasil' / r.arquivo,
                     minimo=int(r.minimo), maximo=int(r.maximo), origem='desenvolvimento_intervalos', grupo=r.arquivo)
                 for r in gab.itertuples()]
    protocolo = dict(imgsz=1280, iou=.7, confiancas=CONFS, candidatos=candidatos,
        pesos={k: dict(caminho=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in pesos.items()},
        conjunto_novo_consultado=False, teste_independente=False, promove_app=False,
        limite='Fotos antigas de desenvolvimento e validacao brasileira ja conhecida. Comparacao exploratoria, nao prova de melhora em imagens novas.')
    (OUT / 'protocolo.json').write_text(json.dumps(protocolo, indent=2), encoding='utf-8')
    linhas = []
    device = 0 if torch.cuda.is_available() else 'cpu'
    for nome, tta in candidatos:
        modelo = RTDETR(str(pesos[nome])) if nome == 'rtdetr_l' else YOLO(str(pesos[nome]))
        classes = [i for i, label in modelo.names.items() if label.lower() in ('cow', 'cattle', 'bovino', 'vaca', 'gado')]
        if not classes and len(modelo.names) == 1:
            classes = [0]
        if not classes:
            raise ValueError(f'Nao foi encontrada classe bovina em {nome}: {modelo.names}')
        chave = nome + ('_tta' if tta else '_simples')
        pasta = OUT / chave
        pasta.mkdir(exist_ok=True)
        for r in catalogo:
            im = cv2.imread(str(r['img']))
            if im is None:
                raise ValueError(f'Imagem ilegivel: {r["img"]}')
            h, w = im.shape[:2]
            cache = pasta / (Path(r['arquivo']).stem + '.npz')
            start = time.perf_counter()
            # Cache isolado por protocolo e candidato. Nao reutiliza contagens antigas.
            resultado = modelo.predict(im, imgsz=1280, conf=.05, iou=.7, classes=classes,
                                       augment=tta, max_det=1000, device=device, verbose=False)[0]
            b = resultado.boxes.xyxy.cpu().numpy().copy() if resultado.boxes is not None else np.empty((0,4))
            s = resultado.boxes.conf.cpu().numpy().copy() if resultado.boxes is not None else np.empty(0)
            segundos = time.perf_counter() - start
            np.savez_compressed(cache, caixas=b, scores=s)
            ref = None
            if 'rot' in r:
                ref = ct.yolo_para_pixels(r['rot'].read_text(encoding='utf-8').splitlines(), w, h)
            for conf in CONFS:
                bb, ss = ct.filtrar(b, s, conf)
                row = dict(candidato=chave, origem=r['origem'], arquivo=r['arquivo'], grupo=r['grupo'],
                           conf=conf, previsto=len(bb), segundos=segundos)
                if ref is not None:
                    row.update(real=len(ref), **comparar_caixas(ref, bb, ss))
                else:
                    row.update(minimo=r['minimo'], maximo=r['maximo'],
                               erro_intervalo=max(r['minimo']-len(bb), 0, len(bb)-r['maximo']))
                linhas.append(row)
            # Painel padronizado em .15, sem escolher o limiar por foto.
            bb, _ = ct.filtrar(b, s, .15)
            cv2.imwrite(str(pasta / r['arquivo']), ct.desenhar(im, bb, ref=ref))
        pd.DataFrame(linhas).to_csv(OUT / 'por_imagem.csv', index=False)
        print(f'{chave}: {len(catalogo)} imagens concluidas', flush=True)
    d = pd.DataFrame(linhas)
    res = []
    for (candidato, origem, conf), g in d.groupby(['candidato', 'origem', 'conf']):
        row = dict(candidato=candidato, origem=origem, conf=conf, imagens=len(g), segundos_por_imagem=float(g.segundos.mean()))
        if origem == 'validacao_brasil':
            tp, fp, fn = (int(g[c].sum()) for c in ('TP','FP','FN'))
            row.update(**metricas_contagem(g.real, g.previsto), **taxas_deteccao(tp,fp,fn), TP=tp, FP=fp, FN=fn)
        else:
            row.update(MAE_intervalo=float(g.erro_intervalo.mean()), dentro=int((g.erro_intervalo == 0).sum()))
        res.append(row)
    resumo = pd.DataFrame(res)
    resumo.to_csv(OUT / 'resumo.csv', index=False)
    print('\nVALIDACAO BRASILEIRA CONHECIDA', flush=True)
    print(resumo[resumo.origem == 'validacao_brasil'].sort_values('MAE')[['candidato','conf','MAE','F1_caixas','FP','FN']].head(15).to_string(index=False), flush=True)
    print('\nDESENVOLVIMENTO ANTIGO - NAO INDEPENDENTE', flush=True)
    print(resumo[resumo.origem != 'validacao_brasil'].sort_values(['MAE_intervalo','dentro'], ascending=[True,False])[['candidato','conf','MAE_intervalo','dentro']].head(15).to_string(index=False), flush=True)


if __name__ == '__main__':
    import sys
    rtdetr = '--rtdetr' in sys.argv
    maior = '--maior' in sys.argv
    if rtdetr:
        OUT = ROOT / 'data/experimentos/alternativas_rtdetr_v1'
    elif maior:
        OUT = ROOT / 'data/experimentos/alternativas_yolo11x_v1'
    executar(rtdetr=rtdetr, maior=maior)
