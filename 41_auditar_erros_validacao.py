"""Exporta maiores divergencias da validacao para revisao visual, sem corrigir rotulos."""
from pathlib import Path
import html
import json

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/experimentos/recortes_escala_v1'


def numerar(img, boxes, cor, titulo):
    out = img.copy()
    for i, box in enumerate(boxes, 1):
        a, b, c, d = np.rint(box).astype(int)
        cv2.rectangle(out, (a, b), (c, d), cor, 2)
        cv2.putText(out, str(i), (max(0, a), max(18, b)), cv2.FONT_HERSHEY_SIMPLEX, .55, cor, 2)
    faixa = np.full((45, out.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(faixa, titulo, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2)
    return np.concatenate([faixa, out], axis=0)


def main():
    torch.set_num_threads(4)
    d = pd.read_csv(OUT / 'por_imagem.csv')
    d = d[(d.modelo == 'ajustado') & (d.estrategia == 'inteira1280') & np.isclose(d.conf, .35)].copy()
    d['erro'] = d.previsto - d.real
    # Amostra deliberada de erros, nao estimativa da frequencia de defeitos.
    escolhidas = pd.concat([d.nlargest(3, 'erro'), d.nsmallest(3, 'erro')]).drop_duplicates('arquivo')
    val = pd.read_csv(OUT / 'validacao.csv').set_index('arquivo')
    pasta = OUT / 'auditoria_visual'
    pasta.mkdir(exist_ok=True)
    modelo = YOLO(str(ROOT / 'data/experimentos/detector_denso_v3/treino/weights/best.pt'))
    device = 0 if torch.cuda.is_available() else 'cpu'
    paginas = []
    for r in escolhidas.itertuples():
        v = val.loc[r.arquivo]
        img = cv2.imread(v.img)
        h, w = img.shape[:2]
        ref = ct.yolo_para_pixels(Path(v.rot).read_text(encoding='utf-8').splitlines(), w, h)
        b, s = ct.detectar_inteira(modelo, img, imgsz=1280, iou_nms=.7, device=device)
        b, s = ct.filtrar(b, s, .35)
        esquerda = numerar(img, ref, (0, 180, 0), f'Rotulos publicados: {len(ref)}')
        direita = numerar(img, b, (0, 0, 220), f'Detector ajustado: {len(b)} - conf .35')
        nome = Path(r.arquivo).stem + '.jpg'
        cv2.imwrite(str(pasta / nome), np.concatenate([esquerda, direita], axis=1))
        np.savez_compressed(pasta / (Path(nome).stem + '.npz'), ref=ref, caixas=b, scores=s)
        paginas.append(f'<h2>{html.escape(r.arquivo)}</h2><img style="max-width:100%" src="{html.escape(nome)}">')
    (pasta / 'galeria.html').write_text('<!doctype html><meta charset="utf-8"><title>Revisao de erros</title><h1>Divergencias na validacao conhecida</h1><p>Selecao por maiores erros, nao amostra aleatoria. Verde: anotacoes publicadas; vermelho: detector. Nenhuma previsao e gabarito automatico.</p>' + ''.join(paginas), encoding='utf-8')
    escolhidas.to_csv(pasta / 'amostra.csv', index=False)
    print('Galeria:', pasta / 'galeria.html', flush=True)


if __name__ == '__main__':
    main()
