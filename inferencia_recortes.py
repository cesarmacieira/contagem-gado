"""Recortes com escala de entrada independente; experimento, sem alterar o app."""
from __future__ import annotations

import numpy as np
from contagem import grade_tiles, nms


def detectar_recortes(modelo, imagem, fracao=.5, imgsz=1280, overlap=.3,
                      iou_local=.7, iou_juncao=.5, conf_min=.05, device=0, classes=None,
                      descartar_borda_interna=False, margem_borda_px=.5, augment=False):
    """Amplia janelas relativas e devolve caixas nas coordenadas do original.

    A fracao se refere ao maior lado. O NMS entre janelas nao usa contencao,
    para nao eliminar um bezerro apenas por estar dentro da caixa de uma vaca.
    """
    if not 0 < fracao <= 1 or imgsz <= 0:
        raise ValueError('fracao deve estar em (0, 1] e imgsz deve ser positivo')
    h, w = imagem.shape[:2]
    tile = max(1, round(max(h, w) * fracao))
    caixas, scores, origens = [], [], []
    for origem, (x, y) in enumerate(grade_tiles(w, h, tile, overlap)):
        janela = imagem[y:y + tile, x:x + tile]
        resultado = modelo.predict(janela, imgsz=imgsz, conf=conf_min,
                                   iou=iou_local, max_det=1000,
                                   device=device, verbose=False, classes=classes, augment=augment)[0]
        if resultado.boxes is None or not len(resultado.boxes):
            continue
        b = resultado.boxes.xyxy.cpu().numpy().copy()
        s = resultado.boxes.conf.cpu().numpy().copy()
        if descartar_borda_interna:
            # Uma predicao cortada artificialmente pode virar uma segunda caixa
            # pequena do mesmo animal. Bordas da foto original permanecem validas.
            jh, jw = janela.shape[:2]
            manter = np.ones(len(b), dtype=bool)
            if x > 0:
                manter &= b[:,0] > margem_borda_px
            if x + jw < w:
                manter &= b[:,2] < jw - margem_borda_px
            if y > 0:
                manter &= b[:,1] > margem_borda_px
            if y + jh < h:
                manter &= b[:,3] < jh - margem_borda_px
            b, s = b[manter], s[manter]
            if not len(b):
                continue
        b[:, [0, 2]] += x
        b[:, [1, 3]] += y
        caixas.append(b)
        scores.append(s)
        origens.append(np.full(len(b), origem))
    if not caixas:
        return np.empty((0, 4)), np.empty(0)
    b, s = np.concatenate(caixas), np.concatenate(scores)
    manter = nms(b, s, iou_max=iou_juncao, origens=np.concatenate(origens))
    return b[manter], s[manter]
