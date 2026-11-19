"""Funções compartilhadas pelo notebook de treino e pelo app: recorte em tiles, rótulos YOLO e contagem por detecção.

Imagens de drone são grandes (4000x3000 px ou mais) e as vacas ocupam poucos pixels. Por isso o modelo é treinado e usado
em recortes (tiles) de tamanho fixo, sem redimensionar, e as detecções dos tiles são unidas com NMS (`detectar`).

Fotos comuns e imagens de instalação (COLO, Open Images) têm a vaca grande e no máximo ~1.150 px de lado: nelas o tile cortaria o
animal em pedaços. Para elas vale `detectar_inteira`: uma passada na imagem inteira, ajustada ao tamanho de entrada do YOLO.
"""
from __future__ import annotations

import numpy as np

CONF_MIN = 0.05  # limiar baixo usado na detecção; o limiar final é aplicado depois (filtrar), o que torna o ajuste instantâneo
VERSAO_INFERENCIA = 2


def linhas_yolo_unicas(linhas: list[str]) -> list[str]:
    """Valida e deduplica após mapear classes; não funde caixas apenas parecidas."""
    unicas = {}
    for linha in linhas:
        if not linha.strip():
            continue
        valores = np.asarray([float(p) for p in linha.split()], dtype=float)
        if (len(valores) != 5 or not np.isfinite(valores).all() or valores[0] != 0
                or np.any(valores[1:] < 0) or np.any(valores[1:] > 1) or np.any(valores[3:] <= 0)):
            raise ValueError(f"Rótulo YOLO inválido: {linha!r}")
        chave = tuple(np.round(valores, 6))
        unicas.setdefault(chave, "0 " + " ".join(f"{v:.6f}" for v in valores[1:]))
    return list(unicas.values())


def tile_tem_fragmento_sem_rotulo(caixas_px, x0, y0, tw, th, min_visivel=0.5):
    """Evita treinar animal visível como fundo. Exclui tiles com fragmento não anotado."""
    b = np.asarray(caixas_px, dtype=float).reshape(-1, 4)
    if not len(b):
        return False
    inter = np.maximum(0, np.minimum(b[:, 2:], [x0 + tw, y0 + th]) - np.maximum(b[:, :2], [x0, y0])).prod(1)
    area = np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)
    fracao = inter / np.maximum(area, 1e-9)
    return bool(np.any((inter > 0) & (fracao < min_visivel)))


# ------------------------------------------------------------------ tiles
def grade_tiles(w: int, h: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    """Cantos (x0, y0) dos tiles que cobrem a imagem inteira, com sobreposição relativa `overlap`."""
    if tile <= 0 or w <= 0 or h <= 0 or not 0 <= overlap < 1:
        raise ValueError("Dimensões positivas e overlap em [0, 1) são obrigatórios")
    passo = max(1, int(tile * (1 - overlap)))

    def eixo(n: int) -> list[int]:
        if n <= tile:
            return [0]
        pos = list(range(0, n - tile + 1, passo))
        if pos[-1] + tile < n:
            pos.append(n - tile)
        return pos

    return [(x, y) for y in eixo(h) for x in eixo(w)]


def rotulos_do_tile(caixas_px: np.ndarray, x0: int, y0: int, tw: int, th: int, min_visivel: float = 0.5) -> list[str]:
    """Linhas YOLO (classe 0) das caixas que ficam ao menos `min_visivel` dentro do tile.

    caixas_px: array (n, 4) em pixels da imagem inteira, formato x1, y1, x2, y2.
    """
    linhas = []
    for a, b, c, d in np.asarray(caixas_px, dtype=float).reshape(-1, 4):
        area = max(c - a, 0) * max(d - b, 0)
        if area <= 0:
            continue
        ix1, iy1, ix2, iy2 = max(a, x0), max(b, y0), min(c, x0 + tw), min(d, y0 + th)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2 - ix1) * (iy2 - iy1) / area < min_visivel:
            continue
        cx, cy = ((ix1 + ix2) / 2 - x0) / tw, ((iy1 + iy2) / 2 - y0) / th
        linhas.append(f"0 {cx:.6f} {cy:.6f} {(ix2 - ix1) / tw:.6f} {(iy2 - iy1) / th:.6f}")
    return linhas


def yolo_para_pixels(linhas: list[str], w: int, h: int) -> np.ndarray:
    """Converte linhas YOLO normalizadas (classe cx cy w h) em array (n, 4) x1, y1, x2, y2 em pixels."""
    out = []
    for l in linhas_yolo_unicas(linhas):
        p = l.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, dtype=float).reshape(-1, 4)


# ------------------------------------------------------------------ NMS
def nms(caixas: np.ndarray, scores: np.ndarray, iou_max: float = 0.5, ioa_max: float | None = None,
        origens: np.ndarray | None = None) -> np.ndarray:
    """Junção entre recortes. Preserva pares do mesmo recorte, já tratados pelo YOLO.

    Contenção (IoA) fica desligada: bezerro dentro da caixa da vaca é um animal distinto.
    ioa_max explícito mantém compatibilidade com experimentos antigos.
    """
    if len(caixas) == 0:
        return np.array([], dtype=int)
    area = (caixas[:, 2] - caixas[:, 0]) * (caixas[:, 3] - caixas[:, 1])
    ordem = np.argsort(-scores)
    mantidos = []
    while ordem.size:
        i = ordem[0]
        mantidos.append(i)
        if ordem.size == 1:
            break
        r = ordem[1:]
        iw = np.clip(np.minimum(caixas[i, 2], caixas[r, 2]) - np.maximum(caixas[i, 0], caixas[r, 0]), 0, None)
        ih = np.clip(np.minimum(caixas[i, 3], caixas[r, 3]) - np.maximum(caixas[i, 1], caixas[r, 1]), 0, None)
        inter = iw * ih
        iou = inter / (area[i] + area[r] - inter + 1e-9)
        ioa = inter / (np.minimum(area[i], area[r]) + 1e-9)
        suprimir = iou > iou_max
        if ioa_max is not None:
            suprimir |= ioa > ioa_max
        if origens is not None:
            suprimir &= origens[r] != origens[i]
        ordem = r[~suprimir]
    return np.array(mantidos, dtype=int)


# ------------------------------------------------------------------ detecção
def detectar(modelo, imagem_bgr: np.ndarray, tile: int = 1024, overlap: float = 0.2, conf_min: float = CONF_MIN,
             iou_nms: float = 0.5, lote: int = 8, device=None, iou_local: float = 0.7,
             max_det: int = 1000, legado: bool = False, classes=None, augment: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Detecta vacas na imagem inteira por tiles. Devolve (caixas Nx4 em pixels, scores N), já sem duplicatas entre tiles."""
    h, w = imagem_bgr.shape[:2]
    cantos = grade_tiles(w, h, tile, overlap)
    if lote < 1:
        raise ValueError("lote deve ser positivo")
    todas, scores, origens = [], [], []
    for i in range(0, len(cantos), lote):
        grupo = cantos[i:i + lote]
        recortes = [imagem_bgr[y:y + tile, x:x + tile] for x, y in grupo]
        res = modelo.predict(recortes, imgsz=tile, conf=conf_min, iou=iou_nms if legado else iou_local,
                             max_det=max_det, verbose=False, device=device, classes=classes, augment=augment)
        for j, ((x, y), r) in enumerate(zip(grupo, res)):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy().copy()
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y
            todas.append(b)
            scores.append(r.boxes.conf.cpu().numpy())
            origens.append(np.full(len(b), i + j, dtype=int))
    if not todas:
        return np.zeros((0, 4)), np.zeros(0)
    b, s = np.concatenate(todas), np.concatenate(scores)
    k = nms(b, s, iou_nms, ioa_max=0.7 if legado else None,
            origens=None if legado else np.concatenate(origens))
    return b[k], s[k]


def detectar_inteira(modelo, imagem_bgr: np.ndarray, imgsz: int = 1024, conf_min: float = CONF_MIN,
                     iou_nms: float = 0.7, device=None, max_det: int = 1000,
                     classes=None, augment: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Detecta na imagem inteira, sem tiles (fotos de solo e imagens de instalação, onde a vaca é grande).

    O YOLO ajusta a imagem a `imgsz` px mantendo a proporção; as caixas voltam em pixels da imagem original.
    iou_nms = 0.7 (padrão do YOLO) porque, com a vaca grande, animais lado a lado se sobrepõem de verdade.
    Devolve (caixas Nx4 em pixels, scores N).
    """
    r = modelo.predict(imagem_bgr, imgsz=imgsz, conf=conf_min, iou=iou_nms, max_det=max_det, verbose=False,
                       device=device, classes=classes, augment=augment)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    return r.boxes.xyxy.cpu().numpy().copy(), r.boxes.conf.cpu().numpy().copy()


def filtrar(caixas: np.ndarray, scores: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray]:
    m = scores >= conf
    return caixas[m], scores[m]


def desenhar(imagem_bgr: np.ndarray, caixas: np.ndarray, cor=(76, 35, 236), cor_ref=None, ref: np.ndarray | None = None) -> np.ndarray:
    """Devolve cópia da imagem com as caixas desenhadas (espessura proporcional ao tamanho da imagem). `ref` = caixas reais."""
    import cv2

    img = imagem_bgr.copy()
    esp = max(2, int(round(max(img.shape[:2]) / 1200)))
    if ref is not None and len(ref):
        for a, b, c, d in ref.astype(int):
            cv2.rectangle(img, (a, b), (c, d), cor_ref or (0, 200, 0), esp)
    for a, b, c, d in caixas.astype(int):
        cv2.rectangle(img, (a, b), (c, d), cor, esp)
    return img
