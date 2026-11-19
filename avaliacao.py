"""Métricas de contagem e correspondência individual; sem dependência do detector."""
from __future__ import annotations

import numpy as np


def metricas_contagem(real, previsto):
    y, p = np.asarray(real, dtype=float), np.asarray(previsto, dtype=float)
    if y.shape != p.shape or y.size == 0 or not np.isfinite([y, p]).all():
        raise ValueError("Contagens devem ter mesmo tamanho, ser finitas e não vazias")
    e, pos = p - y, y > 0
    var = ((y - y.mean()) ** 2).sum()
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), vies=float(e.mean()),
                MAPE_pct=float(np.mean(np.abs(e[pos]) / y[pos]) * 100) if pos.any() else None,
                WAPE_pct=float(np.abs(e).sum() / y.sum() * 100) if y.sum() else None,
                erro_total_pct=float(e.sum() / y.sum() * 100) if y.sum() else None,
                R2=float(1 - (e ** 2).sum() / var) if var > 0 else None,
                exatas_pct=float((e == 0).mean() * 100), n_imagens=int(y.size))


def comparar_caixas(ref, caixas, scores, iou_min=0.5):
    """Matching um-a-um por confiança, com IoU mínimo. TP/FP/FN contra rótulos disponíveis."""
    ref = np.asarray(ref, dtype=float).reshape(-1, 4)
    caixas = np.asarray(caixas, dtype=float).reshape(-1, 4)
    if len(caixas) != len(scores):
        raise ValueError("Uma confiança por caixa é obrigatória")
    usados = set()
    for k in np.argsort(-np.asarray(scores), kind="stable"):
        b = caixas[k]
        inter = np.maximum(0, np.minimum(b[2:], ref[:, 2:]) - np.maximum(b[:2], ref[:, :2])).prod(1)
        area = np.maximum(0, b[2:] - b[:2]).prod()
        areas_ref = np.maximum(0, ref[:, 2:] - ref[:, :2]).prod(1)
        iou = inter / (area + areas_ref - inter + 1e-9)
        if usados:
            iou[list(usados)] = -1
        if len(iou):
            j = int(iou.argmax())
            if iou[j] >= iou_min:
                usados.add(j)
    tp = len(usados)
    return dict(TP=tp, FP=len(caixas) - tp, FN=len(ref) - tp)


def taxas_deteccao(tp, fp, fn):
    return dict(precisao_caixas=tp / (tp + fp) if tp + fp else None,
                recall_caixas=tp / (tp + fn) if tp + fn else None,
                F1_caixas=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
