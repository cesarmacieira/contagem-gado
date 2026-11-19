"""Veto conservador experimental por caixa de outra especie confirmada."""
import numpy as np


def manter_bovinos(caixas, outras_especies, iou_min=.6):
    b=np.asarray(caixas,dtype=float).reshape(-1,4)
    o=np.asarray(outras_especies,dtype=float).reshape(-1,4)
    if not 0 < iou_min <= 1 or not np.isfinite(b).all() or not np.isfinite(o).all():
        raise ValueError('Caixas ou limiar invalidos')
    manter=np.ones(len(b),dtype=bool)
    if not len(b) or not len(o):
        return manter
    if (b[:,2:]<=b[:,:2]).any() or (o[:,2:]<=o[:,:2]).any():
        raise ValueError('Caixa sem area positiva')
    a=np.maximum(b[:,None,:2],o[None,:,:2])
    z=np.minimum(b[:,None,2:],o[None,:,2:])
    inter=np.maximum(z-a,0).prod(axis=2)
    area_b=(b[:,2:]-b[:,:2]).prod(axis=1)
    area_o=(o[:,2:]-o[:,:2]).prod(axis=1)
    iou=inter/(area_b[:,None]+area_o[None,:]-inter)
    return iou.max(axis=1)<iou_min
