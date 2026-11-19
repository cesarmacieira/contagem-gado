"""Associa fragmentos na emenda sem descartar animais apenas por contencao."""
import numpy as np
from recuperacao_multiescala import sobreposicoes
from contagem import nms


def selecionar(caixas,scores,origens,truncadas,conf,iou=.5):
    b=np.asarray(caixas,float).reshape(-1,4);s=np.asarray(scores,float)
    o=np.asarray(origens);t=np.asarray(truncadas,bool)
    if any(x.shape!=(len(b),) for x in [s,o,t]):raise ValueError('Metadados incompativeis')
    if not np.isfinite(b).all() or not np.isfinite(s).all() or (b[:,2:]<=b[:,:2]).any():raise ValueError('Caixas invalidas')
    ids=np.flatnonzero(s>=conf);area=np.prod(b[:,2:]-b[:,:2],axis=1);remove=set()
    for i in ids[t[ids]]:
        others=ids[(o[ids]!=o[i])&~t[ids]&(area[ids]>=1.25*area[i])]
        _,ios=sobreposicoes(b[i],b[others])
        for j in others[ios>=.85]:
            # A distinct body detected alongside this fragment is evidence
            # against treating containment alone as the same animal.
            siblings=ids[(o[ids]==o[i])&(ids!=i)]
            io,_=sobreposicoes(b[j],b[siblings])
            if (io>=.5).any():continue
            remove.add(int(i));break
    candidates=np.array([i for i in ids if i not in remove],int)
    kept=nms(b[candidates],s[candidates],iou_max=iou,origens=o[candidates])
    return candidates[kept],np.asarray(sorted(remove),int)
