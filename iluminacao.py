"""Quatro exposicoes locais, sem gerar detalhes novos ou alterar coordenadas."""
import cv2
import numpy as np
from recuperacao_multiescala import sobreposicoes

VERSAO=1


def variantes(imagem):
    if imagem.dtype!=np.uint8 or imagem.ndim!=3 or imagem.shape[2]!=3:
        raise ValueError('Imagem BGR uint8 esperada')
    lab=cv2.cvtColor(imagem,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(l)
    out={'original':imagem,'contraste_local':cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR)}
    for nome,gamma in [('sombras_claras',.7),('luzes_reduzidas',1.4)]:
        lut=np.rint((np.arange(256)/255.)**gamma*255).astype(np.uint8)
        out[nome]=cv2.LUT(imagem,lut)
    return out


def detectar_passagens(modelo,imagem,imgsz=1024,iou=.5,classes=None,augment=False):
    from contagem import detectar_inteira
    return {nome:detectar_inteira(modelo,im,imgsz=imgsz,conf_min=.05,iou_nms=iou,
        classes=classes,augment=augment) for nome,im in variantes(imagem).items()}


def combinar(passagens,conf=.3,iou=.5,suporte_min=2):
    if 'original' not in passagens or not 0<=conf<=1 or suporte_min<2:
        raise ValueError('Parametros invalidos')
    valid={}
    for nome,(boxes,scores) in passagens.items():
        b=np.asarray(boxes,float).reshape(-1,4);s=np.asarray(scores,float)
        if s.shape!=(len(b),) or not np.isfinite(b).all() or not np.isfinite(s).all() or (b[:,2:]<=b[:,:2]).any():
            raise ValueError('Propostas invalidas')
        keep=s>=conf;valid[nome]=(b[keep],s[keep])
    base,bs=valid['original'];kept=base.tolist();ks=bs.tolist();audit=[]
    candidates=[(float(s[i]),nome,i,b[i]) for nome,(b,s) in valid.items() if nome!='original' for i in range(len(b))]
    candidates.sort(key=lambda v:(-v[0],v[1],v[2]))
    for score,nome,idx,box in candidates:
        overlaps,_=sobreposicoes(box,kept)
        if (overlaps>=iou).any():continue
        votes=[]
        for other,(ob,_) in valid.items():
            agreement,_=sobreposicoes(box,ob)
            if (agreement>=.45).any():votes.append(other)
        if len(votes)<suporte_min:continue
        kept.append(box.tolist());ks.append(score);audit.append(dict(caixa=box.tolist(),score=score,passagens=votes))
    return np.asarray(kept,float).reshape(-1,4),np.asarray(ks,float),audit
