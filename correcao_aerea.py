"""Associacao conservadora para vista externa; nao depende de nomes de imagem."""
import numpy as np
from recuperacao_multiescala import sobreposicoes,selecionar


def complementar(base,candidatos,scores,visual,score_min,visual_min):
    """Adiciona apenas caixas confirmadas e isoladas; preserva a contagem base."""
    existentes=np.asarray(base,float).reshape(-1,4)
    candidatos=np.asarray(candidatos,float).reshape(-1,4)
    scores,visual=np.asarray(scores,float),np.asarray(visual,float)
    if scores.shape!=(len(candidatos),) or visual.shape!=scores.shape:
        raise ValueError('Scores incompativeis')
    if any(not np.isfinite(a).all() for a in (existentes,candidatos,scores,visual)):
        raise ValueError('Valores invalidos')
    if any((a[:,2:]<=a[:,:2]).any() for a in (existentes,candidatos)):
        raise ValueError('Caixa sem area')
    keep=[]
    for i in np.argsort(-scores,kind='stable'):
        if scores[i]<score_min or visual[i]<visual_min:continue
        io,ios=sobreposicoes(candidatos[i],existentes)
        if ((io>=.1)|(ios>=.4)).any():continue
        keep.append(int(i));existentes=np.concatenate([existentes,candidatos[i:i+1]])
    return np.asarray(keep,int)


def deduplicar(caixas,scores):
    b=np.asarray(caixas,float).reshape(-1,4);s=np.asarray(scores,float)
    if len(b)!=len(s) or not np.isfinite(b).all() or not np.isfinite(s).all():
        raise ValueError('Caixas ou scores invalidos')
    if len(b) and (b[:,2:]<=b[:,:2]).any():raise ValueError('Caixa sem area')
    order=np.argsort(-s,kind='stable');kept=[]
    while len(order):
        i=int(order[0]);kept.append(i);rest=order[1:]
        io,ios=sobreposicoes(b[i],b[rest])
        order=rest[~((io>=.3)&(ios>=.85))]
    return np.array(kept,int)


def selecionar_instancias(caixas,scores,verificacao,conf,limiar):
    """Remove caixas redundantes e recupera candidatos isolados confirmados visualmente."""
    b=np.asarray(caixas,float).reshape(-1,4)
    s,v=np.asarray(scores,float),np.asarray(verificacao,float)
    if s.shape!=(len(b),) or v.shape!=(len(b),) or not np.isfinite(v).all():
        raise ValueError('Scores incompativeis')
    if not 0<=conf<=1 or not 0<=limiar<=1 or ((v<0)|(v>1)).any():
        raise ValueError('Limiares ou verificacao invalidos')
    base=np.flatnonzero((s>=conf)&(v>=limiar))
    keep=base[deduplicar(b[base],s[base])].tolist()
    n_base=len(keep)
    low=np.flatnonzero((s>=.15)&(s<conf)&(v>=max(.5,limiar)))
    for i in low[np.argsort(-s[low],kind='stable')]:
        io,ios=sobreposicoes(b[i],b[keep])
        if not ((io>=.1)|(ios>=.4)).any():keep.append(int(i))
    return np.asarray(keep,int),np.arange(len(keep))>=n_base


def selecionar_correcao(caixas,scores,verificacao,gamma_caixas,gamma_scores,conf,limiar):
    b,s,v=np.asarray(caixas),np.asarray(scores),np.asarray(verificacao)
    if len(b)!=len(s) or len(b)!=len(v) or not np.isfinite(v).all():raise ValueError('Scores incompativeis')
    base=np.flatnonzero((s>=conf)&(v>=limiar))
    base=base[deduplicar(b[base],s[base])]
    low=np.flatnonzero((s>=.1)&(s<conf)&(v>=limiar))
    gg=np.asarray(gamma_scores)>=conf
    ids,_=selecionar(b[base],b[low],s[low],np.asarray(gamma_caixas)[gg],np.asarray(gamma_scores)[gg],
        iou_acordo=.5,iou_existente=.3,ios_existente=.7)
    add=low[ids]
    return np.concatenate([base,add]),np.concatenate([np.zeros(len(base),bool),np.ones(len(add),bool)])
