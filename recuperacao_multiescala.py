"""Acrescenta candidatos confirmados em duas escalas, preservando as caixas base."""
import numpy as np


def sobreposicoes(caixa,outras):
    outras=np.asarray(outras,dtype=float).reshape(-1,4)
    if not len(outras):return np.empty(0),np.empty(0)
    caixa=np.asarray(caixa,dtype=float)
    inter=np.maximum(0,np.minimum(caixa[2:],outras[:,2:])-np.maximum(caixa[:2],outras[:,:2])).prod(axis=1)
    area=np.prod(caixa[2:]-caixa[:2]);areas=np.prod(outras[:,2:]-outras[:,:2],axis=1)
    return inter/np.maximum(area+areas-inter,1e-9),inter/np.maximum(np.minimum(area,areas),1e-9)


def selecionar(base,alta,scores_alta,tta,scores_tta,iou_acordo=.5,iou_existente=.3,ios_existente=.7):
    arrays=[np.asarray(x,dtype=float).reshape(-1,4) for x in (base,alta,tta)]
    base,alta,tta=arrays
    sa,st=np.asarray(scores_alta),np.asarray(scores_tta)
    if len(sa)!=len(alta) or len(st)!=len(tta):raise ValueError('Scores incompativeis')
    if any(not np.isfinite(x).all() or (x[:,2:]<=x[:,:2]).any() for x in arrays):
        raise ValueError('Caixas invalidas')
    if not np.isfinite(sa).all() or not np.isfinite(st).all():raise ValueError('Scores invalidos')
    usados=set();selecionados=[];confirmacoes=[]
    existentes=base.copy()
    for i in np.argsort(-sa,kind='stable'):
        acordo,_=sobreposicoes(alta[i],tta)
        if usados:acordo[list(usados)]=-1
        if not len(acordo) or acordo.max()<iou_acordo:continue
        io,ios=sobreposicoes(alta[i],existentes)
        if (io>=iou_existente).any() or (ios>=ios_existente).any():continue
        j=int(acordo.argmax());usados.add(j)
        selecionados.append(int(i));confirmacoes.append(j)
        existentes=np.concatenate([existentes,alta[i:i+1]])
    return np.asarray(selecionados,dtype=int),np.asarray(confirmacoes,dtype=int)


def selecionar_consenso_estavel(base,alta,scores_alta,tta,scores_tta):
    # A tolerancia de sobreposicao acompanha o NMS da rota densa. O consenso
    # precisa existir tambem na quantidade total, nao apenas em caixas isoladas.
    ids,j=selecionar(base,alta,scores_alta,tta,scores_tta,
        iou_acordo=.5,iou_existente=.7,ios_existente=.98)
    limite=min(len(alta),len(tta))-len(base)
    if limite<=0 or abs(len(alta)-len(tta))>1:
        return np.empty(0,dtype=int),np.empty(0,dtype=int)
    return ids[:limite],j[:limite]


def separar_fusoes(base, scores_base, alta, scores_alta, tta, scores_tta):
    """Substitui uma caixa fundida por partes distintas confirmadas nas duas vistas.

    Exige contencao, cobertura do retangulo original e correspondencia um-a-um.
    Nao usa rotulos, contagem esperada, nomes de imagem ou limites por cena.
    """
    from itertools import combinations
    base, alta, tta = [np.asarray(x, dtype=float).reshape(-1, 4) for x in (base, alta, tta)]
    sb, sa, st = [np.asarray(x, dtype=float) for x in (scores_base, scores_alta, scores_tta)]
    # Reutiliza a validacao dos argumentos, sem aceitar nenhuma adicao por ela.
    selecionar(base, alta, sa, tta, st)
    if len(sb) != len(base) or not np.isfinite(sb).all():
        raise ValueError('Scores base invalidos')
    usados_a, usados_t, trocas = set(), set(), []
    for k, b in enumerate(base):
        area = np.prod(b[2:] - b[:2])
        candidatos = []
        for i, a in enumerate(alta):
            if i in usados_a or np.prod(a[2:] - a[:2]) > .8 * area:
                continue
            _, ios = sobreposicoes(a, [b])
            centro = (a[:2] + a[2:]) / 2
            if ios[0] >= .8 and (centro >= b[:2]).all() and (centro <= b[2:]).all():
                candidatos.append(i)
        melhores = None
        for i, j in combinations(candidatos, 2):
            io, _ = sobreposicoes(alta[i], [alta[j]])
            uniao = np.r_[np.minimum(alta[i,:2], alta[j,:2]), np.maximum(alta[i,2:], alta[j,2:])]
            cobertura, _ = sobreposicoes(b, [uniao])
            if io[0] > .25 or cobertura[0] < .7:
                continue
            acordos = []
            for idx in (i, j):
                ia, _ = sobreposicoes(alta[idx], tta)
                if usados_t:
                    ia[list(usados_t)] = -1
                acordos.append(ia)
            pares = [(float(min(acordos[0][u], acordos[1][v])), u, v)
                     for u in np.flatnonzero(acordos[0] >= .5)
                     for v in np.flatnonzero(acordos[1] >= .5) if u != v]
            if not pares:
                continue
            qualidade, u, v = max(pares)
            # Uma parte que ja pertence a outra caixa base nao e nova evidencia.
            outras = np.delete(base, k, axis=0)
            if any((sobreposicoes(alta[idx], outras)[0] >= .5).any() for idx in (i, j)):
                continue
            if melhores is None or qualidade > melhores[0]:
                melhores = (qualidade, i, j, int(u), int(v))
        if melhores is not None:
            _, i, j, u, v = melhores
            usados_a.update((i, j)); usados_t.update((u, v))
            trocas.append(dict(base=k, alta=[i, j], tta=[u, v]))
    removidas = {t['base'] for t in trocas}
    ids_base = [i for i in range(len(base)) if i not in removidas]
    ids_alta = [i for t in trocas for i in t['alta']]
    return (np.concatenate([base[ids_base], alta[ids_alta]]),
            np.concatenate([sb[ids_base], sa[ids_alta]]), trocas)
