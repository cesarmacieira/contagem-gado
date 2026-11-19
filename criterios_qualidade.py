"""Regras de comparacao para a mesma referencia e conjunto de imagens."""
import math


def supera_referencia(candidato,referencia):
    n=candidato['n_imagens']
    if n!=referencia['n_imagens'] or n<=0:raise ValueError('Conjuntos de tamanhos diferentes')
    for m in (candidato,referencia):
        for key in ['MAE','F1_caixas','precisao_caixas','recall_caixas']:
            if m[key] is None or not math.isfinite(m[key]):raise ValueError('Metrica invalida')
    delta=referencia['MAE']-candidato['MAE']
    if candidato['MAE']>.9*referencia['MAE']+1e-12 or delta*n<3-1e-8:return False
    for key in ['F1_caixas','precisao_caixas','recall_caixas']:
        if candidato[key]<referencia[key]:return False
    for key in ['oito_ou_mais','vinte_ou_mais']:
        if key in referencia:
            a,b=candidato.get(key),referencia[key]
            if not a or a['MAE']>b['MAE'] or a['recall_caixas']<b['recall_caixas']:return False
    return True
