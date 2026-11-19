from consenso_modelos import decidir

def alt(boxes,horse=None):
    return dict(caixas=boxes,scores=[.9]*len(boxes),cavalos=horse or [],scores_cavalos=[.9]*len(horse or []))

def test_incerto_nao_corrige():
    assert not decidir([], [alt([[0,0,10,10]])]*2,dict(contagem=1,incerto=True))['corrigido']

def test_zero_exige_evidencia_de_especie():
    original=[[0,0,10,10]]
    assert not decidir(original,[alt([])]*2,dict(contagem=0,incerto=False))['corrigido']
    assert decidir(original,[alt([],original)]*2,dict(contagem=0,incerto=False))['corrigido']

def test_contagens_iguais_em_locais_diferentes_nao_bastam():
    assert not decidir([], [alt([[0,0,10,10]]),alt([[20,20,30,30]])],dict(contagem=1,incerto=False))['corrigido']

def test_consenso_positivo_preserva_original():
    original=[]
    r=decidir(original,[alt([[0,0,10,10]])]*2,dict(contagem=1,incerto=False))
    assert r['corrigido'] and r['contagem']==1 and original==[]


def test_proposta_incerta_continua_incerta():
    from consenso_modelos import propor
    r=propor([], [alt([[0,0,10,10]])]*2,dict(contagem=1,incerto=True))
    assert r['proposta'] and r['incerto'] and not r['corrigido']
