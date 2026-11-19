from decisao_hibrida import decidir

def test_vlm_revisa_sem_terceiro_modelo():
    assert decidir(5,dict(contagem=7,incerto=False))['contagem']==7

def test_incerto_nao_vira_resultado_confirmado():
    assert decidir(5,dict(contagem=7,incerto=True))['contagem'] is None

def test_concordancia():
    assert decidir(7,dict(contagem=7,incerto=False))['estado']=='concordancia'
