import pytest
from criterios_qualidade import supera_referencia


def metricas(mae,n=20,recall=.9):
    return dict(n_imagens=n,MAE=mae,F1_caixas=.9,precisao_caixas=.9,recall_caixas=recall)


def test_um_animal_nao_basta_para_declarar_superioridade():
    assert not supera_referencia(metricas(.15),metricas(.2))


def test_melhor_contagem_nao_compensa_perda_de_animais():
    assert not supera_referencia(metricas(.5,recall=.8),metricas(1))


def test_cenas_densas_nao_podem_regredir():
    a,b=metricas(.5),metricas(1)
    a['vinte_ou_mais']=dict(MAE=2,recall_caixas=.9)
    b['vinte_ou_mais']=dict(MAE=1,recall_caixas=.9)
    assert not supera_referencia(a,b)


def test_melhoria_material_em_mesmas_imagens_e_obrigatoria():
    assert supera_referencia(metricas(.5),metricas(1))
    with pytest.raises(ValueError):supera_referencia(metricas(.5,n=10),metricas(1))
