import numpy as np
from correcao_aerea import deduplicar,selecionar_correcao,selecionar_instancias
from correcao_aerea import complementar
import pytest


def test_nao_remove_bezerro_pequeno_contido():
    assert deduplicar([[0,0,100,100],[20,20,40,40]],[.9,.8]).tolist()==[0,1]


def test_remove_caixas_quase_iguais_e_preserva_score_maior():
    assert deduplicar([[0,0,100,100],[0,0,110,110]],[.7,.9]).tolist()==[1]


def test_recuperacao_exige_outra_inferencia_e_verificador():
    b=np.array([[0,0,100,100],[200,0,300,100]],float)
    idx,r=selecionar_correcao(b,[.9,.18],[.9,.8],b[1:],[.7],.2,.5)
    assert idx.tolist()==[0,1] and r.tolist()==[False,True]
    idx,_=selecionar_correcao(b,[.9,.18],[.9,.8],np.empty((0,4)),[],.2,.5)
    assert idx.tolist()==[0]


def test_instancias_recupera_isolado_mas_nao_cabeca_sobreposta():
    b=np.array([[0,0,100,100],[80,20,110,50],[200,0,300,100]],float)
    idx,r=selecionar_instancias(b,[.9,.25,.18],[.9,.9,.7],.3,.15)
    assert idx.tolist()==[0,2] and r.tolist()==[False,True]


def test_instancias_nao_recupera_sombra_nem_altera_confianca_original():
    b=np.array([[0,0,100,100],[200,0,300,100]],float)
    idx,r=selecionar_instancias(b,[.9,.18],[.9,.1],.3,.15)
    assert idx.tolist()==[0] and not r.any()
    idx,_=selecionar_correcao(b,[.9,.18],[.9,.2],b[1:],[.7],.2,.5)
    assert idx.tolist()==[0]


def test_segunda_escala_nao_adiciona_cabeca_nem_duplicata():
    base=np.array([[0,0,100,100]])
    high=np.array([[0,0,110,110],[80,20,110,50],[200,0,300,100]])
    assert complementar(base,high,np.array([.9,.9,.8]),np.ones(3),.4,.5).tolist()==[2]
    assert base.tolist()==[[0,0,100,100]]


def test_segunda_escala_exige_ambas_confirmacoes():
    high=np.array([[0,0,100,100],[200,0,300,100]])
    assert complementar([],high,np.array([.3,.8]),np.array([.9,.2]),.4,.5).tolist()==[]


def test_segunda_escala_rejeita_valores_invalidos():
    with pytest.raises(ValueError):complementar([],[[0,0,10,10]],[float('nan')],[.9],.4,.5)
