import pytest
from conferencia_especies import manter_bovinos


def test_veto_exige_alta_sobreposicao_sem_remover_vizinho():
    b=[[0,0,100,100],[90,0,190,100]]
    assert manter_bovinos(b,[[0,0,100,100]]).tolist()==[False,True]


def test_sem_outra_especie_preserva_bovinos():
    assert manter_bovinos([[0,0,10,10]],[]).tolist()==[True]
    assert manter_bovinos([],[[0,0,10,10]]).tolist()==[]


def test_rejeita_caixas_invalidas():
    with pytest.raises(ValueError):
        manter_bovinos([[10,0,0,10]],[[0,0,10,10]])
