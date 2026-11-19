import numpy as np
from juncao_aerea import selecionar


def test_fragmento_de_emenda_tem_corpo_confirmado_em_outro_tile():
    b=[[0,0,100,100],[80,10,100,40]]
    kept,removed=selecionar(b,[.8,.9],[0,1],[False,True],.5)
    assert kept.tolist()==[0] and removed.tolist()==[1]


def test_animal_sem_correspondencia_nao_e_descartado_por_borda():
    kept,removed=selecionar([[0,0,100,100]],[.9],[0],[True],.5)
    assert kept.tolist()==[0] and not len(removed)


def test_bezerro_detectado_junto_da_vaca_tem_testemunha():
    b=[[0,0,100,100],[80,10,100,40],[0,0,100,100]]
    kept,removed=selecionar(b,[.9,.8,.7],[0,1,1],[False,True,False],.5)
    assert 1 in kept and not len(removed)


def test_borda_da_foto_original_nao_e_emenda():
    kept,removed=selecionar([[0,0,100,100],[80,10,100,40]],[.9,.8],[0,1],[False,False],.5)
    assert kept.tolist()==[0,1] and not len(removed)
