from recuperacao_multiescala import selecionar,selecionar_consenso_estavel
from recuperacao_multiescala import separar_fusoes


def test_exige_duas_vistas_e_preserva_base():
    i,j=selecionar([[0,0,100,100]],[[200,0,300,100]],[.8],[[202,0,302,100]],[.7])
    assert i.tolist()==[0] and j.tolist()==[0]
    i,_=selecionar([],[[200,0,300,100]],[.8],[],[])
    assert i.tolist()==[]


def test_nao_duplica_caixa_existente_nem_fragmento_contido():
    for b in ([[5,5,105,105]],[[25,25,75,75]]):
        i,_=selecionar([[0,0,100,100]],b,[.8],b,[.7])
        assert i.tolist()==[]


def test_uma_caixa_de_apoio_nao_confirma_dois_animais():
    i,j=selecionar([],[[200,0,300,100],[300,0,400,100]],[.8,.7],[[200,0,400,100]],[.8])
    assert len(i)==len(j)==1


def test_consenso_instavel_na_quantidade_nao_recupera():
    i,_=selecionar_consenso_estavel([],[[0,0,10,10]],[.8],[[0,0,10,10],[20,0,30,10],[40,0,50,10]],[.8,.7,.6])
    assert i.tolist()==[]


def test_consenso_estavel_limita_total_ao_menor_voto():
    i,_=selecionar_consenso_estavel([[0,0,10,10]],[[0,0,10,10],[20,0,30,10]],[.9,.8],
        [[0,0,10,10],[20,0,30,10]],[.9,.8])
    assert i.tolist()==[1]


def test_separa_dois_animais_fundidos_confirmados_em_vistas_distintas():
    partes = [[0,0,45,50],[55,0,100,50]]
    b,s,t = separar_fusoes([[0,0,100,50]],[.8],partes,[.9,.7],partes,[.8,.6])
    assert b.tolist() == partes and s.tolist() == [.9,.7]
    assert t == [dict(base=0,alta=[0,1],tta=[0,1])]


def test_nao_separa_quando_apoio_tem_uma_unica_caixa():
    base = [[0,0,100,50]]
    b,s,t = separar_fusoes(base,[.8],[[0,0,45,50],[55,0,100,50]],[.9,.7],base,[.9])
    assert b.tolist() == base and t == []


def test_nao_separa_duplicando_animal_ja_presente_na_base():
    partes = [[0,0,45,50],[55,0,100,50]]
    base = [[0,0,100,50],partes[1]]
    b,s,t = separar_fusoes(base,[.8,.9],partes,[.9,.7],partes,[.8,.6])
    assert b.tolist() == base and t == []
