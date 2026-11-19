from contagem_resposta_vlm import extrair


def test_numero_explicito_em_frase_ou_por_extenso():
    assert extrair('Há **sete** bovinos na imagem, contando os parcialmente visíveis.')==7
    assert extrair('Há 9 bovinos visíveis.')==9
    assert extrair('Cinco.')==5
    assert extrair('7')==7


def test_nao_escolhe_entre_contagens_ambiguas():
    assert extrair('6 ou 7') is None
    assert extrair('Nao sei') is None
    assert extrair('') is None
