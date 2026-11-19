import pytest
from config_publicacao import chave_autorizada,vaga_video


def test_producao_nao_entrega_chave_servidor_sem_codigo():
    env=dict(CONTAGEM_PUBLICADA='1',OPENAI_API_KEY='segredo',CONTAGEM_VLM_ACCESS_CODE='acesso')
    assert not chave_autorizada(env=env)
    assert not chave_autorizada(codigo='errado',env=env)
    assert chave_autorizada(codigo='acesso',env=env)=='segredo'
    assert chave_autorizada('minha-chave',env=env)=='minha-chave'
    assert not chave_autorizada(env=dict(CONTAGEM_PUBLICADA='1',OPENAI_API_KEY='segredo'))


def test_local_preserva_configuracao_atual():
    assert chave_autorizada(env=dict(OPENAI_API_KEY='local'))=='local'


def test_apenas_um_video_ocupa_servidor():
    with vaga_video():
        with pytest.raises(RuntimeError):
            with vaga_video():pass
    with vaga_video():pass
