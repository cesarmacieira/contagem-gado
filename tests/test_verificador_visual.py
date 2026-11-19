import numpy as np
import pytest
from PIL import Image

from verificador_visual import confirmar, extrair_features, mascara_confirmacao, recorte_quadrado, carregar_binario


def test_abstencao_preserva_objeto_pequeno_sem_aprovar_objeto_grande():
    caixas = [[0, 0, 10, 40], [0, 0, 50, 50], [0, 0, 50, 50]]
    manter, pequenos = mascara_confirmacao(caixas, [.1, .1, .9], .5, 32)
    assert manter.tolist() == [True, False, True]
    assert pequenos.tolist() == [True, False, False]


def test_rejeita_scores_incompativeis():
    with pytest.raises(ValueError):
        mascara_confirmacao([[0, 0, 5, 5]], [], .5)


def test_sem_deteccoes_nao_carrega_modelo():
    assert confirmar(None, None, []).shape == (0,)
    assert extrair_features(None, None, []).shape == (0, 2560)


def test_recorte_quadrado_e_caixa_invalida():
    im = Image.new('RGB', (100, 100))
    crop = recorte_quadrado(im, [10, 20, 30, 60])
    assert crop.width == crop.height == 52
    with pytest.raises(ValueError):
        recorte_quadrado(im, [30, 20, 10, 60])


def test_assinatura_conferida_antes_de_desserializar(tmp_path):
    arquivo = tmp_path / 'modelo.joblib'
    arquivo.write_bytes(b'arquivo invalido')
    with pytest.raises(ValueError, match='alterado'):
        carregar_binario('inexistente', arquivo, 'sha incorreto', device='cpu')


def test_disponibilizacao_conservadora_corresponde_aos_pesos():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'data/modelos/verificador_solo_v1/config.json').read_text())
    assert cfg['ativo_por_padrao'] is False
    assert cfg['visoes_permitidas'] == ['solo_denso']
    for campo in ('extrator', 'classificador'):
        assert hashlib.sha256((root / cfg[campo]).read_bytes()).hexdigest() == cfg[campo + '_sha256']


@pytest.mark.parametrize('contextos', [None, (1.0, 1.05)])
def test_inferencia_respeita_contexto_do_treino(monkeypatch, contextos):
    import verificador_visual as v
    observados=[]
    def extrair(*args, **kwargs):
        observados.append(kwargs['contextos'])
        return np.zeros((1,2560))
    class Classificador:
        def predict_proba(self,x):
            return np.array([[.2,.8]])
    clf=Classificador()
    if contextos is not None:
        clf.contextos_=contextos
    monkeypatch.setattr(v,'extrair_features',extrair)
    assert v.confirmar((None,clf),None,[[0,0,10,10]],device='cpu').tolist()==[.8]
    assert observados==[contextos or (1.3,2.0)]
