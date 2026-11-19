import hashlib
from types import SimpleNamespace
import joblib
import numpy as np
import pytest
from PIL import Image
from verificador_nucleo import recorte_estrito,mascara,carregar,assinatura_receita


def test_nucleo_nao_inclui_animal_fora_da_caixa():
    pixels=np.full((40,80,3),255,np.uint8);pixels[10:20,20:50]=0
    crop=np.array(recorte_estrito(Image.fromarray(pixels),[20,10,50,20]))
    assert crop.shape==(10,30,3) and not crop.any()


def test_recorte_nao_cria_padding_externo_e_rejeita_caixa_invalida():
    im=Image.new('RGB',(10,10),'white')
    assert recorte_estrito(im,[-5,-5,4,4]).size==(4,4)
    with pytest.raises(ValueError):recorte_estrito(im,[12,0,15,5])
    with pytest.raises(ValueError):recorte_estrito(im,[5,5,3,3])


def test_mascara_preserva_ordem_e_limite_e_rejeita_nan():
    assert mascara([.1,.5,.8],.5).tolist()==[False,True,True]
    with pytest.raises(ValueError):mascara([float('nan')],.5)


def test_carregamento_rejeita_receita_de_contexto_quadrado(tmp_path):
    e=tmp_path/'encoder';e.write_bytes(b'encoder')
    c=tmp_path/'clf';joblib.dump(SimpleNamespace(n_features_in_=2560,classes_=[0,1],
        receita_extracao_='contexto_quadrado'),c)
    with pytest.raises(ValueError,match='incompativel'):
        carregar(e,c,hashlib.sha256(e.read_bytes()).hexdigest(),hashlib.sha256(c.read_bytes()).hexdigest(),assinatura_receita())
