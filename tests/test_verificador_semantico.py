import numpy as np
import pytest
from verificador_semantico import manter,candidatas_montaria,assinatura_receita


def test_rejeicao_exige_duas_vistas_concordantes():
    m=np.array([[.08,.06],[.08,-.02],[.01,.07],[-.1,-.1]])
    assert manter(m,.03).tolist()==[False,True,True,True]


def test_sem_caixas_retorna_mascara_vazia():
    assert manter(np.empty((0,2))).shape==(0,)


def test_margens_invalidas_nao_sao_aceitas():
    with pytest.raises(ValueError):manter([[float('nan'),.1]])
    with pytest.raises(ValueError):manter([[.2]])


def test_montaria_exige_pessoa_acima_e_alinhada():
    animal=[[20,100,100,220]]
    assert candidatas_montaria(animal,[[30,20,90,170]]).tolist()==[True]
    assert candidatas_montaria(animal,[[120,20,170,170]]).tolist()==[False]
    assert candidatas_montaria(animal,[[30,110,90,220]]).tolist()==[False]
    assert candidatas_montaria(animal,[]).tolist()==[False]


def carregar_funcao_app():
    import ast
    import cv2
    import contagem as ct
    from pathlib import Path
    arvore=ast.parse((Path(__file__).resolve().parents[1]/'app.py').read_text(encoding='utf-8'))
    fn=next(n for n in arvore.body if isinstance(n,ast.FunctionDef) and n.name=='conferir_montaria_bytes')
    fn.decorator_list=[]
    ns=dict(np=np,cv2=cv2,ct=ct)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'app.py','exec'),ns)
    return ns['conferir_montaria_bytes'],ct


def test_app_preserva_candidato_sem_pessoa_mesmo_com_semantica_negativa(monkeypatch):
    import cv2
    import verificador_semantico as v
    fn,ct=carregar_funcao_app()
    monkeypatch.setattr(v,'margens',lambda *args:(np.array([[.04,.05],[.04,.05]]),None))
    chamadas=[]
    def pessoas(*args,**kwargs):
        chamadas.append(kwargs)
        return np.array([[30,20,90,170]]),np.array([.9])
    monkeypatch.setattr(ct,'detectar_inteira',pessoas)
    dados=cv2.imencode('.png',np.zeros((256,320,3),dtype=np.uint8))[1].tobytes()
    cfg=dict(regra_geometrica=1,limiar=.008,pessoa_imgsz=1280,pessoa_conf=.5)
    mask,_,regiao,_=fn(dados,np.array([[20,100,100,220],[200,100,280,220]]),cfg,None,None,'cpu')
    assert mask.tolist()==[False,True] and regiao.tolist()==[True,False]
    assert chamadas[0]['classes']==[0] and chamadas[0]['augment'] is False


def test_app_vazio_nao_precisa_de_modelos():
    import cv2
    fn,_=carregar_funcao_app()
    dados=cv2.imencode('.png',np.zeros((8,8,3),dtype=np.uint8))[1].tobytes()
    cfg=dict(regra_geometrica=1,limiar=.008)
    mask,mm,_,_=fn(dados,np.empty((0,4)),cfg,None,None,'cpu')
    assert mask.shape==(0,) and mm.shape==(0,2)


def test_receita_alterada_invalida_identidade_do_cache(monkeypatch):
    import verificador_semantico as v
    anterior=assinatura_receita()
    monkeypatch.setattr(v,'POSITIVOS',v.POSITIVOS+['a photo of an ox.'])
    assert assinatura_receita()!=anterior


def test_csv_preserva_indices_apos_dois_filtros(tmp_path):
    import ast
    import pandas as pd
    from pathlib import Path
    from types import SimpleNamespace
    arvore=ast.parse((Path(__file__).resolve().parents[1]/'app.py').read_text(encoding='utf-8'))
    bloco=next(n for n in ast.walk(arvore) if isinstance(n,ast.If)
        and 'montaria_cfg.get' in ast.unparse(n.test))
    for nome in ('clip.pt','pessoas.pt'):(tmp_path/nome).write_bytes(b'teste de alinhamento')
    cfg=dict(ativo=True,peso_clip='clip.pt',peso_pessoas='pessoas.pt',sha_clip='a',
        receita_sha256='b',sha_pessoas='c')
    caixas=np.arange(16,dtype=float).reshape(4,4)
    detalhes=pd.DataFrame(caixas,columns=['x1','y1','x2','y2'])
    detalhes['mantida']=[False,True,False,True]
    mm=np.array([[.04,.05],[-.02,.01]])
    ns=dict(PROJ=tmp_path,montaria_cfg=cfg,detalhes_confirmacao=detalhes,
        b=caixas[[1,3]].copy(),s=np.array([.4,.8]),score_confirmado=np.array([.9,.95]),
            dados=b'imagem simulada',np=np,st=SimpleNamespace(caption=lambda *a:None),avisar=lambda *a:None,
        ler_conferencia_montaria=lambda *a:(None,None,'cpu'),
        conferir_montaria_bytes=lambda *a:(np.array([False,True]),mm,np.array([True,False]),np.empty((0,4))))
    exec(compile(ast.Module(body=[bloco],type_ignores=[]),'app.py','exec'),ns)
    assert detalhes['mantida'].tolist()==[False,False,False,True]
    assert detalhes['rejeitada_por_montaria'].tolist()==[False,True,False,False]
    np.testing.assert_array_equal(ns['b'],caixas[[3]])
    assert ns['score_confirmado'].tolist()==[.95]
    np.testing.assert_array_equal(ns['margens_montaria'],mm[[1]])
