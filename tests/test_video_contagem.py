import json
import cv2
import numpy as np
import pytest
from video_contagem import analisar


def video(tmp_path):
    path=tmp_path/'fonte.avi';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10.,(64,48))
    assert writer.isOpened()
    for i in range(12):writer.write(np.full((48,64,3),i*15,np.uint8))
    writer.release();return path


def detector(frame):
    return np.array([[5,5,20,20]],float),np.array([.9])


def test_amostra_percorre_ate_ultimo_quadro(tmp_path):
    r=analisar(video(tmp_path),detector,tmp_path/'saida',.5)
    assert [q['quadro'] for q in r['quadros']]==[0,5,10,11]
    assert r['meta']['quadros_decodificados']==12 and r['meta']['concluido']
    assert r['meta']['animais_unicos'] is None
    assert r['meta']['max_simultaneo']==1
    assert r['quadros'][-1]['segundos']==pytest.approx(1.1)
    assert json.loads((tmp_path/'saida/resultado.json').read_text(encoding='utf-8'))['meta']['quadros_analisados']==4


def test_modo_todos_realmente_analisa_todos(tmp_path):
    chamadas=[]
    def contar(f):chamadas.append(1);return detector(f)
    r=analisar(video(tmp_path),contar,tmp_path/'saida',0.)
    assert len(chamadas)==12 and len(r['quadros'])==12
    assert r['meta']['modo']=='todos_os_quadros'


def test_limite_nao_vira_analise_parcial_aprovada(tmp_path):
    with pytest.raises(ValueError,match='excederia'):analisar(video(tmp_path),detector,tmp_path/'saida',0.,max_amostras=5)
    assert not (tmp_path/'saida/resultado.json').exists()


def test_arquivo_invalido_rejeitado(tmp_path):
    path=tmp_path/'ruim.mp4';path.write_bytes(b'nao e um video')
    with pytest.raises(ValueError):analisar(path,detector,tmp_path/'saida')
