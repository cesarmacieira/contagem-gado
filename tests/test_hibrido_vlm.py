import io,json
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
import hibrido_vlm as hv


def foto(cor='white'):
    f=io.BytesIO();Image.new('RGB',(200,200),cor).save(f,'PNG');return f.getvalue()


def resposta(n=2):
    return dict(status='completed',usage=dict(input_tokens=1000,output_tokens=100),output=[dict(content=[dict(type='output_text',text=json.dumps(dict(contagem=n,incerto=False,observacao='Dois corpos distintos.')))])])


def test_gatilhos_nao_dependem_so_de_confianca():
    assert not hv.gatilhos([[30,30,70,70]],[.9],200,200)['recomendada']
    assert 'animal na borda' in hv.gatilhos([[0,10,50,50]],[.99],200,200)['motivos']
    assert 'sobreposicao a conferir' in hv.gatilhos([[30,30,90,90],[35,35,95,95]],[.99,.98],200,200)['motivos']
    assert 'baixa confianca' in hv.gatilhos([[30,30,70,70]],[.4],200,200)['motivos']
    assert hv.gatilhos([],[],200,200)['recomendada']
    assert 'propostas proximas ao limiar' in hv.gatilhos([[30,30,70,70]],[.9],200,200,.3,[.25,.9])['motivos']


def test_preserva_detector_e_cache_nao_cobra_novamente(tmp_path):
    chave='sk-ficticia-nao-e-credencial'
    with patch.object(hv.rv,'enviar',return_value=resposta()) as envio:
        a=hv.consultar(foto(),[[30,30,70,70]],[.9],chave,tmp_path)
        b=hv.consultar(foto(),[[30,30,70,70]],[.9],chave,tmp_path)
        assert a['detector']==1 and a['contagem']==2 and a['situacao']=='divergencia'
        assert b['cache'] and envio.call_count==1
        payload=envio.call_args.args[0]
        assert not payload['store'] and 'detector' not in payload['input'][0]['content'][0]['text']
        assert all(chave not in p.read_text(encoding='utf-8') for p in tmp_path.glob('*.json'))


def test_resposta_incompleta_preservada_sem_repetir(tmp_path):
    with patch.object(hv.rv,'enviar',return_value=dict(status='incomplete',usage={} )) as envio:
        for _ in range(2):
            with pytest.raises(RuntimeError):hv.consultar(foto(),[],[],'sk-ficticia-nao-e-credencial',tmp_path)
        assert envio.call_count==1 and len(list(tmp_path.glob('*.resposta.json')))==1


def test_orcamento_impede_nova_consulta(tmp_path):
    with patch.object(hv,'LIMITE_DIARIO_USD',.005),patch.object(hv.rv,'enviar',return_value=resposta()) as envio:
        hv.consultar(foto(),[],[],'sk-ficticia-nao-e-credencial',tmp_path)
        with pytest.raises(RuntimeError,match='diario'):hv.consultar(foto('black'),[],[],'sk-ficticia-nao-e-credencial',tmp_path)
        assert envio.call_count==1


def test_incerteza_nao_vira_concordancia():
    assert hv.comparar(2,dict(contagem=2,incerto=True))=='incerto'
    assert hv.comparar(2,dict(contagem=2,incerto=False))=='concordam'
    with pytest.raises(ValueError):hv.validar(dict(contagem=True,incerto=False,observacao=''))


def test_no_maximo_tres_ampliacoes():
    b=np.array([[20,20,40,40]]*10)
    assert len(hv.preparar(foto(),b,list(range(10))))==4
