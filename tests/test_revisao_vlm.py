import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import revisao_vlm as rv


class RevisaoVLMTests(unittest.TestCase):
    def foto(self, cor='white'):
        b=io.BytesIO();Image.new('RGB',(487,588),cor).save(b,'PNG');return b.getvalue()

    def resposta(self):
        r=dict(minimo=1,maximo=1,animais=[dict(x=.5,y=.5,evidencia='corpo e cabeca',incerto=False)],nota='Revisar')
        return dict(status='completed',output=[dict(content=[dict(type='output_text',text=json.dumps(r))])],usage=dict(input_tokens=1000,output_tokens=100))

    def test_cache_orcamento_e_sigilo(self):
        chave='sk-chave-ficticia-para-teste'
        with tempfile.TemporaryDirectory() as pasta, patch.object(rv,'enviar',return_value=self.resposta()) as mock:
            a=rv.consultar(self.foto(),chave,pasta)
            self.assertEqual(a,rv.consultar(self.foto(),chave,pasta))
            self.assertEqual(mock.call_count,1)
            with self.assertRaises(RuntimeError):rv.consultar(self.foto('black'),chave,pasta)
            self.assertEqual(mock.call_count,1)
            for p in Path(pasta).glob('*.json'):self.assertNotIn(chave,p.read_text(encoding='utf-8'))
            payload=mock.call_args.args[0]
            self.assertFalse(payload['store'])
            self.assertEqual(payload['max_output_tokens'],800)
            self.assertNotIn('7',payload['input'][0]['content'][0]['text'])

    def test_falha_nao_repete(self):
        with tempfile.TemporaryDirectory() as pasta, patch.object(rv,'enviar',side_effect=RuntimeError('offline')) as mock:
            for _ in range(2):
                with self.assertRaises(RuntimeError):rv.consultar(self.foto(),'sk-chave-ficticia-para-teste',pasta)
            self.assertEqual(mock.call_count,1)

    def test_contagem_inconsistente_rejeitada(self):
        with self.assertRaises(ValueError):rv.validar(dict(minimo=2,maximo=2,animais=[]))

    def test_resposta_invalida_preservada_sem_repeticao(self):
        resposta=self.resposta()
        resposta['output'][0]['content'][0]['text']=json.dumps(dict(minimo=0,maximo=1,animais=[dict(x=.5,y=.5,evidencia='animal',incerto=False)],nota=''))
        with tempfile.TemporaryDirectory() as pasta, patch.object(rv,'enviar',return_value=resposta) as mock:
            with self.assertRaises(ValueError):rv.consultar(self.foto(),'sk-chave-ficticia-para-teste',pasta)
            self.assertEqual(len(list(Path(pasta).glob('*.resposta.json'))),1)
            with self.assertRaises(RuntimeError):rv.consultar(self.foto(),'sk-chave-ficticia-para-teste',pasta)
            self.assertEqual(mock.call_count,1)

    def test_orcamento_pequeno(self):
        self.assertLess(rv.ESTIMATIVA_MAX_USD,.01)

    def test_env_nao_executa_e_preserva_ambiente(self):
        import os
        with tempfile.TemporaryDirectory() as pasta, patch.dict(os.environ,{},clear=True):
            Path(pasta,'.env').write_text('OUTRA=ignorada\nOPENAI_API_KEY="chave-local-ficticia"\n',encoding='utf-8')
            rv.carregar_chave_ambiente(pasta)
            self.assertEqual(os.environ['OPENAI_API_KEY'],'chave-local-ficticia')
            self.assertNotIn('OUTRA',os.environ)
            os.environ['OPENAI_API_KEY']='existente'
            rv.carregar_chave_ambiente(pasta)
            self.assertEqual(os.environ['OPENAI_API_KEY'],'existente')


if __name__=='__main__':unittest.main()
