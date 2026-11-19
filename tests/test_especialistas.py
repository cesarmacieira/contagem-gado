import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import contagem as ct
from especialistas import carregar_catalogo, resolver_especialista


class EspecialistasTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'data/modelos').mkdir(parents=True)
        self.catalogo = dict(versao=1, visoes={})
        for nome, dados in [('solo', b'peso solo'), ('drone', b'peso drone')]:
            peso = self.root / f'{nome}.pt'
            peso.write_bytes(dados)
            os.utime(peso, ns=(1000000000,1000000000))
            self.catalogo['visoes'][nome] = dict(rotulo=nome, disponivel=True,
                pesos=peso.name, sha256=hashlib.sha256(dados).hexdigest(), arquitetura='yolo', classes=[0],
                config=dict(modo='inteira', imgsz=1024, tile=1024, conf=.3, iou=.7, overlap=.2))
        self.salvar()

    def salvar(self):
        (self.root / 'data/modelos/especialistas.json').write_text(json.dumps(self.catalogo))

    def test_visoes_podem_usar_pesos_diferentes_com_mesmo_mtime(self):
        a = resolver_especialista(self.root, 'solo')
        b = resolver_especialista(self.root, 'drone')
        self.assertEqual(a['mt'], b['mt'])
        self.assertNotEqual(a['pesos_absolutos'], b['pesos_absolutos'])
        self.assertNotEqual(a['chave_modelo'], b['chave_modelo'])

    def test_modelo_bloqueado_nao_e_liberado_por_outra_visao(self):
        self.catalogo['visoes']['solo']['bloqueado'] = True
        self.salvar()
        with self.assertRaisesRegex(ValueError, 'bloqueado'):
            resolver_especialista(self.root, 'solo')
        self.assertEqual(resolver_especialista(self.root, 'drone')['rotulo'], 'drone')

    def test_peso_trocado_exige_nova_avaliacao(self):
        (self.root / 'solo.pt').write_bytes(b'outro peso')
        with self.assertRaisesRegex(ValueError, 'alterados'):
            resolver_especialista(self.root, 'solo')

    def test_candidato_indisponivel_nao_aparece(self):
        self.catalogo['visoes']['solo']['disponivel'] = False
        self.salvar()
        self.assertNotIn('solo', carregar_catalogo(self.root))

    def test_modelo_de_outra_arquitetura_tem_classes_proprias(self):
        self.catalogo['visoes']['solo'].update(arquitetura='rtdetr', classes=[19])
        self.salvar()
        p = resolver_especialista(self.root, 'solo')
        self.assertEqual(p['arquitetura'], 'rtdetr')
        self.assertEqual(p['classes'], (19,))

    def test_inferencia_encaminha_classes_e_augment(self):
        chamadas = []
        def predict(*args, **kwargs):
            chamadas.append(kwargs)
            return [SimpleNamespace(boxes=None)]
        modelo = SimpleNamespace(predict=predict)
        ct.detectar_inteira(modelo, np.zeros((32,32,3), dtype=np.uint8), classes=(19,), augment=True)
        self.assertEqual(chamadas[0]['classes'], (19,))
        self.assertTrue(chamadas[0]['augment'])

    def test_cache_do_app_inclui_identidade_do_modelo(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'app.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'detectar_bytes')
        nomes = [n.arg for n in fn.args.args]
        self.assertIn('modelo_chave', nomes)
        self.assertIn('classes', nomes)
        self.assertIn('augment', nomes)


if __name__ == '__main__':
    unittest.main()
