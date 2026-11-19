import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from integracao_denso import conferir


class IntegracaoDensoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        (self.base / 'images').mkdir()
        (self.base / 'labels').mkdir()
        self.manifesto, self.gab = [], []
        self.coco = dict(images=[], annotations=[], categories=[dict(id=1, name='bovino')])
        for i, split in enumerate(('train', 'val', 'test'), 1):
            nome = f'{split}.png'
            Image.new('RGB', (100, 80), (i * 40, 10, 20)).save(self.base / 'images' / nome)
            (self.base / 'labels' / f'{split}.txt').write_text('0 0.5 0.5 0.4 0.5\n')
            self.manifesto.append(dict(arquivo=nome, conjunto=split, grupo=f'cena{i}', revisado='sim',
                autor='Revisor teste', licenca='CC0', url='https://example.test/imagem'))
            self.gab.append(dict(arquivo=nome, n_bovinos=1, conjunto=split, grupo=f'cena{i}'))
            self.coco['images'].append(dict(id=i, file_name=nome, width=100, height=80))
            self.coco['annotations'].append(dict(id=i, image_id=i, category_id=1, bbox=[30, 20, 40, 40], iscrowd=0))
        self.salvar()

    def salvar(self):
        for nome, rows in [('manifesto.csv', self.manifesto), ('gabarito.csv', self.gab)]:
            campos = list(dict.fromkeys(k for r in rows for k in r))
            with (self.base / nome).open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=campos)
                w.writeheader()
                w.writerows(rows)
        (self.base / 'anotacoes_coco.json').write_text(json.dumps(self.coco))
        (self.base / 'dicionario.json').write_text('{"classe_0":"bovino"}')

    def test_lote_coerente_tem_snapshot_sem_executar_modelo(self):
        r = conferir(self.base)
        self.assertTrue(r['pronto'], r['erros'])
        self.assertEqual(r['caixas'], 3)
        self.assertFalse(r['inferencia_executada'])
        self.assertEqual(len(r['snapshot_sha256']), 64)

    def test_snapshot_muda_quando_a_proveniencia_muda(self):
        antes = conferir(self.base)['snapshot_sha256']
        self.manifesto[0]['url'] = 'https://example.test/outra-origem'
        self.salvar()
        self.assertNotEqual(antes, conferir(self.base)['snapshot_sha256'])

    def test_grupo_nao_atravessa_divisoes(self):
        self.manifesto[1]['grupo'] = 'cena1'
        self.salvar()
        r = conferir(self.base)
        self.assertFalse(r['pronto'])
        self.assertTrue(any('Grupo em mais' in e for e in r['erros']))

    def test_mesma_foto_rebatizada_e_rejeitada(self):
        (self.base / 'images/val.png').write_bytes((self.base / 'images/train.png').read_bytes())
        r = conferir(self.base)
        self.assertFalse(r['pronto'])
        self.assertTrue(any('identica repetida' in e for e in r['erros']))

    def test_caixa_fora_da_imagem_e_rejeitada(self):
        (self.base / 'labels/train.txt').write_text('0 0.95 0.5 0.4 0.5\n')
        r = conferir(self.base)
        self.assertFalse(r['pronto'])
        self.assertTrue(any('fora da imagem' in e for e in r['erros']))

    def test_coco_nao_pode_omitir_uma_caixa(self):
        self.coco['annotations'].pop()
        self.salvar()
        r = conferir(self.base)
        self.assertFalse(r['pronto'])
        self.assertTrue(any('COCO inconsistente' in e for e in r['erros']))

    def test_gabarito_nao_pode_compensar_duplicacao(self):
        self.gab[0]['n_bovinos'] = 2
        self.salvar()
        self.assertFalse(conferir(self.base)['pronto'])

    def test_n_bovinos_nao_pode_ocultar_intervalo_ambiguo(self):
        self.gab[0].update(minimo=1, maximo=2, ambiguos=1)
        self.salvar()
        r = conferir(self.base)
        self.assertFalse(r['pronto'])
        self.assertTrue(any('ambiguidades' in e for e in r['erros']))

    def test_imagem_ambigua_nao_pode_ser_aprovada(self):
        self.manifesto[0]['status'] = 'parcial'
        self.salvar()
        r = conferir(self.base)
        self.assertFalse(r['pronto'])
        self.assertTrue(any('parcial/ambigua' in e for e in r['erros']))

    def test_pendente_nao_exige_rotulos_e_nao_entra_no_lote(self):
        self.manifesto.append(dict(arquivo='pendente.png', conjunto='', grupo='', revisado='nao',
            autor='', licenca='', url=''))
        self.salvar()
        r = conferir(self.base)
        self.assertTrue(r['pronto'], r['erros'])
        self.assertEqual(r['imagens_pendentes'], 1)
        self.assertEqual(r['imagens_aprovadas'], 3)


if __name__ == '__main__':
    unittest.main()
