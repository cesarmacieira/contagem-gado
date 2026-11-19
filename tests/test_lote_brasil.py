"""Integridade dos dados novos, sem depender de GPU nem carregar pesos."""
import csv
import json
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class LoteBrasilTest(unittest.TestCase):
    def setUp(self):
        self.base=ROOT/'data/brasil_revisado'
        if not (self.base/'manifesto.csv').exists(): self.skipTest('Lote novo ainda nao exportado')
        with (self.base/'manifesto.csv').open(encoding='utf-8-sig',newline='') as f:
            self.rows=list(csv.DictReader(f))

    def test_autores_separados_e_parciais_fora(self):
        vistos={}
        for r in self.rows:
            self.assertEqual(vistos.setdefault(r['grupo'],r['conjunto']),r['conjunto'])
        fonte=json.loads((ROOT/'data/brasil_lote_01/anotacoes.json').read_text(encoding='utf-8'))
        usados={r['arquivo'] for r in self.rows}
        for f in fonte:
            if f['status']!='completa': self.assertNotIn(f.get('arquivo_exportado',f['arquivo']),usados)

    def test_yolo_coco_gabarito_concordam(self):
        coco=json.loads((self.base/'anotacoes_coco.json').read_text(encoding='utf-8'))
        images={r['file_name']:r for r in coco['images']}
        for r in self.rows:
            image=images[r['arquivo']];n=0
            for line in (self.base/'labels'/(Path(r['arquivo']).stem+'.txt')).read_text().splitlines():
                cl,x,y,w,h=map(float,line.split()); self.assertEqual(cl,0)
                self.assertTrue(0<w<=1 and 0<h<=1)
                self.assertGreaterEqual(x-w/2,-1e-7);self.assertLessEqual(x+w/2,1+1e-7)
                self.assertGreaterEqual(y-h/2,-1e-7);self.assertLessEqual(y+h/2,1+1e-7)
                n+=1
            annotations=[a for a in coco['annotations'] if a['image_id']==image['id']]
            self.assertEqual(n,len(annotations));self.assertEqual(n,int(r['n_bovinos']))

    def test_divisao_congelada_preservada(self):
        freeze=json.loads((ROOT/'data/brasil_lote_01/divisao_final_pre_treino.json').read_text(encoding='utf-8'))
        self.assertEqual({(x['arquivo'],x['conjunto']) for x in freeze['imagens']},{(x['arquivo'],x['conjunto']) for x in self.rows})


if __name__=='__main__': unittest.main()
