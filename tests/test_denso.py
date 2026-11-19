import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
import torch
import densidade

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('treino_densidade',ROOT/'24_treinar_densidade.py')
treino=importlib.util.module_from_spec(spec);spec.loader.exec_module(treino)


class MapaConstante(torch.nn.Module):
    def forward(self,x):return torch.ones((len(x),1,x.shape[2]//8,x.shape[3]//8),device=x.device)


class DensoTest(unittest.TestCase):
    def test_mosaico_preserva_quantidade_dos_quatro_quadrantes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'imagem.jpg';cv2.imwrite(str(p),np.zeros((64,64,3),np.uint8))
            df=pd.DataFrame([dict(img=str(p),stem='a',arquivo='a.jpg',grupo='g',fonte='cattle_pontos',n_bovinos=4)])
            ds=treino.Dados(df,{'a':[[10,10],[40,10],[10,40],[40,40]]},True,True);ds.brasil_denso=[0]
            with patch.object(random,'random',side_effect=[0.,.9,.9,.9]),patch.object(random,'uniform',return_value=1.):
                im,den,n,*_=ds[0]
            self.assertEqual(n,16)
            self.assertAlmostEqual(float(den.sum())/densidade.ESCALA,16,places=4)
            for y,x in ((0,0),(0,40),(24,0),(24,40)):
                self.assertAlmostEqual(float(den[0,y:y+24,x:x+40].sum())/densidade.ESCALA,4,places=3)

    def test_negativo_nao_ganha_pontos_e_validacao_e_deterministica(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'imagem.jpg';cv2.imwrite(str(p),np.zeros((64,64,3),np.uint8))
            df=pd.DataFrame([dict(img=str(p),stem='a',arquivo='a.jpg',grupo='g',fonte='cattle_pontos',n_bovinos=0)])
            ds=treino.Dados(df,{'a':[]});a=ds[0];b=ds[0]
            self.assertEqual(a[2],0);self.assertEqual(float(a[1].sum()),0)
            self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[1],b[1]))

    def test_mapa_adaptativo_conserva_massa_no_reposicionamento(self):
        img=np.zeros((600,800,3),np.uint8)
        caixas=np.array([[100+i*60,250,140+i*60,300] for i in range(8)],float)
        n,mapa=densidade.prever_adaptativo(MapaConstante(),img,None,32,'cpu',(caixas,np.ones(8)))
        self.assertEqual(mapa.shape,img.shape[:2]);self.assertAlmostEqual(float(mapa.sum()),n,places=4)
        self.assertGreater(n,0);self.assertEqual(float(mapa[:100].sum()),0)

    def test_grupos_separados_e_cobertura_densa_no_teste(self):
        p=ROOT/'data/denso_cattle/divisao_final_densa.csv'
        if not p.exists():self.skipTest('Dados externos nao obtidos')
        df=pd.read_csv(p);self.assertEqual(len(df),850)
        self.assertEqual(df.groupby('grupo').conjunto.nunique().max(),1)
        self.assertEqual(df.groupby('sha256').conjunto.nunique().max(),1)
        self.assertGreaterEqual(df[df.conjunto=='test'].n_bovinos.max(),75)
        pontos=json.loads((p.parent/'pontos.json').read_text())
        self.assertEqual(sum(len(v) for v in pontos.values()),int(df.n_bovinos.sum()))


if __name__=='__main__':unittest.main()
