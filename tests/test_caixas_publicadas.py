from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('linhagens',ROOT/'36_fechar_linhagens_caixas.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class CaixasPublicadasTest(unittest.TestCase):
    def test_versoes_do_mesmo_video_tem_mesma_linhagem(self):
        self.assertEqual(mod.linhagem('simmental_new1-2712_jpg'),mod.linhagem('simmental_new1-122_jpg'))
        self.assertNotEqual(mod.linhagem('simmental_new1-2712_jpg'),mod.linhagem('simmental_new2-2712_jpg'))

    def test_linhagens_e_hashes_nao_atravessam_conjuntos(self):
        path=ROOT/'data/detectors_cows/divisao_linhagens.csv'
        if not path.exists():self.skipTest('Conjunto publico nao obtido')
        df=pd.read_csv(path)
        for col in ('grupo','linhagem','sha256'):
            self.assertEqual(df.groupby(col).conjunto.nunique().max(),1,col)
        self.assertGreaterEqual(df[df.conjunto=='test'].n_bovinos.max(),20)

    def test_rotulos_canonicos_preservam_quantidade_e_limites(self):
        path=ROOT/'data/detectors_cows/divisao_linhagens.csv'
        if not path.exists():self.skipTest('Conjunto publico nao obtido')
        df=pd.read_csv(path)
        for r in df.itertuples():
            linhas=[l for l in Path(r.rot).read_text(encoding='utf-8').splitlines() if l.strip()]
            self.assertEqual(len(linhas),r.n_bovinos,r.stem)
            for l in linhas:
                cl,x,y,w,h=map(float,l.split());self.assertEqual(cl,0)
                self.assertTrue(np.isfinite([x,y,w,h]).all() and w>0 and h>0)
                self.assertGreaterEqual(x-w/2,-1e-6);self.assertLessEqual(x+w/2,1+1e-6)
                self.assertGreaterEqual(y-h/2,-1e-6);self.assertLessEqual(y+h/2,1+1e-6)


if __name__=='__main__':unittest.main()
