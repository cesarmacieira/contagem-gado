import unittest
import numpy as np
from iluminacao import combinar,variantes


class IluminacaoTests(unittest.TestCase):
    def test_nao_muda_geometria_original(self):
        im=np.full((32,24,3),80,np.uint8)
        v=variantes(im)
        self.assertEqual(len(v),4)
        np.testing.assert_array_equal(v['original'],im)
        self.assertTrue(all(x.shape==im.shape for x in v.values()))

    def test_mesma_passagem_nao_duplica_votos(self):
        vazia=(np.empty((0,4)),np.empty(0))
        b,s,a=combinar(dict(original=vazia,contraste=(np.array([[0,0,10,10],[0,0,10,10]]),np.array([.8,.9]))))
        self.assertEqual(len(b),0)

    def test_preserva_base_e_recupera_com_duas_passagens(self):
        base=(np.array([[0,0,10,10]]),np.array([.6]))
        outras=(np.array([[30,30,40,40]]),np.array([.7]))
        b,s,a=combinar(dict(original=base,contraste=outras,sombras=outras))
        self.assertEqual(len(b),2)
        np.testing.assert_array_equal(b[0],base[0][0])
        self.assertEqual(len(a),1)


if __name__=='__main__':unittest.main()
