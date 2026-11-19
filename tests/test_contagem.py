import unittest
import numpy as np

import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem


class ContagemTest(unittest.TestCase):
    def test_cattle_bull_mesma_caixa_contam_uma_vez(self):
        linhas = ["0 0.5 0.5 0.4 0.4", "0 0.500000 0.5 0.4 0.4"]
        self.assertEqual(len(ct.yolo_para_pixels(linhas, 100, 100)), 1)

    def test_bezerro_contido_na_vaca_nao_e_duplicata(self):
        b = np.array([[0, 0, 100, 100], [20, 20, 40, 40]], dtype=float)
        self.assertEqual(len(ct.nms(b, np.array([.9, .8]))), 2)

    def test_mesmo_recorte_preserva_decisao_local(self):
        b = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=float)
        self.assertEqual(len(ct.nms(b, np.array([.9, .8]), origens=np.array([0, 0]))), 2)
        self.assertEqual(len(ct.nms(b, np.array([.9, .8]), origens=np.array([0, 1]))), 1)

    def test_fragmento_nao_vira_fundo_no_treino(self):
        b = np.array([[90, 20, 130, 60]])
        self.assertTrue(ct.tile_tem_fragmento_sem_rotulo(b, 0, 0, 100, 100))
        self.assertFalse(ct.tile_tem_fragmento_sem_rotulo(b, 80, 0, 100, 100))

    def test_matching_nao_compensa_duplicata_com_omissao(self):
        ref = np.array([[0, 0, 10, 10], [30, 30, 40, 40]])
        pred = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        self.assertEqual(comparar_caixas(ref, pred, np.array([.9, .8])), dict(TP=1, FP=1, FN=1))
        self.assertEqual(metricas_contagem([2], [2])["MAE"], 0)

    def test_grade_cobre_bordas(self):
        cantos = ct.grade_tiles(2300, 1300, 1024, .2)
        self.assertIn((1276, 276), cantos)
        with self.assertRaises(ValueError):
            ct.grade_tiles(100, 100, 1024, 1)


if __name__ == "__main__":
    unittest.main()
