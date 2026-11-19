"""Confere coordenadas e duplicatas nas bordas entre janelas."""
import unittest
from types import SimpleNamespace
import numpy as np

from inferencia_recortes import detectar_recortes


class TensorFalso:
    def __init__(self, valores):
        self.valores = np.asarray(valores, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.valores


class CaixasFalsas:
    def __init__(self, caixas):
        self.xyxy = TensorFalso(caixas)
        self.conf = TensorFalso([.9] * len(caixas))

    def __len__(self):
        return len(self.conf.valores)


class RecortesTest(unittest.TestCase):
    def test_recortes_nao_herdam_tta_da_chamada_anterior(self):
        class Modelo:
            def __init__(self):
                self.estado = {'augment': True}

            def predict(self, janela, **kwargs):
                self.estado.update(kwargs)
                return [SimpleNamespace(boxes=None)]

        modelo = Modelo()
        detectar_recortes(modelo,np.zeros((100,150,3),dtype=np.uint8),device='cpu')
        self.assertFalse(modelo.estado['augment'])
        detectar_recortes(modelo,np.zeros((100,150,3),dtype=np.uint8),device='cpu',augment=True)
        self.assertTrue(modelo.estado['augment'])

    def test_desloca_coordenadas_e_remove_copia_na_sobreposicao(self):
        # Duas janelas de 100 px em imagem 150x100, com origem x=0 e x=50.
        # Ambas encontram o mesmo animal, sem mudar o tamanho da caixa.
        class Modelo:
            def __init__(self):
                self.chamadas = 0

            def predict(self, janela, **kwargs):
                caixas = [[60, 20, 90, 70]] if self.chamadas == 0 else [[10, 20, 40, 70]]
                self.chamadas += 1
                assert kwargs['imgsz'] == 1280
                return [SimpleNamespace(boxes=CaixasFalsas(caixas))]

        modelo = Modelo()
        b, s = detectar_recortes(modelo, np.zeros((100, 150, 3), dtype=np.uint8),
                                fracao=2 / 3, overlap=.5, device='cpu')
        self.assertEqual(modelo.chamadas, 2)
        np.testing.assert_allclose(b, [[60, 20, 90, 70]])
        np.testing.assert_allclose(s, [.9])

    def test_janelas_sem_deteccoes_devolvem_array_vazio(self):
        modelo = SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(boxes=None)])
        b, s = detectar_recortes(modelo, np.zeros((30, 40, 3), dtype=np.uint8), device='cpu')
        self.assertEqual(b.shape, (0, 4))
        self.assertEqual(s.shape, (0,))

    def test_fragmento_na_borda_artificial_nao_duplica_animal_inteiro(self):
        class Modelo:
            def __init__(self):
                self.chamadas = 0

            def predict(self, janela, **kwargs):
                caixas = [[80,20,100,60]] if self.chamadas == 0 else [[30,20,70,60]]
                self.chamadas += 1
                return [SimpleNamespace(boxes=CaixasFalsas(caixas))]

        b,s = detectar_recortes(Modelo(), np.zeros((100,150,3), dtype=np.uint8),
            fracao=2/3, overlap=.5, descartar_borda_interna=True, device='cpu')
        np.testing.assert_allclose(b, [[80,20,120,60]])

    def test_animal_cortado_pela_borda_da_foto_original_e_preservado(self):
        class Modelo:
            def __init__(self):
                self.chamadas = 0

            def predict(self, janela, **kwargs):
                self.chamadas += 1
                return [SimpleNamespace(boxes=CaixasFalsas([[0,20,10,60]]))] if self.chamadas == 1 else [SimpleNamespace(boxes=None)]

        b,s = detectar_recortes(Modelo(), np.zeros((100,150,3), dtype=np.uint8),
            fracao=2/3, overlap=.5, descartar_borda_interna=True, device='cpu')
        np.testing.assert_allclose(b, [[0,20,10,60]])


if __name__ == '__main__':
    unittest.main()
