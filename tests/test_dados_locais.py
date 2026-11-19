import tempfile
import unittest
from pathlib import Path
import pandas as pd

from dados_locais import carregar_revisados


class DadosLocaisTest(unittest.TestCase):
    def preparar(self, raiz):
        base = raiz / "data/brasil_revisado"
        (base / "images").mkdir(parents=True)
        (base / "labels").mkdir()
        return base

    def linha(self, arquivo="a.jpg", grupo="cena1", conjunto="train", revisado=True):
        return dict(arquivo=arquivo, grupo=grupo, conjunto=conjunto, revisado=revisado,
                    autor="Autor", licenca="CC0", url="https://commons.wikimedia.org/wiki/File:Teste.jpg")

    def test_imagem_nao_revisada_nao_entra(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp); base = self.preparar(raiz)
            pd.DataFrame([self.linha(revisado=False)]).to_csv(base / "manifesto.csv", index=False)
            self.assertTrue(carregar_revisados(raiz).empty)

    def test_cena_nao_pode_vazar(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp); base = self.preparar(raiz)
            pd.DataFrame([self.linha(), self.linha("b.jpg", conjunto="test")]).to_csv(base / "manifesto.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Cena"):
                carregar_revisados(raiz)

    def test_png_reservado_nao_entra_no_treino(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp); base = self.preparar(raiz)
            teste = raiz / "data/brasil_novo_teste"; teste.mkdir()
            (teste / "reservada.png").write_bytes(b"imagem-reservada")
            (base / "images/a.jpg").write_bytes(b"imagem-reservada")
            (base / "labels/a.txt").write_text("0 .5 .5 .4 .4")
            pd.DataFrame([self.linha()]).to_csv(base / "manifesto.csv", index=False)
            with self.assertRaisesRegex(ValueError, "duplicada"):
                carregar_revisados(raiz)
