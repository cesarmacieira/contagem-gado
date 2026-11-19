"""Entrada controlada de imagens próprias/web revisadas. Não usa predições como gabarito."""
from pathlib import Path
import hashlib
import pandas as pd

import contagem as ct


def carregar_revisados(projeto):
    projeto = Path(projeto)
    base = projeto / "data" / "brasil_revisado"
    ampliado = projeto / "data" / "brasil_dataset_v2"
    if (ampliado / "manifesto.csv").exists():
        base = ampliado
    manifesto = base / "manifesto.csv"
    colunas = ["dominio", "fonte", "subconjunto", "stem", "img", "rot", "n_vacas", "grupo", "conjunto"]
    if not manifesto.exists():
        return pd.DataFrame(columns=colunas)
    df = pd.read_csv(manifesto, keep_default_na=False)
    exigidas = {"arquivo", "grupo", "conjunto", "revisado", "autor", "licenca", "url"}
    if not exigidas.issubset(df.columns):
        raise ValueError(f"Manifesto deve conter {sorted(exigidas)}")
    df = df[df.revisado.astype(str).str.lower().isin(["true", "1", "sim"])].copy()
    if df.empty:
        return pd.DataFrame(columns=colunas)
    if not df.conjunto.isin(["train", "val", "test"]).all() or (df.grupo.str.strip() == "").any():
        raise ValueError("Informe grupo da cena e conjunto train/val/test")
    if df.groupby("grupo").conjunto.nunique().max() > 1 or not df.arquivo.is_unique:
        raise ValueError("Cena em mais de um conjunto ou arquivos repetidos")
    reservados = [projeto / "data/teste_brasil", projeto / "data/brasil_novo_teste"]
    arquivos_teste = [f for pasta in reservados for f in pasta.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    proibidos = {hashlib.sha256(f.read_bytes()).hexdigest() for f in arquivos_teste}
    hashes, registros = {}, []
    for r in df.itertuples():
        nome = Path(r.arquivo)
        if nome.name != r.arquivo:
            raise ValueError("arquivo deve ser apenas um nome, sem pastas")
        img, rot = base / "images" / nome, base / "labels" / (nome.stem + ".txt")
        if not img.is_file() or not rot.is_file():
            raise ValueError(f"Imagem ou rótulo ausente: {nome}")
        if not all(str(x).strip() for x in (r.autor, r.licenca, r.url)):
            raise ValueError(f"Falta proveniência de {nome}")
        digest = hashlib.sha256(img.read_bytes()).hexdigest()
        if digest in proibidos or digest in hashes:
            raise ValueError(f"Imagem duplicada ou do conjunto exploratório: {nome}")
        hashes[digest] = r.conjunto
        linhas = ct.linhas_yolo_unicas(rot.read_text(encoding="utf-8").splitlines())
        registros.append(dict(dominio="brasil_solo", fonte="BrasilRevisado", subconjunto=r.grupo,
                              stem=nome.stem, img=img, rot=rot, n_vacas=len(linhas),
                              grupo=f"brasil/{r.grupo}", conjunto=r.conjunto))
    return pd.DataFrame(registros, columns=colunas)
