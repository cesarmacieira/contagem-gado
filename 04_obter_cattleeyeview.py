#!/usr/bin/env python
"""Converte o CattleEyeView (download manual em .\\CattleEyeView) para YOLO em data/extras/cattleeyeview.

Lê direto dos zips `annotation-*-00N.zip` (Google Takeout, partes disjuntas), sem extrair tudo. Usa a pasta `detect` (caixas, classe única 'cow').
Os quadros vêm de 14 vídeos de uma rampa, vista de cima, 1920x1080; quadros vizinhos são quase iguais, então pega no máximo N quadros por vídeo, espaçados.
A divisão dos autores é por vídeo (test = 01,05,07,09,10; a pasta val repete os quadros do test e NÃO é usada).

Uso:  python 04_obter_cattleeyeview.py [--max-por-video 120] [--origem CattleEyeView] [--inventario]
Saída: data/extras/cattleeyeview/{images,labels}/<video>/ e inventario.csv (retoma de onde parou).
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
import zipfile
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent


def indexar(origem: Path):
    """Devolve {(split, video, stem): (zip, caminho_img)} e {(split, video, stem): (zip, caminho_rotulo)} da pasta detect."""
    zips = sorted(glob.glob(str(origem / "annotation-*.zip")))
    if not zips:
        sys.exit(f"Nenhum annotation-*.zip em {origem}. Baixe o CattleEyeView e coloque os zips nessa pasta.")
    imgs, labs = {}, {}
    for f in zips:
        with zipfile.ZipFile(f) as z:
            for n in z.namelist():
                p = n.split("/")
                if len(p) == 6 and p[1] == "detect" and p[2] in ("images", "labels") and not n.endswith("/"):
                    chave = (p[3], p[4].replace(".mp4", ""), p[5].rsplit(".", 1)[0])
                    destino = imgs if p[2] == "images" else labs
                    assert chave not in destino, f"arquivo repetido entre zips: {n}"
                    destino[chave] = (f, n)
    return imgs, labs


def ler_rotulo(texto: str, nome: str) -> list[str]:
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    for l in linhas:
        v = l.split()
        assert len(v) == 5 and v[0] == "0", f"rótulo fora do formato 'cow x y w h' em {nome}: {l[:60]}"
        assert all(0.0 <= float(x) <= 1.0 for x in v[1:]), f"coordenada fora de [0,1] em {nome}: {l[:60]}"
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origem", type=Path, default=PROJ / "CattleEyeView")
    ap.add_argument("--saida", type=Path, default=PROJ / "data" / "extras" / "cattleeyeview")
    ap.add_argument("--max-por-video", type=int, default=120, help="máximo de quadros por vídeo, espaçados (0 = todos)")
    ap.add_argument("--inventario", action="store_true", help="só recontar o que já existe em --saida")
    a = ap.parse_args()
    inv_csv = a.saida / "inventario.csv"

    if not a.inventario:
        imgs, labs = indexar(a.origem)
        # a pasta val deve repetir a test (README do dataset); confere antes de ignorá-la
        for k in [k for k in imgs if k[0] == "val"]:
            assert ("test",) + k[1:] in imgs, f"val/{k[1]}/{k[2]} não existe em test: a premissa 'val = test' não vale"
        pares = {}
        for k in imgs:
            if k[0] in ("train", "test") and k in labs:
                pares.setdefault((k[0], k[1]), []).append(k)
        sem_img = sum(1 for k in labs if k[0] in ("train", "test") and k not in imgs)
        print(f"{sum(len(v) for v in pares.values())} quadros com imagem e rótulo; {sem_img} rótulos sem imagem (não usados)")
        (a.saida / "images").mkdir(parents=True, exist_ok=True)
        (a.saida / "labels").mkdir(parents=True, exist_ok=True)
        rng_idx = {}
        for (split, video), chaves in sorted(pares.items(), key=lambda x: x[0][1]):
            chaves = sorted(chaves, key=lambda k: k[2])
            if a.max_por_video and len(chaves) > a.max_por_video:
                chaves = [chaves[i] for i in np.unique(np.linspace(0, len(chaves) - 1, a.max_por_video).round().astype(int))]
            rng_idx[(split, video)] = chaves
        abertos = {}
        feitos = 0
        for (split, video), chaves in rng_idx.items():
            di, dl = a.saida / "images" / f"v{video}", a.saida / "labels" / f"v{video}"
            di.mkdir(exist_ok=True); dl.mkdir(exist_ok=True)
            for k in chaves:
                stem = f"v{video}_{k[2]}"
                ji, jl = di / f"{stem}.jpg", dl / f"{stem}.txt"
                if ji.exists() and jl.exists():
                    continue
                zi, ni = imgs[k]; zl, nl = labs[k]
                for f in (zi, zl):
                    abertos.setdefault(f, zipfile.ZipFile(f))
                linhas = ler_rotulo(abertos[zl].read(nl).decode("utf-8"), nl)
                ji.write_bytes(abertos[zi].read(ni))
                jl.write_text("\n".join(linhas), encoding="utf-8")
                feitos += 1
            print(f"vídeo {video} ({split}): {len(chaves)} quadros", flush=True)
        for z in abertos.values():
            z.close()
        print(f"{feitos} quadros novos gravados")
        autores = {v: s for (s, v) in rng_idx}
    else:
        autores = {}

    # inventário a partir do que existe em disco (reflete a execução atual)
    TESTE_AUTORES = {"01", "05", "07", "09", "10"}
    linhas_inv = []
    for img in sorted((a.saida / "images").glob("v*/*.jpg")):
        video = img.parent.name[1:]
        rot = a.saida / "labels" / img.parent.name / f"{img.stem}.txt"
        assert rot.exists(), f"sem rótulo: {img}"
        n = sum(1 for l in rot.read_text(encoding="utf-8").splitlines() if l.strip())
        linhas_inv.append(dict(video=video, split_autores="test" if video in TESTE_AUTORES else "train", stem=img.stem, n_vacas=n))
    with open(inv_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "split_autores", "stem", "n_vacas"])
        w.writeheader(); w.writerows(linhas_inv)
    por_video = {}
    for r in linhas_inv:
        v = por_video.setdefault(r["video"], [0, 0]); v[0] += 1; v[1] += r["n_vacas"]
    print("\nvídeo  quadros  caixas")
    for v, (ni, nc) in sorted(por_video.items()):
        print(f"  {v}  {ni:7d}  {nc:6d}")
    print(f"TOTAL  {len(linhas_inv)} quadros, {sum(r['n_vacas'] for r in linhas_inv)} caixas. Inventário: {inv_csv}")


if __name__ == "__main__":
    main()
