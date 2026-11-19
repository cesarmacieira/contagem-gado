"""ETAPA 1b - Dados extras para um modelo GERAL de gado (qualquer raça, qualquer ângulo).

Complementa o 01_obter_dados.py (Zenodo 11048412, pasto aéreo). Só baixa e converte para YOLO. Não treina.

Fontes (todas conferidas nas páginas originais):
  openimages : Open Images V7, classes "Cattle" (/m/01xq0k1) e "Bull" (/m/0cnyhnx). Fotos comuns, vários ângulos.
               Só entram imagens SEM caixa de grupo (IsGroupOf) e SEM desenho/ilustração (IsDepiction),
               porque uma caixa de grupo cobre várias vacas e estraga a contagem.
  colo       : Niche-Squad/COLO (Hugging Face), vistas 1_top, 2_side e 3_external, splits train e test.
               Vacas leiteiras em instalação. Usa só o "bbox" (pixels, COCO x,y,w,h); a "segmentation" do
               dataset está deslocada e NÃO é usada.
Fora do script (não dá para automatizar):
  CattleEyeView : o repositório só oferece um formulário do Google. Veja o README.
  Zenodo 11048412: use 01_obter_dados.py (rode no Windows; o Zenodo bloqueia alguns ambientes).

Uso (na pasta do projeto):
    python 03_obter_dados_extras.py                       # tudo (Open Images com limite de treino padrão)
    python 03_obter_dados_extras.py --fonte openimages    # só Open Images
    python 03_obter_dados_extras.py --fonte colo          # só COLO
    python 03_obter_dados_extras.py --max-train 500       # teste rápido (poucas imagens de treino do Open Images)
    python 03_obter_dados_extras.py --csv-train caminho/oidv6-train-annotations-bbox.csv   # se já baixou o CSV (1,2 GB)
    python 03_obter_dados_extras.py --inventario          # só recontar o que já existe em data/extras

Saída (tudo em data/extras, formato YOLO, classe única 0 = gado):
    openimages/images|labels/<split>/   colo/images|labels/<config>_<split>/
    inventario_extras.csv               contagem por fonte e subconjunto
Pode interromper (Ctrl+C) e rodar de novo: continua de onde parou.

Citações:
  Open Images V7 (Google): anotações CC BY 4.0; as imagens têm licenças próprias (em geral CC BY 2.0).
  COLO: Niche-Squad, arXiv 2407.20372 (licença MIT no Hugging Face).
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from contagem import linhas_yolo_unicas

BASE = Path(__file__).resolve().parent / "data" / "extras"
UA = {"User-Agent": "Mozilla/5.0 (projeto-contagem-gado; uso academico)"}

OI_CLASSES = {"/m/01xq0k1": "Cattle", "/m/0cnyhnx": "Bull"}
OI_CSV = {
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
}
OI_IMG = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"
OI_COLS = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction", "IsOccluded", "IsTruncated"]

CSV_LOCAL: dict[str, str] = {}  # split -> caminho de um CSV de caixas já baixado (opção --csv-train)

COLO_URL = "https://huggingface.co/datasets/Niche-Squad/COLO/resolve/main/{cfg}/{split}-00000-of-00001.parquet"
COLO_CFGS = ["1_top", "2_side", "3_external"]
COLO_SPLITS = ["train", "test"]


# ------------------------------------------------------------------ utilitários
def baixar_arquivo(url: str, destino: Path, tentativas: int = 5) -> Path:
    """Baixa um arquivo inteiro (usa .parte e só renomeia no fim)."""
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".parte")
    ultimo = None
    for t in range(tentativas):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r, open(parcial, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if parcial.stat().st_size == 0:
                raise IOError("arquivo vazio")
            parcial.replace(destino)
            return destino
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            ultimo = e
            time.sleep(min(2 ** t, 20))
    raise RuntimeError(f"falha ao baixar {url}: {ultimo}")


def escrever_yolo(caminho: Path, caixas: list[tuple[float, float, float, float]]) -> int:
    """caixas = (xmin, ymin, xmax, ymax) normalizadas 0-1. Devolve quantas caixas válidas foram escritas."""
    linhas = []
    for x0, y0, x1, y1 in caixas:
        x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in (x0, y0, x1, y1))
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        linhas.append(f"0 {x0 + w / 2:.6f} {y0 + h / 2:.6f} {w:.6f} {h:.6f}")
    linhas = linhas_yolo_unicas(linhas)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")
    return len(linhas)


# ------------------------------------------------------------------ Open Images
def filtrar_csv_oi(split: str) -> Path:
    """Lê o CSV de caixas em fluxo (sem guardar o arquivo inteiro) e guarda só as linhas de Cattle/Bull."""
    cache = BASE / "openimages" / "_cache" / f"{split}_alvo.csv"
    if cache.exists() and cache.stat().st_size > 0:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    parcial = cache.with_suffix(".csv.parte")
    ultimo = None
    for t in range(4):
        try:
            print(f"[openimages] lendo anotações de {split} em fluxo (pode levar alguns minutos no train)...")
            if CSV_LOCAL.get(split):  # CSV já baixado (ex.: com curl -C -), lido do disco
                fonte = open(CSV_LOCAL[split], "rb")
            else:
                fonte = urllib.request.urlopen(urllib.request.Request(OI_CSV[split], headers=UA), timeout=120)
            total = guardadas = 0
            with fonte as r, open(parcial, "w", newline="", encoding="utf-8") as saida:
                leitor = csv.DictReader(io.TextIOWrapper(r, encoding="utf-8", newline=""))
                faltam = [c for c in OI_COLS if c not in (leitor.fieldnames or [])]
                if faltam:
                    sys.exit(f"ERRO: CSV do Open Images sem as colunas {faltam}. Formato mudou; me mostre esta mensagem.")
                escritor = csv.writer(saida)
                escritor.writerow(OI_COLS)
                for linha in leitor:
                    total += 1
                    if linha["LabelName"] in OI_CLASSES:
                        escritor.writerow([linha[c] for c in OI_COLS])
                        guardadas += 1
            parcial.replace(cache)
            print(f"[openimages] {split}: {total} linhas lidas, {guardadas} caixas de Cattle/Bull")
            return cache
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            ultimo = e
            time.sleep(min(2 ** t * 3, 30))
    raise RuntimeError(f"falha ao ler anotações de {split}: {ultimo}")


def selecionar_oi(split: str) -> dict[str, list[tuple[float, float, float, float]]]:
    """Imagens utilizáveis: sem caixa de grupo e sem desenho. Devolve {id: [(xmin, ymin, xmax, ymax), ...]}."""
    cache = filtrar_csv_oi(split)
    caixas: dict[str, list] = defaultdict(list)
    excluidas: dict[str, str] = {}
    with open(cache, newline="", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            i = l["ImageID"]
            if l["IsGroupOf"] == "1":
                excluidas[i] = "grupo"
            elif l["IsDepiction"] == "1":
                excluidas.setdefault(i, "desenho")
            caixas[i].append((float(l["XMin"]), float(l["YMin"]), float(l["XMax"]), float(l["YMax"])))
    bons = {i: c for i, c in caixas.items() if i not in excluidas}
    n_grupo = sum(1 for v in excluidas.values() if v == "grupo")
    print(f"[openimages] {split}: {len(caixas)} imagens com gado | descartadas: {n_grupo} com caixa de grupo, "
          f"{len(excluidas) - n_grupo} com desenho | utilizáveis: {len(bons)}")
    return bons


def baixar_oi(split: str, limite: int, threads: int, seed: int) -> None:
    bons = selecionar_oi(split)
    ids = sorted(bons)
    random.Random(seed).shuffle(ids)
    if limite and len(ids) > limite:
        ids = ids[:limite]
    img_dir, lab_dir = BASE / "openimages" / "images" / split, BASE / "openimages" / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lock, feito, falhas = threading.Lock(), [0], []

    def um(i: str) -> None:
        destino = img_dir / f"{i}.jpg"
        if not (destino.exists() and destino.stat().st_size > 0):
            parcial = destino.with_suffix(".jpg.parte")
            ultimo = None
            for t in range(4):
                try:
                    req = urllib.request.Request(OI_IMG.format(split=split, id=i), headers=UA)
                    with urllib.request.urlopen(req, timeout=60) as r:
                        dados = r.read()
                    if dados[:3] != b"\xff\xd8\xff":
                        raise IOError("não é JPEG")
                    parcial.write_bytes(dados)
                    parcial.replace(destino)
                    break
                except (urllib.error.URLError, OSError, TimeoutError) as e:
                    ultimo = e
                    time.sleep(min(2 ** t, 10))
            else:
                raise RuntimeError(f"{i}: {ultimo}")
        escrever_yolo(lab_dir / f"{i}.txt", bons[i])

    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(um, i): i for i in ids}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                falhas.append(str(e))
            with lock:
                feito[0] += 1
                print(f"\r    {split}: {feito[0]}/{len(ids)} imagens ({feito[0] / max(time.time() - t0, 1e-9):.1f}/s)", end="", flush=True)
    print()
    if falhas:
        print(f"[openimages] {split}: {len(falhas)} imagens falharam (rode de novo para tentar outra vez). Ex.: {falhas[:2]}")


# ------------------------------------------------------------------ COLO
def converter_colo(cfg: str, split: str) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("ERRO: falta o pyarrow para ler o COLO. Rode: python -m pip install pyarrow")
    pasta = f"{cfg}_{split}"
    pq_path = baixar_arquivo(COLO_URL.format(cfg=cfg, split=split), BASE / "colo" / "_parquet" / f"{cfg}_{split}.parquet")
    img_dir, lab_dir = BASE / "colo" / "images" / pasta, BASE / "colo" / "labels" / pasta
    img_dir.mkdir(parents=True, exist_ok=True)
    tabela = pq.read_table(pq_path)
    n_img = n_cx = 0
    for linha in tabela.to_pylist():
        nome = Path(linha["filename"]).stem
        w, h = float(linha["width"]), float(linha["height"])
        destino = img_dir / f"{nome}.jpg"
        if not destino.exists():
            destino.write_bytes(linha["image"]["bytes"])
        caixas = []
        for x, y, bw, bh in linha["annotations"]["bbox"]:  # COCO em pixels: x, y (canto superior esquerdo), largura, altura
            caixas.append((x / w, y / h, (x + bw) / w, (y + bh) / h))
        n_cx += escrever_yolo(lab_dir / f"{nome}.txt", caixas)
        n_img += 1
    print(f"[colo] {pasta}: {n_img} imagens, {n_cx} caixas")


# ------------------------------------------------------------------ inventário
def inventario() -> None:
    linhas = []
    for fonte in sorted(p for p in BASE.glob("*") if (p / "labels").is_dir()):
        for sub in sorted((fonte / "labels").iterdir()):
            txts = list(sub.glob("*.txt"))
            caixas = sum(1 for t in txts for l in t.read_text(encoding="utf-8").splitlines() if l.strip())
            com_gado = sum(1 for t in txts if t.stat().st_size > 0)
            imgs = len(list((fonte / "images" / sub.name).glob("*.jpg")))
            linhas.append({"fonte": fonte.name, "subconjunto": sub.name, "imagens": imgs, "rotulos": len(txts),
                           "imagens_com_gado": com_gado, "caixas": caixas})
    if not linhas:
        print("Nada em data/extras ainda.")
        return
    BASE.mkdir(parents=True, exist_ok=True)
    saida = BASE / "inventario_extras.csv"
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"\nInventário salvo em {saida}")
    for l in linhas:
        print(f"  {l['fonte']:11s} {l['subconjunto']:20s} imagens={l['imagens']:5d} com gado={l['imagens_com_gado']:5d} caixas={l['caixas']:6d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", choices=["tudo", "openimages", "colo"], default="tudo")
    ap.add_argument("--max-train", type=int, default=2000, help="[openimages] limite de imagens do split train (0 = todas)")
    ap.add_argument("--threads", type=int, default=8, help="[openimages] downloads em paralelo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv-train", default="", help="[openimages] caminho do oidv6-train-annotations-bbox.csv já baixado (opcional)")
    ap.add_argument("--inventario", action="store_true", help="só recontar o que já existe")
    a = ap.parse_args()
    if a.csv_train:
        CSV_LOCAL["train"] = a.csv_train
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "classes.txt").write_text("gado\n", encoding="utf-8")
    if not a.inventario:
        if a.fonte in ("tudo", "openimages"):
            for split in ("validation", "test", "train"):
                baixar_oi(split, a.max_train if split == "train" else 0, a.threads, a.seed)
        if a.fonte in ("tudo", "colo"):
            for cfg in COLO_CFGS:
                for split in COLO_SPLITS:
                    converter_colo(cfg, split)
    inventario()
    print("\nPRONTO. Lembretes:"
          "\n  - CattleEyeView: baixe pelo formulário do repositório (github.com/AnimalEyeQ/CattleEyeView), veja o README."
          "\n  - Zenodo 11048412: python 01_obter_dados.py --seletivo")
