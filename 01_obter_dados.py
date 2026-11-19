"""ETAPA 1 - Obter os dados reais (imagens de drone de gado em pastagem, Zenodo 11048412, CC BY 4.0).

Só baixa e confere. Não treina nada.

Dataset: "Drone images and their annotations of grazing cows" (Institut de l'Elevage, projeto ICAERUS, v2 abr/2024)
  - 1.385 imagens RGB nadir (4000x3000 ou 5280x3956 px), 60-100 m de altura, DJI Mavic 3
  - 4.941 caixas de vacas em formato YOLO e PASCAL VOC
  - 1 arquivo: Cattle_drone_images_042024.zip (16,6 GB, MD5 publicado no Zenodo)

Uso (na pasta do projeto):
    python 01_obter_dados.py             # baixa o zip completo (~16,6 GB), confere o MD5. Lento se o Zenodo limitar a velocidade
    python 01_obter_dados.py --seletivo  # RÁPIDO: lê o zip pela rede e baixa só as imagens com vaca + uma parte das sem vaca
    python 01_obter_dados.py --listar    # só mostra o inventário do zip que já está em data/raw

Saída:
    data/raw/Cattle_drone_images_042024.zip   (modo completo) arquivo original, conferido por MD5
    data/raw/Cattle_drone_images_selecao.zip  (modo --seletivo) todos os rótulos + as imagens escolhidas
    data/raw/inventario.csv                   imagens e rótulos por fazenda/voo (para conferência)

Pode interromper (Ctrl+C) e rodar de novo: os dois modos continuam de onde pararam.
Citação: Helary, L., Lebreton, A. (2024). Drone images and their annotations of grazing cows (v2).
         Zenodo. doi:10.5281/zenodo.11048412
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

NOME = "Cattle_drone_images_042024.zip"
NOME_SELECAO = "Cattle_drone_images_selecao.zip"
MD5 = "c18911fb11cb58741b9cfa16516ca8cd"  # publicado na página do registro no Zenodo
URL = f"https://zenodo.org/records/11048412/files/{NOME}?download=1"
RAW = Path(__file__).resolve().parent / "data" / "raw"
CACHE = RAW / "selecao_parcial"  # arquivos baixados no modo seletivo (permite retomar)
UA = {"User-Agent": "Mozilla/5.0 (projeto-contagem-gado; uso academico)"}
EXT_IMG = (".jpg", ".jpeg", ".png")


# ------------------------------------------------------------------ modo completo
def md5(path: Path) -> str:
    h = hashlib.md5()
    tam, lido = path.stat().st_size, 0
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
            lido += len(chunk)
            print(f"\r    conferindo MD5: {lido / tam * 100:5.1f}%", end="", flush=True)
    print()
    return h.hexdigest()


def baixar() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    final, parcial = RAW / NOME, RAW / f"{NOME}.parte"
    if final.exists():
        if md5(final) == MD5:
            print(f"[ok] {NOME}: já baixado e MD5 confere")
            return final
        print(f"[!] {NOME}: MD5 não confere, baixando de novo")
        final.unlink()
    inicio = parcial.stat().st_size if parcial.exists() else 0
    req = urllib.request.Request(URL, headers={**UA, **({"Range": f"bytes={inicio}-"} if inicio else {})})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # já estava completo
            parcial.rename(final)
            return final
        raise
    if inicio and resp.status != 206:  # servidor ignorou o Range: recomeça
        inicio = 0
    total = inicio + int(resp.headers.get("Content-Length", 0))
    feito = inicio
    t0, b0 = time.time(), feito
    print(f"[baixando] {NOME} ({total / 1e9:.1f} GB)" + (f", retomando de {inicio / 1e9:.2f} GB" if inicio else ""))
    with open(parcial, "ab" if inicio else "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            feito += len(chunk)
            v = (feito - b0) / max(time.time() - t0, 1e-9) / 1e6
            print(f"\r    {feito / 1e9:7.2f} / {total / 1e9:.2f} GB  ({v:4.1f} MB/s)", end="", flush=True)
    print()
    parcial.rename(final)
    if md5(final) != MD5:
        final.unlink()
        sys.exit(f"ERRO: MD5 de {NOME} não confere. Rode de novo.")
    print(f"[ok] {NOME}: MD5 confere")
    return final


# ------------------------------------------------------------------ modo seletivo (HTTP Range)
class RangeFile:
    """Arquivo somente leitura sobre HTTP com cabeçalho Range. Permite abrir um zip remoto sem baixá-lo inteiro."""

    BLOCO = 256 * 1024

    def __init__(self, url: str):
        self.url0, self.url, self.pos, self._buf = url, url, 0, (0, b"")
        self.size = self._descobrir_tamanho()

    def _abrir(self, ini: int, fim: int):
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={ini}-{fim}"})
        return urllib.request.urlopen(req, timeout=90)

    def _descobrir_tamanho(self) -> int:
        with self._abrir(0, 0) as r:
            self.url = r.geturl()  # URL final depois de redirecionamentos
            cr = r.headers.get("Content-Range", "")
            if r.status != 206 or "/" not in cr:
                raise RuntimeError("O servidor não aceita Range (HTTP 206). Use o modo completo: python 01_obter_dados.py")
            return int(cr.rsplit("/", 1)[1])

    def _get(self, ini: int, fim: int) -> bytes:
        ult = None
        for tent in range(8):
            try:
                with self._abrir(ini, fim) as r:
                    dados = r.read()
                if len(dados) != fim - ini + 1:
                    raise IOError(f"resposta incompleta ({len(dados)} de {fim - ini + 1} bytes)")
                return dados
            except urllib.error.HTTPError as e:
                ult = e
                if e.code in (401, 403, 410):  # link redirecionado pode ter expirado
                    self.url = self.url0
                    self._descobrir_tamanho()
            except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as e:  # inclui IncompleteRead e conexão derrubada
                ult = e
            time.sleep(min(2 ** tent, 20))
        raise RuntimeError(f"falha ao ler bytes {ini}-{fim}: {ult}")

    # interface mínima de arquivo usada pelo zipfile
    def seekable(self): return True
    def tell(self): return self.pos
    def close(self): pass

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        ini = self.pos
        bi, bd = self._buf
        if bi <= ini and ini + n <= bi + len(bd):
            out = bd[ini - bi:ini - bi + n]
        else:
            fim = min(self.size, ini + max(n, self.BLOCO))
            dados = self._get(ini, fim - 1)
            self._buf = (ini, dados) if len(dados) <= 4 * self.BLOCO else (0, b"")
            out = dados[:n]
        self.pos += len(out)
        return out


def _zip_remoto() -> zipfile.ZipFile:
    return zipfile.ZipFile(RangeFile(URL))


def seletivo(neg_frac: float, trabalhadores: int, max_pos: int = 0, seed: int = 42) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    print("[seletivo] abrindo o zip pela rede (só o índice, alguns MB)...")
    zr = _zip_remoto()
    infos = {i.filename: i for i in zr.infolist() if not i.is_dir()}
    print(f"[seletivo] itens no zip: {len(infos)}")
    imgs, rots, extras, duplicados = {}, {}, [], []
    for nome in infos:
        p = parse(nome)
        if p is None:
            if nome.endswith("obj.names") or nome.endswith("conditions_summary.xlsx"):
                extras.append(nome)
            continue
        tipo, fazenda, voo, stem = p
        destino = imgs if tipo == "imagem" else rots
        chave = (fazenda, stem.lower())  # imagem e rótulo são pareados por fazenda + nome (o nome tem data e hora)
        if chave in destino:
            duplicados.append(nome)
        destino[chave] = nome
    print(f"[seletivo] imagens: {len(imgs)} | rótulos: {len(rots)} | extras (obj.names, conditions_summary): {len(extras)}")
    if duplicados:
        sys.exit(f"ERRO: {len(duplicados)} nomes repetidos na mesma fazenda (ex.: {duplicados[:2]}). Me mostre esta mensagem.")
    if not imgs or not rots:
        sys.exit("Estrutura do zip diferente da esperada. Rode: python 01_obter_dados.py --estrutura e me mostre a saída.")

    # 1) rótulos (pequenos): baixa todos, em paralelo, e guarda em cache
    local = threading.local()

    def baixar_membro(nome: str) -> Path:
        destino = CACHE / nome
        if destino.exists() and destino.stat().st_size == infos[nome].file_size:
            return destino
        destino.parent.mkdir(parents=True, exist_ok=True)
        dados, ult = None, None
        for _ in range(4):
            try:
                if not hasattr(local, "z"):
                    local.z = _zip_remoto()  # um leitor por thread, reaproveitado
                dados = local.z.read(nome)
                break
            except Exception as e:  # noqa: BLE001  (rede instável: recria o leitor e tenta de novo)
                ult = e
                local.__dict__.pop("z", None)
                time.sleep(2)
        if dados is None:
            raise RuntimeError(f"não consegui baixar {nome}: {ult}")
        tmp = destino.with_suffix(destino.suffix + ".parte")
        tmp.write_bytes(dados)
        tmp.replace(destino)
        return destino

    def paralelo(nomes: list[str], rotulo: str) -> None:
        feito, t0, total_b = 0, time.time(), sum(infos[n].file_size for n in nomes)
        lido = 0
        lock = threading.Lock()
        with ThreadPoolExecutor(trabalhadores) as ex:
            futs = {ex.submit(baixar_membro, n): n for n in nomes}
            for f in as_completed(futs):
                f.result()
                with lock:
                    feito += 1
                    lido += infos[futs[f]].file_size
                    v = lido / max(time.time() - t0, 1e-9) / 1e6
                    print(f"\r    {rotulo}: {feito}/{len(nomes)} | {lido / 1e9:6.2f} / {total_b / 1e9:.2f} GB | {v:5.1f} MB/s", end="", flush=True)
        print()

    paralelo(sorted(rots.values()) + extras, "rótulos")
    # .txt dentro das pastas YOLO que não pertencem a nenhuma imagem (listas, anotações soltas): mostra e ignora
    orfaos = sorted(k for k in rots if k not in imgs)
    for k in orfaos:
        conteudo = (CACHE / rots[k]).read_text(encoding="utf-8", errors="replace").strip().replace("\n", " | ")
        print(f"[seletivo] ignorado (sem imagem correspondente): {rots[k]}  conteúdo: {conteudo[:80]!r}")
    rots = {k: v for k, v in rots.items() if k in imgs}
    sem_rotulo = sorted(k for k in imgs if k not in rots)
    if sem_rotulo:
        sys.exit(f"ERRO: {len(sem_rotulo)} imagens sem arquivo de rótulo (ex.: {[imgs[k] for k in sem_rotulo[:2]]}). Me mostre esta mensagem.")
    com_vaca = {}
    for k, nome in rots.items():
        linhas = [l for l in (CACHE / nome).read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        com_vaca[k] = len(linhas)
    caixas = sum(com_vaca.values())
    pos = sorted(k for k in imgs if com_vaca.get(k, 0) > 0)
    neg = sorted(k for k in imgs if com_vaca.get(k, 0) == 0)
    print(f"[seletivo] caixas nos rótulos: {caixas} (Zenodo declara 4941) | imagens com vaca: {len(pos)} | sem vaca: {len(neg)}")

    # 2) imagens: todas com vaca + parte das sem vaca
    rng = random.Random(seed)
    def ordem(chaves):  # embaralha com semente fixa e põe primeiro o que já está no cache (não rebaixa)
        emb = list(chaves)
        rng.shuffle(emb)
        no_cache = [k for k in emb if (CACHE / imgs[k]).exists()]
        resto = set(no_cache)
        return no_cache + [k for k in emb if k not in resto]

    pos = ordem(pos)
    if max_pos and len(pos) > max_pos:
        pos = pos[:max_pos]
    neg_sel = ordem(neg)[:min(len(neg), int(round(neg_frac * len(pos))))]
    escolha = [imgs[k] for k in pos + neg_sel]
    gb = sum(infos[n].file_size for n in escolha) / 1e9
    print(f"[seletivo] vou baixar {len(pos)} imagens com vaca + {len(neg_sel)} sem vaca = {len(escolha)} imagens ({gb:.1f} GB de {sum(i.file_size for i in infos.values()) / 1e9:.1f} GB)")
    paralelo(escolha, "imagens")

    # 3) monta o zip reduzido (mesmos caminhos do original, para o notebook ler igual)
    final = RAW / NOME_SELECAO
    tmp = final.with_suffix(".zip.parte")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zo:
        for nome in sorted(rots.values()) + extras + escolha:
            zo.write(CACHE / nome, nome)
    tmp.replace(final)
    print(f"[ok] {final.name}: {final.stat().st_size / 1e9:.1f} GB (rótulos completos + imagens escolhidas)")
    print(f"     A pasta {CACHE.name}/ é só cache para retomar; pode apagar depois.")
    return final


def estrutura() -> None:
    """Mostra como o zip está organizado (só lê o índice remoto, não baixa imagens)."""
    zr = _zip_remoto()
    infos = [i for i in zr.infolist() if not i.is_dir()]
    print(f"Itens (arquivos) no zip: {len(infos)}")
    grupos = defaultdict(list)
    for i in infos:
        partes = i.filename.split("/")
        ext = partes[-1].rpartition(".")[2].lower() if "." in partes[-1] else "(sem extensão)"
        pasta = "/".join(p if not p.startswith("DJI_") else "DJI_*" for p in partes[1:-1]) if len(partes) > 2 else "(raiz)"
        grupos[(ext, partes[0] if len(partes) > 1 else "(raiz)", pasta)].append(i)
    print("\nextensão | pasta-raiz | subpastas (voos como DJI_*) -> quantidade, tamanho total")
    for (ext, raiz, pasta), v in sorted(grupos.items()):
        print(f"  .{ext:5s} | {raiz} | {pasta} -> {len(v)} arquivos, {sum(x.file_size for x in v) / 1e6:.1f} MB")
    print("\nExemplos de nomes (até 3 por grupo de extensão):")
    por_ext = defaultdict(list)
    for i in infos:
        por_ext[i.filename.rpartition(".")[2].lower()].append(i.filename)
    for ext, nomes in sorted(por_ext.items()):
        print(f"  .{ext} ({len(nomes)})")
        for n in nomes[:3]:
            print("     ", n)


# ------------------------------------------------------------------ inventário
def parse(nome: str):
    """Devolve (tipo, fazenda, voo, stem) para imagens e rótulos YOLO; None para o resto.

    Estrutura real do zip (conferida com --estrutura):
      Cattle_drone_images/<fazenda>/JPGImages/<voo>/<imagem>.JPG
      Cattle_drone_images/<fazenda>/YOLO_1.1/<voo>/obj_train_data/<imagem>.txt   (em Mauron, alguns ficam direto em <voo>/)
    A pasta dos rótulos se chama YOLO_1.1 ou YOLO1.1. train.txt, obj.names, obj.data não são rótulos.
    """
    partes = [p for p in nome.split("/") if p]
    if not partes or "__MACOSX" in partes or nome.endswith("/"):
        return None
    stem, _, ext = partes[-1].rpartition(".")
    if stem.startswith("~$"):
        return None
    pastas = partes[:-1]
    if "JPGImages" in pastas:
        if "." + ext.lower() not in EXT_IMG:
            return None
        i, tipo = pastas.index("JPGImages"), "imagem"
    else:
        i = next((k for k, p in enumerate(pastas) if p.lower().startswith("yolo")), None)
        if i is None or ext.lower() != "txt" or stem.lower() == "train":
            return None
        tipo = "rotulo"
    fazenda = partes[i - 1] if i > 0 else "(raiz)"
    voo = partes[i + 1] if len(partes) > i + 2 else "(sem voo)"
    return tipo, fazenda, voo, stem


def inventario(zp: Path) -> None:
    cont = defaultdict(lambda: {"imagens": 0, "rotulos": 0, "caixas": 0, "imagens_com_vaca": 0})
    outros = 0
    with zipfile.ZipFile(zp) as z:
        infos = z.infolist()
        print(f"Itens no zip: {len(infos)}")
        for info in infos:
            p = parse(info.filename)
            if p is None:
                if not info.is_dir():
                    outros += 1
                continue
            tipo, fazenda, voo, _ = p
            c = cont[(fazenda, voo)]
            if tipo == "imagem":
                c["imagens"] += 1
            else:
                c["rotulos"] += 1
                linhas = [l for l in z.read(info).decode("utf-8", "replace").splitlines() if l.strip()]
                c["caixas"] += len(linhas)
                c["imagens_com_vaca"] += 1 if linhas else 0
    linhas = [{"fazenda": f, "voo": v, **c} for (f, v), c in sorted(cont.items())]
    if not linhas:
        sys.exit("Nenhuma imagem/rótulo reconhecido. A estrutura do zip é diferente do descrito; mostre as 30 primeiras "
                 "linhas de: python -c \"import zipfile;print(*zipfile.ZipFile(r'%s').namelist()[:30], sep='\\n')\"" % zp)
    saida = RAW / "inventario.csv"
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"\nInventário salvo em {saida}")
    print(f"  Voos: {len(linhas)} | imagens no zip: {sum(l['imagens'] for l in linhas)} | rótulos: {sum(l['rotulos'] for l in linhas)}")
    print(f"  Imagens com vaca: {sum(l['imagens_com_vaca'] for l in linhas)} | caixas: {sum(l['caixas'] for l in linhas)}")
    print(f"  (esperado pela página do Zenodo para o dataset todo: 1385 imagens e 4941 caixas)  | outros arquivos no zip: {outros}")
    for fz in sorted({l["fazenda"] for l in linhas}):
        s = [l for l in linhas if l["fazenda"] == fz]
        print(f"  {fz:14s} voos={len(s):3d} imagens={sum(l['imagens'] for l in s):4d} caixas={sum(l['caixas'] for l in s):5d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true", help="só gera o inventário do zip já baixado")
    ap.add_argument("--estrutura", action="store_true", help="mostra a organização do zip remoto (sem baixar imagens) e sai")
    ap.add_argument("--seletivo", action="store_true", help="baixa só as imagens com vaca + parte das sem vaca (bem mais rápido)")
    ap.add_argument("--neg-frac", type=float, default=0.5, help="[seletivo] imagens sem vaca por imagem com vaca (padrão 0,5)")
    ap.add_argument("--max-pos", type=int, default=0, help="[seletivo] limita as imagens COM vaca (ex.: 150 para um teste rápido); 0 = todas")
    ap.add_argument("--threads", type=int, default=6, help="[seletivo] conexões em paralelo (padrão 6)")
    a = ap.parse_args()
    if a.estrutura:
        estrutura()
        sys.exit(0)
    if a.listar:
        zp = RAW / NOME if (RAW / NOME).exists() else RAW / NOME_SELECAO
        if not zp.exists():
            sys.exit("Nenhum zip em data/raw. Rode sem --listar para baixar.")
    elif a.seletivo:
        zp = seletivo(a.neg_frac, a.threads, a.max_pos)
    else:
        zp = baixar()
    inventario(zp)
    print("\nPRONTO. Próximo passo: jupyter lab 02_treinamento.ipynb")
