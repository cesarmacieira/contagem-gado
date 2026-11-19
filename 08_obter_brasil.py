"""Busca fotos reais no Wikimedia Commons, com licença/autoria, para revisão antes do treino.

python 08_obter_brasil.py --max-imagens 60
Não baixa de sites sem proveniência nem transforma predições em rótulos.
"""
import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent
UA = {"User-Agent": "ContagemGadoResearch/2.0 (public cattle images; attribution preserved)"}


def baixar(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()


def texto(valor):
    return html.unescape(re.sub(r"<[^>]+>", " ", valor)).strip()


def dhash(img):
    cinza = cv2.cvtColor(cv2.resize(img, (9, 8)), cv2.COLOR_BGR2GRAY)
    return cinza[:, 1:] > cinza[:, :-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-imagens", type=int, default=60)
    a = ap.parse_args()
    if a.max_imagens < 1:
        ap.error("--max-imagens deve ser positivo")
    base = PROJ / "data/brasil_candidatos"; (base / "images").mkdir(parents=True, exist_ok=True)
    manifest = base / "fontes.json"
    fontes = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    vistos = {r["titulo"] for r in fontes}
    proibidos, hashes_ref = set(), []
    for pasta in (PROJ / "data/teste_brasil", PROJ / "data/brasil_novo_teste"):
        csv = pasta / "fontes.csv"
        if csv.exists():
            proibidos.update("File:" + str(x) for x in pd.read_csv(csv, encoding="utf-8-sig").titulo)
        for f in pasta.iterdir() if pasta.exists() else []:
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                img = cv2.imread(str(f))
                if img is not None:
                    hashes_ref.append(dhash(img))
    novos = 0
    for query in ('Nelore cattle', 'gado rebanho Brasil', 'cattle herd Brazil'):
        continuar = {}
        while novos < a.max_imagens:
            params = dict(action="query", generator="search", gsrsearch=query, gsrnamespace=6, gsrlimit=50,
                          prop="imageinfo", iiprop="url|extmetadata|size", iiurlwidth=1600, format="json", **continuar)
            try:
                res = json.loads(baixar("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)))
            except Exception as e:
                print(f"Não foi possível consultar o Commons: {e}. As imagens existentes foram preservadas.")
                raise SystemExit(1)
            for page in res.get("query", {}).get("pages", {}).values():
                titulo = page["title"]
                if titulo in vistos or titulo in proibidos or not page.get("imageinfo"):
                    continue
                info = page["imageinfo"][0]; md = info.get("extmetadata", {})
                licenca = texto(md.get("LicenseShortName", {}).get("value", ""))
                autor = texto(md.get("Artist", {}).get("value", ""))
                if not autor or not (licenca.startswith("CC BY") or licenca in ("CC0", "Public domain")):
                    continue
                if info.get("width", 0) < 600 or not urllib.parse.urlparse(info.get("url", "")).path.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                url_img = info.get("thumburl", info["url"])
                nome = f"commons_{page['pageid']}" + Path(urllib.parse.urlparse(info["url"]).path).suffix.lower()
                try:
                    dados = baixar(url_img)
                    im = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
                    if im is None:
                        raise ValueError("não é imagem legível")
                    h = dhash(im)
                    if any(np.count_nonzero(h != r) <= 4 for r in hashes_ref):
                        continue  # evita rebaixamentos/cópias do teste, mesmo em outra resolução
                    digest = hashlib.sha256(dados).hexdigest()
                    if digest in {r.get("sha256") for r in fontes}:
                        continue
                    (base / "images" / nome).write_bytes(dados)
                    grupo = "autor_" + hashlib.sha256(autor.encode()).hexdigest()[:12]
                    fontes.append(dict(arquivo=nome, titulo=titulo, autor=autor, licenca=licenca,
                                       licenca_url=md.get("LicenseUrl", {}).get("value", ""),
                                       url=info.get("descriptionurl", ""), url_imagem=url_img, grupo=grupo,
                                       status="pendente_revisao", sha256=digest))
                    vistos.add(titulo); hashes_ref.append(h); novos += 1
                    manifest.write_text(json.dumps(fontes, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"{novos}/{a.max_imagens}: {titulo} [{licenca}]", flush=True)
                except Exception as e:
                    print(f"Falha em {titulo}: {e}", flush=True)
                if novos >= a.max_imagens:
                    break
                time.sleep(.25)
            continuar = res.get("continue", {})
            if not continuar:
                break
    print(f"{novos} novas imagens. Revise com: streamlit run 09_revisar_brasil.py")


if __name__ == "__main__":
    main()
