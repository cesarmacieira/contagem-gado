#!/usr/bin/env python
"""Avalia o modelo treinado nas imagens de data/teste_brasil contra o gabarito.csv.

O gabarito tem MÍNIMO e MÁXIMO por imagem (contagem visual: em cenas densas ou com animais ocultos não existe um número exato).
Status: 'dentro' = a contagem do modelo está entre o mínimo e o máximo; 'abaixo'/'acima' = fora do intervalo.
O modo (recortes ou imagem inteira) e o limiar vêm do cenário escolhido em --dominio, como no app.

Com --varrer testa várias estratégias de inferência (imagem inteira em tamanhos maiores, NMS mais permissivo, recortes com zoom) e vários limiares,
e mostra qual contagem melhor com o gabarito. Atenção: escolher a melhor entre dezenas de combinações em 10 imagens dá um resultado otimista;
confirme a escolha em imagens novas antes de confiar nela.

Uso:  python 05_avaliar_gabarito.py [--dominio foto_openimages] [--conf 0.25] [--pasta data/teste_brasil] [--varrer]
Saída: tabela no terminal, <pasta>/resultados.csv e imagens anotadas em <pasta>/saida/
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
import hashlib
import sys
from pathlib import Path

import cv2
import pandas as pd
from avaliacao import comparar_caixas

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
import contagem as ct  # noqa: E402


def estrategias():
    for imgsz in (1024, 1280, 1536, 1920):
        for iou in (0.4, 0.5, 0.6, 0.7, 0.85):
            yield f"inteira imgsz={imgsz} iou={iou}", ("inteira", imgsz, iou)
    for esc in (1.5, 2.0):
        for iou in (0.5, 0.7):
            yield f"recortes com zoom x{esc} iou={iou}", ("zoom", esc, iou)


def detectar_estrategia(modelo, img, est, meta, device):
    tipo, p, iou = est
    if tipo == "inteira":
        return ct.detectar_inteira(modelo, img, imgsz=p, iou_nms=iou, device=device)
    h, w = img.shape[:2]
    g = cv2.resize(img, (int(w * p), int(h * p)), interpolation=cv2.INTER_CUBIC)
    b, s = ct.detectar(modelo, g, tile=meta["tile"], overlap=0.3, iou_nms=iou, device=device)
    return b / p, s


def varrer(modelo, gab, pasta: Path, meta: dict, device, saida: Path, dominio: str) -> None:
    imgs = {r.arquivo: cv2.imread(str(pasta / r.arquivo)) for r in gab.itertuples()}
    assert all(v is not None for v in imgs.values()), "imagem ilegível na pasta"
    grade = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    linhas, por_img, detalhes = [], {}, []
    for nome, est in estrategias():
        print("testando:", nome, flush=True)
        det = {r.arquivo: detectar_estrategia(modelo, imgs[r.arquivo], est, meta, device) for r in gab.itertuples()}
        for conf in grade:
            prev = {a: len(ct.filtrar(b, s, conf)[0]) for a, (b, s) in det.items()}
            fora = [0 if r.minimo <= prev[r.arquivo] <= r.maximo else (prev[r.arquivo] - r.minimo if prev[r.arquivo] < r.minimo else prev[r.arquivo] - r.maximo)
                    for r in gab.itertuples()]
            for r in gab.itertuples():
                detalhes.append(dict(estrategia=nome, conf=conf, arquivo=r.arquivo, minimo=r.minimo, maximo=r.maximo, previsto=prev[r.arquivo]))
            meio = (gab.minimo + gab.maximo) / 2
            linhas.append(dict(estrategia=nome, conf=conf, dentro=sum(f == 0 for f in fora), erro_medio_ao_intervalo=round(sum(abs(f) for f in fora) / len(fora), 2),
                               previsto_total=sum(prev.values()), gabarito_meio=float(meio.sum()),
                               viés_pct=round((sum(prev.values()) / float(meio.sum()) - 1) * 100, 1) if meio.sum() else None))
            por_img[(nome, conf)] = prev
    res = pd.DataFrame(linhas).sort_values(["dentro", "erro_medio_ao_intervalo"], ascending=[False, True])
    res.to_csv(saida / "varredura.csv", index=False, encoding="utf-8-sig")
    print("\nMelhores 15 combinações (mais imagens dentro do intervalo; depois menor erro):\n")
    print(res.head(15).to_string(index=False))
    pd.DataFrame(detalhes).to_csv(saida / "varredura_por_imagem.csv", index=False, encoding="utf-8-sig")
    sz = meta.get("imgsz_por_dominio", {}).get(dominio, meta["tile"])
    iou = meta.get("iou_por_dominio", {}).get(dominio, 0.7)
    conf_base = meta["conf_por_dominio"][dominio]
    print(f"\nReferência salva: {dominio}, imgsz={sz}, iou={iou}, conf={conf_base}")
    print("EXPLORATÓRIO: o vencedor foi escolhido neste gabarito; não representa desempenho independente.")
    m = res.iloc[0]
    print(f"\nMelhor: {m.estrategia}, limiar {m.conf}. Contagem por imagem:")
    prev = por_img[(m.estrategia, m.conf)]
    print(pd.DataFrame({"arquivo": gab.arquivo, "minimo": gab.minimo, "maximo": gab.maximo, "previsto": [prev[x] for x in gab.arquivo]}).to_string(index=False))
    print(f"\nTabela completa: {saida / 'varredura.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", type=Path, default=PROJ / "data" / "teste_brasil")
    ap.add_argument("--dominio", default="foto_openimages", help="cenário do modelo: define modo e limiar (veja conf_por_dominio em metricas.json)")
    ap.add_argument("--conf", type=float, default=None, help="limiar de confiança; padrão = o do cenário")
    ap.add_argument("--varrer", action="store_true", help="testa várias estratégias de inferência e limiares (demora alguns minutos)")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--perfil", choices=["validado", "brasil-experimental"], default="validado")
    ap.add_argument("--artefatos", type=Path, default=PROJ / "data/processed")
    ap.add_argument("--saida", type=Path, default=None)
    a = ap.parse_args()
    if a.conf is not None and not ct.CONF_MIN <= a.conf <= 1:
        ap.error(f"--conf deve estar entre {ct.CONF_MIN} e 1")
    if a.iou is not None and not 0 < a.iou < 1:
        ap.error("--iou deve estar entre 0 e 1")
    if a.imgsz is not None and a.imgsz <= 0:
        ap.error("--imgsz deve ser positivo")

    meta_p, pesos = a.artefatos / "metricas.json", a.artefatos / "modelo_gado.pt"
    if not (meta_p.exists() and pesos.exists()):
        sys.exit("Falta data/processed/metricas.json ou modelo_gado.pt. Rode antes o notebook 02_treinamento.ipynb.")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if a.dominio not in meta["conf_por_dominio"]:
        sys.exit(f"Cenário '{a.dominio}' não existe. Opções: {list(meta['conf_por_dominio'])}")
    modo = meta["modo_por_dominio"][a.dominio]
    conf = a.conf if a.conf is not None else float(meta["conf_por_dominio"][a.dominio])
    imgsz = int(meta.get("imgsz_por_dominio", {}).get(a.dominio, meta["tile"]))
    iou = float(meta.get("iou_por_dominio", {}).get(a.dominio, 0.7))
    if a.perfil == "brasil-experimental":
        modo, imgsz, conf, iou = "inteira", 1280, 0.15, 0.7
    imgsz = a.imgsz if a.imgsz is not None else imgsz
    iou = a.iou if a.iou is not None else iou
    conf = a.conf if a.conf is not None else conf
    gab_p = a.pasta / "gabarito.csv"
    if not gab_p.exists():
        sys.exit(f"Sem {gab_p}")
    gab = pd.read_csv(gab_p, encoding="utf-8-sig")

    if gab.empty or not gab.arquivo.is_unique or (gab.minimo < 0).any() or (gab.maximo < gab.minimo).any():
        sys.exit("Gabarito vazio, duplicado ou com intervalos inválidos")
    if any(Path(x).name != x for x in gab.arquivo):
        sys.exit("Gabarito deve usar nomes sem caminhos")

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    modelo = YOLO(str(pesos))
    saida = a.saida or a.pasta / "avaliacoes" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saida.mkdir(parents=True, exist_ok=False)
    (saida / "execucao.json").write_text(json.dumps(dict(perfil=a.perfil, dominio=a.dominio, imgsz=imgsz, conf=conf, iou=iou,
        modo=modo, versao_inferencia=ct.VERSAO_INFERENCIA, exploratorio=True,
        pesos_sha256=hashlib.sha256(pesos.read_bytes()).hexdigest()), indent=2), encoding="utf-8")
    if a.varrer:
        varrer(modelo, gab, a.pasta, meta, device, saida, a.dominio)
        return
    print(f"Cenário: {a.dominio} | modo: {modo} | imgsz: {imgsz} | limiar: {conf} | dispositivo: {device}\n")

    linhas = []
    for r in gab.itertuples():
        img = cv2.imread(str(a.pasta / r.arquivo))
        if img is None:
            sys.exit(f"Imagem ilegível ou ausente: {a.pasta / r.arquivo}")
        if modo == "tiles":
            caixas, scores = ct.detectar(modelo, img, tile=meta["tile"], overlap=meta["overlap"], iou_local=iou, iou_nms=meta.get("iou_juncao", 0.5), device=device,
                                          legado=meta.get("versao_inferencia", 1) == 1 and a.perfil == "validado")
        else:
            caixas, scores = ct.detectar_inteira(modelo, img, imgsz=imgsz, iou_nms=iou, device=device,
                                                   max_det=300 if meta.get("versao_inferencia", 1) == 1 and a.perfil == "validado" else 1000)
        b, s = ct.filtrar(caixas, scores, conf)
        n = len(b)
        status = "dentro" if r.minimo <= n <= r.maximo else ("abaixo" if n < r.minimo else "acima")
        erro = 0 if status == "dentro" else (n - r.minimo if status == "abaixo" else n - r.maximo)
        cv2.imwrite(str(saida / r.arquivo), ct.desenhar(img, b),
                    [cv2.IMWRITE_JPEG_QUALITY, 90] if Path(r.arquivo).suffix.lower() in (".jpg", ".jpeg") else [cv2.IMWRITE_PNG_COMPRESSION, 3])
        rot = a.pasta / "labels" / (Path(r.arquivo).stem + ".txt")
        mm = {}
        if rot.exists():
            h, w = img.shape[:2]
            mm = comparar_caixas(ct.yolo_para_pixels(rot.read_text(encoding="utf-8").splitlines(), w, h), b, s)
        linhas.append(dict(**mm, arquivo=r.arquivo, minimo=r.minimo, maximo=r.maximo, confianca_gabarito=r.confianca, previsto=n, status=status,
                           erro_ao_intervalo=erro, conf_media=round(float(s.mean()), 2) if len(s) else None))
    res = pd.DataFrame(linhas)
    res.to_csv(saida / "resultados.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    n_in = int((res.status == "dentro").sum())
    print("\nConjunto exploratório; dentro do intervalo não é precisão de detecção.")
    print(f"\nDentro do intervalo: {n_in} de {len(res)} | erro médio ao intervalo (vacas): {res.erro_ao_intervalo.abs().mean():.2f}")
    print(f"Total do gabarito: {res.minimo.sum()} a {res.maximo.sum()} | total previsto: {res.previsto.sum()}")
    print(f"Imagens anotadas em: {saida}\nTabela: {saida / 'resultados.csv'}")


if __name__ == "__main__":
    main()
