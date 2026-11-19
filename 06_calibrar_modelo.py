"""Recalibra pesos existentes sem treinar. Escolhe imgsz/conf/NMS só na validação.

python 06_calibrar_modelo.py
python 06_calibrar_modelo.py --promover  # backup e atualização dos artefatos do app
As dez fotos exploratórias do Brasil nunca entram na escolha.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao

PROJ = Path(__file__).resolve().parent


def catalogo_origem(origem):
    p = origem / "catalogo.csv"
    if p.exists():
        cat = pd.read_csv(p)
        for campo in ("img", "rot"):
            cat[campo] = cat[campo].map(Path)
        return cat
    div = pd.read_csv(origem / "divisao.csv")
    registros = []
    for r in div[div.usar].itertuples():
        d = r.dominio
        if d == "aereo_pasto":
            nome = f"{r.fazenda}__{str(r.voo).replace('/', '_')}__{r.stem}"
            base = PROJ / "data/interim" / r.conjunto
            img, rot = base / "imagens" / (nome + ".jpg"), base / "rotulos" / (nome + ".txt")
            grupo, modo = f"zen/{r.fazenda}/{r.voo}", "tiles"
        else:
            nome = f"{d}__{re.sub(r'[^A-Za-z0-9_-]', '_', r.stem)}"
            if d == "foto_openimages":
                sp = "validation" if r.conjunto == "val" else r.conjunto
                base = PROJ / "data/extras/openimages"
                img, rot = base / "images" / sp / (r.stem + ".jpg"), base / "labels" / sp / (r.stem + ".txt")
                grupo = f"oi/{r.stem}"
            elif d.startswith("colo_"):
                base = PROJ / "data/extras/colo"
                imgs = list((base / "images").glob(f"*/{r.stem}.jpg"))
                if len(imgs) != 1:
                    raise ValueError(f"Imagem COLO ambígua/ausente: {r.stem}")
                img = imgs[0]; rot = base / "labels" / img.parent.name / (r.stem + ".txt")
                g = re.match(r"(?:top|side|img)_(\d+)", r.stem)
                if not g:
                    raise ValueError(f"Nome COLO inválido: {r.stem}")
                grupo = f"colo/{'ext' if d == 'colo_externo' else 'cena'}/{g[1]}"
            elif d == "cev_rampa":
                base = PROJ / "data/extras/cattleeyeview"
                img, rot = base / "images" / r.voo / (r.stem + ".jpg"), base / "labels" / r.voo / (r.stem + ".txt")
                grupo = f"cev/{r.voo}"
            else:
                raise ValueError(f"Use catalogo.csv para este domínio: {d}")
            modo = "inteira"
        if not img.exists() or not rot.exists():
            raise FileNotFoundError(f"Imagem/rótulo ausente: {img} / {rot}")
        n = len(ct.linhas_yolo_unicas(rot.read_text(encoding="utf-8").splitlines()))
        registros.append(dict(arquivo=nome, dominio=d, fonte=r.fazenda, fazenda=r.fazenda, voo=r.voo,
                              grupo=grupo, stem=r.stem, conjunto=r.conjunto, n_vacas=n, img=img, rot=rot, modo=modo))
    return pd.DataFrame(registros)


def limpar_json(x):
    if isinstance(x, dict):
        return {str(k): limpar_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [limpar_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origem", type=Path, default=PROJ / "data/processed")
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--promover", action="store_true")
    a = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = a.saida or PROJ / "data/avaliacoes" / ("calibracao_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((a.origem / "metricas.json").read_text(encoding="utf-8"))
    pesos = a.origem / "modelo_gado.pt"
    digest = hashlib.sha256(pesos.read_bytes()).hexdigest()
    cat = catalogo_origem(a.origem)
    assert cat.arquivo.is_unique
    assert cat.groupby("grupo").conjunto.nunique().max() == 1
    # Reconta até catálogos exportados antes da deduplicação.
    cat["n_vacas"] = [len(ct.linhas_yolo_unicas(Path(p).read_text(encoding="utf-8").splitlines())) for p in cat.rot]
    cat.to_csv(out / "catalogo.csv", index=False)
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    modelo = YOLO(str(pesos))
    cache = PROJ / "data/cache_inferencia" / digest[:16]
    cache.mkdir(parents=True, exist_ok=True)
    refs, shapes = {}, {}

    def inferir(r, sz, iou):
        img = cv2.imread(str(r.img))
        if img is None:
            raise ValueError(f"Imagem ilegível: {r.img}")
        h, w = img.shape[:2]; shapes[r.arquivo] = (h, w)
        refs[r.arquivo] = ct.yolo_para_pixels(Path(r.rot).read_text(encoding="utf-8").splitlines(), w, h)
        chave = hashlib.sha256(Path(r.img).read_bytes() + repr((ct.VERSAO_INFERENCIA, r.modo, sz, iou, meta["overlap"], 1000)).encode()).hexdigest()
        p = cache / (chave + ".npz")
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                return z["caixas"].copy(), z["scores"].copy()
        if r.modo == "tiles":
            b, s = ct.detectar(modelo, img, tile=sz, overlap=meta["overlap"], iou_local=iou, device=device)
        else:
            b, s = ct.detectar_inteira(modelo, img, imgsz=sz, iou_nms=iou, device=device)
        np.savez_compressed(p, caixas=b, scores=s)
        return b, s

    grade_conf = np.round(np.arange(0.10, 0.91, 0.05), 2)
    linhas, configs = [], {}
    for dominio, val in cat[cat.conjunto == "val"].groupby("dominio"):
        modo = val.modo.iloc[0]
        tamanhos = (int(meta["tile"]),) if modo == "tiles" else (1024, 1280, 1536)
        print(f"Validação: {dominio}, {len(val)} imagens, {val.grupo.nunique()} grupos", flush=True)
        for sz in tamanhos:
            for iou in (0.5, 0.7, 0.85):
                det = {r.arquivo: inferir(r, sz, iou) for r in val.itertuples()}
                for conf in grade_conf:
                    n, tp, fp, fn = [], 0, 0, 0
                    for r in val.itertuples():
                        b, s = ct.filtrar(*det[r.arquivo], conf); n.append(len(b))
                        mm = comparar_caixas(refs[r.arquivo], b, s)
                        tp += mm["TP"]; fp += mm["FP"]; fn += mm["FN"]
                    linhas.append(dict(dominio=dominio, imgsz=sz, iou=iou, conf=float(conf),
                                       **metricas_contagem(val.n_vacas, n), **taxas_deteccao(tp, fp, fn), TP=tp, FP=fp, FN=fn))
                print(f"  imgsz={sz}, iou={iou} concluído", flush=True)
        v = pd.DataFrame([x for x in linhas if x["dominio"] == dominio])
        best = v.sort_values(["MAE", "F1_caixas", "imgsz", "iou", "conf"], ascending=[True, False, True, True, True]).iloc[0]
        configs[dominio] = dict(imgsz=int(best.imgsz), iou=float(best.iou), conf=float(best.conf))
        print("  Escolhido:", configs[dominio], flush=True)
    varredura = pd.DataFrame(linhas)
    varredura.to_csv(out / "varredura_conf.csv", index=False)
    escolhido = varredura[(varredura.imgsz == varredura.dominio.map({d: c["imgsz"] for d, c in configs.items()})) &
                         (varredura.iou == varredura.dominio.map({d: c["iou"] for d, c in configs.items()}))]
    global_conf = float(escolhido.groupby("conf").WAPE_pct.mean().idxmin())
    teste, det_teste = [], {}
    print("Teste: somente configurações já fixadas na validação", flush=True)
    for r in cat[cat.conjunto == "test"].itertuples():
        c = configs[r.dominio]; b, s = inferir(r, c["imgsz"], c["iou"])
        n_global = len(ct.filtrar(b, s, global_conf)[0])
        b, s = ct.filtrar(b, s, c["conf"])
        det_teste[r.arquivo] = (b, s)
        teste.append(dict(arquivo=r.arquivo, dominio=r.dominio, fazenda=r.fazenda, voo=r.voo, grupo=r.grupo,
                          real=r.n_vacas, previsto=len(b), erro=len(b)-r.n_vacas, previsto_conf_global=n_global,
                          imgsz_usado=c["imgsz"], iou_usado=c["iou"], conf_usada=c["conf"],
                          **comparar_caixas(refs[r.arquivo], b, s)))
    teste = pd.DataFrame(teste)
    teste.to_csv(out / "contagem_teste.csv", index=False)
    teste[["arquivo", "dominio", "TP", "FP", "FN"]].to_csv(out / "deteccao_teste.csv", index=False)
    por_dominio = {}
    for d, g in teste.groupby("dominio"):
        tp, fp, fn = g[["TP", "FP", "FN"]].sum()
        por_dominio[d] = dict(**configs[d], **metricas_contagem(g.real, g.previsto), **taxas_deteccao(tp, fp, fn))
    densidade = teste.assign(faixa=pd.cut(teste.real, [-1, 0, 2, 5, 10, 20, 50, np.inf]))
    pd.DataFrame([dict(faixa=str(f), **metricas_contagem(g.real, g.previsto)) for f, g in densidade.groupby("faixa", observed=True)]).to_csv(out / "por_densidade.csv", index=False)
    catalogo_i = cat.set_index("arquivo"); exemplos = []
    (out / "exemplos").mkdir()
    for d, g in teste.groupby("dominio"):
        for r in g.loc[g.erro.abs().nlargest(3).index].itertuples():
            ln = catalogo_i.loc[r.arquivo]; img = cv2.imread(str(ln.img))
            vis = ct.desenhar(img, det_teste[r.arquivo][0], ref=refs[r.arquivo])
            h, w = vis.shape[:2]; vis = cv2.resize(vis, (min(w, 1600), int(h * min(w, 1600) / w)))
            cv2.imwrite(str(out / "exemplos" / (r.arquivo + ".jpg")), vis)
            exemplos.append(dict(arquivo=r.arquivo + ".jpg", dominio=d, fazenda=r.fazenda, voo=r.voo, real=r.real, previsto=r.previsto))
    pd.DataFrame(exemplos).to_csv(out / "exemplos.csv", index=False)
    shutil.copy2(pesos, out / "modelo_gado.pt")
    for nome in ("curvas_treino.csv", "divisao.csv"):
        shutil.copy2(a.origem / nome, out / nome)
    div = pd.read_csv(out / "divisao.csv")
    for r in cat.itertuples():
        div.loc[(div.dominio == r.dominio) & (div.stem == r.stem), "n_vacas"] = r.n_vacas
    div.to_csv(out / "divisao.csv", index=False)
    meta.update(gerado_em=datetime.now().isoformat(timespec="seconds"), versao_inferencia=ct.VERSAO_INFERENCIA,
                pesos_sha256=digest, recalibrado_sem_retreino=True, revisao_rotulos="deduplicados_v2",
                tile_metricas_historicas=True, conf_otima=global_conf, conf_por_dominio={d:c["conf"] for d,c in configs.items()},
                imgsz_por_dominio={d:c["imgsz"] for d,c in configs.items()}, iou_por_dominio={d:c["iou"] for d,c in configs.items()},
                imgsz_grade=[1024,1280,1536], iou_grade=[0.5,0.7,0.85], iou_juncao=0.5, max_det=1000,
                contagem_teste=dict(**metricas_contagem(teste.real, teste.previsto), conf=global_conf), por_dominio=por_dominio)
    for d, g in cat.groupby("dominio"):
        meta["fontes"][d].update(imagens=len(g), caixas=int(g.n_vacas.sum()))
    meta["caixas_total"] = int(div.n_vacas.sum())
    for cj,g in cat.groupby("conjunto"):
        meta["conjuntos"][cj] = dict(imagens_usadas=len(g), caixas=int(g.n_vacas.sum()), voos=g.grupo.nunique())
    (out / "metricas.json").write_text(json.dumps(limpar_json(meta), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(pd.DataFrame(por_dominio).T.to_string())
    print("Artefatos:", out, flush=True)
    # Porta de liberação: comparar contra as previsões congeladas anteriores, com o MESMO
    # gabarito corrigido. Não tenta outra configuração olhando o teste após uma rejeição.
    base_csv = a.origem / "contagem_teste.csv"
    liberacao = dict(aprovada=False, motivo="Sem referência comparável")
    if base_csv.exists():
        base = pd.read_csv(base_csv)
        if base.arquivo.is_unique and set(base.arquivo) == set(teste.arquivo):
            comp = teste[["arquivo", "real", "previsto"]].merge(base[["arquivo", "previsto"]], on="arquivo", suffixes=("_nova", "_base"), validate="one_to_one")
            antiga = metricas_contagem(comp.real, comp.previsto_base)
            nova = metricas_contagem(comp.real, comp.previsto_nova)
            liberacao = dict(aprovada=nova["MAE"] <= antiga["MAE"], MAE_base=antiga["MAE"], MAE_candidata=nova["MAE"],
                             motivo="Porta de não regressão de MAE no teste; não é nova seleção de hiperparâmetros")
    (out / "liberacao.json").write_text(json.dumps(liberacao, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.promover:
        if not liberacao["aprovada"]:
            print("Candidata NÃO promovida:", liberacao, flush=True)
            return
        alvo = PROJ / "data/processed"
        backup = PROJ / "data/backups" / ("antes_calibracao_" + stamp)
        shutil.copytree(alvo, backup)
        for p in out.iterdir():
            if p.is_dir():
                shutil.copytree(p, alvo / p.name, dirs_exist_ok=True)
            else:
                shutil.copy2(p, alvo / p.name)
        print("App atualizado; backup:", backup)


if __name__ == "__main__":
    main()
