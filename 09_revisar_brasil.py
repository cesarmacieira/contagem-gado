"""Revisão humana das fotos candidatas: streamlit run 09_revisar_brasil.py."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import contagem as ct

PROJ = Path(__file__).resolve().parent
BASE = PROJ / "data/brasil_candidatos"
DEST = PROJ / "data/brasil_revisado"
st.set_page_config(page_title="Revisar bovinos", layout="wide")
st.title("Revisar imagens de bovinos")
st.write("Informe uma caixa por animal visível. Animais ocultos sem evidência visual não devem ser inventados. Imagens ambíguas podem ser descartadas.")
p = BASE / "fontes.json"
if not p.exists():
    st.info("Execute primeiro: python 08_obter_brasil.py"); st.stop()
fontes = json.loads(p.read_text(encoding="utf-8"))
fonte = st.selectbox("Imagem", fontes, format_func=lambda r: r["arquivo"])
arquivo = fonte["arquivo"]
if Path(arquivo).name != arquivo:
    st.error("Nome de arquivo inválido"); st.stop()
img = cv2.imread(str(BASE / "images" / arquivo))
if img is None:
    st.error("Imagem ilegível"); st.stop()
h, w = img.shape[:2]
st.caption(f"{w} × {h} px | {fonte['autor']} | {fonte['licenca']}")
st.markdown(f"[Origem da imagem]({fonte['url']})")
rot = DEST / "labels" / (Path(arquivo).stem + ".txt")
inicial = ct.yolo_para_pixels(rot.read_text(encoding="utf-8").splitlines(), w, h) if rot.exists() else np.zeros((0, 4))
cols = ["x1", "y1", "x2", "y2"]
caixas = st.data_editor(pd.DataFrame(inicial, columns=cols), num_rows="dynamic", key="caixas_" + arquivo,
                        column_config={c: st.column_config.NumberColumn(min_value=0, max_value=max(w, h), step=1) for c in cols})
b = caixas.dropna().to_numpy(dtype=float).reshape(-1, 4)
st.image(cv2.cvtColor(ct.desenhar(img, b), cv2.COLOR_BGR2RGB), caption=f"{len(b)} caixas", use_container_width=True)
grupo = st.text_input("Grupo da cena (mesma fazenda/sessão/vídeo deve usar o mesmo grupo)", fonte.get("grupo", ""))
split = st.selectbox("Conjunto", ["train", "val", "test"])
revisado = st.checkbox("Revisei todas as caixas e todos os bovinos visíveis")
negativo = st.checkbox("Esta imagem não contém nenhum bovino (negativo revisado)")
if st.button("Salvar revisão"):
    if not revisado or not grupo.strip() or len(b) != len(caixas):
        st.error("Confirme a revisão, informe o grupo e preencha todas as coordenadas."); st.stop()
    if not len(b) and not negativo:
        st.error("Confirme explicitamente imagens sem bovinos."); st.stop()
    if len(b) and (negativo or np.any(b[:, 2:] <= b[:, :2]) or np.any(b < 0) or np.any(b[:, [0, 2]] > w) or np.any(b[:, [1, 3]] > h)):
        st.error("Caixas inválidas ou confirmação de negativo incompatível."); st.stop()
    manifesto = DEST / "manifesto.csv"
    df = pd.read_csv(manifesto, keep_default_na=False) if manifesto.exists() else pd.DataFrame()
    if not df.empty and ((df.grupo == grupo) & (df.conjunto != split) & (df.arquivo != arquivo)).any():
        st.error("Este grupo já está em outro conjunto."); st.stop()
    for nome in ("images", "labels"):
        (DEST / nome).mkdir(parents=True, exist_ok=True)
    linhas = [f"0 {(a+c)/2/w:.6f} {(bb+d)/2/h:.6f} {(c-a)/w:.6f} {(d-bb)/h:.6f}" for a, bb, c, d in b]
    linhas = ct.linhas_yolo_unicas(linhas)
    # Exige confirmação antes da inclusão no treino; o manifesto é o ponto de admissão.
    registro = dict(arquivo=arquivo, grupo=grupo, conjunto=split, revisado=True, autor=fonte["autor"], licenca=fonte["licenca"], url=fonte["url"])
    if not df.empty:
        df = df[df.arquivo != arquivo]
    df = pd.concat([df, pd.DataFrame([registro])], ignore_index=True)
    shutil.copy2(BASE / "images" / arquivo, DEST / "images" / arquivo)
    rot.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")
    tmp = manifesto.with_suffix(".csv.tmp"); df.to_csv(tmp, index=False); tmp.replace(manifesto)
    st.success(f"Revisão salva: {len(linhas)} bovinos, grupo {grupo}, conjunto {split}.")
