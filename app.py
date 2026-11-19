"""Contagem de gado (qualquer ângulo) — app Streamlit (dados públicos: Zenodo 11048412, COLO e Open Images).

Execução:  streamlit run app.py   ->  http://localhost:8501   (logs em logs\\app.log)
O app LÊ os arquivos gerados pelo notebook 02_treinamento (pasta data/processed) e usa o modelo treinado para contar vacas em imagens enviadas.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import warnings
from importlib import metadata, reload
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import contagem as ct
from especialistas import carregar_catalogo, resolver_especialista
ct = reload(ct)  # Streamlit mantém módulos importados entre recargas do app

warnings.filterwarnings("ignore", category=FutureWarning)
PROJ = Path(__file__).resolve().parent
from revisao_vlm import carregar_chave_ambiente
carregar_chave_ambiente(PROJ)
from config_publicacao import carregar_segredos
carregar_segredos(st.secrets)
RAW, PROC, LOGS = PROJ / "data" / "raw", PROJ / os.environ.get("CONTAGEM_ARTEFATOS", "data/processed"), PROJ / "logs"
LOGS.mkdir(exist_ok=True)
logging.basicConfig(filename=LOGS / "app.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("contagem-gado")

st.set_page_config(page_title="Contagem de gado", page_icon="🐄", layout="wide")

# ------------------------------------------------------------------ design system CAST
PRIMARIA, SECUNDARIA = "#4C23EC", "#17104F"
pio.templates["cast"] = go.layout.Template(layout=dict(
    colorway=[PRIMARIA, SECUNDARIA, "#8B6CF6", "#0F9D6B", "#F59E0B", "#B91C1C", "#64748B"],
    font=dict(family="Inter, sans-serif", color="#1F2937"), title=dict(font=dict(size=15, color=SECUNDARIA)),
    xaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB"), yaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB")))
pio.templates.default = "plotly_white+cast"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1340px; }
h1, h2, h3, [data-testid="stHeading"] { font-family: 'Inter', sans-serif !important; }
[data-testid="stHeading"], [data-testid="stHeading"] * { color: #17104F !important; }
.stMarkdown a, .stMarkdown a:visited { color: #4C23EC !important; }
h1 { color: #17104F; font-weight: 800; letter-spacing: -0.01em; }
h2, h3 { color: #17104F; font-weight: 700; }
[data-testid="stSidebar"] { background: #F3F4F6; border-right: 1px solid #E5E7EB; }
[data-testid="stSidebar"] h1 { font-size: 1.25rem; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1.5px solid #E5E7EB; border-radius: 14px; padding: 0.9rem 1.1rem; transition: box-shadow .2s; }
[data-testid="stMetric"]:hover { box-shadow: 0 2px 12px rgba(76, 35, 236, .10); }
[data-testid="stMetricLabel"] { color: #6B7280; font-weight: 600; }
[data-testid="stMetricValue"] { color: #17104F; font-weight: 800; font-size: 1.75rem; }
[data-testid="stMetricValue"] > div { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
.stButton > button, .stDownloadButton > button { background: #4C23EC; color: #fff; border: none; border-radius: 10px; font-weight: 600; transition: background .2s; }
.stButton > button:hover, .stDownloadButton > button:hover { background: #3B18C4 !important; color: #fff !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #E5E7EB; }
.stTabs [data-baseweb="tab"] { font-weight: 600; color: #6B7280; }
.stTabs [aria-selected="true"] { color: #4C23EC !important; border-bottom: 2px solid #4C23EC !important; }
[data-testid="stAlert"] { border-radius: 12px; }
code { background: #EEF2FF !important; color: #4C23EC !important; border-radius: 5px; padding: 1px 6px !important; font-size: .85em !important; }
hr { border-top: 1.5px solid #E5E7EB; }
</style>
""", unsafe_allow_html=True)

ARQUIVOS = {
    "data/raw/Cattle_drone_images_042024.zip": ("Etapa 1", "01_obter_dados.py"),
    "data/raw/inventario.csv": ("Etapa 1", "01_obter_dados.py"),
    "data/extras/inventario_extras.csv": ("Etapa 1b", "03_obter_dados_extras.py"),
    "data/extras/cattleeyeview/inventario.csv": ("Etapa 1c (opcional)", "04_obter_cattleeyeview.py"),
    "data/processed/modelo_gado.pt": ("Etapa 2", "02_treinamento.ipynb, seção 11"),
    "data/processed/metricas.json": ("Etapa 2", "02_treinamento.ipynb, seção 11"),
    "data/processed/contagem_teste.csv": ("Etapa 2", "02_treinamento.ipynb, seção 11"),
    "data/processed/varredura_conf.csv": ("Etapa 2", "02_treinamento.ipynb, seção 11"),
    "data/processed/curvas_treino.csv": ("Etapa 2", "02_treinamento.ipynb, seção 11"),
    "data/processed/divisao.csv": ("Etapa 2", "02_treinamento.ipynb, seção 11"),
    "data/processed/exemplos.csv": ("Etapa 2", "02_treinamento.ipynb, seção 10"),
}


NOMES_DOMINIO = {"aereo_pasto": "Drone vertical em pasto (Zenodo)",
                 "colo_topo": "Vista de cima em instalação (COLO)",
                 "colo_lateral": "Vista lateral em instalação (COLO)",
                 "colo_externo": "Vista externa (COLO)",
                 "cev_rampa": "Vista de cima em rampa de curral (CattleEyeView)",
                 "foto_openimages": "Foto comum (Open Images)", "brasil_solo": "Rebanho brasileiro de solo (revisado)"}
OUTRO = "Outro / escolher modo de imagem"


def nome_dominio(d: str) -> str:
    return NOMES_DOMINIO.get(d, d)


# ------------------------------------------------------------------ utilidades
def num(x, d=1, suf=""):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".") + suf


def mtime(rel: str) -> float:
    p = PROC / Path(rel).name if rel.startswith("data/processed/") else PROJ / rel
    return p.stat().st_mtime_ns if p.exists() else 0


def pacote(nome: str) -> str:
    try:
        return metadata.version(nome)
    except metadata.PackageNotFoundError:
        return "NÃO INSTALADO"


@st.cache_data(show_spinner=False)
def ler_csv(rel: str, mt: float) -> pd.DataFrame:
    return pd.read_csv(PROC / Path(rel).name if rel.startswith("data/processed/") else PROJ / rel)


@st.cache_resource(show_spinner="Carregando o modelo…")
def ler_modelo(mt: float, pesos: str | None = None, arquitetura: str = "yolo"):
    from ultralytics import YOLO, RTDETR

    classe = RTDETR if arquitetura == "rtdetr" else YOLO
    return classe(pesos or str(PROC / "modelo_gado.pt"))


def carregar():
    """Carrega os artefatos da etapa 2. Devolve (dict, faltando)."""
    obrig = [r for r in ARQUIVOS if r.startswith("data/processed")]
    faltando = [r for r in obrig if not (PROC / Path(r).name).exists()]
    if faltando:
        return None, faltando
    try:
        d = {k: ler_csv(f"data/processed/{k}.csv", mtime(f"data/processed/{k}.csv"))
             for k in ["contagem_teste", "varredura_conf", "curvas_treino", "divisao", "exemplos"]}
        d["meta"] = json.loads((PROC / "metricas.json").read_text(encoding="utf-8"))
        return d, None
    except Exception as e:  # noqa: BLE001
        log.exception("falha ao carregar artefatos")
        return None, [f"Erro ao ler os arquivos: {e}"]


@st.cache_data(show_spinner="Detectando vacas (a 1ª vez em CPU pode levar alguns minutos)…", max_entries=4)
def detectar_bytes(dados: bytes, modo: str, tile: int, overlap: float, imgsz: int, _modelo, mt: float,
                   iou: float = 0.7, iou_juncao: float = 0.5, max_det: int = 1000, versao: int = ct.VERSAO_INFERENCIA, legado: bool = False,
                   modelo_chave: str = "", classes=None, augment: bool = False):
    """modo: 'tiles' (imagem grande de drone), 'inteira' (foto comum) ou 'auto' (decide pelo tamanho)."""
    img = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível ler a imagem (use JPG ou PNG).")
    if modo == "auto":
        modo = "inteira"
    if legado:
        max_det = 300
    if modo == "tiles":
        caixas, scores = ct.detectar(_modelo, img, tile=tile, overlap=overlap, iou_local=iou, iou_nms=iou_juncao, max_det=max_det, legado=legado,
                                    classes=classes, augment=augment)
    else:
        caixas, scores = ct.detectar_inteira(_modelo, img, imgsz=imgsz, iou_nms=iou, max_det=max_det, classes=classes, augment=augment)
    return caixas, scores, img.shape[:2], modo


def para_exibir(img_bgr: np.ndarray, largura: int = 1800) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w > largura:
        img_bgr = cv2.resize(img_bgr, (largura, int(h * largura / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def painel_revisao_vlm(dados, caixas, scores, conf=.3, candidatos_scores=None):
    from ui_hibrido import painel_revisao
    return painel_revisao(dados, caixas, scores, PROJ, conf, candidatos_scores)


@st.cache_resource(show_spinner="Carregando conferência visual...")
def ler_confirmador_visual(extrator: str, mt_extrator: int, sha_extrator: str,
                          classificador: str, mt_classificador: int, sha_classificador: str):
    import torch
    from verificador_visual import carregar_binario
    if hashlib.sha256(Path(extrator).read_bytes()).hexdigest() != sha_extrator:
        raise ValueError("Extrator visual alterado desde a avaliação")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return carregar_binario(extrator,classificador,sha_classificador,device=device),device


@st.cache_data(show_spinner="Conferindo objetos parecidos com gado...", max_entries=8)
def pontuar_confirmacao(dados: bytes, caixas: np.ndarray, verificador_chave: str, _verificador, device: str):
    from verificador_visual import confirmar
    img = cv2.imdecode(np.frombuffer(dados,np.uint8),cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem inválida na conferência")
    return confirmar(_verificador,img,caixas,device=device)


@st.cache_resource(show_spinner="Carregando conferência de sombras...")
def ler_verificador_nucleo(extrator, mt_extrator, sha_extrator,
                          classificador, mt_classificador, sha_classificador, receita_sha):
    import torch
    from verificador_nucleo import carregar
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return carregar(extrator,classificador,sha_extrator,sha_classificador,receita_sha,device=device)


@st.cache_data(show_spinner="Conferindo sombras...", max_entries=8)
def pontuar_nucleo_bytes(dados, caixas, identidade_verificador, _verificador):
    from verificador_nucleo import pontuar
    img = cv2.imdecode(np.frombuffer(dados,np.uint8),cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem inválida na conferência de sombras")
    return pontuar(_verificador,img,caixas)


@st.cache_resource(show_spinner="Carregando conferência de montaria...")
def ler_conferencia_montaria(peso_clip: str, mt_clip: int, sha_clip: str, receita_sha: str,
                            peso_pessoas: str, mt_pessoas: int, sha_pessoas: str):
    import torch
    from ultralytics import YOLO
    from verificador_semantico import carregar,assinatura_receita
    if assinatura_receita()!=receita_sha:
        raise ValueError("Receita semântica diferente da avaliação")
    if hashlib.sha256(Path(peso_clip).read_bytes()).hexdigest()!=sha_clip:
        raise ValueError("Modelo semântico diferente da avaliação")
    if hashlib.sha256(Path(peso_pessoas).read_bytes()).hexdigest()!=sha_pessoas:
        raise ValueError("Detector de pessoas diferente da avaliação")
    device="cuda" if torch.cuda.is_available() else "cpu"
    semantico=carregar(peso_clip,device=device)
    pessoas=YOLO(peso_pessoas)
    if pessoas.names[0]!="person":
        raise ValueError("Classe de pessoa incompatível")
    return semantico,pessoas,device


@st.cache_data(show_spinner="Conferindo montarias...", max_entries=8)
def conferir_montaria_bytes(dados: bytes, caixas: np.ndarray, config: dict,
                           _semantico, _pessoas, device: str):
    from verificador_semantico import margens,manter,candidatas_montaria
    img=cv2.imdecode(np.frombuffer(dados,np.uint8),cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem inválida na conferência de montaria")
    if config.get("regra_geometrica")!=1:
        raise ValueError("Regra de montaria incompatível")
    mm,_=margens(_semantico,img,caixas)
    rejeitar=~manter(mm,config["limiar"])
    pessoas=np.empty((0,4))
    if rejeitar.any():
        pessoas,_=ct.detectar_inteira(_pessoas,img,imgsz=config["pessoa_imgsz"],
            conf_min=config["pessoa_conf"],iou_nms=.5,max_det=300,classes=[0],augment=False,
            device=0 if device=="cuda" else "cpu")
    regiao=candidatas_montaria(caixas,pessoas)
    return ~(rejeitar & regiao),mm,regiao,pessoas


# ------------------------------------------------------------------ barra lateral
PAGINAS = ["Resumo", "Contar rebanho", "Rótulos Brasil", "Desempenho do modelo", "Exemplos do teste", "Origem dos dados"]
with st.sidebar:
    st.title("🐄 Contagem de gado")
    st.caption("Visão computacional · dados reais públicos")
    pagina = st.radio("Navegação", PAGINAS, label_visibility="collapsed")
    st.divider()

D, faltando = carregar()
if D is not None:
    with st.sidebar:
        m = D["meta"]
        st.caption(f"Modelo treinado em {m['gerado_em'][:16].replace('T', ' ')}")
        st.caption(f"Base: {m['modelo_base']} · tile {m['tile']} px · {m['epocas_rodadas']} épocas")
        st.caption("Dados: Zenodo 11048412 (drone), COLO e Open Images; veja *Origem dos dados*")


def posprocessar_deteccoes(caixas, scores, conf, dados, modelo, modelo_mt, modelo_chave, classes_modelo, tile_usado, overlap_usado, especialista=None, confirmacao_meta=None, usar_confirmacao=False, avisar=None):
    avisar = avisar or (lambda mensagem: None)
    b, s = ct.filtrar(caixas, scores, conf)
    detalhes_confirmacao,score_confirmado,margens_montaria = None,None,None
    if usar_confirmacao:
        try:
            from verificador_visual import mascara_confirmacao
            extrator_path = (PROJ / confirmacao_meta["extrator"]).resolve()
            classificador_path = (PROJ / confirmacao_meta["classificador"]).resolve()
            if not extrator_path.is_relative_to(PROJ) or not classificador_path.is_relative_to(PROJ):
                raise ValueError("Arquivos de conferência fora do projeto")
            verificador,device_confirmacao = ler_confirmador_visual(str(extrator_path),extrator_path.stat().st_mtime_ns,
                confirmacao_meta["extrator_sha256"],str(classificador_path),classificador_path.stat().st_mtime_ns,
                confirmacao_meta["classificador_sha256"])
            id_confirmacao = f"{confirmacao_meta['versao']}:{confirmacao_meta['extrator_sha256']}:{confirmacao_meta['classificador_sha256']}"
            valores = pontuar_confirmacao(dados,b,id_confirmacao,verificador,device_confirmacao)
            manter,abstencoes = mascara_confirmacao(b,valores,confirmacao_meta["limiar"],confirmacao_meta.get("min_lado_px",0))
            detalhes_confirmacao = pd.DataFrame(b,columns=["x1","y1","x2","y2"])
            detalhes_confirmacao["confianca_detector"] = s
            detalhes_confirmacao["score_confirmacao"] = valores
            detalhes_confirmacao["preservada_por_tamanho_pequeno"] = abstencoes
            detalhes_confirmacao["mantida"] = manter
            avisar(f"Conferência: {int((~manter).sum())} detecções não confirmadas; {int(abstencoes.sum())} pequenas preservadas sem rejeição automática.")
            b,s,score_confirmado = b[manter],s[manter],valores[manter]
            montaria_cfg=confirmacao_meta.get("montaria",{})
            if montaria_cfg.get("ativo",False) and len(b):
                clip_path=(PROJ/montaria_cfg["peso_clip"]).resolve()
                pessoas_path=(PROJ/montaria_cfg["peso_pessoas"]).resolve()
                if not clip_path.is_relative_to(PROJ) or not pessoas_path.is_relative_to(PROJ):
                    raise ValueError("Modelos de montaria fora do projeto")
                semantico,pessoas,device_montaria=ler_conferencia_montaria(str(clip_path),clip_path.stat().st_mtime_ns,
                    montaria_cfg["sha_clip"],montaria_cfg["receita_sha256"],str(pessoas_path),pessoas_path.stat().st_mtime_ns,
                    montaria_cfg["sha_pessoas"])
                mask_m,mm,regiao_m,_=conferir_montaria_bytes(dados,b,montaria_cfg,semantico,pessoas,device_montaria)
                indices_visuais=detalhes_confirmacao.index[detalhes_confirmacao["mantida"]]
                detalhes_confirmacao["vantagem_equino_inteira"]=np.nan
                detalhes_confirmacao["vantagem_equino_superior"]=np.nan
                detalhes_confirmacao["pessoa_em_posicao_montaria"]=False
                detalhes_confirmacao["rejeitada_por_montaria"]=False
                detalhes_confirmacao.loc[indices_visuais,"vantagem_equino_inteira"]=mm[:,0]
                detalhes_confirmacao.loc[indices_visuais,"vantagem_equino_superior"]=mm[:,1]
                detalhes_confirmacao.loc[indices_visuais,"pessoa_em_posicao_montaria"]=regiao_m
                detalhes_confirmacao.loc[indices_visuais,"rejeitada_por_montaria"]=~mask_m
                detalhes_confirmacao.loc[indices_visuais,"mantida"]=mask_m
                avisar(f"Conferência de montaria: {int((~mask_m).sum())} detecções removidas.")
                b,s,score_confirmado,margens_montaria=b[mask_m],s[mask_m],score_confirmado[mask_m],mm[mask_m]
        except Exception as e:
            log.exception("falha na conferencia visual")
            raise RuntimeError(f"Nao consegui concluir a conferencia: {e}") from e
    if especialista and especialista.get("verificador_nucleo"):
        try:
            from verificador_nucleo import mascara
            nucleo = especialista["verificador_nucleo"]
            ep,cp = (PROJ/nucleo["extrator"]).resolve(),(PROJ/nucleo["classificador"]).resolve()
            if not ep.is_relative_to(PROJ) or not cp.is_relative_to(PROJ):
                raise ValueError("Arquivos de sombras fora do projeto")
            recurso = ler_verificador_nucleo(str(ep),ep.stat().st_mtime_ns,nucleo["extrator_sha256"],
                str(cp),cp.stat().st_mtime_ns,nucleo["classificador_sha256"],nucleo["receita_sha256"])
            identidade = json.dumps(nucleo,sort_keys=True)+especialista["sha256"]
            valores = pontuar_nucleo_bytes(dados,caixas,identidade,recurso)
            if especialista.get("correcao_instancias"):
                from correcao_aerea import selecionar_instancias
                indices,recuperadas = selecionar_instancias(caixas,scores,valores,conf,nucleo["limiar"])
                manter = np.zeros(len(caixas),dtype=bool)
                manter[indices] = True
            else:
                manter = (scores>=conf)&mascara(valores,nucleo["limiar"])
                indices = np.flatnonzero(manter)
                recuperadas = np.zeros(len(indices),dtype=bool)
            recuperacao_escala = especialista.get("recuperacao_escala")
            origem_escala = np.full(len(caixas),False)
            if recuperacao_escala:
                from correcao_aerea import complementar
                extras,extras_scores,_,_ = detectar_bytes(dados,"inteira",tile_usado,overlap_usado,
                    recuperacao_escala["imgsz"],modelo,modelo_mt,.7,.5,1000,ct.VERSAO_INFERENCIA,
                    False,modelo_chave,classes_modelo,False)
                extras_visual = pontuar_nucleo_bytes(dados,extras,identidade,recurso)
                ids = complementar(caixas[indices],extras,extras_scores,extras_visual,
                    max(conf,recuperacao_escala["confianca_min"]),max(nucleo["limiar"],recuperacao_escala["visual_min"]))
                n_original = len(caixas)
                origem_escala = np.r_[origem_escala,np.ones(len(extras),dtype=bool)]
                caixas = np.concatenate([caixas,extras])
                scores = np.concatenate([scores,extras_scores])
                valores = np.concatenate([valores,extras_visual])
                indices = np.r_[indices,n_original+ids]
                recuperadas = np.r_[recuperadas,np.ones(len(ids),dtype=bool)]
                manter = np.zeros(len(caixas),dtype=bool)
                manter[indices] = True
                avisar(f"Recuperadas na segunda resolução: {len(ids)}.")
            detalhes_confirmacao = pd.DataFrame(caixas,columns=["x1","y1","x2","y2"])
            detalhes_confirmacao["segunda_resolucao"] = origem_escala
            detalhes_confirmacao["confianca_detector"] = scores
            detalhes_confirmacao["score_confirmacao"] = valores
            detalhes_confirmacao["mantida"] = manter
            detalhes_confirmacao["recuperada_por_conferencia"] = False
            detalhes_confirmacao.loc[indices,"recuperada_por_conferencia"] = recuperadas
            b,s,score_confirmado = caixas[indices],scores[indices],valores[indices]
            avisar(f"Recuperadas pela conferência visual: {int(recuperadas.sum())}.")
            avisar(f"Filtragem de confiança, sombras e duplicações: {int((~manter).sum())} propostas descartadas.")
        except Exception as e:
            log.exception("falha na conferencia de sombras")
            raise RuntimeError(f"Nao consegui concluir a conferencia de sombras: {e}") from e
    return caixas, scores, b, s, detalhes_confirmacao, score_confirmado, margens_montaria


def exigir_dados():
    if D is None:
        st.error("Os resultados do treinamento ainda não foram gerados (execute `02_treinamento.ipynb`; veja o README).")
        st.write("Arquivos que faltam:", faltando)
        st.stop()


# ================================================================== RESUMO
if pagina == "Resumo":
    st.title("Contagem de gado em imagens")
    if D and D["meta"].get("versao_inferencia", 1) == 1:
        st.caption("Números históricos do treino original. A nova calibração ficou separada porque aumentou o erro no teste.")
    st.markdown(
        "Projeto de pesquisa de **detecção e contagem de bovinos** em imagens de vários ângulos, treinado com **dados reais públicos**: drone em pasto (Zenodo), "
        "instalações vistas de cima, de lado e de fora (COLO) e fotos comuns (Open Images). Imagens grandes de drone são divididas em recortes e as detecções "
        "são somadas sem repetir as da borda; fotos menores entram inteiras. O erro é medido **por tipo de cenário**."
    )
    c = st.columns(3)
    c[0].success("**O que o piloto entrega**\n\nContagem por imagem com erro medido em voos que o modelo não viu; caixas desenhadas; exportação das detecções.")
    c[1].warning("**Gado brasileiro**\n\nO lote novo tem fotos brasileiras com caixas revisadas visualmente e teste separado por autor. A amostra ainda é pequena para comprovar desempenho em rebanhos grandes.")
    c[2].info("**Princípio**\n\nNenhum dado é sintético. O erro é medido em imagens de teste separadas por voo (drone), por cena (COLO) e pela divisão oficial (Open Images).")
    exigir_dados()
    m = D["meta"]
    ct_ = m["contagem_teste"]
    if m.get("rapido"):
        st.warning("Este modelo foi treinado no **modo rápido** (modelo pequeno, poucas épocas, parte das imagens) só para validar o fluxo. Os números não valem como resultado final.")
    st.header("Principais números")
    k = st.columns(4)
    k[0].metric("Imagens no dataset", f"{m['imagens_total']:,}".replace(",", "."))
    k[1].metric("Vacas anotadas", f"{m['caixas_total']:,}".replace(",", "."))
    k[2].metric("Erro médio por imagem (teste)", f"{num(ct_['MAE'], 2)} vacas", help="Média do erro absoluto entre contagem real e prevista, em imagens inteiras de voos não vistos no treino.")
    k[3].metric("Viés (teste)", f"{ct_['vies']:+.2f} vacas".replace(".", ","), help="Positivo = o modelo conta a mais; negativo = conta a menos.")
    k = st.columns(4)
    k[0].metric("Erro no total do teste", f"{ct_['erro_total_pct']:+.1f}%".replace(".", ","), help="Soma prevista contra soma real em todas as imagens de teste.")
    k[1].metric("R² da contagem", num(ct_["R2"], 2))
    k[2].metric("mAP50 por tile (teste)", num(m["tile_metricas"]["teste"]["mAP50"], 3))
    k[3].metric("Limiar de confiança global", num(m["conf_otima"], 2), help="Menor erro relativo médio entre os cenários na validação. Cada cenário tem o seu limiar (página Desempenho).")
    if m.get("por_dominio"):
        st.subheader("Erro de contagem no teste, por cenário")
        pd_ = pd.DataFrame(m["por_dominio"]).T
        tab_ = pd.DataFrame({"Cenário": [nome_dominio(d) for d in pd_.index], "Imagens de teste": pd_["n_imagens"].astype(int).values,
                             "Erro médio (vacas)": pd_["MAE"].round(2).values, "Erro relativo (WAPE, %)": pd_["WAPE_pct"].round(1).values,
                             "Viés (vacas)": pd_["vies"].round(2).values, "Limiar": pd_["conf"].round(2).values})
        st.dataframe(tab_, use_container_width=True, hide_index=True)
        st.caption("Os cenários não são comparáveis entre si: o número de vacas por imagem muda muito de um para outro. No COLO há vacas visíveis sem caixa no rótulo, então parte do erro é do rótulo.")
    st.header("Como navegar")
    st.markdown("- **Contar rebanho**: envie uma imagem, escolha o tipo e veja a contagem e as caixas.\n"
                "- **Desempenho do modelo**: erro de contagem no teste, escolha do limiar e curvas de treino.\n"
                "- **Exemplos do teste**: imagens reais com a anotação (verde) e o modelo (roxo).\n"
                "- **Origem dos dados**: fonte, licença, limites e estado dos arquivos do projeto.")

elif pagina == "Rótulos Brasil":
    st.title("Fotos brasileiras e gabarito visual")
    base_brasil = PROJ / "data/brasil_dataset_v2"
    if not base_brasil.exists():
        base_brasil = PROJ / "data/brasil_revisado"
    if not (base_brasil / "manifesto.csv").exists():
        st.info("O novo lote ainda não foi preparado.")
        st.stop()
    manifesto_brasil = pd.read_csv(base_brasil / "manifesto.csv")
    gabarito_brasil = pd.read_csv(base_brasil / "gabarito.csv")
    cols_brasil = st.columns(3)
    cols_brasil[0].metric("Fotos com marcação completa", len(manifesto_brasil))
    cols_brasil[1].metric("Bovinos marcados", int(manifesto_brasil.n_bovinos.sum()))
    cols_brasil[2].metric("Fotos de teste reservadas", int((manifesto_brasil.conjunto == "test").sum()))
    st.caption("Primeira revisão visual pelo assistente. Autores separados entre treino, validação e teste; fotos ambíguas ficaram fora do conjunto supervisionado.")
    conjunto_brasil = st.selectbox("Conjunto de imagens", ["Todos", "train", "val", "test"])
    lista_brasil = manifesto_brasil if conjunto_brasil == "Todos" else manifesto_brasil[manifesto_brasil.conjunto == conjunto_brasil]
    foto_brasil = st.selectbox("Foto marcada", lista_brasil.arquivo.tolist())
    registro_brasil = lista_brasil[lista_brasil.arquivo == foto_brasil].iloc[0]
    st.image(str(base_brasil / "marcacoes" / foto_brasil), caption=f"{int(registro_brasil.n_bovinos)} bovinos identificados", use_container_width=True)
    st.write(gabarito_brasil[gabarito_brasil.arquivo == foto_brasil].iloc[0].nota)
    st.markdown(f"Autor: {registro_brasil.autor} | Licença: {registro_brasil.licenca} | [Fonte original]({registro_brasil.url})")
    st.dataframe(manifesto_brasil[["arquivo", "conjunto", "n_bovinos", "autor", "licenca"]], hide_index=True, use_container_width=True)
    st.download_button("Baixar gabarito", (base_brasil / "gabarito.csv").read_bytes(), "gabarito_brasil.csv", "text/csv")
    st.download_button("Baixar marcações COCO", (base_brasil / "anotacoes_coco.json").read_bytes(), "anotacoes_brasil_coco.json", "application/json")
    comparacao_brasil = PROJ / "data/experimentos/comparacao_calibracao_brasil/resultados.json"
    if comparacao_brasil.exists():
        resultados_brasil = json.loads(comparacao_brasil.read_text(encoding="utf-8"))
        st.subheader("Comparação nas fotos reservadas")
        st.dataframe(pd.DataFrame([dict(Perfil=nome, Erro_medio=valores["MAE"], Contagens_exatas_pct=valores["exatas_pct"], F1_caixas=valores["F1_caixas"]) for nome, valores in resultados_brasil.items() if isinstance(valores, dict) and "MAE" in valores]), hide_index=True)
        st.caption("Amostra pequena: 13 fotos e 44 bovinos, em duas rodadas. Duas novas versões treinadas não superaram o modelo atual. A melhora acima vem da calibração, e não comprova precisão em rebanhos muito densos.")

# ================================================================== CONTAR REBANHO
elif pagina == "Contar rebanho":
    st.title("Contar rebanho em imagem ou video")
    exigir_dados()
    m = D["meta"]
    try:
        modelo = ler_modelo(mtime("data/processed/modelo_gado.pt"))
    except Exception as e:  # noqa: BLE001
        log.exception("falha ao carregar modelo")
        st.error(f"Não consegui carregar o modelo: {e}")
        st.stop()
    confs = m.get("conf_por_dominio") or {}
    modos = m.get("modo_por_dominio") or {}
    opcoes = [nome_dominio(d) for d in confs] + [OUTRO]
    inverso = {nome_dominio(d): d for d in confs}
    catalogo = carregar_catalogo(PROJ)
    especialista = None
    usar_especialista = st.toggle("Escolher modelo por visão", value=True) if catalogo else False
    if usar_especialista:
        visao = st.selectbox("Visão da imagem", list(catalogo), format_func=lambda k: catalogo[k]["rotulo"])
        try:
            especialista = resolver_especialista(PROJ, visao)
        except (ValueError, OSError) as e:
            st.error(f"Não consegui usar o especialista: {e}")
            st.stop()
        tipo, dom = especialista["rotulo"], especialista["dominio"]
    else:
        tipo = st.selectbox("Tipo de imagem", opcoes, help="Define o modo de detecção e o limiar padrão, ambos escolhidos na validação desse cenário.")
        dom = inverso.get(tipo)
    modo = modos.get(dom, "inteira") if dom else "inteira"
    conf_padrao = float(confs.get(dom, m["conf_otima"]))
    imgsz = int((m.get("imgsz_por_dominio") or {}).get(dom, m["tile"]))
    iou = float((m.get("iou_por_dominio") or {}).get(dom, 0.7))
    iou_juncao = float(m.get("iou_juncao", 0.5))
    perfis = {"validado": "Configuração validada", "solo_experimental": "Rebanho de solo - experimental", "manual": "Ajuste manual"}
    brasil_path = PROJ / "data/experimentos/perfil_brasil_app.json"
    brasil = json.loads(brasil_path.read_text(encoding="utf-8")) if brasil_path.exists() else None
    if brasil and brasil.get("disponivel"):
        perfis = {"validado": perfis["validado"], "brasil_poucos": "Brasil - fotos com poucos animais", **perfis}
    denso_path = PROJ / "data/modelos/perfil_detector_denso.json"
    denso_meta = json.loads(denso_path.read_text(encoding="utf-8")) if denso_path.exists() else None
    if denso_meta and denso_meta.get("disponivel"):
        perfis = {"validado": perfis["validado"], "solo_ajustado": "Rebanho de solo - detector ajustado", **perfis}
    perfil = "especialista" if especialista else st.radio("Perfil de contagem", list(perfis), format_func=perfis.get, horizontal=True)
    modelo_mt = mtime("data/processed/modelo_gado.pt")
    modelo_chave = str(PROC / "modelo_gado.pt")
    classes_modelo, augment_modelo = None, False
    tile_usado, overlap_usado = m["tile"], m["overlap"]
    if perfil == "solo_ajustado":
        pesos_denso = PROJ / denso_meta["pesos"]
        modelo_mt = pesos_denso.stat().st_mtime_ns
        modelo = ler_modelo(modelo_mt, str(pesos_denso))
        modelo_chave = str(pesos_denso)
        cfg_denso = denso_meta["config"]
        modo, imgsz, conf_padrao, iou = "inteira", cfg_denso["imgsz"], cfg_denso["conf"], cfg_denso["iou"]
        st.info(denso_meta["descricao"])
    if perfil == "brasil_poucos":
        pesos_brasil = PROJ / brasil["pesos"]
        modelo_mt = pesos_brasil.stat().st_mtime_ns
        modelo = ler_modelo(modelo_mt, str(pesos_brasil))
        modelo_chave = str(pesos_brasil)
        cfg_brasil = brasil["config"]
        modo, imgsz, conf_padrao, iou = "inteira", cfg_brasil["imgsz"], cfg_brasil["conf"], cfg_brasil["iou"]
        st.info(brasil["descricao"])
    if perfil == "solo_experimental":
        modo, imgsz, conf_padrao, iou = "inteira", 1280, 0.15, 0.7
        st.info("Configuração exploratória: recuperou animais nas fotos brasileiras, mas pode duplicar caixas. Confira a imagem anotada.")
    if especialista:
        cfg_esp = especialista["config"]
        modelo_mt = especialista["mt"]
        modelo_chave = especialista["chave_modelo"]
        modelo = ler_modelo(modelo_mt, especialista["pesos_absolutos"], especialista["arquitetura"])
        modo, imgsz, conf_padrao, iou = cfg_esp["modo"], cfg_esp["imgsz"], cfg_esp["conf"], cfg_esp["iou"]
        tile_usado, overlap_usado = cfg_esp["tile"], cfg_esp["overlap"]
        iou_juncao = cfg_esp.get("iou_juncao", .5)
        classes_modelo, augment_modelo = especialista["classes"], cfg_esp.get("augment", False)
        st.info(especialista["descricao"])
    if (not dom and perfil not in ("brasil_poucos", "solo_ajustado", "solo_experimental")) or perfil == "manual":
        modo = st.selectbox("Como a imagem foi capturada?", ["inteira", "tiles"],
                            index=0 if modo == "inteira" else 1,
                            format_func=lambda x: "Foto de solo / imagem inteira" if x == "inteira" else "Drone vertical / recortes")
    if perfil == "manual":
        if modo == "inteira":
            imgsz = st.select_slider("Resolução de entrada", options=[640, 1024, 1280, 1536, 1920], value=imgsz if imgsz in [640, 1024, 1280, 1536, 1920] else 1024)
        iou = st.slider("Sobreposição permitida entre caixas (NMS)", 0.30, 0.90, iou, 0.05,
                        help="Valores altos preservam animais sobrepostos, mas também podem manter duplicações.")
    legado_usado = especialista["config"].get("legado", False) if especialista else perfil == "validado" and m.get("versao_inferencia", 1) == 1
    iou_exibido = 0.5 if modo == "tiles" and legado_usado else iou
    st.caption(f"{'Recortes' if modo == 'tiles' else 'Imagem inteira'} | entrada {tile_usado if modo == 'tiles' else imgsz} px | NMS {iou_exibido:.2f}")
    confirmacao_meta,usar_confirmacao = None,False
    confirmacao_path = PROJ / "data/modelos/verificador_solo_v1/config.json"
    if especialista and confirmacao_path.exists():
        confirmacao_candidata = json.loads(confirmacao_path.read_text(encoding="utf-8"))
        if (visao in confirmacao_candidata.get("visoes_permitidas", [])
                and especialista["sha256"] == confirmacao_candidata.get("detector_sha256")):
            confirmacao_meta = confirmacao_candidata
            usar_confirmacao = st.checkbox("Conferir detecções — experimental", value=False,
                help="Pode reduzir objetos confundidos com gado. Ainda não comprovado em novos rebanhos densos.")
    from ui_hibrido import exemplos, painel_video
    entrada_tipo = st.radio("Entrada", ["Imagem", "Video"], horizontal=True)
    if entrada_tipo == "Video":
        conf_video = st.slider("Limiar do detector no video", .05, .95, min(max(conf_padrao,.05),.95), .01)
        config_video = dict(modo=modo,tile=tile_usado,overlap=overlap_usado,imgsz=imgsz,iou=iou,iou_juncao=iou_juncao,
                            legado=legado_usado,modelo=modelo_chave,modelo_mt=modelo_mt,classes=classes_modelo,augment=augment_modelo,
                            especialista=especialista,confirmacao=confirmacao_meta if usar_confirmacao else None,versao_inferencia=ct.VERSAO_INFERENCIA)
        def inferir_quadro_video(frame):
            ok,encoded = cv2.imencode('.png',frame)
            if not ok:raise ValueError('Nao consegui converter quadro para inferencia')
            bytes_quadro=encoded.tobytes()
            cb,cs,_,_=detectar_bytes(bytes_quadro,modo,tile_usado,overlap_usado,imgsz,modelo,modelo_mt,iou,iou_juncao,1000,ct.VERSAO_INFERENCIA,legado_usado,modelo_chave,classes_modelo,augment_modelo)
            _,_,bb,ss,_,_,_=posprocessar_deteccoes(cb,cs,conf_video,bytes_quadro,modelo,modelo_mt,modelo_chave,classes_modelo,tile_usado,overlap_usado,especialista,confirmacao_meta,usar_confirmacao)
            return bb,ss
        painel_video(PROJ,inferir_quadro_video,config_video,conf_video)
        st.stop()
    origem_imagem=st.radio("Origem da imagem", ["Enviar arquivo", "Exemplos reais da internet"],horizontal=True)
    if origem_imagem == "Exemplos reais da internet":
        dados=exemplos(PROJ)
        if dados is None:st.stop()
    else:
        arq=st.file_uploader("Imagem (JPG ou PNG)",type=["jpg","jpeg","png"])
        if arq is None:
            st.info("Envie uma imagem ou escolha Exemplos reais da internet.")
            st.stop()
        dados=arq.getvalue()
    chave = hashlib.md5(dados).hexdigest()[:8]
    try:
        caixas, scores, (h, w), modo_usado = detectar_bytes(dados, modo, tile_usado, overlap_usado, imgsz, modelo, modelo_mt, iou, iou_juncao, 1000, ct.VERSAO_INFERENCIA,
            legado_usado, modelo_chave, classes_modelo, augment_modelo)
    except Exception as e:  # noqa: BLE001
        log.exception("falha na detecção")
        st.error(f"Falha ao processar a imagem: {e}")
        st.stop()
    conf = st.slider("Limiar de confiança", 0.05, 0.95, min(max(conf_padrao, 0.05), 0.95), 0.01,
                     help="Mais alto descarta detecções; mais baixo também pode acrescentar duplicações.",
                     key=f"conf_{dom}_{perfil}_{modelo_chave}_{imgsz}_{iou}_{conf_padrao}")
    b, s = ct.filtrar(caixas, scores, conf)
    try:
        caixas, scores, b, s, detalhes_confirmacao, score_confirmado, margens_montaria = posprocessar_deteccoes(caixas, scores, conf, dados, modelo, modelo_mt, modelo_chave, classes_modelo, tile_usado, overlap_usado, especialista, confirmacao_meta, usar_confirmacao, st.caption)
    except Exception as erro:
        st.error(str(erro))
        st.stop()
    log.info("contagem arquivo=%s tamanho=%sx%s tipo=%s modo=%s conf=%.2f n=%d", chave, w, h, tipo, modo_usado, conf, len(b))
    k = st.columns(4)
    k[0].metric("Vacas contadas", len(b))
    k[1].metric("Imagem", f"{w}×{h} px")
    k[2].metric("Confiança média", num(float(s.mean()), 2) if len(s) else "—")
    k[3].metric("Entre limiar mínimo e escolhido", int(((scores >= ct.CONF_MIN) & (scores < conf)).sum()),
                help="Propostas abaixo do limiar escolhido. A conferência visual pode recuperar candidatos isolados; consulte o CSV de conferência.")
    img = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
    vis = ct.desenhar(img, b)
    st.image(para_exibir(vis), caption=f"{len(b)} vacas detectadas", width=min(900, vis.shape[1], max(1, int(560 * vis.shape[1] / vis.shape[0]))))
    c = st.columns(2)
    df = pd.DataFrame(b, columns=["x1", "y1", "x2", "y2"]).round(1)
    df["confianca"] = np.round(s, 3)
    if score_confirmado is not None:
        df["score_confirmacao"] = np.round(score_confirmado,4)
    if margens_montaria is not None:
        df["vantagem_equino_inteira"]=np.round(margens_montaria[:,0],5)
        df["vantagem_equino_superior"]=np.round(margens_montaria[:,1],5)
    c[0].download_button("Baixar detecções (CSV)", df.to_csv(index=False).encode("utf-8"), file_name=f"deteccoes_{chave}.csv", mime="text/csv")
    c[1].download_button("Baixar imagem anotada (JPG)", cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(),
                         file_name=f"anotada_{chave}.jpg", mime="image/jpeg")
    if detalhes_confirmacao is not None:
        st.download_button("Baixar todas as detecções e a conferência (CSV)",detalhes_confirmacao.to_csv(index=False).encode("utf-8"),
            file_name=f"conferencia_{chave}.csv",mime="text/csv")
    painel_revisao_vlm(dados, b, s, conf, scores)
    st.warning("A contagem vale **para esta imagem**. Imagens da mesma cena ou missão se sobrepõem: somar as contagens de várias imagens "
               "conta duas vezes as vacas que aparecem em mais de uma. O app não identifica indivíduos nem une imagens.")

# ================================================================== DESEMPENHO
elif pagina == "Desempenho do modelo":
    st.title("Desempenho do modelo")
    exigir_dados()
    m, ctt, vv = D["meta"], D["contagem_teste"], D["varredura_conf"]
    if m.get("versao_inferencia", 1) != ct.VERSAO_INFERENCIA:
        st.info("Resultados históricos anteriores às correções de rótulos e recortes. Execute 06_calibrar_modelo.py para medir a versão atual.")
    mt = m["contagem_teste"]
    if "dominio" not in ctt.columns:
        ctt["dominio"] = "aereo_pasto"
    if "dominio" not in vv.columns:
        vv["dominio"] = "aereo_pasto"
    ctt["cenario"] = ctt.dominio.map(nome_dominio)
    vv["cenario"] = vv.dominio.map(nome_dominio)
    ims = m.get("imgsz_por_dominio")
    if ims and "imgsz" in vv.columns:      # mostra cada cenário só no tamanho de entrada escolhido
        vv = vv[vv.imgsz == vv.dominio.map(ims)].copy()
    if m.get("iou_por_dominio") and "iou" in vv.columns:
        vv = vv[vv.iou == vv.dominio.map(m["iou_por_dominio"])].copy()
    st.markdown(f"Medido em **{mt['n_imagens']:.0f} imagens inteiras de teste**, de voos, cenas ou divisões que o modelo não viu no treino nem na escolha do limiar. "
                "Cada cenário usa o limiar escolhido na sua validação. Os números abaixo juntam todos os cenários; veja a aba *Por cenário*.")
    k = st.columns(5)
    k[0].metric("MAE", f"{num(mt['MAE'], 2)} vacas")
    k[1].metric("RMSE", f"{num(mt['RMSE'], 2)} vacas")
    k[2].metric("Viés", f"{mt['vies']:+.2f}".replace(".", ","))
    k[3].metric("MAPE (imagens com vaca)", f"{num(mt['MAPE_pct'], 1)}%")
    k[4].metric("R²", num(mt["R2"], 2))
    t0, t1, t2, t3, t4 = st.tabs(["Por cenário", "Real × previsto", "Escolha do limiar", "Treino", "Divisão dos dados"])
    with t0:
        if {"TP", "FP", "FN"}.issubset(ctt.columns):
            st.subheader("Correspondência com os animais anotados")
            st.dataframe(ctt.groupby("cenario")[["TP", "FP", "FN"]].sum(), use_container_width=True)
            st.caption("TP: animal localizado; FP: caixa sem correspondência; FN: animal anotado perdido. Correspondência individual com IoU mínimo 0,5.")
        if m.get("por_dominio"):
            pd_ = pd.DataFrame(m["por_dominio"]).T
            pd_.index = [nome_dominio(d) for d in pd_.index]
            pd_ = pd_[["n_imagens", "conf", "MAE", "RMSE", "vies", "WAPE_pct", "MAPE_pct", "erro_total_pct", "R2"]].astype(float).round(2)
            pd_.columns = ["Imagens", "Limiar", "MAE", "RMSE", "Viés", "WAPE %", "MAPE %", "Erro no total %", "R²"]
            st.dataframe(pd_, use_container_width=True)
            st.caption("MAE e RMSE em vacas por imagem; WAPE = soma dos erros absolutos ÷ soma do real. Compare cenários pelo WAPE, não pelo MAE. "
                       "No COLO, vacas visíveis sem caixa no rótulo aparecem como excesso de contagem.")
        else:
            st.info("Os resultados por cenário não existem nestes arquivos; rode `02_treinamento.ipynb` de novo.")
    with t1:
        mx = float(max(ctt.real.max(), ctt.previsto.max())) + 2
        fig = px.scatter(ctt, x="real", y="previsto", color="cenario", hover_data=["fazenda", "voo", "arquivo", "erro"], labels={"real": "Contagem real", "previsto": "Contagem prevista"})
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#6B7280", dash="dash"), name="perfeito"))
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
        por_faz = ctt.groupby("fazenda").agg(imagens=("arquivo", "size"), real=("real", "sum"), previsto=("previsto", "sum"), MAE=("erro", lambda s: s.abs().mean())).round(2)
        por_faz["erro_total_pct"] = ((por_faz.previsto - por_faz.real) / por_faz.real.replace(0, np.nan) * 100).round(1)
        st.subheader("Por fazenda")
        st.dataframe(por_faz, use_container_width=True)
        st.caption("Fazendas com poucas imagens de teste têm erro pouco confiável; olhe o número de imagens.")
        st.subheader("Maiores erros")
        st.dataframe(ctt.reindex(ctt.erro.abs().sort_values(ascending=False).index).head(10), use_container_width=True, hide_index=True)
    with t2:
        fig = px.line(vv, x="conf", y="WAPE_pct" if "WAPE_pct" in vv.columns else "MAE", color="cenario", markers=True,
                      labels={"conf": "Limiar de confiança", "WAPE_pct": "Erro relativo na validação (WAPE, %)", "MAE": "MAE na validação (vacas/imagem)", "cenario": "Cenário"})
        fig.add_vline(x=m["conf_otima"], line_dash="dash", line_color="#B91C1C", annotation_text="limiar global")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vv.round(2), use_container_width=True, hide_index=True)
    with t3:
        h = D["curvas_treino"]
        c = st.columns(2)
        c[0].plotly_chart(px.line(h, x="epoch", y=["train/box_loss", "val/box_loss"], title="Perda de caixa"), use_container_width=True)
        c[1].plotly_chart(px.line(h, x="epoch", y=["metrics/mAP50(B)", "metrics/mAP50-95(B)"], title="Validação por tile"), use_container_width=True)
        st.caption("mAP por tile mede detecção em recortes; a medida de contagem é a das outras abas.")
        if m.get("tile_metricas_historicas"):
            st.caption("mAP histórico do treino original; não recalculado após revisão dos rótulos. As métricas de contagem são da avaliação atual.")
        st.json(m["tile_metricas"])
    with t4:
        dv = D["divisao"]
        if "dominio" not in dv.columns:
            dv["dominio"] = "aereo_pasto"
        dv = dv[dv.usar.astype(bool)] if "usar" in dv.columns else dv
        dv["cenario"] = dv.dominio.map(nome_dominio)
        resumo = dv.groupby(["cenario", "conjunto"]).agg(imagens=("stem", "size"), com_vaca=("n_vacas", lambda s: int((s > 0).sum())), caixas=("n_vacas", "sum"))
        st.dataframe(resumo, use_container_width=True)
        st.caption("A divisão é por grupo: por voo (drone), por cena ou sequência (COLO) e pela divisão oficial (Open Images). Nenhum grupo aparece em dois conjuntos. "
                   "No drone, nem todas as imagens sem vaca entram (veja o notebook, seção 5).")

# ================================================================== EXEMPLOS
elif pagina == "Exemplos do teste":
    st.title("Exemplos do conjunto de teste")
    exigir_dados()
    ex = D["exemplos"]
    st.caption("Verde = anotação real · roxo = modelo. Em cada cenário: os dois maiores erros e uma amostra aleatória.")
    for r in ex.itertuples():
        p = PROC / "exemplos" / r.arquivo
        if p.exists():
            st.image(str(p), caption=f"{nome_dominio(getattr(r, 'dominio', 'aereo_pasto'))} · {r.fazenda} · {r.voo} · real {r.real} × previsto {r.previsto}", use_container_width=True)

# ================================================================== ORIGEM DOS DADOS
elif pagina == "Origem dos dados":
    st.title("Origem dos dados")
    st.table(pd.DataFrame({
        "Fonte": ["Zenodo 11048412 (ICAERUS / Institut de l'Elevage)", "COLO (Hugging Face, Roboflow)", "CattleEyeView (SUTD / AnimalEYEQ)", "Open Images V7"],
        "O que é": ["Imagens RGB de drone (4000×3000 ou 5280×3956 px), 60–100 m, pastos franceses. Caixas de vacas em 1.385 imagens; o zip tem 4.904 caixas em 1.387 rótulos; 2 rótulos (157 caixas) não têm imagem, então o projeto usa 4.747 caixas em 1.385 imagens (o registro declara 4.941; os 37 restantes não foram explicados)",
                    "1.254 imagens de vacas em instalações: vista de cima, lateral e externa. Caixas em pixels; a divisão train/test dos autores foi refeita por cena porque vaza",
                    "Quadros de vídeo de uma rampa de curral, vista de cima, 1920×1080, 1–3 vacas avermelhadas por quadro; 14 vídeos. Usa os quadros da pasta detect, espaçados (no máximo 120 por vídeo); divisão dos autores por vídeo",
                    "Fotos comuns com a classe Cattle (e Bull), com caixa. Só imagens sem grupo e sem desenho; amostra de 2.000 no treino, validação e teste oficiais inteiros"],
        "Licença": ["CC BY 4.0", "MIT", "Ver o repositório do dataset (não conferida aqui)", "Anotações CC BY 4.0; cada imagem tem a sua licença"],
        "Modo no modelo": ["Recortes de 1024 px, divisão por voo", "Imagem inteira, divisão por grupo", "Imagem inteira, divisão por vídeo", "Imagem inteira, divisão oficial"]}))
    st.subheader("Limites")
    st.markdown("- **CattleEyeView:** uma única vista (rampa de curral, de cima) e poucas vacas por quadro; quadros do mesmo vídeo são quase iguais, então o erro medido tem menos informação do que o número de imagens sugere. Não vale como prova para Nelore.\n"
                "- **Nelore e pasto brasileiro:** nenhuma fonte aberta com caixas foi encontrada. O desempenho nesse gado **não está medido**; valide com imagens suas.\n"
                "- **COLO:** há vacas visíveis sem caixa no rótulo, então o modelo parece contar a mais do que o real.\n"
                "- **Open Images:** não foi feito para contagem (fotos com poucos animais) e a classe inclui búfalo. Fotos de rebanho denso estão fora da amostra.\n"
                "- **Sem negativos de solo:** fotos sem nenhum bovino quase não existem no treino de solo; falsos positivos em cenas vazias são possíveis.\n"
                "- O modelo conta o que aparece na imagem: vacas sob árvores ou na sombra podem ser perdidas.\n"
                "- Não há identificação individual; imagens com sobreposição não podem ser somadas.\n"
                "- O limiar de confiança é um compromisso entre falsos positivos e vacas perdidas.")
    st.subheader("Arquivos do projeto")
    linhas = [{"Arquivo": r, "Gerado por": f"{e} — {s}", "Existe": "sim" if (PROJ / r).exists() else "NÃO"} for r, (e, s) in ARQUIVOS.items()]
    st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)
    st.subheader("Ambiente")
    st.code(f"Python {sys.version.split()[0]} · {platform.system()}\n" +
            "\n".join(f"{p} {pacote(p)}" for p in ["streamlit", "ultralytics", "torch", "opencv-python-headless", "pandas", "plotly"]))
