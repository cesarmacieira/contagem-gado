"""Diagnostico na validacao conhecida. Nao consulta teste ou lote novo."""
from pathlib import Path
import hashlib
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao
from inferencia_recortes import detectar_recortes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/experimentos/recortes_escala_v1'
CONFIGS = {
    'inteira1280': dict(tipo='inteira', imgsz=1280),
    'recorte075': dict(tipo='recortes', fracao=.75, imgsz=1280),
    'recorte050': dict(tipo='recortes', fracao=.5, imgsz=1280),
}


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / 'data/experimentos/detector_denso_v3/catalogo_treino_validacao.csv')
    df = df[(df.conjunto == 'val') & (df.n_bovinos >= 8)].copy()
    assert len(df) > 0 and (df.conjunto == 'val').all()
    df.to_csv(OUT / 'validacao.csv', index=False)
    pesos = {
        'original': ROOT / 'data/processed/modelo_gado.pt',
        'ajustado': ROOT / 'data/experimentos/detector_denso_v3/treino/weights/best.pt',
    }
    protocolo = dict(configs=CONFIGS, confiancas=[.10, .15, .25, .35, .45],
        iou_local=.7, iou_juncao=.5, imagens=len(df),
        grupos=int(df.grupo.nunique()),
        pesos={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in pesos.items()},
        limite='Diagnostico e selecao em validacao ja conhecida. Nao e teste independente. Nao acessa brasil_denso_novo.')
    (OUT / 'protocolo.json').write_text(json.dumps(protocolo, indent=2), encoding='utf-8')
    linhas = []
    device = 0 if torch.cuda.is_available() else 'cpu'
    for nome, peso in pesos.items():
        modelo = YOLO(str(peso))
        for estrategia, config in CONFIGS.items():
            for i, r in enumerate(df.itertuples(), 1):
                img = cv2.imread(r.img)
                if img is None:
                    raise ValueError(f'Imagem ilegivel: {r.img}')
                h, w = img.shape[:2]
                ref = ct.yolo_para_pixels(Path(r.rot).read_text(encoding='utf-8').splitlines(), w, h)
                start = time.perf_counter()
                if config['tipo'] == 'inteira':
                    b, s = ct.detectar_inteira(modelo, img, imgsz=config['imgsz'], iou_nms=.7, device=device)
                else:
                    b, s = detectar_recortes(modelo, img, fracao=config['fracao'], imgsz=config['imgsz'], device=device)
                segundos = time.perf_counter() - start
                for conf in protocolo['confiancas']:
                    bb, ss = ct.filtrar(b, s, conf)
                    linhas.append(dict(modelo=nome, estrategia=estrategia, conf=conf,
                        arquivo=r.arquivo, grupo=r.grupo, real=len(ref), previsto=len(bb),
                        segundos=segundos, **comparar_caixas(ref, bb, ss)))
                if i % 10 == 0 or i == len(df):
                    pd.DataFrame(linhas).to_csv(OUT / 'por_imagem.csv', index=False)
                    print(f'{nome} {estrategia}: {i}/{len(df)}', flush=True)
    dados = pd.DataFrame(linhas)
    resumir(dados, df)


def resumir(dados, df):
    auditoria = OUT / 'auditoria_visual/achados.json'
    qualidade_pendente = auditoria.exists()
    resumo = []
    for (modelo, estrategia, conf), d in dados.groupby(['modelo', 'estrategia', 'conf']):
        tp, fp, fn = (int(d[c].sum()) for c in ('TP', 'FP', 'FN'))
        erro = d.assign(erro=np.abs(d.previsto - d.real))
        resumo.append(dict(modelo=modelo, estrategia=estrategia, conf=conf,
            **metricas_contagem(d.real, d.previsto), **taxas_deteccao(tp, fp, fn),
            TP=tp, FP=fp, FN=fn, MAE_macro_grupos=float(erro.groupby('grupo').erro.mean().mean()),
            segundos_por_imagem=float(d.segundos.mean())))
    tabela = pd.DataFrame(resumo).sort_values(['MAE', 'F1_caixas'], ascending=[True, False])
    tabela.to_csv(OUT / 'resumo_validacao.csv', index=False)
    texto = ['# Diagnostico de escala dos recortes', '',
        '**ATENCAO: rotulagem incompleta confirmada na validacao. Numeros abaixo sao provisorios contra as anotacoes publicadas; nao representam erros reais integralmente revisados.**' if qualidade_pendente else 'Auditoria visual completa dos rotulos ainda nao realizada.', '',
        f'Validacao conhecida: {len(df)} imagens densas, {df.grupo.nunique()} grupos. ',
        'Nenhuma imagem do teste ou do lote brasil_denso_novo foi consultada.', '',
        '| Modelo | Estrategia | Confianca | MAE | F1 | Perdidos | Falsos positivos |',
        '|---|---|---:|---:|---:|---:|---:|']
    escolhas = {}
    for nome, grupo in tabela.groupby('modelo'):
        ordenado = grupo.assign(score=(grupo.MAE + grupo.MAE_macro_grupos) / 2).sort_values(['score', 'F1_caixas'], ascending=[True, False])
        base = ordenado[ordenado.estrategia == 'inteira1280'].iloc[0]
        for estrategia in CONFIGS:
            r = ordenado[ordenado.estrategia == estrategia].iloc[0]
            texto.append(f'| {nome} | {estrategia} | {r.conf:.2f} | {r.MAE:.3f} | {r.F1_caixas:.3f} | {r.FN} | {r.FP} |')
        candidato = ordenado.iloc[0]
        escolhas[nome] = dict(estrategia=candidato.estrategia, conf=float(candidato.conf),
            melhora_MAE_e_F1_na_validacao=bool(candidato.MAE < base.MAE and candidato.F1_caixas >= base.F1_caixas),
            qualidade_gabarito_pendente=qualidade_pendente,
            promove_app=False)
    texto += ['', 'Escolha por media do MAE por imagem e do MAE macro por grupo; F1 como desempate.',
        'Os limiares foram selecionados nesta validacao. Isso nao comprova generalizacao para Nelore denso.',
        'Os pesos e os padroes do aplicativo nao foram alterados. Recortes podem aumentar falsos positivos e custo.']
    (OUT / 'relatorio.md').write_text('\n'.join(texto), encoding='utf-8')
    (OUT / 'candidatos_validacao.json').write_text(json.dumps(escolhas, indent=2), encoding='utf-8')
    print(tabela[['modelo', 'estrategia', 'conf', 'MAE', 'F1_caixas', 'TP', 'FP', 'FN']].to_string(index=False), flush=True)


if __name__ == '__main__':
    import sys
    if '--resumir' in sys.argv:
        resumir(pd.read_csv(OUT / 'por_imagem.csv'), pd.read_csv(OUT / 'validacao.csv'))
    else:
        main()
