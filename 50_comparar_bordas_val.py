"""Ablacao de duplicacoes de bordas usando somente a validacao congelada."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao
from inferencia_recortes import detectar_recortes

ROOT = Path(__file__).resolve().parent
VAL = ROOT / 'data/experimentos/solo_denso_validacao_v1'
OUT = ROOT / 'data/experimentos/solo_bordas_validacao_v1'


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    modelo = YOLO(str(ROOT / 'data/processed/modelo_gado.pt'))
    val = json.loads((VAL / 'catalogo_val.json').read_text(encoding='utf-8'))
    linhas = []
    for borda in (False,True):
        for r in val:
            im = cv2.imread(str(VAL / 'validacao/images' / r['arquivo']))
            h,w = im.shape[:2]
            ref = ct.yolo_para_pixels((VAL / 'validacao/labels' / (Path(r['arquivo']).stem+'.txt')).read_text().splitlines(),w,h)
            b,s = detectar_recortes(modelo,im,classes=[0],descartar_borda_interna=borda)
            np.savez_compressed(OUT / (f'borda{int(borda)}_'+Path(r['arquivo']).stem+'.npz'),caixas=b,scores=s,ref=ref)
            for conf in (.05,.10,.15,.20,.25,.30,.40):
                bb,ss = ct.filtrar(b,s,conf)
                linhas.append(dict(borda_interna=borda,arquivo=r['arquivo'],conf=conf,real=len(ref),previsto=len(bb),**comparar_caixas(ref,bb,ss)))
    d = pd.DataFrame(linhas)
    d.to_csv(OUT / 'por_imagem.csv',index=False)
    resumo = []
    for (borda,conf),g in d.groupby(['borda_interna','conf']):
        tp,fp,fn = (int(g[k].sum()) for k in ('TP','FP','FN'))
        resumo.append(dict(borda_interna=bool(borda),conf=float(conf),**metricas_contagem(g.real,g.previsto),
            **taxas_deteccao(tp,fp,fn),TP=tp,FP=fp,FN=fn))
    tabela = pd.DataFrame(resumo)
    tabela['objetivo'] = tabela.MAE+(tabela.FP+tabela.FN)/tabela.n_imagens
    tabela = tabela.sort_values(['objetivo','MAE','F1_caixas'],ascending=[True,True,False])
    tabela.to_csv(OUT / 'resumo.csv',index=False)
    (OUT / 'limites.json').write_text(json.dumps(dict(teste_inferido=False,promovido=False,
        grupos_validacao=len({r['grupo'] for r in val}),
        limite='Remover caixas tocando a borda interna pode eliminar um animal se ele nao for detectado na janela vizinha. Comparacao apenas exploratoria.'),indent=2),encoding='utf-8')
    print(tabela[['borda_interna','conf','MAE','F1_caixas','TP','FP','FN','objetivo']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
