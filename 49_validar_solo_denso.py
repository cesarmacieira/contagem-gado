"""Compara modelos/escala SOMENTE na nova validacao brasileira, sem teste.

Congela uma copia apenas das imagens de val. O teste novo nao sofre inferencia
e nenhum candidato e promovido ao aplicativo por esta comparacao.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao
from integracao_denso import conferir
from inferencia_recortes import detectar_recortes

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'data/brasil_denso_novo'
CONFS = (.05,.10,.15,.20,.25,.30,.40,.50,.60)


def congelar(out):
    resumo = conferir(BASE)
    if not resumo['pronto']:
        raise ValueError(f'Lote ainda inconsistente: {resumo.get("erros", resumo.get("arquivos_ausentes"))}')
    val = [r for r in resumo['registros'] if r['conjunto'] == 'val']
    assert val and any(r['n_bovinos'] >= 20 for r in val), 'Validacao ainda sem rebanho denso'
    for r in val:
        for pasta, nome, digest in [('images',r['arquivo'],r['sha256']),
                                    ('labels',Path(r['arquivo']).stem+'.txt',r['sha256_label'])]:
            destino = out / 'validacao' / pasta / nome
            destino.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(BASE / pasta / nome,destino)
            assert hashlib.sha256(destino.read_bytes()).hexdigest() == digest, 'Fonte mudou durante a copia; nao executar inferencia'
    (out / 'catalogo_val.json').write_text(json.dumps(val,ensure_ascii=False,indent=2),encoding='utf-8')
    return val,resumo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saida',type=Path,default=ROOT / 'data/experimentos/solo_denso_validacao_v1')
    a = ap.parse_args()
    out = a.saida
    if (out / 'protocolo.json').exists():
        raise ValueError('Experimento existente preservado; use --saida com uma pasta nova')
    torch.set_num_threads(4)
    out.mkdir(parents=True,exist_ok=True)
    val,fonte = congelar(out)
    pesos = {
        'original': (ROOT / 'data/processed/modelo_gado.pt',[0]),
        'ajustado_externo': (ROOT / 'data/experimentos/detector_denso_v3/treino/weights/best.pt',[0]),
        'yolo11x_geral': (ROOT / 'data/modelos_candidatos/yolo11x.pt',[19]),
    }
    estrategias = {
        'inteira1280': dict(tipo='inteira',imgsz=1280,iou=.7,augment=False),
        'inteira1920': dict(tipo='inteira',imgsz=1920,iou=.7,augment=False),
        'tta1280': dict(tipo='inteira',imgsz=1280,iou=.7,augment=True),
        'recorte050': dict(tipo='recortes',imgsz=1280,iou=.7,augment=False,fracao=.5),
        'recorte050_tta': dict(tipo='recortes',imgsz=1280,iou=.7,augment=True,fracao=.5),
    }
    protocolo = dict(visao='solo_denso',modelos={k:dict(pesos=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),classes=c) for k,(p,c) in pesos.items()},
        estrategias=estrategias,confiancas=CONFS,validacao=val,grupos=len({r['grupo'] for r in val}),
        fingerprint_fonte=fonte['snapshot_sha256'],criterio='MAE por imagem; maior F1 como desempate. Sem promocao antes de avaliacao separada.',
        teste_consultado_para_inferencia=False,usa_gabaritos_como_rotulos_de_treino=False,promove_app=False,
        limite='Duas imagens de validacao podem ser recortes da mesma foto. Isso nao equivale a duas cenas independentes. Nenhuma contagem de teste e prevista.')
    (out / 'protocolo.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    linhas = []
    device = 0 if torch.cuda.is_available() else 'cpu'
    for nome,(peso,classes) in pesos.items():
        modelo = YOLO(str(peso))
        for estrategia,cfg in estrategias.items():
            pasta = out / nome / estrategia
            pasta.mkdir(parents=True,exist_ok=True)
            for r in val:
                im = cv2.imread(str(out / 'validacao/images' / r['arquivo']))
                h,w = im.shape[:2]
                ref = ct.yolo_para_pixels((out / 'validacao/labels' / (Path(r['arquivo']).stem+'.txt')).read_text().splitlines(),w,h)
                start = time.perf_counter()
                if cfg['tipo'] == 'inteira':
                    b,s = ct.detectar_inteira(modelo,im,imgsz=cfg['imgsz'],iou_nms=cfg['iou'],device=device,classes=classes,augment=cfg['augment'])
                else:
                    b,s = detectar_recortes(modelo,im,fracao=cfg['fracao'],imgsz=cfg['imgsz'],device=device,classes=classes,augment=cfg['augment'])
                segundos = time.perf_counter()-start
                np.savez_compressed(pasta / (Path(r['arquivo']).stem+'.npz'),caixas=b,scores=s,ref=ref)
                for conf in CONFS:
                    bb,ss = ct.filtrar(b,s,conf)
                    linhas.append(dict(modelo=nome,estrategia=estrategia,conf=conf,arquivo=r['arquivo'],grupo=r['grupo'],
                        real=len(ref),previsto=len(bb),segundos=segundos,**comparar_caixas(ref,bb,ss)))
            pd.DataFrame(linhas).to_csv(out / 'por_imagem.csv',index=False)
            print(nome,estrategia,'concluido',flush=True)
    dados = pd.DataFrame(linhas)
    resumo = []
    for (nome,estrategia,conf),d in dados.groupby(['modelo','estrategia','conf']):
        tp,fp,fn = (int(d[k].sum()) for k in ('TP','FP','FN'))
        resumo.append(dict(modelo=nome,estrategia=estrategia,conf=conf,
            **metricas_contagem(d.real,d.previsto),**taxas_deteccao(tp,fp,fn),TP=tp,FP=fp,FN=fn,
            segundos_por_imagem=float(d.segundos.mean())))
    tabela = pd.DataFrame(resumo).sort_values(['MAE','F1_caixas'],ascending=[True,False])
    tabela.to_csv(out / 'resumo.csv',index=False)
    melhor = tabela.iloc[0]
    base = tabela[(tabela.modelo=='original') & (tabela.estrategia=='inteira1280') & np.isclose(tabela.conf,.15)].iloc[0]
    candidato = dict(visao='solo_denso',modelo=melhor.modelo,pesos=protocolo['modelos'][melhor.modelo],
        estrategia=estrategias[melhor.estrategia],confianca=float(melhor.conf),validacao=melhor.to_dict(),
        referencia=base.to_dict(),grupos_validacao=protocolo['grupos'],
        melhora_MAE_e_F1_na_validacao=bool(melhor.MAE<base.MAE and melhor.F1_caixas>=base.F1_caixas),
        promovido=False,teste_positivo_ainda_necessario=True)
    (out / 'candidato_validacao.json').write_text(json.dumps(candidato,indent=2),encoding='utf-8')
    for nome_est,modelo,estrategia,conf in [('referencia','original','inteira1280',.15),
                                          ('candidato',melhor.modelo,melhor.estrategia,float(melhor.conf))]:
        for r in val:
            cache = np.load(out / modelo / estrategia / (Path(r['arquivo']).stem+'.npz'))
            b,s = ct.filtrar(cache['caixas'],cache['scores'],conf)
            im = cv2.imread(str(out / 'validacao/images' / r['arquivo']))
            cv2.imwrite(str(out / (nome_est+'_'+r['arquivo'])),ct.desenhar(im,b,ref=cache['ref']))
    print(tabela[['modelo','estrategia','conf','MAE','F1_caixas','FP','FN']].head(18).to_string(index=False),flush=True)
    print('Referencia',base.to_dict(),flush=True)
    print('Promovido: False; falta teste brasileiro positivo independente e validacao mais variada.',flush=True)


if __name__ == '__main__':
    main()
