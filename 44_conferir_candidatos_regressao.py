"""Escolhas da validacao em regressao brasileira reutilizada, sem promocao."""
from pathlib import Path
import hashlib
import json

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/experimentos/alternativas_regressao_v1'


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    perfil = json.loads((ROOT / 'data/experimentos/perfil_brasil_app.json').read_text(encoding='utf-8'))
    candidatos = {
        'app_atual_poucos': dict(pesos=perfil['pesos'], **perfil['config'], augment=False, classes=[0]),
        'original_tta_validacao': dict(pesos='data/processed/modelo_gado.pt', imgsz=1280, iou=.7, conf=.45, augment=True, classes=[0]),
        'yolo11x_tta_validacao': dict(pesos='data/modelos_candidatos/yolo11x.pt', imgsz=1280, iou=.7, conf=.45, augment=True, classes=[19]),
    }
    protocolo = dict(candidatos=candidatos, fonte='brasil_dataset_v2, conjunto=test, 13 fotos ja examinadas em rodadas anteriores',
        escolha='TTA e limiar .45 selecionados na validacao de nove fotos antes desta comparacao; referencia e perfil atual do app',
        teste_independente=False, consulta_lote_novo=False, promove_app=False,
        sha256={k:hashlib.sha256((ROOT / v['pesos']).read_bytes()).hexdigest() for k,v in candidatos.items()})
    (OUT / 'escolha_antes_regressao.json').write_text(json.dumps(protocolo, indent=2), encoding='utf-8')
    manifest = pd.read_csv(ROOT / 'data/brasil_dataset_v2/manifesto.csv')
    df = manifest[manifest.conjunto == 'test']
    assert len(df) == 13
    rows, resumo = [], {}
    device = 0 if torch.cuda.is_available() else 'cpu'
    for nome, cfg in candidatos.items():
        modelo = YOLO(str(ROOT / cfg['pesos']))
        pasta = OUT / nome
        pasta.mkdir(exist_ok=True)
        for r in df.itertuples():
            im = cv2.imread(str(ROOT / 'data/brasil_dataset_v2/images' / r.arquivo))
            h, w = im.shape[:2]
            ref = ct.yolo_para_pixels((ROOT / 'data/brasil_dataset_v2/labels' / (Path(r.arquivo).stem + '.txt')).read_text(encoding='utf-8').splitlines(), w, h)
            resultado = modelo.predict(im, imgsz=cfg['imgsz'], iou=cfg['iou'], conf=cfg['conf'],
                augment=cfg['augment'], classes=cfg['classes'], max_det=1000, device=device, verbose=False)[0]
            b, s = resultado.boxes.xyxy.cpu().numpy(), resultado.boxes.conf.cpu().numpy()
            rows.append(dict(candidato=nome, arquivo=r.arquivo, real=len(ref), previsto=len(b), **comparar_caixas(ref,b,s)))
            cv2.imwrite(str(pasta / r.arquivo), ct.desenhar(im,b,ref=ref))
        dados = pd.DataFrame([r for r in rows if r['candidato'] == nome])
        tp, fp, fn = (int(dados[c].sum()) for c in ('TP','FP','FN'))
        resumo[nome] = dict(**metricas_contagem(dados.real,dados.previsto), **taxas_deteccao(tp,fp,fn), TP=tp, FP=fp, FN=fn,
                            contagens_exatas=int((dados.real == dados.previsto).sum()))
        pd.DataFrame(rows).to_csv(OUT / 'por_imagem.csv', index=False)
        (OUT / 'resultados.json').write_text(json.dumps(resumo,indent=2), encoding='utf-8')
        print(nome, json.dumps(resumo[nome]), flush=True)


if __name__ == '__main__':
    main()
