"""Novo treino de deteccao com o lote revisado completo, sem promover o app.

Congela protocolo e dados antes do treino. Avalia apenas validacao e regressao
historica, nunca apresentadas como um novo teste independente.
"""
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/experimentos/especialista_solo_revisado_v2'
VAL = ROOT / 'data/experimentos/solo_denso_validacao_v2'
PESO = ROOT / 'data/processed/modelo_gado.pt'


def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU indisponivel; nenhum treino iniciado')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'resultado.json').exists():
        print('Experimento concluido e preservado', flush=True)
        return
    spec = spec_from_file_location('preparo', ROOT/'51_ajustar_especialista_solo.py')
    prep = module_from_spec(spec); spec.loader.exec_module(prep)
    prep.EXP, prep.VAL = OUT, VAL
    protocolo = OUT/'protocolo_treino.json'
    if not protocolo.exists():
        prep.preparar()
        d = json.loads((OUT/'protocolo.json').read_text())
        d.update(epochs=16, freeze=10, batch=8, lr0=.0001, imgsz=1280,
                 confiancas_fixas=dict(solo_denso=.15, solo_poucos=.30),
                 justificativa='Lote revisado completo: 62 caixas contra 53 no treino anterior. '
                 'Congelar as dez primeiras camadas limita alteracao do extrator com poucos dados novos.',
                 selecao='Checkpoint best por avaliacao YOLO em val; sem busca de hiperparametros apos ver a regressao.',
                 limites='Duas imagens de validacao da mesma cena. Comparacoes adicionais sao regressao conhecida.',
                 promovido=False)
        protocolo.write_text(json.dumps(d, indent=2), encoding='utf-8')
        paths = list((OUT/'dataset').rglob('*.txt'))
        (OUT/'hashes_antes_treino.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}, indent=2), encoding='utf-8')
    d = json.loads(protocolo.read_text())
    assert hashlib.sha256(PESO.read_bytes()).hexdigest() == d['pesos_iniciais_sha256']
    for nome, sha in json.loads((OUT/'hashes_antes_treino.json').read_text()).items():
        assert hashlib.sha256((OUT/nome).read_bytes()).hexdigest() == sha, nome
    last = OUT/'treino/weights/last.pt'
    if not last.exists():
        YOLO(str(PESO)).train(data=str(OUT/'dataset.yaml'), project=str(OUT), name='treino', exist_ok=False,
            epochs=d['epochs'], imgsz=d['imgsz'], batch=d['batch'], device=0, workers=0, freeze=d['freeze'],
            optimizer='AdamW', lr0=d['lr0'], lrf=.2, warmup_epochs=1, warmup_bias_lr=0,
            patience=16, mosaic=.3, close_mosaic=2, scale=.3, translate=.05, fliplr=.5, flipud=0,
            degrees=0, hsv_s=.3, hsv_v=.2, mixup=0, copy_paste=0, seed=1121,
            deterministic=True, cache=False, amp=True, save_period=4, plots=False, verbose=False)
    elif int(pd.read_csv(OUT/'treino/results.csv').epoch.iloc[-1]) < d['epochs']:
        YOLO(str(last)).train(resume=True, device=0, workers=0)
    candidate = OUT/'treino/weights/best.pt'
    selection = dict(peso=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                     criterio=d['selecao'], config=d['confiancas_fixas'])
    (OUT/'escolha_antes_regressao.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    # Duas imagens de val e as 13 fotos historicas. Nao consulta teste novo.
    cases = []
    for r in json.loads((VAL/'catalogo_val.json').read_text()):
        cases.append((VAL/'validacao/images'/r['arquivo'], VAL/'validacao/labels'/(Path(r['arquivo']).stem+'.txt'), 'val_densa',1280,.15,.7))
    antigo = ROOT/'data/brasil_dataset_v2'
    for r in pd.read_csv(antigo/'manifesto.csv').itertuples():
        if r.conjunto=='test':
            cases.append((antigo/'images'/r.arquivo, antigo/'labels'/(Path(r.arquivo).stem+'.txt'), 'regressao_poucos',1024,.30,.5))
    gab = pd.read_csv(ROOT/'data/teste_brasil/gabarito.csv')
    for r in gab.itertuples():
        cases.append((ROOT/'data/teste_brasil'/r.arquivo,None,'intervalos_densos',1280,.15,.7))
    rows = []
    for nome, peso in [('referencia',PESO),('candidato',candidate)]:
        modelo = YOLO(str(peso))
        for img, lab, origem, imgsz, conf, iou in cases:
            im = cv2.imread(str(img))
            if im is None:
                raise ValueError(f'Imagem ausente: {img}')
            b,s = ct.detectar_inteira(modelo,im,imgsz=imgsz,conf_min=conf,iou_nms=iou,device=0,classes=[0],augment=False)
            row = dict(modelo=nome, arquivo=img.name, origem=origem, previsto=len(b))
            if lab is not None:
                h,w=im.shape[:2]; ref=ct.yolo_para_pixels(lab.read_text(encoding='utf-8-sig').splitlines(),w,h)
                row.update(real=len(ref), erro=abs(len(b)-len(ref)), **comparar_caixas(ref,b,s))
            else:
                g=gab[gab.arquivo==img.name].iloc[0]
                row.update(minimo=int(g.minimo),maximo=int(g.maximo),erro=max(g.minimo-len(b),0,len(b)-g.maximo))
            rows.append(row)
            np.savez(OUT/(nome+'__'+img.stem+'.npz'),caixas=b,scores=s)
        print('Avaliacao concluida:',nome,flush=True)
    tabela=pd.DataFrame(rows); tabela.to_csv(OUT/'por_imagem.csv',index=False)
    resumo={}
    for (origem,nome),g in tabela.groupby(['origem','modelo']):
        m=dict(MAE=float(g.erro.mean()),n_imagens=len(g))
        if g.real.notna().all():
            tp,fp,fn=(int(g[k].sum()) for k in ('TP','FP','FN'))
            m.update(**metricas_contagem(g.real,g.previsto),TP=tp,FP=fp,FN=fn,**taxas_deteccao(tp,fp,fn))
        resumo.setdefault(origem,{})[nome]=m
    a=tabela[tabela.modelo=='referencia'].set_index(['origem','arquivo'])
    b=tabela[tabela.modelo=='candidato'].set_index(['origem','arquivo'])
    resultado=dict(metricas=resumo,sha256=selection['sha256'],promovido=False,
        piorou_contagem=bool((b.erro>a.erro).any()),perdeu_correspondencias=bool((b.TP<a.TP).any()),
        limite=d['limites'])
    (OUT/'resultado.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    print(json.dumps(resultado,indent=2),flush=True)


if __name__=='__main__':
    main()
