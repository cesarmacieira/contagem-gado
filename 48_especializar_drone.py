"""Ajuste e avaliacao exclusivos de drone, sem tocar nos modelos de solo.

Reconstroi recortes de rotulos originais evitando fragmentos sem anotacao.
Seleciona checkpoint/limiar na validacao. Teste historico e regressao, nao novo
teste independente. So promove a rota aerea se superar ambas as referencias.
"""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao

ROOT = Path(__file__).resolve().parent
EXP = ROOT / 'data/experimentos/especialista_drone_v1'
PESO = ROOT / 'data/processed/modelo_gado.pt'
CONFS = (.15,.25,.35,.45,.55,.65)


def catalogo():
    d = pd.read_csv(ROOT / 'data/processed/divisao.csv')
    d = d[(d.dominio == 'aereo_pasto') & d.usar.astype(str).str.lower().isin(['true','1'])]
    rows = []
    for r in d.itertuples():
        nome = f"{r.fazenda}__{str(r.voo).replace('/', '_')}__{r.stem}"
        base = ROOT / 'data/interim' / r.conjunto
        img, rot = base / 'imagens' / (nome+'.jpg'), base / 'rotulos' / (nome+'.txt')
        if not img.is_file() or not rot.is_file():
            raise FileNotFoundError(nome)
        rows.append(dict(arquivo=nome,img=str(img),rot=str(rot),grupo=f'{r.fazenda}/{r.voo}',
                         conjunto=r.conjunto,n_vacas=int(r.n_vacas)))
    df = pd.DataFrame(rows)
    assert df.groupby('grupo').conjunto.nunique().max() == 1
    return df


def selecionar(df, limite, seed):
    # Diversidade de voos primeiro; nenhum resultado do detector participa.
    d = df.sample(frac=1,random_state=seed).copy()
    d['ordem_grupo'] = d.groupby('grupo').cumcount()
    return d.sort_values('ordem_grupo', kind='stable').head(limite)


def preparar(df):
    manifesto = []
    selecionados = {}
    for split, limite, max_tiles in [('train',80,512),('val',24,96)]:
        fonte = selecionar(df[df.conjunto == split], limite, 811 if split == 'train' else 812)
        selecionados[split] = fonte
        for pasta in ('images','labels'):
            (EXP / 'dataset' / pasta / split).mkdir(parents=True,exist_ok=True)
        exportados = 0
        rng = np.random.default_rng(813 if split == 'train' else 814)
        for r in fonte.itertuples():
            im = cv2.imread(r.img)
            h,w = im.shape[:2]
            bb = ct.yolo_para_pixels(Path(r.rot).read_text(encoding='utf-8').splitlines(),w,h)
            opcoes = []
            for x,y in ct.grade_tiles(w,h,1024,.2):
                tw,th = min(1024,w-x),min(1024,h-y)
                if ct.tile_tem_fragmento_sem_rotulo(bb,x,y,tw,th):
                    continue
                linhas = ct.rotulos_do_tile(bb,x,y,tw,th)
                opcoes.append((x,y,linhas))
            # Ate seis tiles positivos e dois negativos por imagem.
            positivos = sorted([v for v in opcoes if v[2]],key=lambda v:len(v[2]),reverse=True)[:6]
            negativos = [v for v in opcoes if not v[2]]
            if negativos:
                negativos = [negativos[k] for k in rng.choice(len(negativos),min(2,len(negativos)),replace=False)]
            for x,y,linhas in positivos+negativos:
                if exportados >= max_tiles:
                    break
                nome = f'{r.arquivo}__x{x}_y{y}'
                cv2.imwrite(str(EXP / 'dataset/images' / split / (nome+'.jpg')),im[y:y+1024,x:x+1024], [cv2.IMWRITE_JPEG_QUALITY,95])
                (EXP / 'dataset/labels' / split / (nome+'.txt')).write_text('\n'.join(linhas)+ ('\n' if linhas else ''),encoding='utf-8')
                manifesto.append(dict(arquivo=nome,original=r.img,grupo=r.grupo,conjunto=split,n_bovinos=len(linhas)))
                exportados += 1
        print('Preparado',split,exportados,'recortes de',len(fonte),'originais',flush=True)
    m = pd.DataFrame(manifesto)
    assert m.groupby('grupo').conjunto.nunique().max() == 1
    assert len(m[m.conjunto=='train']) >= 32 and len(m[m.conjunto=='val']) >= 8
    m.to_csv(EXP / 'tiles.csv',index=False)
    selecionados['val'].to_csv(EXP / 'validacao_inteira.csv',index=False)
    df[df.conjunto=='test'].to_csv(EXP / 'regressao_historica.csv',index=False)
    arquivo = EXP / 'dataset.yaml'
    arquivo.write_text(f'path: {EXP.as_posix()}/dataset\ntrain: images/train\nval: images/val\nnames:\n  0: vaca\n',encoding='utf-8')
    (EXP / 'preparacao.json').write_text(json.dumps(dict(pesos_iniciais_sha256=hashlib.sha256(PESO.read_bytes()).hexdigest(),
        visao='drone_vertical', epochs=8, imgsz=1024, criterio='Ajuste exclusivo de drone; sem fragmentos visiveis sem rotulo nos tiles',
        dataset_novo_brasileiro_usado=False, teste_independente=False,
        train_tiles=int((m.conjunto=='train').sum()),val_tiles=int((m.conjunto=='val').sum()),
        n_validacao_inteira=len(selecionados['val']), n_regressao_historica=int((df.conjunto=='test').sum())),indent=2),encoding='utf-8')
    return arquivo


def avaliar(peso, df, confs, legado=False):
    modelo = YOLO(str(peso))
    rows = []
    for i,r in enumerate(df.itertuples(),1):
        im = cv2.imread(r.img)
        h,w = im.shape[:2]
        ref = ct.yolo_para_pixels(Path(r.rot).read_text(encoding='utf-8').splitlines(),w,h)
        b,s = ct.detectar(modelo,im,tile=1024,overlap=.2,iou_local=.7,iou_nms=.5,legado=legado,
                          max_det=300 if legado else 1000,device=0,classes=[0])
        for c in confs:
            bb,ss = ct.filtrar(b,s,c)
            rows.append(dict(arquivo=r.arquivo,grupo=r.grupo,conf=c,real=len(ref),previsto=len(bb),**comparar_caixas(ref,bb,ss)))
        if i % 10 == 0:
            print(peso.name,'legado' if legado else 'atual',i,'/',len(df),flush=True)
    d = pd.DataFrame(rows)
    resumo = {}
    for c,g in d.groupby('conf'):
        tp,fp,fn = (int(g[k].sum()) for k in ('TP','FP','FN'))
        e = g.assign(erro=(g.previsto-g.real).abs())
        m = dict(**metricas_contagem(g.real,g.previsto),**taxas_deteccao(tp,fp,fn),TP=tp,FP=fp,FN=fn,
                 MAE_macro_grupos=float(e.groupby('grupo').erro.mean().mean()))
        m['score'] = (m['MAE']+m['MAE_macro_grupos'])/2
        densas = g[g.real>=8]
        if len(densas):
            dtp,dfp,dfn = (int(densas[k].sum()) for k in ('TP','FP','FN'))
            m['oito_ou_mais'] = dict(**metricas_contagem(densas.real,densas.previsto),**taxas_deteccao(dtp,dfp,dfn))
        resumo[float(c)] = m
    return resumo,d


def main():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    EXP.mkdir(parents=True,exist_ok=True)
    if (EXP / 'resultado.json').exists():
        print('Experimento concluido; resultados preservados',flush=True)
        return
    if not (EXP / 'preparacao.json').exists():
        preparar(catalogo())
    if not (EXP / 'treino/weights/last.pt').exists():
        YOLO(str(PESO)).train(data=str(EXP / 'dataset.yaml'),project=str(EXP),name='treino',exist_ok=False,
            epochs=8,imgsz=1024,batch=8,device=0,workers=2,optimizer='AdamW',lr0=.00004,lrf=.2,
            warmup_epochs=1,warmup_bias_lr=0,patience=8,mosaic=.3,close_mosaic=2,scale=.2,translate=.05,
            fliplr=.5,flipud=.5,degrees=0,mixup=0,copy_paste=0,seed=815,deterministic=True,
            cache=False,amp=True,save_period=4,plots=False,verbose=False)
    else:
        progresso = pd.read_csv(EXP / 'treino/results.csv')
        if int(progresso.epoch.iloc[-1]) < 8:
            print('Retomando checkpoint aereo salvo',flush=True)
            YOLO(str(EXP / 'treino/weights/last.pt')).train(resume=True,device=0,workers=2)
    val = pd.read_csv(EXP / 'validacao_inteira.csv')
    escolhas = []
    candidatos = [PESO, EXP / 'treino/weights/best.pt', EXP / 'treino/weights/last.pt']
    for peso in candidatos:
        m,rows = avaliar(peso,val,CONFS)
        rows.to_csv(EXP / f'validacao_{peso.parent.name}_{peso.stem}.csv',index=False)
        conf = min(m,key=lambda c:(m[c]['score'],-(m[c]['F1_caixas'] or 0)))
        escolhas.append(dict(peso=str(peso),sha256=hashlib.sha256(peso.read_bytes()).hexdigest(),conf=conf,validacao=m[conf]))
        print('Escolha validacao',peso.name,conf,m[conf]['MAE'],flush=True)
    baseline = escolhas[0]
    novo = min(escolhas[1:],key=lambda r:(r['validacao']['score'],-(r['validacao']['F1_caixas'] or 0)))
    (EXP / 'escolha_antes_regressao.json').write_text(json.dumps(dict(original=baseline,especialista=novo,
        todos=escolhas,visao='drone_vertical',nao_exige_melhora_no_solo=True,teste_independente=False),indent=2),encoding='utf-8')
    teste = pd.read_csv(EXP / 'regressao_historica.csv')
    resultados = {}
    for nome,p,c,legado in [('original_calibrado',Path(baseline['peso']),baseline['conf'],False),
                             ('especialista',Path(novo['peso']),novo['conf'],False),
                             ('app_anterior',PESO,.5,True)]:
        m,rows = avaliar(p,teste,[c],legado)
        resultados[nome] = m[c]
        rows.to_csv(EXP / f'regressao_{nome}.csv',index=False)
    a = resultados['especialista']
    passou = all(a['MAE'] < resultados[k]['MAE'] and (a['F1_caixas'] or 0) >= (resultados[k]['F1_caixas'] or 0)
                 for k in ('original_calibrado','app_anterior'))
    if a.get('oito_ou_mais'):
        passou &= all(a['oito_ou_mais']['MAE'] <= resultados[k]['oito_ou_mais']['MAE']
            and (a['oito_ou_mais']['recall_caixas'] or 0) >= (resultados[k]['oito_ou_mais']['recall_caixas'] or 0)
            for k in ('original_calibrado','app_anterior'))
    resultado = dict(visao='drone_vertical',resultados=resultados,passou_na_visao=bool(passou),
        promove_solo=False,teste_independente=False,escolhido=novo,promovido=False)
    if passou:
        destino = ROOT / 'data/modelos/drone_vertical_v1.pt'
        shutil.copy2(novo['peso'],destino)
        regpath = ROOT / 'data/modelos/especialistas.json'
        backup = ROOT / 'data/backups' / ('especialistas_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
        backup.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(regpath,backup)
        reg = json.loads(regpath.read_text(encoding='utf-8'))
        visao = reg['visoes']['drone_vertical']
        visao.update(pesos='data/modelos/drone_vertical_v1.pt',sha256=hashlib.sha256(destino.read_bytes()).hexdigest(),
            estado='avaliado_em_regressao',descricao=f"Especialista aereo: erro medio {a['MAE']:.2f} por imagem na regressao historica aerea. Nao e teste brasileiro novo nem referencia para fotos de solo.")
        visao['config'].update(conf=novo['conf'],iou=.7,legado=False)
        regpath.write_text(json.dumps(reg,ensure_ascii=False,indent=2),encoding='utf-8')
        resultado['promovido'] = True
    (EXP / 'resultado.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    print(json.dumps(resultado,indent=2),flush=True)


if __name__ == '__main__':
    main()
