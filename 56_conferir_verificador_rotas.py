"""Regressao por perfil de solo antes de disponibilizar confirmacao opcional."""
from pathlib import Path
import hashlib
import json
import shutil
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas
from verificador_visual import carregar_binario,confirmar,mascara_confirmacao

ROOT = Path(__file__).resolve().parent
BIN = ROOT / 'data/experimentos/verificador_binario_v1'
OUT = ROOT / 'data/experimentos/verificador_compatibilidade_v1'
EXTRATOR = ROOT / 'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-lado',type=float,default=0)
    ap.add_argument('--saida',type=Path,default=OUT)
    ap.add_argument('--binario',type=Path,default=BIN)
    ap.add_argument('--sem-publicar',action='store_true')
    args = ap.parse_args()
    out = args.saida
    if (out / 'resultado.json').exists():
        raise ValueError('Comparacao existente preservada; escolher nova pasta --saida')
    torch.set_num_threads(4)
    out.mkdir(parents=True,exist_ok=True)
    binario = args.binario
    cfg = json.loads((binario / 'escolha_antes_teste.json').read_text())
    verificador = carregar_binario(EXTRATOR,binario / 'classificador.joblib',cfg['classificador_sha256'])
    modelo = YOLO(str(ROOT / 'data/processed/modelo_gado.pt'))
    manifest = pd.read_csv(ROOT / 'data/brasil_dataset_v2/manifesto.csv')
    amostras = [dict(arquivo=r.arquivo,img=ROOT / 'data/brasil_dataset_v2/images' / r.arquivo,
        rot=ROOT / 'data/brasil_dataset_v2/labels' / (Path(r.arquivo).stem+'.txt'),origem='regressao_13_poucos',rota='solo_poucos')
        for r in manifest[manifest.conjunto=='test'].itertuples()]
    piloto = json.loads((ROOT / 'data/experimentos/solo_piloto_separado_v1/protocolo_antes_teste.json').read_text())
    for rota in ('solo_poucos','solo_denso'):
        for r in piloto['imagens_teste']:
            base = ROOT / 'data/experimentos/solo_piloto_separado_v1/teste'
            amostras.append(dict(arquivo=r['arquivo'],img=base / 'images' / r['arquivo'],
                rot=base / 'labels' / (Path(r['arquivo']).stem+'.txt'),origem='regressao_piloto',rota=rota))
    val = json.loads((ROOT / 'data/experimentos/solo_denso_validacao_v2/catalogo_val.json').read_text())
    for r in val:
        base = ROOT / 'data/experimentos/solo_denso_validacao_v2/validacao'
        amostras.append(dict(arquivo=r['arquivo'],img=base / 'images' / r['arquivo'],rot=base / 'labels' / (Path(r['arquivo']).stem+'.txt'),
            origem='val_densa',rota='solo_denso'))
    gab = pd.read_csv(ROOT / 'data/teste_brasil/gabarito.csv')
    amostras += [dict(arquivo=r.arquivo,img=ROOT / 'data/teste_brasil' / r.arquivo,
        minimo=r.minimo,maximo=r.maximo,origem='desenvolvimento_intervalos',rota='solo_denso') for r in gab.itertuples()]
    amostras += [dict(arquivo=r.arquivo,img=ROOT / 'data/teste_brasil' / r.arquivo,
        minimo=r.minimo,maximo=r.maximo,origem='desenvolvimento_intervalos',rota='solo_poucos') for r in gab.itertuples() if r.maximo<=10]
    rows = []
    for r in amostras:
        rota = r['rota']
        conf,sz,iou = (.30,1024,.5) if rota=='solo_poucos' else (.15,1280,.7)
        im = cv2.imread(str(r['img']))
        h,w = im.shape[:2]
        b,s = ct.detectar_inteira(modelo,im,imgsz=sz,iou_nms=iou,classes=[0],augment=False,device=0)
        b,s = ct.filtrar(b,s,conf)
        v = confirmar(verificador,im,b)
        ref = ct.yolo_para_pixels(r['rot'].read_text().splitlines(),w,h) if 'rot' in r else None
        for ver in (False,True):
            confirmado,absteve = mascara_confirmacao(b,v,cfg['limiar'],args.min_lado)
            mask = confirmado if ver else np.ones(len(b),dtype=bool)
            bb,ss = b[mask],s[mask]
            row = dict(rota=rota,origem=r['origem'],arquivo=r['arquivo'],verificado=ver,previsto=len(bb))
            if ref is not None:
                row.update(real=len(ref),erro=abs(len(bb)-len(ref)),**comparar_caixas(ref,bb,ss))
            else:
                row.update(erro=max(r['minimo']-len(bb),0,len(bb)-r['maximo']))
            rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(out / 'por_imagem.csv',index=False)
    liberadas = []
    resumo = {}
    for rota,g in d.groupby('rota'):
        antes = g[~g.verificado].set_index(['origem','arquivo'])
        depois = g[g.verificado].set_index(['origem','arquivo'])
        comparavel = antes.TP.notna()
        perdeu_tp = bool((depois.loc[comparavel,'TP'] < antes.loc[comparavel,'TP']).any())
        piorou_erro = bool((depois.erro > antes.erro).any())
        melhorou_fp = bool(depois.loc[comparavel,'FP'].sum() < antes.loc[comparavel,'FP'].sum())
        apto = not perdeu_tp and not piorou_erro and melhorou_fp
        resumo[rota] = dict(perdeu_correspondencia_positiva=perdeu_tp,piorou_erro_em_alguma_imagem=piorou_erro,
            reduziu_falsos_positivos=melhorou_fp,disponivel_apenas_com_opt_in=apto,
            FP_antes=int(antes.loc[comparavel,'FP'].sum()),FP_depois=int(depois.loc[comparavel,'FP'].sum()))
        if apto:
            liberadas.append(rota)
    # Opcao sempre desligada por padrao; nao troca os pesos do detector.
    destino = ROOT / 'data/modelos/verificador_solo_v1'
    destino.mkdir(parents=True,exist_ok=True)
    if not args.sem_publicar:
        shutil.copy2(EXTRATOR,destino / 'extrator.pth')
        shutil.copy2(binario / 'classificador.joblib',destino / 'classificador.joblib')
    meta = dict(versao=2 if args.min_lado else 1,experimental=True,ativo_por_padrao=False,visoes_permitidas=liberadas,
        extrator='data/modelos/verificador_solo_v1/extrator.pth',
        extrator_sha256=cfg['extrator_sha256'],classificador='data/modelos/verificador_solo_v1/classificador.joblib',
        classificador_sha256=cfg['classificador_sha256'],limiar=cfg['limiar'],min_lado_px=args.min_lado,
        detector_sha256=hashlib.sha256((ROOT / 'data/processed/modelo_gado.pt').read_bytes()).hexdigest(),
        descricao='Conferencia experimental de objetos parecidos com gado. Nao recupera animais omitidos pelo detector.',
        limite='Positivos de validacao de um grupo; regressao conhecida e apenas quatro negativos novos de teste. Sem validacao em rebanho brasileiro denso novo.')
    if not args.sem_publicar:
        (destino / 'config.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (out / 'resultado.json').write_text(json.dumps(dict(rotas=resumo,config=meta),indent=2),encoding='utf-8')
    print(json.dumps(resumo,indent=2),flush=True)
    print(d[d.origem=='regressao_piloto'][['rota','arquivo','verificado','previsto','TP','FP','FN']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
