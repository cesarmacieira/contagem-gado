"""Compara padroes preexistentes com a calibracao congelada na validacao nova."""
from pathlib import Path
import json
import cv2
import pandas as pd
from ultralytics import YOLO
import contagem as ct
from avaliacao import metricas_contagem, comparar_caixas, taxas_deteccao

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'data/experimentos/comparacao_calibracao_brasil';out.mkdir(exist_ok=True,parents=True)
    meta=json.loads((ROOT/'data/processed/metricas.json').read_text(encoding='utf-8'))
    congelado=json.loads((ROOT/'data/experimentos/brasil_lote_01_v2/avaliacao/config_congelada_original.json').read_text(encoding='utf-8'))
    df=pd.read_csv(ROOT/'data/brasil_revisado/manifesto.csv');df['base']=str(ROOT/'data/brasil_revisado');df['rodada']='primeira'
    extra=pd.read_csv(ROOT/'data/brasil_lote_02/revisado/manifesto.csv');extra['rodada']='complemento'
    df=pd.concat([df[df.conjunto=='test'],extra[extra.conjunto=='test']],ignore_index=True)
    cfgs=dict(app_atual=dict(imgsz=int(meta.get('imgsz_por_dominio',{}).get('foto_openimages',meta['tile'])),iou=float(meta.get('iou_por_dominio',{}).get('foto_openimages',.7)),conf=float(meta['conf_por_dominio']['foto_openimages'])),referencia_035=dict(imgsz=1024,iou=.7,conf=.35),calibrado_validacao=congelado['config'])
    modelo=YOLO(str(ROOT/'data/processed/modelo_gado.pt'));linhas=[];res={}
    for nome,cfg in cfgs.items():
        real=[];prev=[];tp=fp=fn=0
        for r in df.itertuples():
            origem=Path(r.base);im=cv2.imread(str(origem/'images'/r.arquivo));h,w=im.shape[:2]
            ref=ct.yolo_para_pixels((origem/'labels'/(Path(r.arquivo).stem+'.txt')).read_text().splitlines(),w,h)
            b,s=ct.detectar_inteira(modelo,im,imgsz=cfg['imgsz'],iou_nms=cfg['iou'],conf_min=.05,device=0)
            b,s=ct.filtrar(b,s,cfg['conf']);m=comparar_caixas(ref,b,s);tp+=m['TP'];fp+=m['FP'];fn+=m['FN']
            real.append(len(ref));prev.append(len(b));linhas.append(dict(perfil=nome,arquivo=r.arquivo,rodada=r.rodada,real=len(ref),previsto=len(b),**m))
        res[nome]=dict(config=cfg,**metricas_contagem(real,prev),TP=tp,FP=fp,FN=fn,**taxas_deteccao(tp,fp,fn))
    res['protocolo']='Configs antigas fixas comparadas com config escolhida e congelada nas nove fotos de validacao. Nenhum parametro escolhido nestes 13 testes. Maximo 10 animais por imagem neste teste, sem demonstracao de rebanho denso.'
    (out/'resultados.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(linhas).to_csv(out/'contagem_por_imagem.csv',index=False)
    print(json.dumps(res,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
