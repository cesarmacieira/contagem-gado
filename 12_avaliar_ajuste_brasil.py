"""Calibra so na validacao; compara modelos no teste reservado uma unica rodada."""
from pathlib import Path
import hashlib
import json
import importlib.util
import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import contagem as ct
from avaliacao import metricas_contagem, comparar_caixas, taxas_deteccao

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/brasil_revisado'
OUT=ROOT/'data/experimentos/brasil_lote_01_v2/avaliacao'


def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experimento',default='brasil_lote_01_v2')
    ap.add_argument('--teste-novo',action='store_true')
    args=ap.parse_args()
    assert Path(args.experimento).name==args.experimento
    OUT=ROOT/'data/experimentos'/args.experimento/'avaliacao'
    OUT.mkdir(parents=True,exist_ok=True)
    resultado=OUT/'resultados.json'
    if resultado.exists():
        print(resultado.read_text(encoding='utf-8'));return
    df=pd.read_csv(BASE/'manifesto.csv')
    df['base']=str(BASE)
    teste_reutilizado=df[df.conjunto=='test'].copy()
    if args.teste_novo:
        extra=pd.read_csv(ROOT/'data/brasil_lote_02/revisado/manifesto.csv')
        df=pd.concat([df[df.conjunto!='test'],extra[extra.conjunto=='test']],ignore_index=True)
    assert df.groupby('grupo').conjunto.nunique().max()==1
    resumo={};linhas=[]; grade=[]
    paths=dict(original=ROOT/'data/processed/modelo_gado.pt',ajustado=OUT.parent/'treino/weights/best.pt')
    for nome,peso in paths.items():
        modelo=YOLO(str(peso));cache={}
        def det(r,sz,iou):
            key=(r.arquivo,sz,iou)
            if key not in cache:
                origem=Path(r.base)
                img=cv2.imread(str(origem/'images'/r.arquivo));h,w=img.shape[:2]
                ref=ct.yolo_para_pixels((origem/'labels'/(Path(r.arquivo).stem+'.txt')).read_text().splitlines(),w,h)
                b,s=ct.detectar_inteira(modelo,img,imgsz=sz,conf_min=.05,iou_nms=iou,device=0)
                cache[key]=(b,s,ref)
            return cache[key]
        def medir(part,sz,iou,conf,exportar=False):
            real=[];prev=[];tp=fp=fn=0
            for r in part.itertuples():
                b,s,ref=det(r,sz,iou);b,s=ct.filtrar(b,s,conf)
                met=comparar_caixas(ref,b,s);tp+=met['TP'];fp+=met['FP'];fn+=met['FN']
                real.append(len(ref));prev.append(len(b))
                if exportar:
                    linhas.append(dict(modelo=nome,arquivo=r.arquivo,conjunto=r.conjunto,real=len(ref),previsto=len(b),erro=len(b)-len(ref),**met))
                    image=cv2.imread(str(Path(r.base)/'images'/r.arquivo))
                    draw=ct.desenhar(image,b,ref=ref)
                    (OUT/'comparacoes'/nome).mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(OUT/'comparacoes'/nome/r.arquivo),draw)
            return dict(**metricas_contagem(real,prev),TP=tp,FP=fp,FN=fn,**taxas_deteccao(tp,fp,fn))
        val=df[df.conjunto=='val'];configs=[]
        print('Calibrando validacao:',nome,flush=True)
        for sz in (1024,1280,1536):
            for iou in (.5,.7):
                for conf in np.round(np.arange(.10,.71,.05),2):
                    met=medir(val,sz,iou,float(conf))
                    configs.append(dict(imgsz=sz,iou=iou,conf=float(conf),**met))
        configs.sort(key=lambda x:(x['MAE'],-(x['F1_caixas'] or 0),x['imgsz']))
        melhor=configs[0];cfg={k:melhor[k] for k in ('imgsz','iou','conf')}
        resumo[nome]=dict(pesos=str(peso),sha256=hashlib.sha256(peso.read_bytes()).hexdigest(),config=cfg,validacao=melhor)
        grade.extend(dict(modelo=nome,**c) for c in configs)
        # Salvar escolha antes de ler a primeira imagem de teste.
        (OUT/f'config_congelada_{nome}.json').write_text(json.dumps(resumo[nome],indent=2),encoding='utf-8')
        print('Configuracao congelada',nome,cfg,flush=True)
        resumo[nome]['teste']=medir(df[df.conjunto=='test'],cfg['imgsz'],cfg['iou'],cfg['conf'],True)
        # Comparacao controla inferencia: mesmos parametros nos dois modelos.
        resumo[nome]['teste_config_fixa']=medir(df[df.conjunto=='test'],1280,.7,.15)
        if args.teste_novo:
            resumo[nome]['teste_anterior_reutilizado']=medir(teste_reutilizado,cfg['imgsz'],cfg['iou'],cfg['conf'])
        # As dez fotos antigas ja foram usadas em varreduras: apenas diagnostico.
        pasta_antiga=ROOT/'data/teste_brasil'
        antigos=pd.read_csv(pasta_antiga/'gabarito.csv')
        diagnostico=[]
        for r in antigos.itertuples():
            img=cv2.imread(str(pasta_antiga/r.arquivo))
            if img is None: continue
            b,s=ct.detectar_inteira(modelo,img,imgsz=cfg['imgsz'],conf_min=.05,iou_nms=cfg['iou'],device=0)
            n=len(ct.filtrar(b,s,cfg['conf'])[0])
            erro=max(r.minimo-n,0,n-r.maximo)
            diagnostico.append(dict(arquivo=r.arquivo,minimo=int(r.minimo),maximo=int(r.maximo),previsto=n,erro_ao_intervalo=erro))
        resumo[nome]['exploratorio_antigo']=dict(dentro=sum(r['erro_ao_intervalo']==0 for r in diagnostico),total=len(diagnostico),erro_medio_ao_intervalo=float(np.mean([r['erro_ao_intervalo'] for r in diagnostico])),imagens=diagnostico)
        print('Teste:',nome,resumo[nome]['teste'],flush=True)
    pd.DataFrame(grade).to_csv(OUT/'grade_validacao.csv',index=False)
    pd.DataFrame(linhas).to_csv(OUT/'contagem_teste.csv',index=False)
    resumo['protocolo']=dict(validacao_imagens=int((df.conjunto=='val').sum()),teste_imagens=int((df.conjunto=='test').sum()),separacao='autor',rotulos='primeira revisao visual assistente',teste_usado_para_ajustar=False,teste_novo=args.teste_novo,limitacao='Amostra pequena; nao demonstra confiabilidade em rebanhos grandes. Precisao das caixas depende de revisao adicional do gabarito.')
    resumo['melhorou_teste']=resumo['ajustado']['teste']['MAE']<resumo['original']['teste']['MAE']
    resultado.write_text(json.dumps(resumo,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(resumo,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
