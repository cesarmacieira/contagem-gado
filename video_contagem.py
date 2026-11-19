"""Decodifica o video ate o fim e analisa todos os quadros ou amostras temporais."""
from pathlib import Path
import json,math
import cv2
import numpy as np


def analisar(path,inferir,pasta,intervalo=1.,progresso=None,max_amostras=5000):
    if intervalo<0 or not math.isfinite(intervalo):raise ValueError('Intervalo invalido')
    pasta=Path(pasta);pasta.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError('Nao foi possivel abrir o video. Use MP4, MOV ou AVI com codec suportado.')
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps<=0:raise ValueError('Video sem taxa de quadros valida; nao e possivel construir a linha do tempo')
        passo=max(1,round(intervalo*fps)) if intervalo else 1
        previstos=(total-1)//passo+1+int((total-1)%passo!=0) if total>0 else 0
        if previstos>max_amostras:raise ValueError(f'Analise excederia {max_amostras} quadros. Aumente o intervalo entre amostras.')
        registros=[];indice=0;ultimo=None;ultimo_id=-1
        def processar(frame,i):
            if len(registros)>=max_amostras:raise ValueError('Limite de quadros atingido antes do fim; analise nao concluida. Aumente o intervalo.')
            b,s=inferir(frame);b=np.asarray(b,float).reshape(-1,4);s=np.asarray(s,float)
            if s.shape!=(len(b),) or not np.isfinite(b).all() or not np.isfinite(s).all():raise ValueError('Detector retornou caixas invalidas')
            nome=f'quadro_{i:08d}.jpg'
            if not cv2.imwrite(str(pasta/nome),frame,[cv2.IMWRITE_JPEG_QUALITY,90]):raise IOError('Nao consegui salvar o quadro analisado')
            r=dict(quadro=i,segundos=i/fps,contagem=len(b),confianca_media=float(s.mean()) if len(s) else None,
                largura=frame.shape[1],altura=frame.shape[0],caixas=b.tolist(),scores=s.tolist(),arquivo=nome)
            registros.append(r)
            if progresso:progresso(len(registros),i,total)
        while True:
            ok,frame=cap.read()
            if not ok:break
            ultimo=frame;ultimo_id=indice
            if indice%passo==0:processar(frame,indice)
            indice+=1
        if ultimo is None:raise ValueError('Video nao contem quadros decodificaveis')
        # Truncamento detectavel pelo conteiner nao pode virar sucesso silencioso.
        if total>0 and indice<total-max(2,round(total*.01)):raise ValueError('Decodificacao terminou antes do numero de quadros informado; video pode estar truncado')
        if registros[-1]['quadro']!=ultimo_id:processar(ultimo,ultimo_id)
        meta=dict(concluido=True,fps=fps,quadros_declarados=total,quadros_decodificados=indice,quadros_analisados=len(registros),
            duracao_segundos=indice/fps,ultimo_instante_analisado=ultimo_id/fps,intervalo_solicitado=intervalo,
            intervalo_efetivo=passo/fps,modo='todos_os_quadros' if passo==1 else 'amostragem_temporal',
            min_simultaneo=min(r['contagem'] for r in registros),max_simultaneo=max(r['contagem'] for r in registros),
            media_simultanea=sum(r['contagem'] for r in registros)/len(registros),animais_unicos=None,
            nota='Contagem simultanea nos quadros analisados; nao representa individuos unicos ou total que atravessou a cena.')
        (pasta/'resultado.json').write_text(json.dumps(dict(meta=meta,quadros=registros),ensure_ascii=False,indent=2),encoding='utf-8')
        return dict(meta=meta,quadros=registros)
    finally:cap.release()
