"""Segunda opiniao conservadora. Nunca altera pesos ou a deteccao original."""
from pathlib import Path
import numpy as np
from avaliacao import comparar_caixas


def decidir(original, alternativas, vlm):
    original=np.asarray(original,float).reshape(-1,4)
    saida=dict(contagem=len(original),corrigido=False,motivo='Sem consenso seguro',caixas=original.tolist())
    if len(original)>14 or len(alternativas)!=2:return saida
    a,b=alternativas
    n=len(a['caixas'])
    if n>14 or n!=len(b['caixas']) or vlm.get('incerto',True) or vlm.get('contagem')!=n:return saida
    if n==len(original):
        saida['motivo']='Contagens concordam; original preservado'
        return saida
    if n:
        pares=comparar_caixas(a['caixas'],b['caixas'],b['scores'],.3)
        if pares['TP']!=n:return saida
    else:
        # Ausencia de deteccoes nao prova ausencia de bovinos.
        # Exigir cavalos explicitamente identificados cobrindo cada proposta original.
        for alt in alternativas:
            cavalos=alt.get('cavalos',[])
            scores=alt.get('scores_cavalos',[])
            if len(cavalos)!=len(original) or comparar_caixas(original,cavalos,scores,.3)['TP']!=len(original):return saida
    saida.update(contagem=n,corrigido=True,motivo='Dois detectores e VLM concordam, com correspondencia espacial',caixas=a['caixas'])
    return saida


def inferir(dados,root):
    import cv2
    from ultralytics import YOLO,RTDETR
    img=cv2.imdecode(np.frombuffer(dados,np.uint8),cv2.IMREAD_COLOR)
    if img is None:raise ValueError('Imagem invalida')
    resultados=[]
    for nome,classe in [('yolo11x.pt',YOLO),('rtdetr-l.pt',RTDETR)]:
        path=Path(root)/'data/modelos_candidatos'/nome
        if not path.is_file():raise ValueError(f'Peso local ausente: {nome}')
        r=classe(str(path)).predict(img,imgsz=1024,conf=.3,iou=.5,classes=[17,19],verbose=False)[0]
        caixas=r.boxes.xyxy.cpu().numpy();scores=r.boxes.conf.cpu().numpy();ids=r.boxes.cls.cpu().numpy()
        bovinos=ids==19;cavalos=(ids==17)&(scores>=.6)
        resultados.append(dict(modelo=nome,caixas=caixas[bovinos].tolist(),scores=scores[bovinos].tolist(),cavalos=caixas[cavalos].tolist(),scores_cavalos=scores[cavalos].tolist()))
    return resultados


def propor(original, alternativas, vlm):
    """Proposta localizada mesmo com VLM incerto, sem transforma-la em confirmacao."""
    saida=decidir(original,alternativas,vlm)
    if saida['corrigido'] or len(alternativas)!=2:return saida
    a,b=alternativas;n=len(a['caixas'])
    if not 0<n<=14 or n!=len(b['caixas']) or n!=vlm.get('contagem') or n==len(original):return saida
    pares=comparar_caixas(a['caixas'],b['caixas'],b['scores'],.3)['TP']
    if pares<max(1,n-1):return saida
    return dict(contagem=n,corrigido=False,proposta=True,incerto=True,correspondencias=pares,motivo=f'Dois detectores e a leitura VLM indicam {n}; {pares} caixas correspondem entre detectores. Proposta incerta, exige conferencia dos animais sobrepostos.',caixas=a['caixas'])
