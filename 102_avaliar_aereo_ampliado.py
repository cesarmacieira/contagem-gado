"""Escolhe especialista por validacao, sem consultar teste nem promover sozinho."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np,pandas as pd,torch
from ultralytics import YOLO
from especialistas import resolver_especialista
from verificador_nucleo import carregar,pontuar
from correcao_aerea import selecionar_instancias
from avaliacao import comparar_caixas
import contagem as ct

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/aereo_ampliado_v2'


def main():
    if (OUT/'escolha_validacao.json').exists():raise ValueError('Escolha preservada')
    profile=resolver_especialista(ROOT,'curral_externo_topo');v=profile['verificador_nucleo'];torch.set_num_threads(4)
    vf=carregar(ROOT/v['extrator'],ROOT/v['classificador'],v['extrator_sha256'],v['classificador_sha256'],v['receita_sha256'])
    protocol=dict(confs=[.15,.2,.3,.4,.5],modos=['detector','verificador_instancias'],
        baseline=dict(TP=317,FP=20,FN=17,MAE=.21875),
        escopo_baseline='Referencia exploratoria do curral_externo_topo com segunda escala; nao e gate de promocao drone_vertical. As receitas nao sao iguais.',
        gate='TP >=317, FP <=20 e MAE <=0.21875, pelo menos uma melhora estrita. Escolher menor FP+FN e MAE. Sem teste.',
        pesos=['best','last'],val_ja_usada_desenvolvimento=True,rotulos_revisados_apos_inspecao=True)
    (OUT/'protocolo_antes_avaliacao.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];hashes={}
    for name in protocol['pesos']:
        weight=OUT/'treino/weights'/(name+'.pt');hashes[name]=hashlib.sha256(weight.read_bytes()).hexdigest();model=YOLO(str(weight))
        folder=OUT/('previsoes_'+name);folder.mkdir(exist_ok=True)
        for n,f in enumerate(sorted((OUT/'dataset/images/val').glob('*.jpg'))):
            im=cv2.imread(str(f));h,w=im.shape[:2];ref=ct.yolo_para_pixels((ROOT/'data/experimentos/gabarito_aereo_revisado_v2/labels'/(f.stem+'.txt')).read_text().splitlines(),w,h)
            b,s=ct.detectar_inteira(model,im,imgsz=1024,conf_min=.05,iou_nms=.7,device=0,classes=[0],augment=False)
            prob=pontuar(vf,im,b);np.savez(folder/(f.stem+'.npz'),caixas=b,scores=s,visual=prob,ref=ref)
            for conf in protocol['confs']:
                for mode in protocol['modos']:
                    if mode=='detector':ix=np.flatnonzero(s>=conf)
                    else:ix,_=selecionar_instancias(b,s,prob,conf,v['limiar'])
                    rows.append(dict(peso=name,modo=mode,conf=conf,arquivo=f.name,erro=abs(len(ix)-len(ref)),**comparar_caixas(ref,b[ix],s[ix])))
            if (n+1)%24==0:print(name,n+1,flush=True)
    tab=pd.DataFrame(rows);tab.to_csv(OUT/'validacao_comparada.csv',index=False);summary=[]
    for (name,mode,conf),g in tab.groupby(['peso','modo','conf']):
        summary.append(dict(peso=name,modo=mode,conf=conf,sha256=hashes[name],TP=int(g.TP.sum()),FP=int(g.FP.sum()),FN=int(g.FN.sum()),MAE=float(g.erro.mean())))
    good=[r for r in summary if r['TP']>=317 and r['FP']<=20 and r['MAE']<=protocol['baseline']['MAE'] and (r['TP']>317 or r['FP']<20 or r['MAE']<protocol['baseline']['MAE'])]
    chosen=min(good,key=lambda r:(r['FP']+r['FN'],r['MAE'])) if good else None
    result=dict(candidatos=summary,selecionado=chosen,promovido=False,teste_consultado=False)
    (OUT/'escolha_validacao.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
