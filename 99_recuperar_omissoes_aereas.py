"""Investiga segunda escala so na validacao; preserva base e nao consulta teste."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np,pandas as pd,torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas
from especialistas import resolver_especialista
from correcao_aerea import selecionar_instancias
from recuperacao_multiescala import sobreposicoes
from verificador_nucleo import carregar,pontuar

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/recuperacao_aerea_escala_v1'


from correcao_aerea import complementar


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    profile=resolver_especialista(ROOT,'curral_externo_topo');cfg=profile['config'];v=profile['verificador_nucleo']
    protocolo=dict(perfil=profile,imgsz_adicional=1536,politicas=[dict(score=s,visual=p) for s in [.4,.6] for p in [.5,.7]],
        gate='FN menor que atual e FP no maximo igual; menor FP+FN, depois MAE. Sem consulta a teste.',
        preserva_caixas_atuais=True,treino=False)
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    torch.set_num_threads(4);model=YOLO(profile['pesos_absolutos'])
    verifier=carregar(ROOT/v['extrator'],ROOT/v['classificador'],v['extrator_sha256'],v['classificador_sha256'],v['receita_sha256'])
    fonte=ROOT/'data/experimentos/pipeline_sombras_validacao_v1';imgs=ROOT/'data/experimentos/especialista_drone_v1/dataset/images/val';rows=[]
    for n,f in enumerate(sorted(imgs.glob('*.jpg'))):
        d=np.load(fonte/(f.stem+'.npz'));b,s,prob,ref=d['caixas'],d['scores'],d['verificador'],d['ref']
        ix,_=selecionar_instancias(b,s,prob,cfg['conf'],v['limiar']);base,bs=b[ix],s[ix]
        im=cv2.imread(str(f));cache=OUT/(f.stem+'.npz')
        if cache.exists():
            a=np.load(cache);bb,ss,vv=a['caixas'],a['scores'],a['visual']
        else:
            bb,ss=ct.detectar_inteira(model,im,imgsz=1536,conf_min=.1,iou_nms=.7,device=0,classes=[0],augment=False)
            vv=pontuar(verifier,im,bb);np.savez(cache,caixas=bb,scores=ss,visual=vv)
        options=[('atual',base,bs)]
        for policy in protocolo['politicas']:
            ids=complementar(base,bb,ss,vv,policy['score'],policy['visual'])
            options.append((f"s{policy['score']}_v{policy['visual']}",np.concatenate([base,bb[ids]]),np.concatenate([bs,ss[ids]])))
        for name,boxes,scores in options:
            rows.append(dict(arquivo=f.name,politica=name,erro=abs(len(boxes)-len(ref)),**comparar_caixas(ref,boxes,scores)))
        if (n+1)%24==0:print('Validacao',n+1,flush=True)
    table=pd.DataFrame(rows);table.to_csv(OUT/'validacao.csv',index=False);summ=[]
    for name,g in table.groupby('politica'):
        summ.append(dict(politica=name,TP=int(g.TP.sum()),FP=int(g.FP.sum()),FN=int(g.FN.sum()),MAE=float(g.erro.mean())))
    atual=next(r for r in summ if r['politica']=='atual');good=[r for r in summ if r['FN']<atual['FN'] and r['FP']<=atual['FP']]
    chosen=min(good,key=lambda r:(r['FP']+r['FN'],r['MAE'])) if good else None
    result=dict(validacao=summ,selecionada=chosen,promovida=False,teste_consultado=False,
        limite='Validacao aerea conhecida usada para desenvolvimento; sem conclusao de novas fazendas brasileiras.')
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
