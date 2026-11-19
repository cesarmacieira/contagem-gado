"""Mede associacao de emendas em fotos completas, sem treino nem teste."""
from pathlib import Path
import json
import cv2,numpy as np,pandas as pd,torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas,metricas_contagem,taxas_deteccao
from juncao_aerea import selecionar
from criterios_qualidade import supera_referencia

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/juncao_aerea_v1'


def inferir(model,im):
    h,w=im.shape[:2];all_b=[];all_s=[];all_o=[];all_t=[]
    corners=ct.grade_tiles(w,h,1024,.2)
    for start in range(0,len(corners),8):
        chunk=corners[start:start+8];crops=[im[y:y+1024,x:x+1024] for x,y in chunk]
        results=model.predict(crops,imgsz=1024,conf=.05,iou=.7,max_det=1000,device=0,classes=[0],verbose=False)
        for j,((x,y),crop,r) in enumerate(zip(chunk,crops,results)):
            b=r.boxes.xyxy.cpu().numpy().copy();s=r.boxes.conf.cpu().numpy();ch,cw=crop.shape[:2]
            t=((b[:,0]<=2)&(x>0))|((b[:,1]<=2)&(y>0))|((b[:,2]>=cw-2)&(x+cw<w))|((b[:,3]>=ch-2)&(y+ch<h))
            b+=np.array([x,y,x,y]);all_b.extend(b);all_s.extend(s);all_o.extend([start+j]*len(b));all_t.extend(t)
    return np.asarray(all_b).reshape(-1,4),np.asarray(all_s),np.asarray(all_o),np.asarray(all_t)


def resumir(g):
    tp,fp,fn=[int(g[k].sum()) for k in ['TP','FP','FN']]
    result=dict(**metricas_contagem(g.real,g.previsto),**taxas_deteccao(tp,fp,fn),TP=tp,FP=fp,FN=fn)
    result['MAE_macro_grupos']=float(g.assign(erro=(g.real-g.previsto).abs()).groupby('grupo').erro.mean().mean())
    for key,lim in [('oito_ou_mais',8),('vinte_ou_mais',20)]:
        subset=g[g.real>=lim]
        if len(subset):
            dtp,dfp,dfn=[int(subset[k].sum()) for k in ['TP','FP','FN']]
            result[key]=dict(**metricas_contagem(subset.real,subset.previsto),**taxas_deteccao(dtp,dfp,dfn))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    confs=[.15,.25,.35,.45,.55,.65]
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(dict(confs=confs,contencao=.85,area_relativa=1.25,margem_px=2,
        pesos='original',treino=False,teste=False,promocao=False,protege_codeteccao_mesmo_tile=True),indent=2),encoding='utf-8')
    torch.set_num_threads(4);model=YOLO(str(ROOT/'data/processed/modelo_gado.pt'))
    val=pd.read_csv(ROOT/'data/experimentos/gabarito_aereo_revisado_v2/validacao_completa.csv');rows=[]
    for r in val.itertuples():
        im=cv2.imread(r.img);h,w=im.shape[:2];ref=ct.yolo_para_pixels(Path(r.rot).read_text().splitlines(),w,h)
        b,s,o,t=inferir(model,im);np.savez(OUT/(r.arquivo+'.npz'),caixas=b,scores=s,origens=o,truncadas=t,ref=ref)
        # Actual active drone profile: confidence .5, legacy inference.
        active,ascore=ct.detectar(model,im,conf_min=.5,iou_local=.7,iou_nms=.5,legado=True,max_det=300,device=0,classes=[0])
        rows.append(dict(arquivo=r.arquivo,grupo=r.grupo,modo='app_atual',conf=.5,real=len(ref),previsto=len(active),fragmentos_removidos=0,**comparar_caixas(ref,active,ascore)))
        for c in confs:
            for mode in ['nms_atual','bordas_confirmadas']:
                if mode=='nms_atual':ids=np.flatnonzero(s>=c);ids=ids[ct.nms(b[ids],s[ids],.5,origens=o[ids])];removed=[]
                else:ids,removed=selecionar(b,s,o,t,c)
                rows.append(dict(arquivo=r.arquivo,grupo=r.grupo,modo=mode,conf=c,real=len(ref),previsto=len(ids),fragmentos_removidos=len(removed),**comparar_caixas(ref,b[ids],s[ids])))
        print('Completa',r.arquivo,flush=True)
    table=pd.DataFrame(rows);table.to_csv(OUT/'por_imagem.csv',index=False);summary=[]
    for (mode,c),g in table.groupby(['modo','conf']):summary.append(dict(modo=mode,conf=float(c),metricas=resumir(g),removidas=int(g.fragmentos_removidos.sum())))
    baseline=next(x for x in summary if x['modo']=='app_atual')
    normal=min([x for x in summary if x['modo']=='nms_atual'],key=lambda x:(x['metricas']['MAE']+x['metricas']['MAE_macro_grupos'],-x['metricas']['F1_caixas']))
    candidate=min([x for x in summary if x['modo']=='bordas_confirmadas'],key=lambda x:(x['metricas']['MAE']+x['metricas']['MAE_macro_grupos'],-x['metricas']['F1_caixas']))
    passed=supera_referencia(candidate['metricas'],baseline['metricas']) and supera_referencia(candidate['metricas'],normal['metricas'])
    result=dict(app_atual=baseline,nms_calibrada=normal,candidata=candidate,passou=passed,promovida=False,teste_consultado=False)
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
