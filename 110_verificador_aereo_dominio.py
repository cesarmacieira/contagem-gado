"""Aprende verificacao no dominio aereo, calibra em fotos completas e confirma sem teste."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib,json
import cv2,joblib,numpy as np,pandas as pd,torch
from sklearn.linear_model import LogisticRegression
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas
from criterios_qualidade import supera_referencia
from verificador_visual import carregar_verificador
from verificador_nucleo import features,assinatura_receita

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/verificador_aereo_dominio_v1'
DATA=ROOT/'data/experimentos/aereo_ampliado_v2'
PESO=ROOT/'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth'


def modulo(name,path):
    sp=spec_from_file_location(name,ROOT/path);m=module_from_spec(sp);sp.loader.exec_module(m);return m


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    recipe=dict(C=[.1,1.],confs=[.25,.35,.45,.55],limiares=[.1,.3,.5,.7],max_tiles_positivos=400,max_tiles_negativos=200,
        negativos='Tres nucleos por tile vazio ja presente no treino do detector. Rotulos originais, sem nova revisao humana.',
        positivos='Ate dois GT por tile; excluir somente dois GT contidos sinalizados para revisao, sem alterar o dataset.',
        comparacao='Mesma receita moderna de tiles1024/NMS0.5, seguida de verificador; comparar tambem app ativo legado. Confirmar peso/limiar congelados em outros quadros de val.',
        teste=False,campanario=False,seed=1110,promocao_automatica=False)
    (OUT/'protocolo_antes_ajuste.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    torch.set_num_threads(4);encoder=carregar_verificador(PESO);rng=np.random.default_rng(1110)
    manifest=pd.read_csv(DATA/'treino_manifesto.csv');positive=manifest[manifest.n>0].sample(frac=1,random_state=1110)
    negative=manifest[manifest.n==0].sample(frac=1,random_state=1111)
    # One tile per original first, then a second tile if needed.
    positive=positive.assign(ordem=positive.groupby('original').cumcount()).sort_values('ordem',kind='stable').head(400)
    negative=negative.drop_duplicates('original').head(200)
    suspects=pd.read_csv(ROOT/'data/experimentos/estrategia_qualidade_v1/rotulos_treino_suspeitos.csv')
    xs=[];ys=[];origins=[]
    for label,subset in [(1,positive),(0,negative)]:
        for n,r in enumerate(subset.itertuples()):
            im=cv2.imread(str(DATA/'dataset/images/train'/(r.id+'.jpg')));h,w=im.shape[:2]
            if label:
                b=ct.yolo_para_pixels((DATA/'dataset/labels/train'/(r.id+'.txt')).read_text().splitlines(),w,h)
                excluded=set(suspects.loc[suspects.id==r.id,'gt'])
                ids=[i for i in range(len(b)) if i not in excluded]
                if not ids:continue
                ids=rng.choice(ids,min(2,len(ids)),replace=False);boxes=b[ids]
            else:
                boxes=[]
                for _ in range(3):
                    bw,bh=rng.integers(32,min(257,w)+1),rng.integers(32,min(257,h)+1)
                    x,y=rng.integers(0,w-bw+1),rng.integers(0,h-bh+1);boxes.append([x,y,x+bw,y+bh])
                boxes=np.array(boxes)
            for blur in [False,True]:
                f=features(encoder,im,boxes,blur=blur);xs.append(f);ys.extend([label]*len(f));origins.extend([r.id]*len(f))
            if (n+1)%100==0:print('Features',label,n+1,flush=True)
    x,y=np.concatenate(xs),np.asarray(ys);np.savez_compressed(OUT/'features_treino.npz',x=x,y=y,origens=np.asarray(origins))
    clfs={}
    for c in recipe['C']:
        clf=LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=1110).fit(x,y)
        clf.receita_extracao_='retangulo_rgb_cinza_v1';joblib.dump(clf,OUT/(f'clf_C{c}.joblib'));clfs[c]=clf
    avaliar_modelos(encoder,clfs,recipe,y)


def avaliar_modelos(encoder,clfs,recipe,y):
    helper=modulo('juncao_eval','109_avaliar_juncao_aerea.py');rows=[]
    val=pd.read_csv(ROOT/'data/experimentos/gabarito_aereo_revisado_v2/validacao_completa.csv')
    for r in val.itertuples():
        d=np.load(ROOT/'data/experimentos/juncao_aerea_v1'/(r.arquivo+'.npz'));b,s,o=d['caixas'],d['scores'],d['origens']
        ids=ct.nms(b,s,.5,origens=o);b,s=b[ids],s[ids];im=cv2.imread(r.img);f=features(encoder,im,b)
        np.savez_compressed(OUT/(r.arquivo+'.npz'),caixas=b,scores=s,features=f,ref=d['ref'])
        for c,clf in clfs.items():
            probabilities=clf.predict_proba(f)[:,1] if len(f) else np.empty(0)
            for conf in recipe['confs']:
                for threshold in recipe['limiares']:
                    keep=(s>=conf)&(probabilities>=threshold)
                    rows.append(dict(C=c,conf=conf,limiar=threshold,arquivo=r.arquivo,grupo=r.grupo,real=len(d['ref']),previsto=int(keep.sum()),**comparar_caixas(d['ref'],b[keep],s[keep])))
    tab=pd.DataFrame(rows);tab.to_csv(OUT/'calibracao.csv',index=False);summ=[]
    baseline=json.loads((ROOT/'data/experimentos/juncao_aerea_v1/resultado.json').read_text())
    for (c,conf,th),g in tab.groupby(['C','conf','limiar']):summ.append(dict(C=c,conf=conf,limiar=th,metricas=helper.resumir(g)))
    good=[r for r in summ if all(supera_referencia(r['metricas'],baseline[k]['metricas']) for k in ['app_atual','nms_calibrada'])]
    chosen=min(good,key=lambda r:(r['metricas']['MAE']+r['metricas']['MAE_macro_grupos'],-r['metricas']['F1_caixas'])) if good else None
    (OUT/'escolha_antes_confirmacao.json').write_text(json.dumps(dict(escolhida=chosen,todas=summ,positivos=int(y.sum()),negativos=int((y==0).sum())),indent=2),encoding='utf-8')
    result=dict(escolhida=chosen,confirmacao=None,promovido=False,teste_consultado=False)
    if chosen:
        model=YOLO(str(ROOT/'data/processed/modelo_gado.pt'));clf=clfs[chosen['C']]
        ds=modulo('dados_confirmacao','48_especializar_drone.py').catalogo();remaining=ds[(ds.conjunto=='val')&~ds.arquivo.isin(val.arquivo)]
        remaining.to_csv(OUT/'quadros_confirmacao_antes_inferencia.csv',index=False);rr=[]
        for r in remaining.itertuples():
            im=cv2.imread(r.img);h,w=im.shape[:2];ref=ct.yolo_para_pixels(Path(r.rot).read_text().splitlines(),w,h)
            b,s=ct.detectar(model,im,conf_min=.05,iou_local=.7,iou_nms=.5,device=0,classes=[0]);f=features(encoder,im,b)
            probs=clf.predict_proba(f)[:,1] if len(f) else np.empty(0);keep=(s>=chosen['conf'])&(probs>=chosen['limiar'])
            old,old_s=ct.detectar(model,im,conf_min=.5,iou_local=.7,iou_nms=.5,device=0,classes=[0],legado=True,max_det=300)
            for name,bb,ss in [('app_atual',old,old_s),('candidata',b[keep],s[keep])]:rr.append(dict(modo=name,arquivo=r.arquivo,grupo=r.grupo,real=len(ref),previsto=len(bb),**comparar_caixas(ref,bb,ss)))
        pd.DataFrame(rr).to_csv(OUT/'confirmacao.csv',index=False)
        result['confirmacao']={name:helper.resumir(g) for name,g in pd.DataFrame(rr).groupby('modo')}
        result['passou_confirmacao']=supera_referencia(result['confirmacao']['candidata'],result['confirmacao']['app_atual'])
        if result['passou_confirmacao']:
            result['proposta']=dict(visao='drone_vertical',confianca=chosen['conf'],legado=False,classificador=str(OUT/(f"clf_C{chosen['C']}.joblib")),limiar=chosen['limiar'],receita_sha256=assinatura_receita())
    result['limites']='Treino rotulado existente; backgrounds sem nova revisao humana. Calibracao conhecida com tres GT revisados apos previsoes; confirmacao em outros quadros historicos das mesmas fazendas/voos. Nao e novo teste brasileiro.'
    if recipe.get('limites'):result['limites']=recipe['limites']
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
