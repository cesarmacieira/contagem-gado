"""Ablacao isolada de dois nucleos revisados; nao altera o aplicativo."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import ast,hashlib,json
import cv2,joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageOps
from sklearn.linear_model import LogisticRegression
from torchvision.transforms.functional import to_tensor,normalize
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas
from verificador_visual import carregar_verificador

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/verificador_sombras_nucleo_v1'
PESO=ROOT/'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth'


def features(encoder,im,boxes,blur=False):
    if not len(boxes):return np.empty((0,2560),np.float32)
    model,transform,_=encoder;pil=Image.fromarray(im[:,:,::-1]);patches=[]
    for box in boxes:
        # Nunca expande para fora da regiao rotulada.
        crop=pil.crop(tuple(box))
        if blur:crop=crop.resize((24,24),Image.Resampling.BICUBIC)
        for view in (crop,ImageOps.grayscale(crop).convert('RGB')):
            patches.append(normalize(to_tensor(view.resize((224,224),Image.Resampling.BICUBIC)),transform.mean,transform.std))
    encoded=[]
    with torch.inference_mode():
        for j in range(0,len(patches),32):
            f=model.avgpool(model.features(torch.stack(patches[j:j+32]).cuda())).flatten(1)
            encoded.append(torch.nn.functional.normalize(f,dim=1).cpu().numpy())
    return np.concatenate(encoded).reshape(-1,2560)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Experimento preservado')
    torch.set_num_threads(4);encoder=carregar_verificador(PESO)
    source=ROOT/'data/candidatos_sombras_aereas_v1'
    cand=pd.read_csv(source/'candidatos.csv').set_index('id')
    review=pd.read_csv(ROOT/'data/revisao_sombras_aereas_ia2/revisao_sombras_ia2.csv').set_index('id')
    assert all(review.loc[k,'classe_ia2']=='sombra_com_fundo' for k in ('S01','S02'))
    (OUT/'protocolo_antes_ajuste.json').write_text(json.dumps(dict(
        recorte='Retangulo estrito RGB+cinza; nenhuma expansao ao contexto',C=1.0,
        comparacao='Mesmos dados base, sem versus com dois nucleos novos',
        calibracao='99% do menor score de correspondencia correta em tiles aereos de val',
        novos_negativos=['S01','S02'],usa_Campanario_no_treino=False,promover=False),indent=2),encoding='utf-8')
    xs=[];ys=[];origens=[]
    def add(im,b,y,origin):
        for blur in (False,True):
            f=features(encoder,im,b,blur);xs.append(f);ys.extend([y]*len(f));origens.extend([origin]*len(f))
    pos=ROOT/'data/experimentos/verificador_auxiliar_v3/positivos'
    for r in pd.read_csv(pos/'catalogo.csv').itertuples():
        if r.conjunto!='train':continue
        p=next((pos/'dataset/images/train').glob(r.nome+'.*'));im=cv2.imread(str(p));h,w=im.shape[:2]
        b=ct.yolo_para_pixels((pos/'dataset/labels/train'/(r.nome+'.txt')).read_text().splitlines(),w,h)
        if len(b):add(im,b,1,r.nome)
    neg=ROOT/'data/negativos_solo_v1'
    for r in pd.read_csv(neg/'manifesto.csv').itertuples():
        if r.conjunto!='train' or r.status!='aprovado_sem_bovinos':continue
        im=cv2.imread(str(neg/'images'/r.arquivo));h,w=im.shape[:2]
        add(im,np.array([[0,0,w,h],[0,0,w/2,h/2],[w/2,0,w,h/2],[0,h/2,w/2,h],[w/2,h/2,w,h]],float),0,r.arquivo)
    base=ROOT/'data/confundidores_especies_v1'
    for r in json.loads((base/'revisao_visual.json').read_text()):
        if r['conjunto']!='train' or not r['revisado']:continue
        im=cv2.imread(str(base/'images'/r['arquivo']));h,w=im.shape[:2]
        add(im,(np.array(r['bbox_rel'])*[w,h,w,h]).reshape(1,4),0,r['id'])
    nbase=len(ys)
    for k in ('S01','S02'):
        r=cand.loc[k];p=ROOT/r.origem
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r.sha256
        box=ast.literal_eval(r.xyxy);im=cv2.imread(str(p));x1,y1,x2,y2=box
        core=im[y1:y2,x1:x2];cv2.imwrite(str(OUT/(k+'_nucleo.png')),core)
        h,w=core.shape[:2];add(core,np.array([[0,0,w,h]],float),0,k)
    x=np.concatenate(xs);y=np.array(ys)
    np.savez_compressed(OUT/'features_treino.npz',x=x,y=y,origens=np.array(origens),n_base=nbase)
    modelos={}
    for name,end in [('sem_nucleos',nbase),('com_nucleos',len(y))]:
        clf=LogisticRegression(C=1,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=991).fit(x[:end],y[:end])
        clf.receita_extracao_='retangulo_rgb_cinza_v1'
        joblib.dump(clf,OUT/(name+'_retangulo.joblib'));modelos[name]=clf
    spec=spec_from_file_location('matching',ROOT/'53_verificar_falsos_positivos.py')
    matcher=module_from_spec(spec);spec.loader.exec_module(matcher)
    detector=YOLO(str(ROOT/'data/processed/modelo_gado.pt'));anchors={k:[] for k in modelos}
    val=ROOT/'data/experimentos/especialista_drone_v1/dataset'
    for p in sorted((val/'images/val').glob('*.jpg')):
        im=cv2.imread(str(p));h,w=im.shape[:2]
        ref=ct.yolo_para_pixels((val/'labels/val'/(p.stem+'.txt')).read_text().splitlines(),w,h)
        b,s=ct.detectar_inteira(detector,im,imgsz=1024,conf_min=.2,iou_nms=.7,device=0,classes=[0],augment=False)
        f=features(encoder,im,b);ids=matcher.indices_correspondentes(ref,b,s)
        if not len(f):continue
        for name,clf in modelos.items():anchors[name].extend(clf.predict_proba(f)[ids,1].tolist())
    assert all(anchors.values())
    thresholds={name:float(min(a)*.99) for name,a in anchors.items()}
    (OUT/'limiares_antes_regressao.json').write_text(json.dumps(dict(limiares=thresholds,TP_val={k:len(a) for k,a in anchors.items()}),indent=2),encoding='utf-8')
    # Reutiliza previsoes anteriores: nenhum detector e reexecutado na cena conhecida.
    case=ROOT/'data/experimentos/avaliacao_0359_curral_v1'
    data=np.load(case/'previsoes.npz');b,s=data['caixas'],data['scores']
    img=ROOT/'data/fontes_pesquisa/nellore_inspecao/revisao_0359_recorte16'
    im=cv2.imread(str(img/'recorte.png'));f=features(encoder,im,b)
    pts=json.loads((img/'gabarito_revisado_antes_detector.json').read_text())['pontos_xy']
    spec=spec_from_file_location('pontos',ROOT/'67_verificar_especies_no_denso.py')
    pm=module_from_spec(spec);spec.loader.exec_module(pm)
    rows=[]
    for name,clf in modelos.items():
        scores=clf.predict_proba(f)[:,1];keep=scores>=thresholds[name]
        hits=pm.correspondencias_pontos(pts,b[keep]);removed=np.flatnonzero(~keep).tolist()
        rows.append(dict(modelo=name,previsto=int(keep.sum()),pontos_cobertos=hits,removidas=removed,limiar=thresholds[name]))
        np.savez(OUT/(name+'_0359.npz'),caixas=b,scores=s,probabilidade=scores,manter=keep)
    result=dict(amostras_base=nbase,nucleos_novos=2,novas_amostras_com_variacoes=len(y)-nbase,
        cena_conhecida=rows,promovido=False,limite='Ajuste de pesquisa com apenas duas sombras de pasto; '
        '0359 ja e regressao conhecida e nao treina. Nao resolve omissao do bovino escuro nem valida confinamento brasileiro.')
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
