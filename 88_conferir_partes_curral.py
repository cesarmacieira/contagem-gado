"""Regra de parte versus corpo calibrada em val; nao promove o aplicativo."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
import verificador_semantico as sem
from avaliacao import comparar_caixas

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/partes_curral_v1'
NOVA=ROOT/'data/fontes_pesquisa/nellore_inspecao/avaliacao_recorte_25'


def geometria(b):
    area=np.prod(b[:,2:]-b[:,:2],axis=1);mask=np.zeros(len(b),bool)
    for i in range(len(b)):
        inter=np.maximum(0,np.minimum(b[i,2:],b[:,2:])-np.maximum(b[i,:2],b[:,:2])).prod(axis=1)
        apoio=(inter/np.maximum(area,1e-9)>=.1)&(area>=area[i])
        apoio[i]=False;mask[i]=apoio.any()
    return mask


def main():
    estado=NOVA/'ESTADO_GABARITO.json'
    if estado.exists() and not json.loads(estado.read_text())['valido_para_contagem_exata']:
        raise ValueError('Gabarito da cena brasileira invalidado; nao calibrar correcao de extras')
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    sem.POSITIVOS=['an aerial photo of a whole cow with its body.',
        'a photo of the torso and back of a cow.','an aerial photo of cattle bodies.']
    sem.NEGATIVOS=['a close-up photo of a cow head without its body.',
        'a photo of only the head and neck of a cow.','a close-up photo of cattle faces.']
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(dict(
        positivos=sem.POSITIVOS,negativos=sem.NEGATIVOS,receita=sem.assinatura_receita(),
        regra='Rejeitar somente parte nas duas vistas e sobreposta a uma caixa de area maior ou igual.',
        limiar='Maxima margem minima das correspondencias corretas de val, mais .001; minimo zero.',
        promover=False),indent=2),encoding='utf-8')
    checker=sem.carregar(ROOT/'data/modelos_candidatos/ViT-B-32.pt')
    model=YOLO(str(ROOT/'data/processed/modelo_gado.pt'))
    spec=spec_from_file_location('matching',ROOT/'53_verificar_falsos_positivos.py')
    m=module_from_spec(spec);spec.loader.exec_module(m)
    def obter(p,split):
        im=cv2.imread(str(p));h,w=im.shape[:2]
        b,s=ct.detectar_inteira(model,im,imgsz=1024,conf_min=.2,iou_nms=.7,device=0,classes=[0],augment=False)
        ref=ct.yolo_para_pixels((ROOT/f'data/yolo/labels/{split}'/(p.stem+'.txt')).read_text().splitlines(),w,h)
        mm,_=sem.margens(checker,im,b)
        return im,b,s,ref,mm
    vals=[]
    for p in sorted((ROOT/'data/yolo/images/val').glob('colo_topo__*.jpg')):
        _,b,s,ref,mm=obter(p,'val');ids=m.indices_correspondentes(ref,b,s)
        vals.extend(mm[ids].min(axis=1).tolist())
    if not vals:raise ValueError('Sem correspondencias de val')
    limiar=float(max(0,max(vals)+.001))
    (OUT/'limiar_antes_regressao.json').write_text(json.dumps(dict(limiar=limiar,TP_val=len(vals))),encoding='utf-8')
    rows=[]
    for p in sorted((ROOT/'data/yolo/images/test').glob('colo_topo__*.jpg')):
        _,b,s,ref,mm=obter(p,'test');keep=~((mm>=limiar).all(axis=1)&geometria(b))
        for nome,bb,ss in [('antes',b,s),('depois',b[keep],s[keep])]:
            rows.append(dict(arquivo=p.name,versao=nome,real=len(ref),previsto=len(bb),
                erro=abs(len(bb)-len(ref)),**comparar_caixas(ref,bb,ss)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'historico.csv',index=False)
    im=cv2.imread(str(NOVA/'recorte.png'));d=np.load(NOVA/'curral_topo.npz');b,s=d['caixas'],d['scores']
    mm,_=sem.margens(checker,im,b);keep=~((mm>=limiar).all(axis=1)&geometria(b))
    spec=spec_from_file_location('pontos',ROOT/'67_verificar_especies_no_denso.py')
    pts=module_from_spec(spec);spec.loader.exec_module(pts)
    pontos=json.loads((NOVA/'gabarito_antes_inferencia.json').read_text())['pontos_xy']
    a=df[df.versao=='antes'].set_index('arquivo');z=df[df.versao=='depois'].set_index('arquivo')
    resultado=dict(limiar=limiar,TP_val_preservados=len(vals),antes=len(b),depois=int(keep.sum()),
        pontos_cobertos=pts.correspondencias_pontos(pontos,b[keep]),
        historico_piorou_erro=bool((z.erro>a.erro).any()),historico_perdeu_TP=bool((z.TP<a.TP).any()),
        historico_removidas=int(a.previsto.sum()-z.previsto.sum()),promovido=False)
    np.savez(OUT/'cena25.npz',caixas=b,scores=s,margens=mm,manter=keep)
    (OUT/'resultado.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    print(json.dumps(resultado,indent=2))


if __name__=='__main__':main()
