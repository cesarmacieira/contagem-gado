"""Seleciona uma de tres confiancas em val; a cena conhecida nao seleciona nada."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib,json
import cv2,joblib
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas
from verificador_visual import carregar_verificador

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/pipeline_sombras_validacao_v1'
TRAIN=ROOT/'data/experimentos/verificador_sombras_nucleo_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    cfg=json.loads((TRAIN/'limiares_antes_regressao.json').read_text());thr=cfg['limiares']['com_nucleos']
    cpf=TRAIN/'com_nucleos_retangulo.joblib'
    (OUT/'protocolo_antes_validacao.json').write_text(json.dumps(dict(
        confiancas=[.05,.1,.2],imgsz=1024,iou=.7,limiar_verificador=thr,
        classificador_sha256=hashlib.sha256(cpf.read_bytes()).hexdigest(),
        criterio='Menor FP+FN em val, depois menor MAE, depois maior confianca. Sem ajuste em 0359.',
        promover=False),indent=2),encoding='utf-8')
    spec=spec_from_file_location('strict',ROOT/'93_ajustar_verificador_sombras.py')
    strict=module_from_spec(spec);spec.loader.exec_module(strict)
    torch.set_num_threads(4);enc=carregar_verificador(strict.PESO);clf=joblib.load(cpf)
    model=YOLO(str(ROOT/'data/processed/modelo_gado.pt'));val=ROOT/'data/experimentos/especialista_drone_v1/dataset'
    rows=[]
    for p in sorted((val/'images/val').glob('*.jpg')):
        im=cv2.imread(str(p));h,w=im.shape[:2]
        ref=ct.yolo_para_pixels((val/'labels/val'/(p.stem+'.txt')).read_text().splitlines(),w,h)
        b,s=ct.detectar_inteira(model,im,imgsz=1024,conf_min=.05,iou_nms=.7,device=0,classes=[0],augment=False)
        f=strict.features(enc,im,b);v=clf.predict_proba(f)[:,1] if len(f) else np.empty(0)
        np.savez_compressed(OUT/(p.stem+'.npz'),caixas=b,scores=s,verificador=v,ref=ref)
        for conf in (.05,.1,.2):
            keep=(s>=conf)&(v>=thr)
            rows.append(dict(arquivo=p.name,conf=conf,real=len(ref),previsto=int(keep.sum()),
                **comparar_caixas(ref,b[keep],s[keep])))
    t=pd.DataFrame(rows);t.to_csv(OUT/'por_imagem_validacao.csv',index=False)
    summary=[]
    for conf,g in t.groupby('conf'):
        summary.append(dict(conf=float(conf),TP=int(g.TP.sum()),FP=int(g.FP.sum()),FN=int(g.FN.sum()),
            erros_instancias=int((g.FP+g.FN).sum()),MAE=float((g.previsto-g.real).abs().mean())))
    selected=sorted(summary,key=lambda r:(r['erros_instancias'],r['MAE'],-r['conf']))[0]
    (OUT/'escolha_antes_regressao.json').write_text(json.dumps(dict(selecionada=selected,todas=summary),indent=2),encoding='utf-8')
    base=ROOT/'data/fontes_pesquisa/nellore_inspecao/revisao_0359_recorte16'
    im=cv2.imread(str(base/'recorte.png'));b,s=ct.detectar_inteira(model,im,imgsz=1024,conf_min=.05,iou_nms=.7,device=0,classes=[0],augment=False)
    v=clf.predict_proba(strict.features(enc,im,b))[:,1];keep=(s>=selected['conf'])&(v>=thr)
    spec=spec_from_file_location('pontos',ROOT/'67_verificar_especies_no_denso.py')
    pm=module_from_spec(spec);spec.loader.exec_module(pm)
    pts=json.loads((base/'gabarito_revisado_antes_detector.json').read_text())['pontos_xy']
    result=dict(validacao=summary,selecionada=selected,cena_conhecida=dict(real=16,previsto=int(keep.sum()),
        pontos_cobertos=pm.correspondencias_pontos(pts,b[keep])),promovido=False,
        limite='Calibracao em tiles aereos, nao galpoes. 0359 ja conhecida, mesma fazenda fora de treino. '
        'Uma melhora nela nao valida a aplicacao final.')
    np.savez(OUT/'0359.npz',caixas=b,scores=s,verificador=v,manter=keep)
    cv2.imwrite(str(OUT/'0359_resultado.jpg'),ct.desenhar(im,b[keep]))
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
