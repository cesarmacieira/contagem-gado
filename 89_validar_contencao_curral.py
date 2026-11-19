"""Verifica uma unica regra existente de contencao, somente em curral_topo."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import json
import hashlib
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/contencao_curral_v1'
NOVA=ROOT/'data/fontes_pesquisa/nellore_inspecao/avaliacao_recorte_25'


def main():
    estado=NOVA/'ESTADO_GABARITO.json'
    if estado.exists() and not json.loads(estado.read_text())['valido_para_contagem_exata']:
        raise ValueError('Gabarito da cena brasileira invalidado; nao promover por esta contagem')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    protocolo=dict(imgsz=1024,conf=.2,iou=.7,ioa=.7,visao='curral_topo',
        criterio='Nao perder TP nem piorar erro em nenhuma imagem de val/regressao; melhorar a cena conhecida de 25.',
        peso_sha256=hashlib.sha256((ROOT/'data/processed/modelo_gado.pt').read_bytes()).hexdigest(),promover=False)
    (OUT/'protocolo_antes_regressao.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    torch.set_num_threads(4);model=YOLO(str(ROOT/'data/processed/modelo_gado.pt'));rows=[]
    for split in ('val','test'):
        for p in sorted((ROOT/f'data/yolo/images/{split}').glob('colo_topo__*.jpg')):
            im=cv2.imread(str(p));h,w=im.shape[:2]
            rot=ROOT/f'data/yolo/labels/{split}'/(p.stem+'.txt')
            ref=ct.yolo_para_pixels(rot.read_text().splitlines(),w,h)
            b,s=ct.detectar_inteira(model,im,imgsz=1024,conf_min=.2,iou_nms=.7,device=0,classes=[0],augment=False)
            k=ct.nms(b,s,iou_max=.7,ioa_max=.7)
            np.savez(OUT/(split+'__'+p.stem+'.npz'),caixas=b,scores=s,manter=k,ref=ref)
            for nome,bb,ss in [('antes',b,s),('depois',b[k],s[k])]:
                rows.append(dict(split=split,arquivo=p.name,versao=nome,real=len(ref),previsto=len(bb),
                    erro=abs(len(bb)-len(ref)),**comparar_caixas(ref,bb,ss)))
    t=pd.DataFrame(rows);t.to_csv(OUT/'por_imagem.csv',index=False)
    a=t[t.versao=='antes'].set_index(['split','arquivo']);z=t[t.versao=='depois'].set_index(['split','arquivo'])
    d=np.load(NOVA/'curral_topo.npz');b,s=d['caixas'],d['scores'];k=ct.nms(b,s,iou_max=.7,ioa_max=.7)
    spec=spec_from_file_location('pts',ROOT/'67_verificar_especies_no_denso.py')
    pts=module_from_spec(spec);spec.loader.exec_module(pts)
    pontos=json.loads((NOVA/'gabarito_antes_inferencia.json').read_text())['pontos_xy']
    hits=pts.correspondencias_pontos(pontos,b[k])
    resultado=dict(n_imagens=len(a),perdeu_TP=bool((z.TP<a.TP).any()),piorou_contagem=bool((z.erro>a.erro).any()),
        removidas_historico=int(a.previsto.sum()-z.previsto.sum()),cena25_antes=len(b),cena25_depois=len(k),
        cena25_pontos=hits,passou=bool(not (z.TP<a.TP).any() and not (z.erro>a.erro).any() and hits==25 and len(k)<len(b)),
        promovido=False,limite='Validacao e regressao conhecida, incluindo uma cena brasileira de uma fazenda.')
    (OUT/'resultado.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    print(json.dumps(resultado,indent=2))


if __name__=='__main__':main()
