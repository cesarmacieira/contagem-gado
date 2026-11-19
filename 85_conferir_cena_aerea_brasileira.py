"""Primeira avaliacao de uma cena nova, com recorte/gabarito antes do detector.

Uso isolado de pesquisa. A fonte anota dorso/cabeca: suas caixas nao sao
comparaveis diretamente com as caixas de corpo inteiro do aplicativo.
"""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import contagem as ct
from especialistas import resolver_especialista

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/fontes_pesquisa/nellore_inspecao'
OUT=BASE/'avaliacao_recorte_25'


def main():
    estado=OUT/'ESTADO_GABARITO.json'
    if estado.exists() and not json.loads(estado.read_text())['valido_para_contagem_exata']:
        raise ValueError('Gabarito de contagem invalidado; revisar instancias antes de reavaliar')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Primeira avaliacao preservada')
    source=BASE/'2024-10-05_DJI_0209.jpg'
    expected='5e3424a8c9b730bbf6e8f71cce1d9787d294e25df26ea52c0fc771b2de95c772'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    original=cv2.imread(str(source));h,w=original.shape[:2]
    rect=[1300,1200,4100,h];x1,y1,x2,y2=rect
    im=original[y1:y2,x1:x2].copy()
    pontos=[];ids=[]
    linhas=[list(map(float,s.split())) for s in source.with_suffix('.txt').read_text().splitlines() if s.strip()]
    classe0=[r for r in linhas if int(r[0])==0]
    for j,r in enumerate(classe0):
        x,y=r[1]*w,r[2]*h
        if x1<=x<x2 and y1<=y<y2:
            pontos.append([x-x1,y-y1]);ids.append(j+1)
    assert ids==list(range(4,29)),ids
    assert len(pontos)==25
    cv2.imwrite(str(OUT/'recorte.png'),im)
    frozen=dict(fonte_sha256=expected,recorte_xyxy=rect,n_bovinos=25,pontos_xy=pontos,
        ids_dorsos_fonte=ids,grupo='fazenda_campanario_todas_datas',conjunto='pesquisa_avaliacao',
        revisao='Uma revisao visual do assistente antes de qualquer detector; grupo central de 25 bovinos.',
        politica='Contar cada bovino individualizavel do grupo central. Um ponto no dorso por animal. '
        'Nao somar as classes dorso e cabeca. Nao usar caixas de dorso como corpo inteiro.',
        limite='Uma imagem/uma fazenda, sem segundo anotador. Fonte isolada de pesquisa, licenca nao confirmada.',
        inferencia_executada=False,
        perfis='Somente curral_topo e drone_vertical atuais, fixados no catalogo antes desta imagem')
    (OUT/'gabarito_antes_inferencia.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    marked=im.copy()
    for j,(x,y) in enumerate(pontos):
        cv2.circle(marked,(round(x),round(y)),8,(0,0,255),-1)
        cv2.putText(marked,str(j+1),(round(x)+8,round(y)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
    cv2.imwrite(str(OUT/'gabarito_pontos.jpg'),marked)
    perfis={nome:resolver_especialista(ROOT,nome) for nome in ('curral_topo','drone_vertical')}
    # Identidades e parametros gravados antes de carregar o detector.
    (OUT/'perfis_antes_inferencia.json').write_text(json.dumps(perfis,indent=2,default=str),encoding='utf-8')
    spec=spec_from_file_location('pontos',ROOT/'67_verificar_especies_no_denso.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    torch.set_num_threads(4)
    rows=[]
    for nome,p in perfis.items():
        cfg=p['config'];model=YOLO(p['pesos_absolutos'])
        if cfg['modo']=='inteira':
            b,s=ct.detectar_inteira(model,im,imgsz=cfg['imgsz'],conf_min=cfg['conf'],
                iou_nms=cfg['iou'],device=0,classes=p['classes'],augment=cfg['augment'])
        else:
            # O preset aereo usa o mesmo caminho legado de tiles do aplicativo.
            b,s=ct.detectar(model,im,tile=cfg['tile'],overlap=cfg['overlap'],
                conf_min=.05,iou_local=cfg['iou'],iou_nms=cfg['iou_juncao'],
                device=0,classes=p['classes'],legado=cfg['legado'],augment=cfg['augment'])
            b,s=ct.filtrar(b,s,cfg['conf'])
        hits=mod.correspondencias_pontos(pontos,b)
        rows.append(dict(visao=nome,real=25,previsto=len(b),erro_absoluto=abs(len(b)-25),
            pontos_cobertos_um_a_um=hits,pontos_sem_caixa=25-hits,caixas_sem_ponto=len(b)-hits))
        np.savez(OUT/(nome+'.npz'),caixas=b,scores=s)
        cv2.imwrite(str(OUT/(nome+'.jpg')),ct.desenhar(im,b))
    result=dict(resultados=rows,treino_executado_nessa_fazenda=False,
        limite=frozen['limite'],nota='Correspondencia por ponto, nao F1 de caixas. '
        'Primeira avaliacao desta cena; consultas futuras sao regressao conhecida.')
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
