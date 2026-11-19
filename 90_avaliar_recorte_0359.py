"""Uma avaliacao do perfil atual, com duas leituras concordantes anteriores."""
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
BASE=ROOT/'data/fontes_pesquisa/nellore_inspecao/revisao_0359_recorte16'
OUT=ROOT/'data/experimentos/avaliacao_0359_curral_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Primeira avaliacao preservada')
    g=json.loads((BASE/'gabarito_revisado_antes_detector.json').read_text())
    assert g['aprovado_para_avaliacao'] and g['n_bovinos']==len(g['pontos_xy'])==16
    assert hashlib.sha256((BASE/'recorte.png').read_bytes()).hexdigest()==g['recorte_sha256']
    perfil=resolver_especialista(ROOT,'curral_topo');cfg=perfil['config']
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(dict(perfil=perfil,
        gabarito_sha256=hashlib.sha256((BASE/'gabarito_revisado_antes_detector.json').read_bytes()).hexdigest(),
        sem_ajuste_de_parametros=True,treino=False,promover=False),indent=2,default=str),encoding='utf-8')
    torch.set_num_threads(4);model=YOLO(perfil['pesos_absolutos'])
    im=cv2.imread(str(BASE/'recorte.png'))
    b,s=ct.detectar_inteira(model,im,imgsz=cfg['imgsz'],conf_min=.05,iou_nms=cfg['iou'],
        device=0,classes=perfil['classes'],augment=cfg['augment'])
    b,s=ct.filtrar(b,s,cfg['conf'])
    spec=spec_from_file_location('pontos',ROOT/'67_verificar_especies_no_denso.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    hits=mod.correspondencias_pontos(g['pontos_xy'],b)
    r=dict(visao='curral_topo',real=16,previsto=len(b),erro_absoluto=abs(len(b)-16),
        pontos_cobertos_um_a_um=hits,pontos_sem_caixa=16-hits,caixas_sem_ponto=len(b)-hits,
        modelo_alterado=False,parametros_alterados=False,
        limite='Uma imagem de uma fazenda, duas leituras de assistentes, sem revisao humana independente. '
        '16 bovinos, abaixo da meta de 20–50. Contagem e correspondencia por pontos, nao F1 de caixas. '
        'Fonte isolada de pesquisa; licenca de incorporacao nao confirmada. Consultas futuras sao regressao conhecida.')
    np.savez(OUT/'previsoes.npz',caixas=b,scores=s)
    cv2.imwrite(str(OUT/'resultado_visual.jpg'),ct.desenhar(im,b))
    (OUT/'resultado.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
