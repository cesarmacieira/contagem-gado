"""Teste parcial: minimo visual congelado; nao inventa contagem exata."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/busca_teste_denso_independente'
OUT=BASE/'avaliacao_minimo_bernard'


def preparar():
    OUT.mkdir(parents=True,exist_ok=True)
    destino=OUT/'gabarito_antes_inferencia.json'
    if destino.exists():
        raise ValueError('Gabarito existente preservado')
    # Coordenadas marcadas visualmente sobre ampliacoes, sem detector.
    esquerda=[(230,328),(520,335),(614,317),(742,273),(1380,285),(1495,231),(1673,249)]
    direita=[(132,214),(407,227),(500,219),(567,301),(687,294),(826,207),(922,214),
             (1032,267),(1192,286),(1306,264),(1486,328),(1605,325),(1720,401)]
    pontos=[[round(x*2800/1800),round(950+y*1100/707)] for x,y in esquerda]
    pontos += [[round(2600+x*2872/1800),round(950+y*1100/689)] for x,y in direita]
    arquivo=BASE/'bernard_28059238281.jpg'
    cfg=json.loads((ROOT/'data/modelos/verificador_solo_v1/config.json').read_text())
    protocolo=dict(data=datetime.now(timezone.utc).isoformat(),arquivo=arquivo.name,
        sha256=hashlib.sha256(arquivo.read_bytes()).hexdigest(),grupo='autor_bernard_dupont_from_france',
        conjunto='test',criterio='Minimo visual de bovinos individualizaveis, ancorados na cabeca. Outros animais visiveis nao anotados; nunca usar como treino ou gabarito completo.',
        minimo=len(pontos),maximo=None,contagem_exata=False,pontos_xy=pontos,
        detector=dict(pesos='data/processed/modelo_gado.pt',imgsz=1280,iou=.7,conf=.15,augment=False,
            sha256=hashlib.sha256((ROOT/'data/processed/modelo_gado.pt').read_bytes()).hexdigest()),
        verificador=cfg,limite='Um autor ja reservado para teste; nao equivale a benchmark completo, nem mede precisao de caixas ou quantidade total.')
    destino.write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    im=Image.open(arquivo).convert('RGB')
    d=ImageDraw.Draw(im)
    for i,(x,y) in enumerate(pontos,1):
        d.ellipse((x-22,y-22,x+22,y+22),outline='red',width=7)
        d.text((x+24,y-24),str(i),fill='yellow',stroke_width=2,stroke_fill='black',font_size=35)
    im.thumbnail((1800,900));im.save(OUT/'minimo_visual.jpg')
    print('Minimo visual preparado:',len(pontos),flush=True)


def avaliar():
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    import contagem as ct
    from verificador_visual import carregar_binario,confirmar,mascara_confirmacao
    if (OUT/'resultado.json').exists():
        raise ValueError('Avaliacao existente preservada; teste agora conhecido')
    p=json.loads((OUT/'gabarito_antes_inferencia.json').read_text())
    assert p['minimo']==len(p['pontos_xy']) and not p['contagem_exata']
    arquivo=BASE/p['arquivo'];peso=ROOT/p['detector']['pesos'];cfg=p['verificador']
    for caminho,sha in ((arquivo,p['sha256']),(peso,p['detector']['sha256']),
        (ROOT/cfg['extrator'],cfg['extrator_sha256']),(ROOT/cfg['classificador'],cfg['classificador_sha256'])):
        assert hashlib.sha256(caminho.read_bytes()).hexdigest()==sha
    torch.set_num_threads(4)
    im=cv2.imread(str(arquivo))
    modelo=YOLO(str(peso))
    b,s=ct.detectar_inteira(modelo,im,imgsz=1280,iou_nms=.7,classes=[0],augment=False,device=0)
    b,s=ct.filtrar(b,s,.15)
    ver=carregar_binario(ROOT/cfg['extrator'],ROOT/cfg['classificador'],cfg['classificador_sha256'])
    v=confirmar(ver,im,b)
    manter,_=mascara_confirmacao(b,v,cfg['limiar'],cfg.get('min_lado_px',0))
    linhas=[]
    for nome,bb in (('detector_atual',b),('com_verificador',b[manter])):
        linhas.append(dict(modelo=nome,previsto=len(bb),minimo_visual=p['minimo'],
            deficit_minimo=max(0,p['minimo']-len(bb)),atingiu_minimo=len(bb)>=p['minimo']))
        cv2.imwrite(str(OUT/(nome+'.jpg')),ct.desenhar(im,bb))
    (OUT/'resultado.json').write_text(json.dumps(dict(resultados=linhas,
        conclusao='Atingir o minimo nao comprova acerto; falsos positivos podem compensar omissoes. Nao mede erro absoluto da contagem total.',
        contagem_exata=False,teste_agora_consumido=True),indent=2),encoding='utf-8')
    print(json.dumps(linhas,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--avaliar',action='store_true');args=ap.parse_args()
    avaliar() if args.avaliar else preparar()
