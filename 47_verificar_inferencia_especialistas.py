"""Paridade das rotas reais do app; nao afirma teste visual completo da interface."""
from pathlib import Path
import ast
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import contagem as ct
from especialistas import resolver_especialista

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    arvore = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
    fn = next(n for n in arvore.body if isinstance(n, ast.FunctionDef) and n.name == 'detectar_bytes')
    fn.decorator_list = []
    ns = dict(np=np, cv2=cv2, ct=ct)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'app.py', 'exec'), ns)
    amostras = {'solo_poucos': ROOT / 'data/teste_brasil/brasil_12.jpg',
                'solo_denso': ROOT / 'data/teste_brasil/brasil_08.jpg',
                'drone_vertical': next((ROOT / 'data/yolo/images/val').glob('Jalogny__*.jpg')),
                'curral_topo': next((ROOT / 'data/yolo/images/val').glob('colo_topo__*.jpg'))}
    modelos, registros = {}, []
    for visao, img in amostras.items():
        p = resolver_especialista(ROOT, visao)
        cfg = p['config']
        peso = p['pesos_absolutos']
        if peso not in modelos:
            modelos[peso] = YOLO(peso)
        modelo = modelos[peso]
        b,s,shape,modo = ns['detectar_bytes'](img.read_bytes(), cfg['modo'], cfg['tile'], cfg['overlap'], cfg['imgsz'],
            modelo, p['mt'], cfg['iou'], cfg['iou_juncao'], 1000, ct.VERSAO_INFERENCIA, cfg['legado'],
            p['chave_modelo'], p['classes'], cfg['augment'])
        im = cv2.imread(str(img))
        max_det = 300 if cfg['legado'] else 1000
        if cfg['modo'] == 'tiles':
            esperado, score = ct.detectar(modelo, im, tile=cfg['tile'], overlap=cfg['overlap'], iou_local=cfg['iou'],
                iou_nms=cfg['iou_juncao'], max_det=max_det, legado=cfg['legado'], classes=p['classes'], augment=cfg['augment'])
        else:
            esperado, score = ct.detectar_inteira(modelo, im, imgsz=cfg['imgsz'], iou_nms=cfg['iou'], max_det=max_det,
                classes=p['classes'], augment=cfg['augment'])
        np.testing.assert_allclose(b, esperado, atol=1e-5)
        np.testing.assert_allclose(s, score, atol=1e-6)
        assert tuple(shape) == im.shape[:2] and modo == cfg['modo']
        registros.append(dict(visao=visao, imagem=str(img), pesos=peso, chave=p['chave_modelo'],
            paridade_caixas=True, contagem=int((s >= cfg['conf']).sum())))
        print(visao, 'OK', registros[-1]['contagem'], flush=True)
    out = ROOT / 'data/experimentos/especialistas_verificacao.json'
    out.write_text(json.dumps(dict(rotas=registros, lote_novo_consultado=False,
        pesos_originais_preservados=True, teste_visual_interface=False,
        nota='AppTest foi interrompido por travamento; validada a funcao real de inferencia por AST e as rotas com modelos reais.'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
