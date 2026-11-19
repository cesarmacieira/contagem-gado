"""Executa funcoes reais do app sem servidor; nao substitui teste visual."""
from pathlib import Path
import ast
import hashlib
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import contagem as ct
from especialistas import resolver_especialista
from verificador_visual import confirmar, mascara_confirmacao

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/experimentos/confirmacao_app_v2'


def main():
    torch.set_num_threads(4)
    if (OUT / 'resultado.json').exists():
        raise ValueError('Verificacao existente preservada; escolher nova pasta de saida')
    arvore = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
    nomes = {'detectar_bytes', 'ler_confirmador_visual', 'pontuar_confirmacao'}
    funcoes = [n for n in arvore.body if isinstance(n, ast.FunctionDef) and n.name in nomes]
    assert len(funcoes) == len(nomes)
    for n in funcoes:
        n.decorator_list = []
    ns = dict(np=np, cv2=cv2, ct=ct, hashlib=hashlib, Path=Path)
    exec(compile(ast.Module(body=funcoes, type_ignores=[]), 'app.py', 'exec'), ns)
    meta = json.loads((ROOT / 'data/modelos/verificador_solo_v1/config.json').read_text())
    assert meta['visoes_permitidas'] == ['solo_denso'] and not meta['ativo_por_padrao']
    ext, clf = ROOT / meta['extrator'], ROOT / meta['classificador']
    verificador, device = ns['ler_confirmador_visual'](str(ext), ext.stat().st_mtime_ns,
        meta['extrator_sha256'], str(clf), clf.stat().st_mtime_ns, meta['classificador_sha256'])
    p = resolver_especialista(ROOT, 'solo_denso')
    cfg = p['config']
    modelo = YOLO(p['pesos_absolutos'])
    fotos = [ROOT / 'data/teste_brasil/brasil_01.jpg']
    fotos += [ROOT / 'data/experimentos/solo_piloto_separado_v1/teste/images' / f for f in ('bdn_19.jpg', 'bdn_21.jpg')]
    OUT.mkdir(parents=True, exist_ok=True)
    linhas = []
    for foto in fotos:
        dados = foto.read_bytes()
        b, s, _, _ = ns['detectar_bytes'](dados, cfg['modo'], cfg['tile'], cfg['overlap'], cfg['imgsz'],
            modelo, p['mt'], cfg['iou'], cfg['iou_juncao'], 1000, ct.VERSAO_INFERENCIA,
            cfg['legado'], p['chave_modelo'], p['classes'], cfg['augment'])
        b, s = ct.filtrar(b, s, cfg['conf'])
        chave = f"{meta['versao']}:{meta['extrator_sha256']}:{meta['classificador_sha256']}"
        scores = ns['pontuar_confirmacao'](dados, b, chave, verificador, device)
        im = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_allclose(scores, confirmar(verificador, im, b, device=device))
        manter, _ = mascara_confirmacao(b, scores, meta['limiar'], meta.get('min_lado_px', 0))
        cv2.imwrite(str(OUT / foto.name), ct.desenhar(im, b[manter]))
        linha = dict(arquivo=foto.name, antes=len(b), depois=int(manter.sum()), paridade_scores=True)
        linhas.append(linha)
        print(linha, flush=True)
    assert [(r['antes'], r['depois']) for r in linhas] == [(8, 8), (10, 0), (11, 11)]
    (OUT / 'resultado.json').write_text(json.dumps(dict(imagens=linhas, teste_visual_interface=False,
        bloqueio_interface='Python asyncio trava em socket._fallback_socketpair ao criar loop local; servidor nao iniciou.',
        modelo_sha256=meta['classificador_sha256']), indent=2), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--saida',type=Path,default=OUT)
    args=ap.parse_args();OUT=args.saida.resolve()
    main()
