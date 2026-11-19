"""Contrato de entrada e verificacao independente do lote brasileiro novo.

Nao altera o lote, nao executa detectores e nao transforma previsoes em rotulos.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import defaultdict

from PIL import Image

COLUNAS = {'arquivo', 'grupo', 'conjunto', 'revisado', 'autor', 'licenca', 'url'}
SIM = {'1', 'true', 'sim'}


def ler_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def nome_seguro(nome):
    return bool(nome and nome not in ('.', '..') and not any(c in nome for c in '/\\:'))


def conferir(base):
    base = Path(base)
    erros, avisos, registros = [], [], []
    resumo = dict(base=str(base), pronto=False, inferencia_executada=False,
                  erros=erros, avisos=avisos, imagens_aprovadas=0, caixas=0, por_conjunto={})
    exigidos = ['manifesto.csv', 'gabarito.csv', 'anotacoes_coco.json', 'dicionario.json']
    faltam = [f for f in exigidos if not (base / f).is_file()]
    if faltam:
        resumo.update(estado='aguardando_entrega', arquivos_ausentes=faltam)
        return resumo
    manifesto = ler_csv(base / 'manifesto.csv')
    if not manifesto or not COLUNAS.issubset(manifesto[0]):
        erros.append(f'Manifesto vazio ou sem colunas obrigatorias: {sorted(COLUNAS)}')
        resumo['estado'] = 'reprovado_integridade'
        return resumo
    grupos, nomes, hashes = {}, set(), {}
    gab_rows = ler_csv(base / 'gabarito.csv')
    gab = {}
    for r in gab_rows:
        nome = r.get('arquivo', '')
        if nome in gab:
            erros.append(f'Gabarito duplicado: {nome}')
        gab[nome] = r
    try:
        coco = json.loads((base / 'anotacoes_coco.json').read_text(encoding='utf-8-sig'))
        json.loads((base / 'dicionario.json').read_text(encoding='utf-8-sig'))
    except (ValueError, OSError) as e:
        erros.append(f'JSON invalido: {e}')
        resumo['estado'] = 'reprovado_integridade'
        return resumo
    imagens_coco, anns = {}, defaultdict(list)
    ids_img, ids_ann = set(), set()
    categorias = {c.get('id'): c.get('name') for c in coco.get('categories', [])}
    if categorias != {1: 'bovino'}:
        erros.append('COCO deve ter categoria unica id=1, name=bovino; YOLO usa classe 0')
    for im in coco.get('images', []):
        if im.get('id') in ids_img or im.get('file_name') in imagens_coco:
            erros.append('COCO com id de imagem ou file_name duplicado')
        ids_img.add(im.get('id'))
        imagens_coco[im.get('file_name')] = im
    for a in coco.get('annotations', []):
        if a.get('id') in ids_ann or a.get('image_id') not in ids_img:
            erros.append('COCO com id de anotacao duplicado ou imagem inexistente')
        ids_ann.add(a.get('id'))
        if a.get('category_id') != 1 or a.get('iscrowd', 0) != 0:
            erros.append('COCO deve conter bovinos individuais, category_id=1 e iscrowd=0')
        anns[a.get('image_id')].append(a)
    aprovados = [r for r in manifesto if r.get('revisado', '').strip().lower() in SIM]
    resumo['imagens_pendentes'] = len(manifesto) - len(aprovados)
    # Mesmo uma imagem ainda pendente nao pode compartilhar grupo com outra divisao.
    for r in manifesto:
        g, split = r.get('grupo', '').strip(), r.get('conjunto', '').strip()
        if g and split in ('train', 'val', 'test'):
            anterior = grupos.setdefault(g, split)
            if anterior != split:
                erros.append(f'Grupo em mais de um conjunto: {g}')
    for r in aprovados:
        nome = r['arquivo']
        if not nome_seguro(nome) or nome in nomes:
            erros.append(f'Nome de imagem invalido ou repetido: {nome}')
            continue
        nomes.add(nome)
        split, grupo = r['conjunto'].strip(), r['grupo'].strip()
        if split not in ('train', 'val', 'test') or not grupo:
            erros.append(f'Grupo ou conjunto invalido: {nome}')
        if not all(r[k].strip() for k in ('autor', 'licenca', 'url')):
            erros.append(f'Origem/licenca ausente: {nome}')
        if r.get('status', '').strip().lower() in ('parcial', 'pendente', 'ambigua', 'ambiguo'):
            erros.append(f'Imagem parcial/ambigua marcada revisado=sim: {nome}')
        img, rot = base / 'images' / nome, base / 'labels' / (Path(nome).stem + '.txt')
        if not img.is_file() or not rot.is_file():
            erros.append(f'Imagem ou label ausente: {nome}')
            continue
        sha = hashlib.sha256(img.read_bytes()).hexdigest()
        if sha in hashes:
            erros.append(f'Imagem identica repetida: {nome} / {hashes[sha]}')
        hashes[sha] = nome
        if r.get('sha256', '').strip() and r['sha256'].strip().lower() != sha:
            erros.append(f'SHA-256 nao corresponde ao original: {nome}')
        try:
            with Image.open(img) as im:
                im.load()
                w, h = im.size
            caixas, vistas = [], set()
            for linha in rot.read_text(encoding='utf-8-sig').splitlines():
                if not linha.strip():
                    continue
                v = tuple(float(x) for x in linha.split())
                if len(v) != 5 or not all(math.isfinite(x) for x in v) or v[0] != 0:
                    raise ValueError('cada linha deve ser classe 0 e quatro coordenadas finitas')
                _, x, y, bw, bh = v
                if bw <= 0 or bh <= 0 or min(x-bw/2, y-bh/2) < -1e-5 or max(x+bw/2, y+bh/2) > 1+1e-5:
                    raise ValueError('caixa fora da imagem ou sem area positiva')
                if v in vistas:
                    raise ValueError('caixa exatamente duplicada no gabarito')
                vistas.add(v)
                caixas.append(((x-bw/2)*w, (y-bh/2)*h, bw*w, bh*h))
        except (OSError, ValueError) as e:
            erros.append(f'Imagem/YOLO invalido em {nome}: {e}')
            continue
        n = len(caixas)
        g = gab.get(nome)
        try:
            if g is None:
                raise ValueError('imagem ausente')
            valores = [g[k] for k in ('n_bovinos', 'minimo', 'maximo') if k in g]
            if not valores or ('minimo' in g) != ('maximo' in g):
                raise ValueError('informar n_bovinos ou o par minimo/maximo')
            if any(float(v) != n for v in valores):
                raise ValueError('contagem exata deve coincidir com todas as caixas YOLO; intervalo com ambiguidades nao e gabarito exato')
            if g.get('ambiguos', '').strip() and float(g['ambiguos']) > 0:
                raise ValueError('ambiguidades ainda presentes; resolver ou manter a imagem pendente')
            for k in ('conjunto', 'grupo'):
                if k in g and g[k] != r[k]:
                    raise ValueError(f'{k} diverge do manifesto')
        except (ValueError, KeyError) as e:
            erros.append(f'Gabarito inconsistente em {nome}: {e}')
        c = imagens_coco.get(nome)
        if c is None or c.get('width') != w or c.get('height') != h:
            erros.append(f'Imagem/dimensoes ausentes ou divergentes no COCO: {nome}')
        else:
            itens = anns[c['id']]
            try:
                bb = [tuple(float(x) for x in a['bbox']) for a in itens]
                if any(len(v) != 4 or not all(math.isfinite(x) for x in v) for v in bb):
                    raise ValueError('bbox invalida')
                # Comparacao um-a-um, independente da ordem e tolerante ao arredondamento YOLO.
                restantes = list(bb)
                for a in caixas:
                    j = next((i for i, b in enumerate(restantes) if all(abs(x-y) <= .02 for x, y in zip(a, b))), None)
                    if j is None:
                        raise ValueError('caixas COCO e YOLO divergem')
                    restantes.pop(j)
                if restantes:
                    raise ValueError('COCO tem caixas adicionais')
            except (ValueError, KeyError, TypeError) as e:
                erros.append(f'COCO inconsistente em {nome}: {e}')
        registros.append(dict(arquivo=nome, conjunto=split, grupo=grupo, n_bovinos=n,
                              autor=r['autor'], licenca=r['licenca'], url=r['url'],
                              sha256=sha, sha256_label=hashlib.sha256(rot.read_bytes()).hexdigest()))
    if not aprovados:
        erros.append('Nenhuma imagem com revisado=sim')
    # O lote deve conter ao menos uma imagem em cada divisao antes de integrar.
    for split in ('train', 'val', 'test'):
        itens = [r for r in registros if r['conjunto'] == split]
        resumo['por_conjunto'][split] = dict(imagens=len(itens), caixas=sum(r['n_bovinos'] for r in itens),
                                           grupos=len({r['grupo'] for r in itens}),
                                           max_bovinos=max((r['n_bovinos'] for r in itens), default=0),
                                           imagens_20_ou_mais=sum(r['n_bovinos'] >= 20 for r in itens))
        if not itens:
            erros.append(f'Conjunto sem imagem aprovada: {split}')
        elif not any(r['n_bovinos'] >= 20 for r in itens):
            avisos.append(f'{split} ainda sem imagem de 20 ou mais bovinos; nao representa por si rebanhos densos')
    resumo['imagens_aprovadas'] = len(registros)
    resumo['caixas'] = sum(r['n_bovinos'] for r in registros)
    resumo['sha256_arquivos_controle'] = {f: hashlib.sha256((base / f).read_bytes()).hexdigest() for f in exigidos}
    conteudo_snapshot = dict(registros=sorted(registros, key=lambda r:r['arquivo']),
                             arquivos_controle=resumo['sha256_arquivos_controle'])
    resumo['snapshot_sha256'] = hashlib.sha256(json.dumps(conteudo_snapshot, sort_keys=True).encode()).hexdigest()
    resumo['pronto'] = not erros
    resumo['estado'] = 'pronto_para_revisao_final' if not erros else 'reprovado_integridade'
    avisos.append('Integridade nao certifica exaustividade visual das caixas, verdade da licenca ou independencia entre fazendas.')
    resumo['registros'] = registros
    return resumo
