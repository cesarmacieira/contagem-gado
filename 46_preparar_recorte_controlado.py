"""Proposta de recorte legivel a partir de anotacoes humanas, sem detector.

Nao altera o lote novo e nao decide divisao. A fonte e o recorte devem ficar
no mesmo grupo. Exportacao e proposta para segunda revisao, nao treino automatico.
"""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'data/brasil_denso_novo'
OUT = ROOT / 'data/apoio_coleta_denso/recorte_bdn09'


def intersecao(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def main():
    origem = BASE / 'images/bdn_09.jpg'
    anotacao = BASE / 'anotacoes/bdn_09.json'
    raw = anotacao.read_bytes()
    a = json.loads(raw)
    with Image.open(origem) as f:
        im = f.convert('RGB')
    regiao = (600, 1500, im.width, min(2400, im.height))
    assert regiao[3] > regiao[1]
    caixas, excluidas = [], []
    for b in a['caixas']:
        area = (b[2]-b[0]) * (b[3]-b[1])
        overlap = intersecao(b, regiao)
        if overlap == 0:
            excluidas.append(b)
            continue
        # Nao introduzir fragmento cortado sem rotulo no recorte.
        if abs(overlap-area) > .01:
            raise ValueError(f'Regiao corta animal anotado: {b}')
        caixas.append([b[0]-regiao[0],b[1]-regiao[1],b[2]-regiao[0],b[3]-regiao[1]])
    ambiguos = a.get('ambiguos', [])
    if any(intersecao(b['caixa'], regiao) > 0 for b in ambiguos):
        raise ValueError('Uma ambiguidade anotada permanece dentro do recorte')
    assert len(caixas) >= 1
    OUT.mkdir(parents=True, exist_ok=True)
    crop = im.crop(regiao)
    crop.save(OUT / 'proposta_recorte.jpg', quality=97)
    w, h = crop.size
    linhas = ['0 ' + ' '.join(f'{v:.8f}' for v in ((x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h))
              for x1,y1,x2,y2 in caixas]
    (OUT / 'proposta_recorte.txt').write_text('\n'.join(linhas) + '\n', encoding='utf-8')
    marcado = crop.copy()
    draw = ImageDraw.Draw(marcado)
    for i, b in enumerate(caixas, 1):
        draw.rectangle(b, outline='red', width=3)
        draw.text((b[0]+2, b[1]+2), str(i), fill='yellow', stroke_width=1, stroke_fill='black')
    marcado.save(OUT / 'conferencia_numerada.jpg', quality=95)
    fontes = json.loads((BASE / 'fontes_brutas.json').read_text(encoding='utf-8-sig'))
    fonte = next(r for r in fontes if r['arquivo'] == 'bdn_09.jpg')
    meta = dict(status='proposta_pendente_segunda_revisao', fonte=fonte,
        original_sha256=hashlib.sha256(origem.read_bytes()).hexdigest(),
        anotacao_origem_sha256=hashlib.sha256(raw).hexdigest(),
        regiao_original_xyxy=regiao, tamanho_recorte=crop.size, caixas=caixas,
        n_caixas_herdadas=len(caixas), n_caixas_fora_regiao=len(excluidas),
        ambiguidades_conhecidas_removidas=len(ambiguos), inferencia_executada=False,
        conjunto='nao_definido', obrigacao='Fonte original e recorte devem compartilhar grupo e conjunto',
        limite='Excluir as ambiguidades conhecidas nao garante ausencia de outras. Conferir todos os bovinos do recorte antes de aprovar. Sem previsoes como gabarito.')
    (OUT / 'proposta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k not in ('caixas','fonte')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
