"""Apoio somente leitura ao lote de outra IA: integridade, duplicatas e contato."""
from pathlib import Path
import csv
import hashlib
import html
import json
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT / 'data/brasil_denso_novo/triagem'
OUT = ROOT / 'data/apoio_coleta_denso'


def dhash(im):
    pixels = list(im.convert('L').resize((9, 8)).getdata())
    return sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
               for y in range(8) for x in range(8))


def ler(p):
    raw = p.read_bytes()
    with Image.open(p) as im:
        im.load()
        original = ImageOps.exif_transpose(im).convert('RGB')
        return dict(arquivo=p.name, caminho=str(p), largura=original.width,
                    altura=original.height, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    dhash=f'{dhash(original):016x}', status='legivel'), original.copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Snapshot de caminhos: nao altera nem renomeia arquivos de outra IA.
    caminhos = sorted(p for p in ENTRADA.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    atuais, imagens, falhas = [], {}, []
    for p in caminhos:
        try:
            r, im = ler(p)
            atuais.append(r)
            imagens[p.name] = im
        except (OSError, ValueError) as e:
            falhas.append(dict(arquivo=p.name, erro=str(e), acao='Rever apos download; nao remover automaticamente'))
    antigos = []
    for pasta in ('data/teste_brasil', 'data/brasil_dataset_v2/images', 'data/brasil_lote_01/images'):
        for p in sorted((ROOT / pasta).glob('*')):
            if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            try:
                r, _ = ler(p)
                antigos.append(r)
            except (OSError, ValueError):
                continue
    pares = []
    for i, a in enumerate(atuais):
        for b in atuais[:i] + antigos:
            distancia = (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count()
            if a['sha256'] == b['sha256'] or distancia <= 7:
                pares.append(dict(arquivo=a['arquivo'], comparar_com=b['caminho'],
                    sha256_igual=a['sha256'] == b['sha256'], distancia_dhash=distancia,
                    acao='Revisar visualmente; dHash sozinho nao confirma duplicata nem independencia de cena'))
    cols = ['arquivo', 'caminho', 'largura', 'altura', 'bytes', 'sha256', 'dhash', 'status']
    with (OUT / 'inventario_triagem.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(atuais)
    resumo = dict(instante_utc=datetime.now(timezone.utc).isoformat(), arquivos_no_snapshot=len(caminhos),
        imagens_legiveis=len(atuais), falhas=falhas, imagens_brasileiras_antigas_comparadas=len(antigos),
        pares_candidatos=pares, inferencia_executada=False, rotulos_criados=False,
        fontes_licencas_verificadas=False,
        limite='Triagem apenas. Sem gabarito, sem divisao, sem consulta ao detector. Comparacao com acervo brasileiro selecionado, nao acervo inteiro.')
    (OUT / 'auditoria_triagem.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    cartoes = []
    for inicio in range(0, len(atuais), 16):
        grupo = atuais[inicio:inicio + 16]
        folha = Image.new('RGB', (1440, 4 * 260), 'white')
        desenho = ImageDraw.Draw(folha)
        for j, r in enumerate(grupo):
            x, y = (j % 4) * 360, (j // 4) * 260
            mini = ImageOps.contain(imagens[r['arquivo']], (350, 225))
            folha.paste(mini, (x + (360 - mini.width) // 2, y))
            desenho.text((x + 5, y + 228), r['arquivo'], fill='black')
            desenho.text((x + 5, y + 243), f"{r['largura']} x {r['altura']}", fill='black')
        nome = f'contato_{inicio // 16 + 1:02d}.jpg'
        folha.save(OUT / nome, quality=90)
        cartoes.append(f'<h2>Lote {inicio // 16 + 1}</h2><img style="max-width:100%" src="{nome}">')
    (OUT / 'galeria_triagem.html').write_text('<!doctype html><meta charset="utf-8"><title>Apoio a triagem</title><h1>Fotos ja baixadas</h1><p>Sem inferencia ou rotulagem. Nao usar como resultado de teste. Origem e licenca ainda precisam ser conferidas pela IA responsavel pela coleta.</p>' + ''.join(cartoes), encoding='utf-8')
    print(json.dumps(resumo, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
