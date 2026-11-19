"""Galeria autocontida das marcacoes e respectivas fontes/licencas."""
from pathlib import Path
import base64
import csv
import html
import json
import cv2

ROOT=Path(__file__).resolve().parent


def main():
    base=ROOT/'data/brasil_revisado'
    fontes=json.loads((ROOT/'data/brasil_lote_01/anotacoes.json').read_text(encoding='utf-8'))
    for f in fontes: f['_base']=str(base)
    extra=ROOT/'data/brasil_lote_02/anotacoes.json'
    if extra.exists():
        novos=json.loads(extra.read_text(encoding='utf-8'))
        for f in novos: f['_base']=str(extra.parent/'revisado')
        fontes.extend(novos)
    cards=[]
    for f in sorted(fontes,key=lambda f:(f['conjunto'],f['pageid'])):
        if f['status']!='completa': continue
        nome=f['arquivo_exportado']; img=cv2.imread(str(Path(f['_base'])/'marcacoes'/nome))
        h,w=img.shape[:2];s=min(1,800/w); img=cv2.resize(img,None,fx=s,fy=s)
        ok,jpg=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,82]);assert ok
        data=base64.b64encode(jpg).decode('ascii')
        esc=html.escape
        cards.append(f'<article data-split="{f["conjunto"]}"><h2>{esc(nome)} <span>{f["conjunto"]}</span></h2><p><b>{len(f["caixas_xyxy"])} bovinos</b> · {esc(f["autor"])}</p><img loading="lazy" src="data:image/jpeg;base64,{data}"><p>{esc(f["anotacao"]["nota"])}</p><a href="{esc(f["url"],quote=True)}">Fonte original</a> · <a href="{esc(f.get("licenca_url") or f["url"],quote=True)}">{esc(f["licenca"])}</a></article>')
    texto='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Gabarito visual — bovinos</title><style>body{font:16px system-ui;background:#f2f5f3;color:#19332b;max-width:1400px;margin:24px auto;padding:0 20px}h1{font-size:32px}header{margin-bottom:24px}button{padding:9px 20px;margin:4px;border:1px solid #315547;border-radius:8px;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}article{background:white;border-radius:12px;padding:18px}article img{width:100%;height:auto}h2{font-size:17px}span{font-size:12px;background:#dcebdd;padding:4px;border-radius:5px}a{color:#205c48}p{line-height:1.5}@media(max-width:500px){.grid{grid-template-columns:1fr}}</style><header><h1>Gabarito visual do novo lote</h1><p>40 fotos completas · 133 caixas · 21 treino / 9 validação / 10 teste.</p><p>Primeira revisão visual pelo assistente. Uma caixa por bovino identificável, incluindo bezerros, animais parcialmente ocultos e cortados nas bordas. Fotografias ambíguas, duplicadas ou fora de escopo ficaram fora do conjunto supervisionado. As caixas cobrem partes visíveis, sem inventar corpos ocultos.</p><button onclick="filtrar('todos')">Todos</button><button onclick="filtrar('train')">Treino</button><button onclick="filtrar('val')">Validação</button><button onclick="filtrar('test')">Teste</button></header><div class="grid">'''+''.join(cards)+'''</div><script>function filtrar(s){document.querySelectorAll('article').forEach(e=>e.hidden=s!=='todos'&&e.dataset.split!==s)}</script></html>'''
    completos=[f for f in fontes if f['status']=='completa']
    resumo=f"{len(completos)} fotos completas · {sum(len(f['caixas_xyxy']) for f in completos)} caixas · "+' / '.join(f"{sum(f['conjunto']==s for f in completos)} {s}" for s in ('train','val','test'))+'.'
    texto=texto.replace('40 fotos completas · 133 caixas · 21 treino / 9 validação / 10 teste.',resumo)
    (base/'galeria_rotulos.html').write_text(texto,encoding='utf-8')
    print('Galeria gerada:',base/'galeria_rotulos.html')


if __name__=='__main__': main()
