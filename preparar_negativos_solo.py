"""Contato e manifesto de negativos reais, aprovados somente apos revisao visual."""
from pathlib import Path
import csv
import hashlib
import json
from PIL import Image,ImageDraw

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'data/negativos_solo_v1'


def main():
    plano = json.loads((BASE / 'plano_antes_inferencia.json').read_text(encoding='utf-8'))['imagens']
    revisoes = json.loads((BASE / 'revisao_visual.json').read_text(encoding='utf-8')) if (BASE / 'revisao_visual.json').exists() else {}
    rows = []
    for r in plano:
        p = BASE / 'images' / r['arquivo']
        if not p.exists():
            continue
        with Image.open(p) as im:
            im.load()
            w,h = im.size
        revisao = revisoes.get(r['arquivo'],{})
        aprovado = revisao.get('status') == 'aprovado_sem_bovinos'
        row = dict(**r,largura=w,altura=h,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   status=revisao.get('status','pendente_revisao'),nota_revisao=revisao.get('nota',''))
        row['revisado'],row['n_bovinos'] = aprovado,0 if aprovado else ''
        row['transformacao'] = 'Miniatura Commons de ate 1280 px; original_url preservada'
        rows.append(row)
        if aprovado:
            (BASE / 'labels').mkdir(exist_ok=True)
            (BASE / 'labels' / (p.stem+'.txt')).write_text('',encoding='utf-8')
    if rows:
        with (BASE / 'manifesto.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for inicio in range(0,len(rows),6):
        folha = Image.new('RGB',(1280,990),'white')
        desenho = ImageDraw.Draw(folha)
        for k,r in enumerate(rows[inicio:inicio+6]):
            im = Image.open(BASE / 'images' / r['arquivo']).convert('RGB')
            im.thumbnail((630,300))
            x,y = (k%2)*640,(k//2)*330
            folha.paste(im,(x,y))
            desenho.text((x+4,y+303),r['arquivo']+' | '+r['conjunto'],fill='black')
        folha.save(BASE / f'contato_{inicio//6+1}.jpg',quality=93)
    (BASE / 'resumo.json').write_text(json.dumps(dict(baixadas=len(rows),aprovadas=sum(r['revisado'] for r in rows),
        por_conjunto={s:sum(r['revisado'] and r['conjunto']==s for r in rows) for s in ('train','val','test')},
        detector_executado=False),indent=2),encoding='utf-8')
    print('Baixadas',len(rows),'aprovadas',sum(r['revisado'] for r in rows),flush=True)


if __name__ == '__main__':
    main()
