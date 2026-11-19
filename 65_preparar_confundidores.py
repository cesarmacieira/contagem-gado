"""Selecao anterior a inferencia por anotacoes oficiais e autores separados."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import random
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/confundidores_especies_v1'


def main():
    destino=OUT/'selecao_antes_download.json'
    if destino.exists():
        raise ValueError('Selecao existente preservada')
    with (OUT/'validation_images.csv').open(encoding='utf-8',newline='') as f:
        meta={r['ImageID']:r for r in csv.DictReader(f)}
    anns=defaultdict(list);gado=set()
    with (OUT/'validation_bbox.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['LabelName'] in ('/m/01xq0k1','/m/0cnyhnx'):
                gado.add(r['ImageID'])
            if r['LabelName'] in ('/m/03k3r','/m/07bgp') and all(r[k]=='0' for k in ('IsGroupOf','IsDepiction','IsOccluded','IsTruncated')):
                b=[float(r[k]) for k in ('XMin','YMin','XMax','YMax')]
                area=(b[2]-b[0])*(b[3]-b[1])
                if area>=.08:
                    anns[r['ImageID']].append(dict(classe='horse' if r['LabelName']=='/m/03k3r' else 'sheep',bbox_rel=b,area=area))
    grupos=set();selecionados=[]
    rng=random.Random(160991)
    for classe,n in [('horse',24),('sheep',12)]:
        ids=sorted(k for k,v in anns.items() if k not in gado and any(a['classe']==classe for a in v))
        rng.shuffle(ids);cont=0
        for k in ids:
            m=meta.get(k,{})
            autor=m.get('AuthorProfileURL') or m.get('Author')
            if not autor or autor in grupos or m.get('License')!='https://creativecommons.org/licenses/by/2.0/':
                continue
            a=max((x for x in anns[k] if x['classe']==classe),key=lambda x:x['area'])
            # Distribuicao por autores antes de baixar/ver os arquivos.
            train_n,val_n=(16,4) if classe=='horse' else (8,2)
            split='train' if cont<train_n else 'val' if cont<train_n+val_n else 'test'
            selecionados.append(dict(id=k,arquivo=k+'.jpg',classe=classe,conjunto=split,grupo=autor,
                autor=m['Author'],licenca=m['License'],pagina=m['OriginalLandingURL'],
                url_original=m['OriginalURL'],url_download=f'https://open-images-dataset.s3.amazonaws.com/validation/{k}.jpg',
                bbox_rel=a['bbox_rel'],revisado=False))
            grupos.add(autor);cont+=1
            if cont==n:
                break
        assert cont==n,(classe,cont)
    destino.write_text(json.dumps(dict(data=datetime.now(timezone.utc).isoformat(),
        origem='Open Images V5 validation; transferido para tarefa binaria, separado do benchmark brasileiro',
        detector_executado=False,registros=selecionados),indent=2),encoding='utf-8')
    print('Selecionados:',len(selecionados),'autores distintos; train 24, val 6, test 6',flush=True)


if __name__=='__main__':
    main()
