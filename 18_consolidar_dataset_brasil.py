"""Entrega unica com os dois lotes; preserva as versoes e divisoes originais."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import cv2
import pandas as pd
import contagem as ct

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'data/brasil_dataset_v2'
    for d in ('images','labels','marcacoes'): (out/d).mkdir(parents=True,exist_ok=True)
    fontes=[];manifest=[];gabarito=[];imgs=[];anns=[];hashes=set();grupos={}
    for lote,base in [('brasil_lote_01',ROOT/'data/brasil_revisado'),('brasil_lote_02',ROOT/'data/brasil_lote_02/revisado')]:
        fontes.extend(json.loads((ROOT/'data'/lote/'anotacoes.json').read_text(encoding='utf-8')))
        df=pd.read_csv(base/'manifesto.csv',keep_default_na=False)
        for row in df.to_dict('records'):
            row.pop('base',None);row['lote']=lote;manifest.append(row)
            for d in ('images','labels','marcacoes'):
                nome=row['arquivo'] if d!='labels' else Path(row['arquivo']).stem+'.txt'
                shutil.copy2(base/d/nome,out/d/nome)
    completos={f['arquivo_exportado']:f for f in fontes if f['status']=='completa'}
    for r in manifest:
        nome=r['arquivo'];f=completos[nome];m=f['anotacao'];im=cv2.imread(str(out/'images'/nome));h,w=im.shape[:2]
        digest=hashlib.sha256((out/'images'/nome).read_bytes()).hexdigest();assert digest not in hashes;hashes.add(digest)
        assert grupos.setdefault(r['grupo'],r['conjunto'])==r['conjunto']
        caixas=ct.yolo_para_pixels((out/'labels'/(Path(nome).stem+'.txt')).read_text().splitlines(),w,h)
        assert len(caixas)==int(r['n_bovinos'])
        pid=f['pageid'];imgs.append(dict(id=pid,file_name=nome,width=w,height=h,split=r['conjunto'],author=f['autor'],license=f['licenca'],source_url=f['url']))
        for x1,y1,x2,y2 in caixas:
            assert x1>=-.001 and y1>=-.001 and x2<=w+.001 and y2<=h+.001
            anns.append(dict(id=len(anns)+1,image_id=pid,category_id=1,bbox=list(map(float,[x1,y1,x2-x1,y2-y1])),area=float((x2-x1)*(y2-y1)),iscrowd=0,segmentation=[]))
        gabarito.append(dict(arquivo=nome,minimo=len(caixas),maximo=len(caixas),conjunto=r['conjunto'],grupo=r['grupo'],revisor='assistente_visual_v1',nota=m['nota'],url=f['url']))
    for nome,rows in [('manifesto.csv',manifest),('gabarito.csv',gabarito)]:
        campos=list(dict.fromkeys(k for r in rows for k in r))
        with (out/nome).open('w',encoding='utf-8-sig',newline='') as fp:
            wr=csv.DictWriter(fp,fieldnames=campos);wr.writeheader();wr.writerows(rows)
    (out/'anotacoes_coco.json').write_text(json.dumps(dict(images=imgs,annotations=anns,categories=[dict(id=1,name='bovino')]),ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'fontes_e_marcacoes.json').write_text(json.dumps([completos[r['arquivo']] for r in manifest],ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'anotacoes_parciais.json').write_text(json.dumps([f for f in fontes if f['status']=='parcial'],ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(ROOT/'data/brasil_revisado/dicionario_classes.json',out/'dicionario_classes.json')
    shutil.copy2(ROOT/'data/brasil_revisado/galeria_rotulos.html',out/'galeria_rotulos.html')
    for sp in ('train','val','test'):
        (out/f'{sp}.txt').write_text('\n'.join(str(out/'images'/r['arquivo']).replace('\\','/') for r in manifest if r['conjunto']==sp)+'\n',encoding='utf-8')
    (out/'dataset.yaml').write_text(f'path: {out.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: bovino\n',encoding='utf-8')
    resumo=dict(imagens=len(manifest),caixas=len(anns),**{sp:dict(imagens=sum(r['conjunto']==sp for r in manifest),caixas=sum(int(r['n_bovinos']) for r in manifest if r['conjunto']==sp)) for sp in ('train','val','test')})
    (out/'resumo.json').write_text(json.dumps(resumo,indent=2),encoding='utf-8')
    (out/'LEIA_ME.md').write_text('''# Fotos e marcacoes de bovinos brasileiros

44 imagens completas e 152 caixas, em dois lotes. Primeira revisao visual do assistente; nao representa consenso de especialistas. Fontes, autores, licencas e transformacoes de recorte estao em fontes_e_marcacoes.json e na galeria. As fotografias conservam suas licencas de origem.

Uma caixa por bovino individualizavel, incluindo bezerros, animais ocultos e cortados. Cabeca e corpo do mesmo animal nao sao duas instancias. Cavalos, pessoas, aves e objetos ficam fora. As caixas cobrem partes visiveis, sem inferir corpos ocultos. Os quatro exemplos parciais ficam em anotacoes_parciais.json e nao entram no treino ou no teste exato.

Treino: 22 fotos/78 bovinos. Validacao: 9/30. Teste: 13/44, provenientes de duas rodadas (10+3). O teste da primeira rodada ja foi examinado antes da segunda; consultar os relatorios dos experimentos, sem alegar que todas as 13 fotos permaneceram ineditas para ambas as rodadas.

gabarito.csv contem contagens exatas conforme a revisao visual. labels/ usa YOLO (classe 0), anotacoes_coco.json usa categoria 1. marcacoes/ e a galeria mostram cada individuo numerado. dataset.yaml aponta para listas separadas por autor. O dicionario define a classe e a politica de anotacao.
''',encoding='utf-8')
    print(json.dumps(resumo,indent=2))


if __name__=='__main__': main()
