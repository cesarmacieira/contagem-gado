"""Recortes reais adicionais, com transformacoes explicitas e teste novo separado."""
from pathlib import Path
import csv
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    lote=ROOT/'data/brasil_lote_02';out=lote/'revisado'
    manual=json.loads((lote/'rotulos_manuais.json').read_text(encoding='utf-8'))
    fontes=json.loads((lote/'fontes.json').read_text(encoding='utf-8'))
    antigas=json.loads((ROOT/'data/brasil_lote_01/divisao_congelada.json').read_text(encoding='utf-8'))
    fontes.extend(f for f in antigas if str(f['pageid']) in manual)
    manifest=[];anotados=[]
    for f in fontes:
        m=manual.get(str(f['pageid']))
        if not m: continue
        pasta=ROOT/'data'/m.get('origem_lote','brasil_lote_02')
        img=cv2.imread(str(pasta/'images'/f['arquivo']));h,w=img.shape[:2]
        b=np.array(m['caixas'],float).reshape(-1,4)
        if m['coordenadas']=='review':
            rev=cv2.imread(str(pasta/'revisao_limpa'/f"{f['pageid']}.jpg"));rh,rw=rev.shape[:2]
            fator=np.array([w/rw,h/rh,w/rw,h/rh]);b*=fator
        else: fator=np.ones(4)
        crop=m.get('recorte_supervisionado');nome=Path(f['arquivo']).stem+('_recorte' if crop else '')+'.jpg'
        if crop:
            x1,y1,x2,y2=np.rint(np.array(crop)*fator).astype(int)
            img=img[y1:y2,x1:x2].copy();b-=[x1,y1,x1,y1];h,w=img.shape[:2]
        b[:,[0,2]]=b[:,[0,2]].clip(0,w);b[:,[1,3]]=b[:,[1,3]].clip(0,h)
        assert ((b[:,2:]-b[:,:2])>0).all()
        r=dict(f,anotacao=m,caixas_xyxy=b.tolist(),arquivo_exportado=nome,status='parcial' if m.get('qualidade')=='parcial' else 'completa',conjunto=m['conjunto'])
        anotados.append(r)
        draw=img.copy()
        for i,box in enumerate(b,1):
            x1,y1,x2,y2=np.rint(box).astype(int);cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(draw,str(i),(x1,max(20,y1)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
        (out/'marcacoes').mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(out/'marcacoes'/nome),draw)
        if m.get('qualidade')=='parcial': continue
        for d in ('images','labels'): (out/d).mkdir(exist_ok=True)
        cv2.imwrite(str(out/'images'/nome),img,[cv2.IMWRITE_JPEG_QUALITY,96])
        linhas=[]
        for x1,y1,x2,y2 in b:
            linhas.append('0 '+' '.join(f'{v:.8f}' for v in ((x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h)))
        (out/'labels'/(Path(nome).stem+'.txt')).write_text('\n'.join(linhas)+'\n',encoding='utf-8')
        manifest.append(dict(arquivo=nome,grupo=f.get('grupo') or 'autor_'+f['autor'],conjunto=m['conjunto'],revisado=True,autor=f['autor'],licenca=f['licenca'],url=f['url'],n_bovinos=len(b),base=str(out)))
    with (out/'manifesto.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        wr=csv.DictWriter(fp,fieldnames=list(manifest[0]));wr.writeheader();wr.writerows(manifest)
    (lote/'anotacoes.json').write_text(json.dumps(anotados,ensure_ascii=False,indent=2),encoding='utf-8')
    congelado=lote/'divisao_congelada.json'
    if congelado.exists(): assert json.loads(congelado.read_text(encoding='utf-8'))==manifest
    else: congelado.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Complemento:',[(r['arquivo'],r['conjunto'],r['n_bovinos']) for r in manifest])


if __name__=='__main__': main()
