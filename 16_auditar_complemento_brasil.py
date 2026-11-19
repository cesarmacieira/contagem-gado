"""Compara novas fontes com imagens preexistentes por dois hashes perceptuais."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent


def fingerprint(p):
    im=cv2.imread(str(p),cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if im is None: return None
    a=cv2.resize(im,(9,8));d=a[:,1:]>a[:,:-1]
    v=cv2.dct(cv2.resize(im,(32,32)).astype(np.float32))[:8,:8].copy();v[0,0]=0
    return (int.from_bytes(np.packbits(d).tobytes(),'big'),int.from_bytes(np.packbits(v>np.median(v)).tobytes(),'big'))


def main():
    lote=ROOT/'data/brasil_lote_02';novas=[(p,fingerprint(p)) for p in (lote/'images').glob('*.jpg')]
    refs=list((ROOT/'data/extras').rglob('*.jpg'))+list((ROOT/'data/teste_brasil').glob('*.jpg'))+list((ROOT/'data/brasil_novo_teste').glob('*.*'))+list((ROOT/'data/brasil_lote_01/images').glob('*.jpg'))
    encontrados=[];n=0
    for p in refs:
        h=fingerprint(p)
        if h is None: continue
        n+=1
        for nova,nh in novas:
            if nh is None: continue
            d=(h[0]^nh[0]).bit_count();s=(h[1]^nh[1]).bit_count()
            if d<=4 and s<=4: encontrados.append(dict(nova=nova.name,referencia=str(p.relative_to(ROOT)),dhash=d,phash=s))
    resultado=dict(referencias=n,novas=len(novas),candidatos=encontrados,limitacao='Hashes nao provam ausencia de recortes ou alteracoes maiores; autoria e origem tambem foram verificadas.')
    (lote/'auditoria_duplicatas.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(resultado,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
