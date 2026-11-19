"""Propaga tres correcoes revisadas para copias dos rotulos das fotos completas."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from PIL import Image
import contagem as ct

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/gabarito_aereo_revisado_v2'


def main():
    target=OUT/'validacao_completa.csv'
    if target.exists():raise ValueError('Manifesto preservado')
    edits=json.loads((OUT/'resultado.json').read_text())['edicoes']
    df=pd.read_csv(ROOT/'data/experimentos/especialista_drone_v1/validacao_inteira.csv');audit=[]
    (OUT/'labels_completas').mkdir(exist_ok=True)
    for i,r in df.iterrows():
        w,h=Image.open(r['img']).size;src=Path(r['rot']);lines=[s for s in src.read_text().splitlines() if s.strip()]
        bb=ct.yolo_para_pixels(lines,w,h);remove=set()
        for e in edits:
            if e['original']!=r['arquivo']:continue
            idx=np.flatnonzero(np.max(np.abs(bb-np.array(e['global_xyxy'])),axis=1)<=2)
            assert len(idx)==1,(r['arquivo'],e['caso'],idx)
            remove.add(int(idx[0]))
        dest=OUT/'labels_completas'/(r['arquivo']+'.txt');dest.write_text('\n'.join(s for j,s in enumerate(lines) if j not in remove)+'\n',encoding='utf-8')
        df.loc[i,'rot']=str(dest)
        audit.append(dict(arquivo=r['arquivo'],origem=str(src),destino=str(dest),removidos=sorted(remove),
            original_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),revisado_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    assert sum(len(a['removidos']) for a in audit)==3
    df.to_csv(target,index=False);(OUT/'auditoria_fotos_completas.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('24 referencias completas copiadas; 3 fragmentos revisados excluidos; originais preservados.')


if __name__=='__main__':main()
