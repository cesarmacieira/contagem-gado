"""Candidatos geometricos de sombra em treino aereo externo; sem detector/rotulos automaticos."""
from pathlib import Path
import hashlib,json
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/candidatos_sombras_aereas_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'candidatos.csv').exists():raise ValueError('Lote preservado')
    div=pd.read_csv(ROOT/'data/processed/divisao.csv')
    div=div[(div.conjunto=='train')&(div.dominio=='aereo_pasto')&div.usar.astype(str).str.lower().isin(['true','1'])]
    allowed={f'{r.fazenda}__{str(r.voo).replace("/","_")}__{r.stem}' for r in div.itertuples()}
    files=[p for p in sorted((ROOT/'data/yolo/images/train').glob('*.jpg')) if p.stem.rsplit('__x',1)[0] in allowed]
    rng=np.random.default_rng(1121);files=[files[i] for i in rng.permutation(len(files))[:160]]
    candidates=[]
    for p in files:
        lab=ROOT/'data/yolo/labels/train'/(p.stem+'.txt')
        lines=[list(map(float,l.split())) for l in lab.read_text().splitlines() if l.strip()]
        if not 2<=len(lines)<=30:continue
        im=cv2.imread(str(p));h,w=im.shape[:2];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        if np.median(gray)<65 or np.percentile(gray,95)<150:continue
        boxes=[]
        mask=(gray<min(55,float(np.median(gray))*.5)).astype('uint8')
        for _,x,y,bw,bh in lines:
            b=np.round(np.array([x-bw/2,y-bh/2,x+bw/2,y+bh/2])*[w,h,w,h]).astype(int)
            boxes.append(b);x1,y1,x2,y2=b
            mask[max(0,y1-8):min(h,y2+8),max(0,x1-8):min(w,x2+8)]=0
        boxes=np.array(boxes)
        n,components,stats,_=cv2.connectedComponentsWithStats(mask,8)
        options=[]
        for k in range(1,n):
            x,y,bw,bh,area=stats[k]
            if min(bw,bh)<28 or max(bw,bh)>350 or area<300 or area>35000 or area/(bw*bh)<.35:continue
            if x<8 or y<8 or x+bw>w-8 or y+bh>h-8:continue
            center=np.array([x+bw/2,y+bh/2]);bc=(boxes[:,:2]+boxes[:,2:])/2
            dist=np.linalg.norm(bc-center,axis=1);size=np.max(boxes[:,2:]-boxes[:,:2],axis=1)
            if not (dist<2*size).any():continue
            darkness=float(np.median(gray)-gray[components==k].mean())
            options.append((darkness,[int(x),int(y),int(x+bw),int(y+bh)]))
        if options:
            score,box=max(options,key=lambda x:x[0])
            candidates.append(dict(origem=str(p.relative_to(ROOT)),rotulo=str(lab.relative_to(ROOT)),
                conjunto='train',dominio='aereo_pasto',grupo='/'.join(p.stem.split('__')[:2]),
                xyxy=box,contraste=score,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    selected=[];groups=set()
    for r in sorted(candidates,key=lambda r:-r['contraste']):
        if r['grupo'] in groups:continue
        groups.add(r['grupo']);selected.append(r)
        if len(selected)==8:break
    (OUT/'recortes').mkdir(exist_ok=True)
    sheet=np.full((600,1280,3),255,np.uint8)
    for i,r in enumerate(selected):
        im=cv2.imread(str(ROOT/r['origem']));h,w=im.shape[:2];x1,y1,x2,y2=r['xyxy']
        crop=im[max(0,y1-30):min(h,y2+30),max(0,x1-30):min(w,x2+30)]
        r.update(id=f'S{i+1:02d}',arquivo=f'recortes/S{i+1:02d}.jpg',
                 arquivo_nucleo=f'recortes/S{i+1:02d}_nucleo.jpg',classe_revisada='',aprovado=False)
        cv2.imwrite(str(OUT/r['arquivo']),crop)
        cv2.imwrite(str(OUT/r['arquivo_nucleo']),im[y1:y2,x1:x2])
        hh,ww=crop.shape[:2];s=min(310/ww,255/hh);thumb=cv2.resize(crop,(round(ww*s),round(hh*s)))
        x=(i%4)*320;y=(i//4)*300;sheet[y:y+thumb.shape[0],x:x+thumb.shape[1]]=thumb
        cv2.putText(sheet,r['id'],(x+8,y+285),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
    cols=['id','arquivo','arquivo_nucleo','origem','rotulo','conjunto','dominio','grupo','xyxy','contraste','sha256','classe_revisada','aprovado']
    pd.DataFrame(selected,columns=cols).to_csv(OUT/'candidatos.csv',index=False)
    cv2.imwrite(str(OUT/'contato.jpg'),sheet)
    (OUT/'protocolo.json').write_text(json.dumps(dict(imagens_examinadas=len(files),
        candidatos=len(selected),detector_executado=False,treino_executado=False,aprovados=0,
        criterio='Componentes escuros compactos perto de caixas existentes, fora delas. '
        'Revisar recorte E contexto: excluir bovinos escuros, partes e sombras misturadas a animais.',
        limite='Treino aereo externo em pasto; nao e confinamento brasileiro. Nenhuma imagem de val/test ou Campanario.'),indent=2),encoding='utf-8')
    print(json.dumps(dict(examinadas=len(files),candidatos=len(selected),aprovados=0)))


if __name__=='__main__':main()
