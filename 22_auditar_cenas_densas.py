"""Une cameras/quadros com fundo geometricamente correspondente antes de treinar."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/denso_cattle'


def main():
    df=pd.read_csv(BASE/'divisao_congelada.csv');n=len(df);orb=cv2.ORB_create(nfeatures=650)
    desc=[];coords=[];hashes=[]
    for r in df.itertuples():
        im=cv2.imread(r.img,cv2.IMREAD_GRAYSCALE);h,w=im.shape
        im=cv2.resize(im,(640,round(h*640/w)))
        kp,ds=orb.detectAndCompute(im,None);desc.append(ds);coords.append(np.array([k.pt for k in kp],np.float32))
        a=cv2.dct(cv2.resize(im,(32,32)).astype(np.float32))[:8,:8];a[0,0]=0;hashes.append((a>np.median(a)).ravel())
    hashes=np.array(hashes);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[b]=a
    for _,rows in df.groupby('grupo'):
        ix=rows.index.tolist()
        for i in ix[1:]:union(ix[0],i)
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING);pares=set();links=[]
    for i in range(n):
        dist=np.not_equal(hashes,hashes[i]).sum(1)
        for j in np.argsort(dist)[1:17]:
            j=int(j);key=tuple(sorted((i,j)))
            if key in pares or find(i)==find(j):continue
            pares.add(key)
            if desc[i] is None or desc[j] is None:continue
            matches=matcher.knnMatch(desc[i],desc[j],k=2)
            good=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance<.72*b.distance]
            if len(good)<28:continue
            a=np.array([coords[i][m.queryIdx] for m in good]);b=np.array([coords[j][m.trainIdx] for m in good])
            H,mask=cv2.findHomography(a,b,cv2.RANSAC,3.0)
            if H is None:continue
            ok=mask.ravel().astype(bool);inliers=int(ok.sum())
            area=np.ptp(a[ok],axis=0).prod() if ok.any() else 0
            if inliers>=24 and inliers/len(good)>=.45 and area>=640*240*.12:
                union(i,j);links.append(dict(i=i,j=j,inliers=inliers))
    df['grupo']= [f'fundo_{find(i):04d}' for i in range(n)]
    groups=df.groupby('grupo').agg(n=('stem','size'),maximo=('n_bovinos','max'))
    rng=np.random.default_rng(20261116);order=groups.assign(sorteio=rng.random(len(groups))).sort_values(['n','sorteio'],ascending=[False,True])
    assert len(groups)>=8, f'Apenas {len(groups)} cenas: insuficiente para avaliacao proposta'
    totals={'train':0,'val':0,'test':0};target={'train':.65*n,'val':.15*n,'test':.2*n};assign={}
    for g,row in order.iterrows():
        s=min(totals,key=lambda s:totals[s]/target[s]);assign[g]=s;totals[s]+=int(row['n'])
    df['conjunto']=df.grupo.map(assign)
    path=BASE/'divisao_cenas_v2.csv'
    if path.exists():
        old=pd.read_csv(path);assert dict(zip(old.stem,old.conjunto))==dict(zip(df.stem,df.conjunto))
    else:df.to_csv(path,index=False)
    resumo=dict(grupos=len(groups),correspondencias=links,maiores=groups.n.nlargest(12).to_dict(),divisao={s:dict(imagens=int((df.conjunto==s).sum()),grupos=int(df[df.conjunto==s].grupo.nunique()),animais=int(df[df.conjunto==s].n_bovinos.sum()),maximo=int(df[df.conjunto==s].n_bovinos.max())) for s in totals},limitacao='Agrupamento geometrico conservador, sem identificacao de fazenda certificada. Nao prova ausencia de mesma fazenda em conjuntos distintos.')
    (BASE/'auditoria_cenas.json').write_text(json.dumps(resumo,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in resumo.items() if k!='correspondencias'},indent=2),flush=True)


if __name__=='__main__':main()
