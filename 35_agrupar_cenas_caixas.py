"""Fecha a auditoria geometrica de cameras, zooms e espelhamentos antes do treino."""
from pathlib import Path
import json,importlib.util
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;BASE=ROOT/'data/detectors_cows'
spec=importlib.util.spec_from_file_location('prep',ROOT/'31_preparar_caixas_publicadas.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)


def main():
    cv2.setNumThreads(2);df=pd.read_csv(BASE/'divisao_final_caixas.csv');n=len(df)
    orb=cv2.ORB_create(nfeatures=700);matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    desc=[];pts=[];hashes=[];areas=[]
    def caracteristicas(path):
        im=cv2.imread(str(path));h,w=im.shape[:2];im=cv2.resize(im,(640,round(h*640/w)))
        resultados=[]
        for foto in (im,im[:,::-1].copy()):
            kp,ds=orb.detectAndCompute(cv2.cvtColor(foto,cv2.COLOR_BGR2GRAY),None)
            resultados.append((ds,np.array([k.pt for k in kp],np.float32),prep.fp(foto)[0]))
        return resultados,im.shape[0]*im.shape[1]
    for k,r in enumerate(df.itertuples(),1):
        chars,area=caracteristicas(r.img);desc.append([a[0] for a in chars]);pts.append([a[1] for a in chars]);hashes.append([a[2] for a in chars]);areas.append(area)
        if k%1000==0:print('Descritores',k,flush=True)
    hashes=np.array(hashes,np.uint64);pop=np.array([i.bit_count() for i in range(256)],np.uint8)
    def distancia(a,b):
        v=np.bitwise_xor(a,b);return pop[v.view(np.uint8).reshape(v.shape+(8,))].sum(-1)
    def corresponde(a,b,pa,pb,area):
        if a is None or b is None:return 0
        pairs=matcher.knnMatch(a,b,k=2);good=[x for pair in pairs if len(pair)==2 for x,y in [pair] if x.distance<.75*y.distance]
        if len(good)<28:return 0
        p=np.float32([pa[m.queryIdx] for m in good]);q=np.float32([pb[m.trainIdx] for m in good]);H,mask=cv2.findHomography(p,q,cv2.RANSAC,3.)
        if H is None:return 0
        ok=mask.ravel().astype(bool);inside=int(ok.sum());cover=float(np.ptp(p[ok],axis=0).prod()) if inside else 0
        return inside if inside>=24 and inside/len(good)>=.45 and cover>=area*.08 else 0
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[b]=a
    for _,g in df.groupby('grupo'):
        ids=g.index.tolist()
        for j in ids[1:]:union(ids[0],j)
    pares=set();links=[]
    for i in range(n):
        ds=distancia(hashes[:,0],hashes[i,0]);fs=distancia(hashes[:,1],hashes[i,0]);d=np.minimum(ds,fs)
        for j in np.argsort(d)[:17]:
            j=int(j);key=tuple(sorted((i,j)))
            if i==j or key in pares or find(i)==find(j):continue
            pares.add(key);flip=int(fs[j]<ds[j]);m=corresponde(desc[i][0],desc[j][flip],pts[i][0],pts[j][flip],areas[i])
            if m:union(i,j);links.append(dict(i=i,j=j,inliers=m))
        if (i+1)%500==0:print('Pares',i+1,'vinculos',len(links),flush=True)
    # Fotos brasileiras reservadas recebem tambem verificacao geometrica de recortes.
    refs=list((ROOT/'data/teste_brasil').glob('*.jpg'))+list((ROOT/'data/brasil_lote_01/images').glob('*.jpg'))+list((ROOT/'data/brasil_lote_02/images').glob('*.jpg'))
    rchars=[caracteristicas(p)[0] for p in refs];rh=np.array([[c[0][2],c[1][2]] for c in rchars],np.uint64);proibidos=set();excl=[]
    for i in range(n):
        ds=distancia(rh[:,0],hashes[i,0]);fs=distancia(rh[:,1],hashes[i,0]);d=np.minimum(ds,fs)
        for j in np.argsort(d)[:3]:
            j=int(j);flip=int(fs[j]<ds[j]);c=rchars[j][flip]
            if corresponde(desc[i][0],c[0],pts[i][0],c[1],areas[i]):
                proibidos.add(find(i));excl.append(dict(nova=df.iloc[i].img,anterior=str(refs[j])));break
    df['grupo']=[f'geo_{find(i):05d}' for i in range(n)];df=df[[find(i) not in proibidos for i in range(n)]].copy()
    groups=df.groupby('grupo').agg(n=('stem','size'),maximo=('n_bovinos','max'));rng=np.random.default_rng(20261116)
    order=groups.assign(sorteio=rng.random(len(groups))).sort_values(['n','sorteio'],ascending=[False,True])
    totals={'train':0,'val':0,'test':0};targets={'train':.65*len(df),'val':.15*len(df),'test':.2*len(df)};assign={}
    for g,row in order.iterrows():
        s=min(totals,key=lambda s:totals[s]/targets[s]);assign[g]=s;totals[s]+=int(row['n'])
    df['conjunto']=df.grupo.map(assign)
    assert df.groupby('grupo').conjunto.nunique().max()==1
    destino=BASE/'divisao_geometrica.csv'
    if destino.exists():
        old=pd.read_csv(destino);assert dict(zip(old.stem,old.conjunto))==dict(zip(df.stem,df.conjunto))
    else:df.to_csv(destino,index=False)
    resumo={s:dict(imagens=int((df.conjunto==s).sum()),grupos=int(df[df.conjunto==s].grupo.nunique()),caixas=int(df[df.conjunto==s].n_bovinos.sum()),maximo=int(df[df.conjunto==s].n_bovinos.max())) for s in totals}
    (BASE/'auditoria_geometrica.json').write_text(json.dumps(dict(resumo=resumo,vinculos=links,exclusoes=excl,refs_brasileiras=len(refs),limitacao='Correspondencia geometrica de fundos nao comprova fazendas independentes. A auditoria visual detectou repeticoes na divisao preliminar; nenhum treino foi concluido com aquela divisao.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(resumo,indent=2),flush=True)


if __name__=='__main__':main()
