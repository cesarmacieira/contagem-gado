"""Importa pontos publicados; separa parentes de recortes e quase duplicatas."""
from pathlib import Path
import json,re,hashlib
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/denso_cattle'


def main():
    source=BASE/'original/Cattle-dataset-main/data'
    items=[];pointsets={};thumbs=[];ph=[];dh=[]
    for imfile in sorted(source.glob('*/images/*.jpg')):
        ann=imfile.parent.parent/'ground_truth'/(imfile.stem+'.mat')
        im=cv2.imread(str(imfile));assert im is not None
        h,w=im.shape[:2];pts=np.asarray(loadmat(ann)['annPoints'],float).reshape(-1,2)
        if not np.isfinite(pts).all() or ((pts<0).any()) or (pts[:,0]>w+1).any() or (pts[:,1]>h+1).any():
            raise ValueError(f'Pontos invalidos: {imfile}')
        nome=imfile.stem;assert nome not in pointsets
        pointsets[nome]=pts.tolist()
        gray=cv2.cvtColor(cv2.resize(im,(128,96)),cv2.COLOR_BGR2GRAY)
        small=cv2.resize(gray,(9,8));dh.append(np.packbits(small[:,1:]>small[:,:-1]))
        spec=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8];spec[0,0]=0
        ph.append(np.packbits(spec>np.median(spec)))
        thumbs.append(cv2.resize(im,(192,128)))
        items.append(dict(stem=nome,img=str(imfile.resolve()),rot=str(ann.resolve()),largura=w,altura=h,n_bovinos=len(pts),split_autores=imfile.parent.parent.name,familia=re.sub(r'p\d+$','',nome),sha256=hashlib.sha256(imfile.read_bytes()).hexdigest()))
    n=len(items);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[b]=a
    seen={}
    for i,r in enumerate(items):
        for key in [('familia',r['familia']),('hash',r['sha256'])]:
            if key in seen:union(i,seen[key])
            seen[key]=i
    # Dois hashes para unir quadros com enquadramento praticamente igual.
    ph=np.array(ph);dh=np.array(dh)
    for i in range(n):
        p=np.unpackbits(np.bitwise_xor(ph[i+1:],ph[i]),axis=1).sum(1)
        d=np.unpackbits(np.bitwise_xor(dh[i+1:],dh[i]),axis=1).sum(1)
        for off in np.where((p<=10)&(d<=12))[0]:union(i,i+1+int(off))
    for i,r in enumerate(items):r['grupo']=f'cena_{find(i):04d}'
    df=pd.DataFrame(items);groups=df.groupby('grupo').agg(n=('stem','size'),animais=('n_bovinos','sum'),maximo=('n_bovinos','max'))
    rng=np.random.default_rng(20261116)
    # Grupos grandes primeiro, balanceamento de tamanho sem olhar desempenho.
    ordem=groups.assign(sorteio=rng.random(len(groups))).sort_values(['n','sorteio'],ascending=[False,True])
    assigned={};tot={'train':0,'val':0,'test':0};targets={'train':.65*n,'val':.15*n,'test':.20*n}
    for g,row in ordem.iterrows():
        sp=min(tot,key=lambda s:tot[s]/targets[s]);assigned[g]=sp;tot[sp]+=int(row['n'])
    df['conjunto']=df.grupo.map(assigned)
    assert df.groupby('grupo').conjunto.nunique().max()==1
    assert len(groups)>=8, 'Poucos grupos independentes para uma divisao confiavel'
    frozen=BASE/'divisao_congelada.csv'
    if frozen.exists():
        old=pd.read_csv(frozen);assert dict(zip(old.stem,old.conjunto))==dict(zip(df.stem,df.conjunto))
    else:df.to_csv(frozen,index=False)
    (BASE/'pontos.json').write_text(json.dumps(pointsets,ensure_ascii=False),encoding='utf-8')
    df[['stem','n_bovinos','grupo','conjunto','split_autores']].to_csv(BASE/'gabarito.csv',index=False)
    (BASE/'dicionario.json').write_text(json.dumps(dict(classe='bovino',tipo_rotulo='ponto_xy_publicado_pelos_autores',caixas_disponiveis=False,fonte='https://github.com/menjure/Cattle-dataset',citacao='Zhong et al. 2022, Cattle Number Estimation on Smart Pasture Based on Multi-Scale Information Fusion',licenca='Sem arquivo de licenca no repositorio; distribuicao dos autores para pesquisa com citacao. Nao redistribuir imagens como se fossem CC.',limitacao='Contagens conforme os pontos publicados, que podem excluir animais de baias adjacentes parcialmente visiveis. Nao equivale a anotacao completa de caixas.'),ensure_ascii=False,indent=2),encoding='utf-8')
    # Contato de representantes de grupos, mais o grupo denso de cada conjunto.
    reps=df.groupby('grupo').head(1).index.tolist()
    for sp in ('train','val','test'): reps.extend(df[df.conjunto==sp].nlargest(2,'n_bovinos').index.tolist())
    reps=list(dict.fromkeys(reps))
    for pag in range(0,len(reps),24):
        canvas=np.full((6*165,4*320,3),245,np.uint8)
        for k,i in enumerate(reps[pag:pag+24]):
            r=df.loc[i];tile=cv2.resize(thumbs[i],(320,128));y,x=(k//4)*165,(k%4)*320
            canvas[y+32:y+160,x:x+320]=tile
            cv2.putText(canvas,f'{i} {r.grupo} {r.conjunto} n={r.n_bovinos}',(x+2,y+18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        cv2.imwrite(str(BASE/f'contato_{pag//24:02d}.jpg'),canvas)
    resumo={sp:dict(imagens=int((df.conjunto==sp).sum()),grupos=int(df[df.conjunto==sp].grupo.nunique()),animais=int(df[df.conjunto==sp].n_bovinos.sum()),maximo=int(df[df.conjunto==sp].n_bovinos.max())) for sp in tot}
    (BASE/'resumo.json').write_text(json.dumps(resumo,indent=2),encoding='utf-8')
    print(json.dumps(resumo,indent=2));print('GRUPOS',groups.n.sort_values(ascending=False).head(15).to_dict())


if __name__=='__main__':main()
