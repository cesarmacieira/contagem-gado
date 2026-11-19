"""Audita caixas publicadas: remove aumentos repetidos, vazamento e rotulos invalidos."""
from pathlib import Path
import re,json,hashlib
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/detectors_cows';SOURCE=BASE/'dataset_cows'


def fp(im):
    g=cv2.cvtColor(cv2.resize(im,(128,96)),cv2.COLOR_BGR2GRAY)
    a=cv2.dct(cv2.resize(g,(32,32)).astype(np.float32))[:8,:8];a[0,0]=0
    d=cv2.resize(g,(9,8))
    return (int.from_bytes(np.packbits(a>np.median(a)).tobytes(),'big'),int.from_bytes(np.packbits(d[:,1:]>d[:,:-1]).tobytes(),'big'))


def ler_rotulo(p):
    linhas=[];visto=set()
    for l in p.read_text(encoding='utf-8-sig').splitlines():
        if not l.strip():continue
        v=np.array([float(x) for x in l.split()]);assert len(v)==5 and np.isfinite(v).all(),p
        assert v[0]==0 and (v[1:]>=0).all() and (v[1:]<=1).all() and v[3]>0 and v[4]>0,p
        x,y,w,h=v[1:];assert x-w/2>=-.0001 and y-h/2>=-.0001 and x+w/2<=1.0001 and y+h/2<=1.0001,p
        key=tuple(np.round(v,6))
        if key not in visto:linhas.append(' '.join(f'{z:.8f}' for z in v));visto.add(key)
    return linhas


def main():
    cv2.setNumThreads(2);BASE.mkdir(exist_ok=True)
    candidatos={};invalidos=[];total=0
    for p in sorted(SOURCE.glob('*/images/*')):
        if p.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
        rot=p.parent.parent/'labels'/(p.stem+'.txt');total+=1
        try:linhas=ler_rotulo(rot)
        except (AssertionError,ValueError,FileNotFoundError) as e:invalidos.append(dict(img=str(p),erro=str(e)));continue
        familia=re.sub(r'\.rf\.[0-9a-f]+$','',p.stem)
        r=dict(img=str(p.resolve()),rot=str(rot.resolve()),stem=p.stem,familia=familia,n_bovinos=len(linhas),split_autores=p.parent.parent.name)
        if familia not in candidatos or len(linhas)>candidatos[familia]['n_bovinos']:candidatos[familia]=r
    print('Originais/variantes:',total,'familias:',len(candidatos),'invalidos:',len(invalidos),flush=True)
    rows=list(candidatos.values());hashes=[];thumbs=[]
    for r in rows:
        im=cv2.imread(r['img']);assert im is not None
        r['altura'],r['largura']=im.shape[:2];r['sha256']=hashlib.sha256(Path(r['img']).read_bytes()).hexdigest()
        hashes.append([*fp(im),*fp(im[:,::-1])]);thumbs.append(cv2.resize(im,(160,120)))
    # Qualquer imagem de material anterior fica fora do novo benchmark.
    refcache=BASE/'refs_hashes.npz'
    if refcache.exists():
        z=np.load(refcache,allow_pickle=False);refs=z['hashes'];names=z['nomes']
    else:
        paths=list((ROOT/'data/extras').rglob('*.jpg'))+list((ROOT/'data/teste_brasil').glob('*.jpg'))+list((ROOT/'data/brasil_lote_01/images').glob('*.jpg'))+list((ROOT/'data/brasil_lote_02/images').glob('*.jpg'))+list((ROOT/'data/denso_cattle/original').rglob('*.jpg'))
        vals=[];names=[]
        for p in paths:
            im=cv2.imread(str(p),cv2.IMREAD_REDUCED_COLOR_4)
            if im is not None:vals.append(fp(im));names.append(str(p.relative_to(ROOT)))
        refs=np.array(vals,np.uint64);names=np.array(names);np.savez_compressed(refcache,hashes=refs,nomes=names)
    hashes=np.array(hashes,np.uint64);excluidos=[];keep=[];keptthumb=[];keptfp=[]
    pop=np.array([int(i).bit_count() for i in range(256)],np.uint8)
    def distancia(a,b):
        x=np.bitwise_xor(a,b);return pop[x.view(np.uint8).reshape(x.shape+(8,))].sum(-1)
    for i,r in enumerate(rows):
        p=distancia(refs[:,0],hashes[i,0]);d=distancia(refs[:,1],hashes[i,1])
        pf=distancia(refs[:,0],hashes[i,2]);dd=distancia(refs[:,1],hashes[i,3])
        ix=np.where(((p<=6)&(d<=6))|((pf<=6)&(dd<=6)))[0]
        if len(ix):excluidos.append(dict(img=r['img'],referencia=str(names[ix[0]])));continue
        keep.append(r);keptthumb.append(thumbs[i]);keptfp.append(hashes[i,:2])
    print('Sem sobreposicao perceptual com acervo antigo:',len(keep),'excluidas:',len(excluidos),flush=True)
    df=pd.DataFrame(keep);hh=np.array(keptfp,np.uint64);n=len(df);parents=list(range(n))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parents[b]=a
    seen={}
    for i,r in df.iterrows():
        fam=r.familia
        video=re.sub(r'(?:frame|img|image)[_-]?\d+.*$','sequencia',fam,flags=re.I)
        if re.fullmatch(r'\d+_jpg',fam):video='bloco_'+str(int(fam.split('_')[0])//100)
        for key in [('sha',r.sha256),('video',video)]:
            if key in seen:union(i,seen[key])
            seen[key]=i
    for i in range(n):
        p=distancia(hh[i+1:,0],hh[i,0]);d=distancia(hh[i+1:,1],hh[i,1])
        for off in np.where((p<=10)&(d<=12))[0]:union(i,i+1+int(off))
    df['grupo']=[f'publicado_{find(i):05d}' for i in range(n)]
    groups=df.groupby('grupo').agg(n=('stem','size'),maximo=('n_bovinos','max'))
    assert len(groups)>=10
    rng=np.random.default_rng(20261116);ordem=groups.assign(sorteio=rng.random(len(groups))).sort_values(['n','sorteio'],ascending=[False,True])
    totals={'train':0,'val':0,'test':0};targets={'train':.65*n,'val':.15*n,'test':.20*n};sp={}
    for g,row in ordem.iterrows():
        s=min(totals,key=lambda k:totals[k]/targets[k]);sp[g]=s;totals[s]+=int(row['n'])
    df['conjunto']=df.grupo.map(sp)
    assert df.groupby('grupo').conjunto.nunique().max()==1
    destino=BASE/'divisao_congelada.csv'
    if destino.exists():
        old=pd.read_csv(destino);assert dict(zip(old.stem,old.conjunto))==dict(zip(df.stem,df.conjunto))
    else:df.to_csv(destino,index=False)
    resumo={s:dict(imagens=int((df.conjunto==s).sum()),grupos=int(df[df.conjunto==s].grupo.nunique()),caixas=int(df[df.conjunto==s].n_bovinos.sum()),maximo=int(df[df.conjunto==s].n_bovinos.max())) for s in totals}
    (BASE/'auditoria.json').write_text(json.dumps(dict(resumo=resumo,variantes=total,familias=len(rows),invalidos=invalidos,excluidos=excluidos,refs=len(refs),erro_config_original='data.yaml usa a mesma pasta em val e test; nao utilizada',fonte='https://huggingface.co/datasets/AlexeyZhigalov/Detectors',limitacao='Grupos por familia, sequencia nominal e hashes. Nao certifica fazendas distintas.'),ensure_ascii=False,indent=2),encoding='utf-8')
    exemplos=pd.concat([df.nlargest(12,'n_bovinos'),df.groupby('conjunto').head(8)]).drop_duplicates('stem').index.tolist()
    for pagina in range(0,len(exemplos),12):
        canvas=np.full((4*220,3*430,3),245,np.uint8)
        for k,i in enumerate(exemplos[pagina:pagina+12]):
            r=df.loc[i];im=cv2.imread(r.img);h,w=im.shape[:2]
            for l in ler_rotulo(Path(r.rot)):
                _,x,y,bw,bh=map(float,l.split());p1=(round((x-bw/2)*w),round((y-bh/2)*h));p2=(round((x+bw/2)*w),round((y+bh/2)*h));cv2.rectangle(im,p1,p2,(0,255,0),2)
            im=cv2.resize(im,(430,190));y,x=(k//3)*220,(k%3)*430;canvas[y+30:y+220,x:x+430]=im
            cv2.putText(canvas,f'{i} {r.conjunto} n={r.n_bovinos}',(x+4,y+20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        cv2.imwrite(str(BASE/f'contato_{pagina//12:02d}.jpg'),canvas)
    print(json.dumps(resumo,indent=2),flush=True)


if __name__=='__main__':main()
