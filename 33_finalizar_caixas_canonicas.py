"""Retira letterbox artificial e repete a auditoria antes do treino real."""
from pathlib import Path
import json,hashlib,importlib.util
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;BASE=ROOT/'data/detectors_cows'
spec=importlib.util.spec_from_file_location('prep_caixas',ROOT/'31_preparar_caixas_publicadas.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)


def conteudo(im):
    h,w=im.shape[:2];g=cv2.cvtColor(cv2.resize(im,(256,256)),cv2.COLOR_BGR2GRAY);g=cv2.medianBlur(g,3)
    ys=np.where((g>12).mean(1)>.08)[0];xs=np.where((g>12).mean(0)>.08)[0]
    if not len(ys) or not len(xs):return 0,0,w,h
    x1=max(0,int(xs.min()*w/256)-3);x2=min(w,int((xs.max()+1)*w/256)+3)
    y1=max(0,int(ys.min()*h/256)-3);y2=min(h,int((ys.max()+1)*h/256)+3)
    if x2-x1<w*.1 or y2-y1<h*.1:return 0,0,w,h
    return x1,y1,x2,y2


def main():
    cv2.setNumThreads(2);src=pd.read_csv(BASE/'divisao_congelada.csv')
    for d in ('images','labels'):(BASE/'canonico'/d).mkdir(exist_ok=True,parents=True)
    z=np.load(BASE/'refs_hashes.npz',allow_pickle=False);refs=z['hashes'];names=z['nomes']
    pop=np.array([int(i).bit_count() for i in range(256)],np.uint8)
    def dist(a,b):
        v=np.bitwise_xor(a,np.uint64(b));return pop[v.view(np.uint8).reshape(v.shape+(8,))].sum(-1)
    rows=[];fotos=[];fps=[];excl=[]
    for k,r in enumerate(src.itertuples(),1):
        im=cv2.imread(r.img);h,w=im.shape[:2];x1,y1,x2,y2=conteudo(im);crop=im[y1:y2,x1:x2].copy();ch,cw=crop.shape[:2]
        # Mediana tira ruido de aumento somente da impressao digital.
        limpo=cv2.medianBlur(crop,3);p,d=prep.fp(limpo);pf,df=prep.fp(limpo[:,::-1])
        ix=np.where(((dist(refs[:,0],p)<=8)&(dist(refs[:,1],d)<=8))|((dist(refs[:,0],pf)<=8)&(dist(refs[:,1],df)<=8)))[0]
        if len(ix):excl.append(dict(img=r.img,referencia=str(names[ix[0]])));continue
        linhas=[]
        for l in prep.ler_rotulo(Path(r.rot)):
            _,xx,yy,bw,bh=map(float,l.split());a=max(0,(xx-bw/2)*w-x1);b=max(0,(yy-bh/2)*h-y1)
            c=min(cw,(xx+bw/2)*w-x1);d2=min(ch,(yy+bh/2)*h-y1)
            if c<=a or d2<=b:raise ValueError(f'Caixa fora do conteudo: {r.stem}')
            linhas.append('0 '+' '.join(f'{v:.8f}' for v in ((a+c)/2/cw,(b+d2)/2/ch,(c-a)/cw,(d2-b)/ch)))
        img=BASE/'canonico/images'/(r.stem+'.jpg');rot=BASE/'canonico/labels'/(r.stem+'.txt')
        cv2.imwrite(str(img),crop,[cv2.IMWRITE_JPEG_QUALITY,96]);rot.write_text('\n'.join(linhas)+ ('\n' if linhas else ''),encoding='utf-8')
        rows.append(dict(img=str(img.resolve()),rot=str(rot.resolve()),stem=r.stem,familia=r.familia,n_bovinos=len(linhas),grupo_preliminar=r.grupo,origem_img=r.img,recorte_xyxy=json.dumps([x1,y1,x2,y2]),largura=cw,altura=ch,sha256=hashlib.sha256(img.read_bytes()).hexdigest()))
        fotos.append(cv2.resize(crop,(320,160)));fps.append((p,d))
        if k%500==0:print(k,'excluidas',len(excl),flush=True)
    df=pd.DataFrame(rows);n=len(df);fps=np.array(fps,np.uint64);parents=list(range(n));seen={}
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parents[b]=a
    for i,r in df.iterrows():
        for key in [('grupo',r.grupo_preliminar),('sha',r.sha256)]:
            if key in seen:union(i,seen[key])
            seen[key]=i
    for i in range(n):
        p=dist(fps[i+1:,0],fps[i,0]);d=dist(fps[i+1:,1],fps[i,1])
        for off in np.where((p<=10)&(d<=12))[0]:union(i,i+1+int(off))
    df['grupo']=[f'canonico_{find(i):05d}' for i in range(n)]
    groups=df.groupby('grupo').agg(n=('stem','size'),maximo=('n_bovinos','max'))
    rng=np.random.default_rng(20261116);order=groups.assign(sorteio=rng.random(len(groups))).sort_values(['n','sorteio'],ascending=[False,True])
    totals={'train':0,'val':0,'test':0};target={'train':.65*n,'val':.15*n,'test':.2*n};assigned={}
    for g,row in order.iterrows():
        sp=min(totals,key=lambda s:totals[s]/target[s]);assigned[g]=sp;totals[sp]+=int(row['n'])
    df['conjunto']=df.grupo.map(assigned)
    assert df.groupby('grupo').conjunto.nunique().max()==1
    out=BASE/'divisao_final_caixas.csv'
    if out.exists():
        old=pd.read_csv(out);assert dict(zip(old.stem,old.conjunto))==dict(zip(df.stem,df.conjunto))
    else:df.to_csv(out,index=False)
    resumo={s:dict(imagens=int((df.conjunto==s).sum()),grupos=int(df[df.conjunto==s].grupo.nunique()),caixas=int(df[df.conjunto==s].n_bovinos.sum()),maximo=int(df[df.conjunto==s].n_bovinos.max())) for s in totals}
    (BASE/'auditoria_final.json').write_text(json.dumps(dict(resumo=resumo,excluidas_por_sobreposicao=excl,imagens_sem_bordas_artificiais=len(df),fonte='https://huggingface.co/datasets/AlexeyZhigalov/Detectors',limitacao='Agrupamento conservador por nomes, hashes de conteudo sem bordas e grupos anteriores; nao certifica fazendas distintas. Rotulos importados dos autores, com verificacao estrutural e amostral.'),ensure_ascii=False,indent=2),encoding='utf-8')
    ids=pd.concat([df.nlargest(8,'n_bovinos'),df[df.conjunto=='test'].nlargest(4,'n_bovinos')]).drop_duplicates('stem').index.tolist();canvas=np.full((4*200,3*420,3),245,np.uint8)
    for k,i in enumerate(ids[:12]):
        r=df.loc[i];im=cv2.imread(r.img);h,w=im.shape[:2]
        for l in prep.ler_rotulo(Path(r.rot)):
            _,x,y,bw,bh=map(float,l.split());cv2.rectangle(im,(round((x-bw/2)*w),round((y-bh/2)*h)),(round((x+bw/2)*w),round((y+bh/2)*h)),(0,255,0),2)
        y,x=(k//3)*200,(k%3)*420;canvas[y+25:y+200,x:x+420]=cv2.resize(im,(420,175));cv2.putText(canvas,f'{i} {r.conjunto} n={r.n_bovinos}',(x+3,y+19),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    cv2.imwrite(str(BASE/'contato_final.jpg'),canvas);print(json.dumps(resumo,indent=2),flush=True)


if __name__=='__main__':main()
