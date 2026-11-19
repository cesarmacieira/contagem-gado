"""Une familias nominais de origem alem das correspondencias geometricas."""
from pathlib import Path
import re,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;BASE=ROOT/'data/detectors_cows'


def linhagem(nome):
    nome=re.sub(r'_(?:jpg|png|jpeg)$','',str(nome),flags=re.I)
    return re.sub(r'[-_]\d+$','',nome)


def main():
    df=pd.read_csv(BASE/'divisao_geometrica.csv');n=len(df);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[b]=a
    df['linhagem']=df.familia.map(linhagem)
    for col in ('grupo','linhagem'):
        for _,r in df.groupby(col):
            ids=r.index.tolist()
            for i in ids[1:]:union(ids[0],i)
    df['grupo']=[f'linhagem_{find(i):05d}' for i in range(n)]
    groups=df.groupby('grupo').agg(n=('stem','size'),maximo=('n_bovinos','max'))
    rng=np.random.default_rng(20261116);order=groups.assign(sorteio=rng.random(len(groups))).sort_values(['n','sorteio'],ascending=[False,True])
    totals={'train':0,'val':0,'test':0};targets={'train':.65*n,'val':.15*n,'test':.2*n};assign={}
    for g,row in order.iterrows():
        s=min(totals,key=lambda s:totals[s]/targets[s]);assign[g]=s;totals[s]+=int(row['n'])
    df['conjunto']=df.grupo.map(assign)
    assert df.groupby('grupo').conjunto.nunique().max()==1
    assert df.groupby('linhagem').conjunto.nunique().max()==1
    destino=BASE/'divisao_linhagens.csv'
    if destino.exists():
        anterior=pd.read_csv(destino);assert dict(zip(anterior.stem,anterior.conjunto))==dict(zip(df.stem,df.conjunto))
    else:df.to_csv(destino,index=False)
    resumo={s:dict(imagens=int((df.conjunto==s).sum()),grupos=int(df[df.conjunto==s].grupo.nunique()),caixas=int(df[df.conjunto==s].n_bovinos.sum()),maximo=int(df[df.conjunto==s].n_bovinos.max())) for s in totals}
    (BASE/'auditoria_linhagens.json').write_text(json.dumps(dict(resumo=resumo,criterio='Prefixos nominais como simmental_new1 e guzerat_new3 ficam inteiros em um conjunto, mesmo quando mudam camera, zoom ou aumentos. Sem certificado de fazendas.',revisao_antes_de_avaliar_teste=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(resumo,indent=2))


if __name__=='__main__':main()
