"""Treina regressores de contagem; escolhe no val e nao acessa resultados do teste."""
from pathlib import Path
import json,hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from avaliacao import metricas_contagem
import denso

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/denso_cattle'
OUT=ROOT/'data/experimentos/denso_v1'


def metricas(df,p):
    p=np.maximum(0,np.rint(p));r=metricas_contagem(df.n_bovinos,p)
    e=pd.DataFrame(dict(g=df.grupo,e=np.abs(p-df.n_bovinos.to_numpy())))
    r['MAE_macro_grupos']=float(e.groupby('g').e.mean().mean())
    r['score_validacao']=(r['MAE']+r['MAE_macro_grupos'])/2
    return r


def preparar_divisao():
    df=pd.read_csv(BASE/'divisao_cenas_v2.csv')
    # Assegurar que o teste represente 75-100 animais ANTES de treinar/medir.
    g=df[df.conjunto=='train'].groupby('grupo').n_bovinos.agg(['size','max'])
    candidatos=g[(g['max']>=75)&(g['max']<=110)].sort_values(['size','max'],ascending=False)
    grupo=candidatos.index[0];df.loc[df.grupo==grupo,'conjunto']='test'
    destino=BASE/'divisao_final_densa.csv'
    if destino.exists():
        anterior=pd.read_csv(destino);assert dict(zip(anterior.stem,anterior.conjunto))==dict(zip(df.stem,df.conjunto))
    else:df.to_csv(destino,index=False)
    assert df.groupby('grupo').conjunto.nunique().max()==1
    assert df[df.conjunto=='test'].n_bovinos.max()>=75
    (BASE/'criterio_divisao_final.json').write_text(json.dumps(dict(grupo_denso_reservado=str(grupo),motivo='Cobrir rebanhos de 75 a 100 animais no teste; escolhido por tamanho do grupo e gabarito antes de treino e sem consultar desempenho',resumo={s:dict(imagens=int((df.conjunto==s).sum()),grupos=int(df[df.conjunto==s].grupo.nunique()),animais=int(df[df.conjunto==s].n_bovinos.sum()),maximo=int(df[df.conjunto==s].n_bovinos.max())) for s in ('train','val','test')}),indent=2),encoding='utf-8')
    return df


def carregar_atributos():
    master=preparar_divisao().set_index('stem')
    df=pd.read_csv(BASE/'atributos.csv')
    for i in df.index[df.fonte=='cattle_pontos']:
        df.loc[i,'grupo']=master.loc[df.loc[i,'stem'],'grupo']
        df.loc[i,'conjunto']=master.loc[df.loc[i,'stem'],'conjunto']
    return df


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'regressor.joblib').exists():raise RuntimeError('Experimento concluido; preserve os resultados')
    df=carregar_atributos();meta=['stem','arquivo','img','grupo','conjunto','fonte','n_bovinos']
    colunas=[c for c in df.columns if c not in meta];train=df[df.conjunto=='train'];val=df[df.conjunto=='val']
    assert not set(train.grupo)&set(val.grupo)
    X=train[colunas].to_numpy();y=train.n_bovinos.to_numpy();V=val[colunas].to_numpy()
    # Grupos grandes nao recebem centenas de vezes o peso de uma cena isolada.
    tamanhos=train.groupby('grupo').stem.transform('size').to_numpy();peso=1/np.sqrt(tamanhos);peso/=peso.mean()
    candidates={}
    for positive in (False,True):
        for alpha in (1.,100.):
            candidates[f'ridge_{positive}_{alpha}']=make_pipeline(StandardScaler(),Ridge(alpha=alpha,positive=positive))
    for leaves in (7,15):
        candidates[f'boost_{leaves}']=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=leaves,min_samples_leaf=12,learning_rate=.05,l2_regularization=10,random_state=42,early_stopping=False)
    candidates['extra_trees']=ExtraTreesRegressor(n_estimators=250,min_samples_leaf=3,max_features=.8,random_state=42,n_jobs=4)
    resultados=[];vencedor=None;melhor=None
    for nome,modelo in candidates.items():
        if nome.startswith('ridge'):modelo.fit(X,y,ridge__sample_weight=peso)
        else:modelo.fit(X,y,sample_weight=peso)
        pred=modelo.predict(V);m=metricas(val,pred);resultados.append(dict(nome=nome,**m))
        print(nome,m['MAE'],m['MAE_macro_grupos'],flush=True)
        if melhor is None or m['score_validacao']<melhor['score_validacao']:
            melhor=dict(nome=nome,**m);vencedor=modelo
    baselines=[]
    for estr in denso.ESTRATEGIAS:
        for conf in denso.CONFIANCAS:
            col=f'{estr}_c{conf:.2f}';m=metricas(val,val[col]);baselines.append(dict(coluna=col,**m))
    baselines.sort(key=lambda r:r['score_validacao'])
    pesos=ROOT/'data/processed/modelo_gado.pt'
    pacote=dict(regressor=vencedor,colunas=colunas,versao=denso.VERSAO,modelo_detector_sha256=hashlib.sha256(pesos.read_bytes()).hexdigest(),faixa_treino=[int(y.min()),int(y.max())],tipo='estimador_contagem_nao_detector_de_caixas')
    joblib.dump(pacote,OUT/'regressor.joblib')
    protocolo=dict(escolha=melhor,baseline_congelado=baselines[0],avaliacao_teste_executada=False,train_imagens=len(train),val_imagens=len(val),test_imagens=int((df.conjunto=='test').sum()),rotulos='Pontos dos autores mais caixas brasileiras revisadas; sem caixas sinteticas',limitacao='Agrupamento por familia, hashes e correspondencias geometricas; nao certifica fazendas independentes. Cenas da China nao comprovam generalizacao brasileira.')
    (OUT/'config_congelada.json').write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(resultados).to_csv(OUT/'comparacao_validacao.csv',index=False)
    pd.DataFrame(baselines).to_csv(OUT/'baselines_validacao.csv',index=False)
    df[meta].to_csv(OUT/'catalogo_congelado.csv',index=False)
    print(json.dumps(protocolo,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
