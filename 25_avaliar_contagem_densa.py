"""Escolha congelada na validacao; teste reservado so depois dessa escolha."""
from pathlib import Path
import json,hashlib
import joblib
import cv2
import numpy as np
import pandas as pd
import torch
import denso,densidade
from avaliacao import metricas_contagem

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/denso_cattle';OUT=ROOT/'data/experimentos/denso_avaliacao_v1'


def medidas(df,p):
    p=np.maximum(0,np.rint(p));m=metricas_contagem(df.n_bovinos,p)
    err=pd.DataFrame(dict(g=df.grupo,e=np.abs(df.n_bovinos.to_numpy()-p)))
    m['MAE_macro_grupos']=float(err.groupby('g').e.mean().mean());return m


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultados.json').exists():print((OUT/'resultados.json').read_text());return
    regpath=ROOT/'data/experimentos/denso_v1';netpath=ROOT/'data/experimentos/denso_densidade_v1'
    regcfg=json.loads((regpath/'config_congelada.json').read_text(encoding='utf-8'))
    netcfg=json.loads((netpath/'config_congelada.json').read_text(encoding='utf-8'))
    escolha='regressor' if regcfg['escolha']['score_validacao']<=netcfg['escolha']['score_validacao'] else 'densidade'
    baseline=regcfg['baseline_congelado']['coluna']
    freeze=dict(escolha=escolha,baseline=baseline,regressor_validacao=regcfg['escolha'],densidade_validacao=netcfg['escolha'],criterio='Menor media do MAE por imagem e MAE macro por grupo na validacao. Nenhuma escolha olhando teste.')
    (OUT/'escolha_antes_teste.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    print('Escolha congelada:',escolha,flush=True)
    tab=pd.read_csv(BASE/'atributos.csv').set_index('stem')
    catalogo=pd.read_csv(regpath/'catalogo_congelado.csv');test=catalogo[catalogo.conjunto=='test'].copy().reset_index(drop=True)
    # Sempre o conjunto estrito com grupo denso 75-100 reservado antes de treinar.
    assert len(test)==181 and test.n_bovinos.max()>=75
    pacote=joblib.load(regpath/'regressor.joblib')
    X=tab.loc[test.stem,pacote['colunas']].to_numpy()
    preg=np.maximum(0,np.rint(pacote['regressor'].predict(X)))
    torch.set_num_threads(4);rede=densidade.carregar(netpath,'cuda');pnet=[]
    for k,r in enumerate(test.itertuples(),1):
        im=cv2.imread(r.img);n,_=densidade.prever(rede,im,'cuda');pnet.append(max(0,int(np.rint(n))))
        if k%50==0:print('Teste',k,flush=True)
    pb=tab.loc[test.stem,baseline].to_numpy();pnet=np.array(pnet)
    preds=dict(baseline=pb,regressor=preg,densidade=pnet);model=preds[escolha]
    metrics={n:medidas(test,p) for n,p in preds.items()}
    linhas=test[['stem','img','grupo','n_bovinos']].copy()
    for n,p in preds.items():linhas[n]=p.astype(int)
    linhas.to_csv(OUT/'contagem_teste.csv',index=False)
    grupos=linhas.groupby('grupo').apply(lambda g:pd.Series(dict(base=float(np.abs(g.baseline-g.n_bovinos).mean()),novo=float(np.abs(g[escolha]-g.n_bovinos).mean()))),include_groups=False)
    dif=(grupos.novo-grupos.base).to_numpy();rng=np.random.default_rng(42)
    boots=np.mean(rng.choice(dif,(3000,len(dif)),replace=True),axis=1);ci=list(map(float,np.quantile(boots,[.025,.975])))
    faixas=[]
    for label,lo,hi in [('0-9',0,9),('10-19',10,19),('20-49',20,49),('50+',50,99999)]:
        mask=test.n_bovinos.between(lo,hi).to_numpy()
        if mask.any():faixas.append(dict(faixa=label,imagens=int(mask.sum()),baseline=medidas(test[mask],pb[mask]),novo=medidas(test[mask],model[mask])))
    pd.DataFrame([dict(faixa=f['faixa'],imagens=f['imagens'],MAE_base=f['baseline']['MAE'],MAE_novo=f['novo']['MAE']) for f in faixas]).to_csv(OUT/'faixas.csv',index=False)
    denso50=next((f for f in faixas if f['faixa']=='50+'),None)
    passou=metrics[escolha]['MAE']<.9*metrics['baseline']['MAE'] and metrics[escolha]['MAE_macro_grupos']<metrics['baseline']['MAE_macro_grupos'] and denso50 is not None and denso50['novo']['MAE']<denso50['baseline']['MAE']
    result=dict(escolha_congelada=freeze,teste=metrics,faixas=faixas,ic95_delta_MAE_macro_grupos=ci,passou_benchmark=bool(passou),test_imagens=len(test),test_grupos=int(test.grupo.nunique()),test_bovinos=int(test.n_bovinos.sum()),test_maximo=int(test.n_bovinos.max()),validacao_brasileira_densa=False,limitacao='Teste com pontos dos autores na China e grupos geometricos, sem certificacao de fazendas independentes. Nao demonstra ganho brasileiro. Estimativa nao corresponde ao numero de caixas.')
    (OUT/'resultados.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    # Exemplos escolhidos por densidade/grupo, nunca pela qualidade do resultado.
    pontos=json.loads((BASE/'pontos.json').read_text());(OUT/'exemplos').mkdir(exist_ok=True)
    ix=list(test.nlargest(6,'n_bovinos').index)+list(test.groupby('grupo').head(1).index[:6])
    for i in dict.fromkeys(ix):
        r=test.loc[i];im=cv2.imread(r.img);h,w=im.shape[:2]
        for x,y in pontos[r.stem]:cv2.circle(im,(round(x),round(y)),max(3,round(w/300)),(0,255,0),-1)
        im=cv2.resize(im,(960,round(h*960/w)));header=np.full((64,960,3),255,np.uint8)
        cv2.putText(header,f"Gabarito {r.n_bovinos} | Detector {int(pb[i])} | Estimativa {int(model[i])}",(12,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
        cv2.imwrite(str(OUT/'exemplos'/(r.stem+'.jpg')),np.vstack([header,im]))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
