"""Ajuste com nucleos de madeira revisados, nunca recortes da validacao."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib,json
import cv2,joblib,numpy as np,pandas as pd,torch
from sklearn.linear_model import LogisticRegression
from verificador_visual import carregar_verificador
from verificador_nucleo import features

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/verificador_madeira_revisado_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    source=ROOT/'data/candidatos_madeira_treino_v1';reviews=json.loads((source/'revisao_visual.json').read_text())
    meta=pd.read_csv(source/'candidatos.csv').set_index('id')
    protocol=dict(C=[.1,1.],confs=[.25,.35,.45,.55],limiares=[.1,.3,.5,.7],peso_novo_negativo=8,
        novos_negativos=[r['id'] for r in reviews],negativos_genericos_110_usados=False,
        negativos_base='Somente base revisada do experimento 93; positivos base e aereos de treino existentes.',
        revisao_sha256=hashlib.sha256((source/'revisao_visual.json').read_bytes()).hexdigest(),
        usa_val_no_treino=False,usa_teste=False,promocao_automatica=False,
        limites='Nucleos de madeira/feno revisados por assistente, sem revisao humana independente; sem negativos automaticos de ausencia de GT. Calibracao conhecida e confirmacao em outros quadros das mesmas fazendas. Nao e teste brasileiro independente.')
    (OUT/'protocolo_antes_treino.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.set_num_threads(4);encoder=carregar_verificador(ROOT/'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth')
    generic=np.load(ROOT/'data/experimentos/verificador_sombras_nucleo_v1/features_treino.npz')
    aerial=np.load(ROOT/'data/experimentos/verificador_aereo_dominio_v1/features_treino.npz')
    pos=aerial['y']==1
    xs=[generic['x'],aerial['x'][pos]];ys=[generic['y'],aerial['y'][pos]]
    origins=[generic['origens'],aerial['origens'][pos]];weights=[np.ones(len(generic['y'])),np.ones(int(pos.sum()))]
    for r in reviews:
        assert r['aprovado_negativo'] and r['sem_bovino_no_nucleo'] and r['conjunto']=='train'
        row=meta.loc[r['id']];path=Path(row.nucleo)
        assert row.conjunto=='train' and 'interim\\train\\' in row.origem
        assert hashlib.sha256(Path(row.origem).read_bytes()).hexdigest()==row.sha256
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['nucleo_sha256']
        im=cv2.imread(str(path));h,w=im.shape[:2];box=np.array([[0,0,w,h]],float)
        for blur in [False,True]:
            xs.append(features(encoder,im,box,blur=blur));ys.append(np.array([0]));origins.append(np.array([r['id']]));weights.append(np.array([8.]))
    x,y=np.concatenate(xs),np.concatenate(ys);sample_weights=np.concatenate(weights)
    for label in [0,1]:sample_weights[y==label]*=len(y)/(2*sample_weights[y==label].sum())
    np.savez_compressed(OUT/'features_treino.npz',x=x,y=y,origens=np.concatenate(origins),pesos=sample_weights)
    models={}
    for c in protocol['C']:
        clf=LogisticRegression(C=c,solver='liblinear',max_iter=2000,random_state=1121).fit(x,y,sample_weight=sample_weights)
        clf.receita_extracao_='retangulo_rgb_cinza_v1';joblib.dump(clf,OUT/(f'clf_C{c}.joblib'));models[c]=clf
    spec=spec_from_file_location('avaliar_madeira',ROOT/'110_verificador_aereo_dominio.py');m=module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT
    m.avaliar_modelos(encoder,models,protocol,y)


if __name__=='__main__':main()
