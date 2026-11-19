"""Cache de inferencia separado do gabarito; nenhum pseudo-rotulo e criado."""
from pathlib import Path
import json,hashlib
import cv2
import pandas as pd
from ultralytics import YOLO
import denso

ROOT=Path(__file__).resolve().parent


def main():
    base=ROOT/'data/denso_cattle';df=pd.read_csv(base/'divisao_congelada.csv')
    df['fonte']='cattle_pontos';df['arquivo']=df.stem+'.jpg'
    brasil=pd.read_csv(ROOT/'data/brasil_dataset_v2/manifesto.csv')
    # As fotos brasileiras de teste permanecem fora da escolha do modelo novo.
    brasil=brasil[brasil.conjunto.isin(['train','val'])].copy()
    brasil['img']=brasil.arquivo.map(lambda x:str(ROOT/'data/brasil_dataset_v2/images'/x))
    brasil['stem']='br_'+brasil.arquivo.map(lambda x:Path(x).stem);brasil['fonte']='brasil_caixas'
    df=pd.concat([df,brasil],ignore_index=True)
    pesos=ROOT/'data/processed/modelo_gado.pt';digest=hashlib.sha256(pesos.read_bytes()).hexdigest()
    pasta=base/f'cache_atributos_{digest[:12]}_v{denso.VERSAO}';pasta.mkdir(exist_ok=True)
    modelo=YOLO(str(pesos));rows=[]
    for k,r in enumerate(df.itertuples(),1):
        cache=pasta/(r.stem+'.json')
        if cache.exists():values=json.loads(cache.read_text(encoding='utf-8'))
        else:
            im=cv2.imread(str(r.img));assert im is not None
            features,contagens,_=denso.extrair(modelo,im)
            values=dict(features=features,contagens=contagens,sha256_imagem=hashlib.sha256(Path(r.img).read_bytes()).hexdigest())
            cache.write_text(json.dumps(values),encoding='utf-8')
        rows.append(dict(stem=r.stem,arquivo=r.arquivo,img=r.img,grupo=r.grupo,conjunto=r.conjunto,fonte=r.fonte,n_bovinos=int(r.n_bovinos),**values['features']))
        if k%25==0:print(f'{k}/{len(df)}',flush=True)
    pd.DataFrame(rows).to_csv(base/'atributos.csv',index=False)
    print('Concluido',len(rows),flush=True)


if __name__=='__main__':main()
