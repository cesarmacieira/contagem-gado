"""Gabarito e galeria de pontos publicados, sem transforma-los em caixas."""
from pathlib import Path
import csv,json,base64,html
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/denso_cattle'


def main():
    df=pd.read_csv(BASE/'divisao_final_densa.csv');pontos=json.loads((BASE/'pontos.json').read_text())
    df[['stem','n_bovinos','grupo','conjunto','largura','altura','split_autores']].to_csv(BASE/'gabarito_final.csv',index=False)
    with (BASE/'marcacoes_pontos.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        wr=csv.writer(fp);wr.writerow(['imagem','individuo','classe','x','y','grupo','conjunto','anotador'])
        for r in df.itertuples():
            for k,(x,y) in enumerate(pontos[r.stem],1):wr.writerow([r.stem,k,0,x,y,r.grupo,r.conjunto,'autores_do_dataset'])
    selecionados=pd.concat([df.groupby(['conjunto','grupo']).head(1).groupby('conjunto').head(8),df.nlargest(12,'n_bovinos')]).drop_duplicates('stem')
    (BASE/'marcacoes_amostra').mkdir(exist_ok=True);cards=[]
    for r in selecionados.itertuples():
        im=cv2.imread(r.img);h,w=im.shape[:2];scale=800/w;im=cv2.resize(im,(800,round(h*scale)))
        for k,(x,y) in enumerate(pontos[r.stem],1):
            xx,yy=round(x*scale),round(y*scale);cv2.circle(im,(xx,yy),3,(0,255,0),-1)
            cv2.putText(im,str(k),(xx+3,yy-3),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
        cv2.imwrite(str(BASE/'marcacoes_amostra'/(r.stem+'.jpg')),im)
        _,jpg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,85]);data=base64.b64encode(jpg).decode()
        cards.append(f'<article><h2>{html.escape(r.stem)} · {r.conjunto} · {r.n_bovinos} animais</h2><img src="data:image/jpeg;base64,{data}"></article>')
    pagina='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Gabarito de contagem densa</title><style>body{font:16px system-ui;max-width:1400px;margin:auto;padding:25px;background:#f4f6f4}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:18px}article{background:white;border-radius:12px;padding:12px}img{width:100%}h2{font-size:15px}p{line-height:1.6}</style><h1>Contagem densa: pontos do gabarito</h1><p>850 imagens e 18.403 pontos publicados pelos pesquisadores. Os numeros verdes/vermelhos indicam os pontos do gabarito; nao sao previsoes do modelo nem caixas. Esta galeria mostra uma amostra escolhida por grupo e densidade, antes de considerar o desempenho.</p><p>Origem: <a href="https://github.com/menjure/Cattle-dataset">Zhong et al., Cattle-dataset</a>. Imagens da China, nao de rebanhos brasileiros. Licenca de reuso nao informada no repositorio; citacao dos autores preservada. Contagens seguem os pontos publicados: alguns animais parcialmente visiveis em baias adjacentes podem ficar fora da regiao anotada.</p><section>'''+''.join(cards)+'</section></html>'
    (BASE/'galeria_pontos.html').write_text(pagina,encoding='utf-8')
    dic=json.loads((BASE/'dicionario.json').read_text(encoding='utf-8'));dic['licenca']='Nao informada no repositorio; os autores solicitam citacao. Nao atribuir licenca CC inexistente.'
    dic.update(classes={'0':'bovino'},campos_ponto={'x':'pixel horizontal da imagem original','y':'pixel vertical da imagem original'},anotador='autores do conjunto; importacao e verificacao estrutural pelo assistente',pontos=sum(len(v) for v in pontos.values()),imagens=len(df))
    (BASE/'dicionario.json').write_text(json.dumps(dic,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Exportado:',len(df),'imagens',sum(len(v) for v in pontos.values()),'pontos',len(selecionados),'exemplos visuais')


if __name__=='__main__':main()
