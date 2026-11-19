"""Confere paridade do peso exportado no caminho real do app e gera marcacoes."""
from pathlib import Path
import ast,json,hashlib
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import contagem as ct

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'data/experimentos/detector_denso_v3'


def main():
    meta=json.loads((ROOT/'data/modelos/perfil_detector_denso.json').read_text(encoding='utf-8'))
    peso=ROOT/meta['pesos'] if meta['disponivel'] else Path(meta['pesos_checkpoint'])
    assert hashlib.sha256(peso.read_bytes()).hexdigest()==meta.get('sha256',meta['sha256_checkpoint'])
    assert hashlib.sha256((ROOT/'data/processed/modelo_gado.pt').read_bytes()).hexdigest()=='afd71a25b852f1c53bf5a7649f60326302f364d464682a86ad9aca66df21564e'
    tree=ast.parse((ROOT/'app.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='detectar_bytes');fn.decorator_list=[]
    ns=dict(np=np,cv2=cv2,ct=ct);exec(compile(ast.Module(body=[fn],type_ignores=[]),'app.py','exec'),ns)
    cfg=meta['config'];model=YOLO(str(peso))
    df=pd.read_csv(EXP/'avaliacao/contagem_teste.csv');df=df[df.modelo=='ajustado'].sort_values(['real','stem'])
    # Posicoes fixas na distribuicao de quantidade; nao selecionar por acerto.
    amostra=df.iloc[np.linspace(0,len(df)-1,8,dtype=int)]
    out=EXP/'avaliacao/marcacoes';out.mkdir(exist_ok=True);linhas=[]
    catalogo=pd.read_csv(EXP/'teste_congelado.csv').set_index('stem')
    for k,r in enumerate(amostra.itertuples()):
        imgpath=Path(r.img)
        b,s,_,_=ns['detectar_bytes'](imgpath.read_bytes(),'inteira',1024,.2,cfg['imgsz'],model,peso.stat().st_mtime_ns,cfg['iou'],.5,1000,ct.VERSAO_INFERENCIA,False)
        b,s=ct.filtrar(b,s,cfg['conf']);assert len(b)==r.previsto,(r.stem,len(b),r.previsto)
        im=cv2.imread(str(imgpath));h,w=im.shape[:2]
        ref=ct.yolo_para_pixels(Path(catalogo.loc[r.stem,'rot']).read_text().splitlines(),w,h)
        marcado=ct.desenhar(im,b,ref=ref,cor_ref=(0,180,0));cv2.imwrite(str(out/f'{k:02d}.jpg'),marcado)
        linhas.append(dict(stem=r.stem,real=int(r.real),previsto=len(b),marcacao=f'marcacoes/{k:02d}.jpg'))
    res=dict(paridade_app=len(linhas),perfil_liberado=meta['disponivel'],originais_preservados=True,interface_visual_completa_testada=False,imagens=linhas)
    (EXP/'avaliacao/verificacao_app.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
    html='<meta charset="utf-8"><title>Conferencia do detector</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}img{max-width:100%}section{margin:32px 0}</style><h1>Conferencia do detector</h1><p>Verde: gabarito publicado. Roxo: deteccoes. Oito posicoes fixas na distribuicao de quantidades do teste, sem escolher por acerto.</p>'
    for r in linhas:html+=f'<section><h2>Gabarito {r["real"]}, previsto {r["previsto"]}</h2><p>{r["stem"]}</p><img src="{r["marcacao"]}"></section>'
    (EXP/'avaliacao/galeria_conferencia.html').write_text(html,encoding='utf-8')
    print(json.dumps(res,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
