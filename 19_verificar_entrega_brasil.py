"""Verifica integridade consolidada e executa o caminho real de inferencia do app."""
from pathlib import Path
import ast
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import contagem as ct
from dados_locais import carregar_revisados

ROOT=Path(__file__).resolve().parent


def main():
    base=ROOT/'data/brasil_dataset_v2';df=carregar_revisados(ROOT)
    assert len(df)==44 and df.n_vacas.sum()==152
    assert df.groupby('grupo').conjunto.nunique().max()==1
    coco=json.loads((base/'anotacoes_coco.json').read_text(encoding='utf-8'))
    for r in df.itertuples():
        im=cv2.imread(str(r.img));assert im is not None
        h,w=im.shape[:2];b=ct.yolo_para_pixels(r.rot.read_text().splitlines(),w,h)
        c=next(i for i in coco['images'] if i['file_name']==r.img.name)
        assert len(b)==sum(a['image_id']==c['id'] for a in coco['annotations'])
        assert (b[:,2:]>b[:,:2]).all() and (b>=-.001).all()
        assert (b[:,[0,2]]<=w+.001).all() and (b[:,[1,3]]<=h+.001).all()
    tree=ast.parse((ROOT/'app.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='detectar_bytes');fn.decorator_list=[]
    ns=dict(np=np,cv2=cv2,ct=ct)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'app.py','exec'),ns)
    perfil=json.loads((ROOT/'data/experimentos/perfil_brasil_app.json').read_text(encoding='utf-8'))
    peso=ROOT/perfil['pesos'];digest=hashlib.sha256(peso.read_bytes()).hexdigest();assert digest==perfil['sha256']
    assert digest=='afd71a25b852f1c53bf5a7649f60326302f364d464682a86ad9aca66df21564e'
    cfg=perfil['config'];modelo=YOLO(str(peso))
    comparacao=pd.read_csv(ROOT/'data/experimentos/comparacao_calibracao_brasil/contagem_por_imagem.csv')
    esperados=comparacao[comparacao.perfil=='calibrado_validacao'].set_index('arquivo').previsto.to_dict()
    linhas=[]
    for r in df[df.conjunto=='test'].itertuples():
        b,s,shape,modo=ns['detectar_bytes'](r.img.read_bytes(),'inteira',1024,.2,cfg['imgsz'],modelo,peso.stat().st_mtime_ns,cfg['iou'],.5,1000,ct.VERSAO_INFERENCIA,False)
        n=len(ct.filtrar(b,s,cfg['conf'])[0]);assert n==esperados[r.img.name],(r.img.name,n,esperados[r.img.name])
        linhas.append(dict(arquivo=r.img.name,real=int(r.n_vacas),previsto=n))
    try: ns['detectar_bytes'](b'invalido','inteira',1024,.2,1024,modelo,0)
    except ValueError: pass
    else: raise AssertionError('Upload invalido foi aceito')
    res=dict(dataset_imagens=len(df),dataset_caixas=int(df.n_vacas.sum()),paridade_app=len(linhas),contagens_exatas=sum(r['real']==r['previsto'] for r in linhas),pesos_originais_preservados=True,upload_invalido_rejeitado=True,teste_visual_interface='AppTest interrompido por travamento; verificada a funcao real de inferencia, sem afirmar teste visual completo',imagens=linhas)
    out=ROOT/'data/experimentos/comparacao_calibracao_brasil/verificacao_entrega.json'
    out.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(res,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
