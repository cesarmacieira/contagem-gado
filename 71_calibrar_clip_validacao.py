"""Calibra por positivos de val; compara apenas desenvolvimento/regressao."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas
from verificador_semantico import carregar,margens,manter,POSITIVOS,NEGATIVOS,SHA_PESO
from verificador_visual import carregar_binario,confirmar,mascara_confirmacao

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/verificador_clip_calibrado_v2'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado concluido preservado')
    torch.set_num_threads(4)
    clip=carregar(ROOT/'data/modelos_candidatos/ViT-B-32.pt')
    base=ROOT/'data/experimentos/solo_denso_validacao_v2/validacao'
    referencias=[];calib=[]
    for nome in ('bdn_08_c1.jpg','bdn_08_c2.jpg'):
        imagem=base/'images'/nome;rotulo=base/'labels'/(Path(nome).stem+'.txt')
        im=cv2.imread(str(imagem));h,w=im.shape[:2]
        ref=ct.yolo_para_pixels(rotulo.read_text().splitlines(),w,h)
        mm,_=margens(clip,im,ref)
        calib.extend(mm.min(axis=1).tolist())
        referencias.append(dict(arquivo=nome,sha256=hashlib.sha256(imagem.read_bytes()).hexdigest(),
            sha_rotulo=hashlib.sha256(rotulo.read_bytes()).hexdigest(),n=len(ref),margens=mm.tolist()))
    assert len(calib)==32
    limiar=float(max(0,max(calib)+.001))
    cfg=json.loads((ROOT/'data/modelos/verificador_solo_v1/config.json').read_text())
    protocolo=dict(criterio='Margem minima nao negativa que preserva os 32 bovinos anotados de val, com folga .001.',
        limiar=limiar,calibracao=referencias,positivos=POSITIVOS,negativos=NEGATIVOS,
        modelo_sha256=SHA_PESO,verificador_anterior=cfg,
        limites='Positivos de apenas uma cena. Avaliacao seguinte e regressao conhecida; sem teste independente novo.')
    (OUT/'protocolo_antes_regressao.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    print('Margem calibrada:',limiar,'32 bovinos de validacao',flush=True)
    ver=carregar_binario(ROOT/cfg['extrator'],ROOT/cfg['classificador'],cfg['classificador_sha256'])
    detector=YOLO(str(ROOT/'data/processed/modelo_gado.pt'))
    amostras=[]
    d=pd.read_csv(ROOT/'data/brasil_dataset_v2/manifesto.csv')
    for r in d[d.conjunto=='test'].itertuples():
        amostras.append(dict(imagem=ROOT/'data/brasil_dataset_v2/images'/r.arquivo,
            rotulo=ROOT/'data/brasil_dataset_v2/labels'/(Path(r.arquivo).stem+'.txt'),rota='solo_poucos',origem='rotulados_antigos'))
    for nome in ('bdn_19.jpg','bdn_21.jpg'):
        p=ROOT/'data/experimentos/solo_piloto_separado_v1/teste'
        amostras.append(dict(imagem=p/'images'/nome,rotulo=p/'labels'/(Path(nome).stem+'.txt'),rota='solo_denso',origem='piloto_conhecido'))
    for nome in ('bdn_08_c1.jpg','bdn_08_c2.jpg'):
        amostras.append(dict(imagem=base/'images'/nome,rotulo=base/'labels'/(Path(nome).stem+'.txt'),rota='solo_denso',origem='val_positiva'))
    gab=pd.read_csv(ROOT/'data/teste_brasil/gabarito.csv')
    for r in gab.itertuples():
        for rota in ['solo_denso']+(['solo_poucos'] if r.maximo<=10 else []):
            amostras.append(dict(imagem=ROOT/'data/teste_brasil'/r.arquivo,rota=rota,origem='intervalos_conhecidos',minimo=r.minimo,maximo=r.maximo))
    p=ROOT/'data/busca_teste_denso_independente'
    amostras.append(dict(imagem=p/'bernard_28059238281.jpg',rota='solo_denso',origem='bernard_conhecido',minimo=20))
    rows=[];detalhes=[]
    for r in amostras:
        dense=r['rota']=='solo_denso'
        im=cv2.imread(str(r['imagem']));h,w=im.shape[:2]
        b,s=ct.detectar_inteira(detector,im,imgsz=1280 if dense else 1024,iou_nms=.7 if dense else .5,
            classes=[0],augment=False,device=0)
        b,s=ct.filtrar(b,s,.15 if dense else .30)
        if dense:
            v=confirmar(ver,im,b);mask,_=mascara_confirmacao(b,v,cfg['limiar'],cfg.get('min_lado_px',0))
            b,s=b[mask],s[mask]
        mm,sims=margens(clip,im,b);mask=manter(mm,limiar)
        ref=ct.yolo_para_pixels(r['rotulo'].read_text().splitlines(),w,h) if 'rotulo' in r else None
        for aplicado,bb,ss in ((False,b,s),(True,b[mask],s[mask])):
            linha=dict(arquivo=r['imagem'].name,rota=r['rota'],origem=r['origem'],clip=aplicado,previsto=len(bb))
            if ref is not None:
                linha.update(real=len(ref),erro=abs(len(bb)-len(ref)),**comparar_caixas(ref,bb,ss))
            elif 'maximo' in r:
                linha.update(erro=max(r['minimo']-len(bb),0,len(bb)-r['maximo']))
            rows.append(linha)
        if r['origem']=='bernard_conhecido':
            spec=spec_from_file_location('pontos',ROOT/'67_verificar_especies_no_denso.py')
            mod=module_from_spec(spec);spec.loader.exec_module(mod)
            pts=json.loads((p/'avaliacao_minimo_bernard/gabarito_antes_inferencia.json').read_text())['pontos_xy']
            cavalo=np.array([1400,1090,1820,1980],float)
            inter=np.maximum(np.minimum(b[:,2:],cavalo[2:])-np.maximum(b[:,:2],cavalo[:2]),0).prod(axis=1)
            iou=inter/((b[:,2:]-b[:,:2]).prod(axis=1)+np.prod(cavalo[2:]-cavalo[:2])-inter)
            detalhes=dict(antes=len(b),depois=int(mask.sum()),pontos_antes=mod.correspondencias_pontos(pts,b),
                pontos_depois=mod.correspondencias_pontos(pts,b[mask]),montaria_antes=int((iou>.5).sum()),montaria_depois=int(((iou>.5)&mask).sum()))
            cv2.imwrite(str(OUT/'bernard_depois.jpg'),ct.desenhar(im,b[mask]))
        np.savez(OUT/(r['rota']+'__'+r['imagem'].stem+'.npz'),caixas=b,scores=s,margens=mm,similaridades=sims,manter=mask)
        print(r['rota'],r['imagem'].name,len(b),'->',int(mask.sum()),flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'por_imagem.csv',index=False)
    resumo={}
    for rota,g in d[d.origem!='bernard_conhecido'].groupby('rota'):
        a=g[~g['clip']].set_index(['origem','arquivo']);b=g[g['clip']].set_index(['origem','arquivo'])
        resumo[rota]=dict(perdeu_tp=bool((b.TP<a.TP).any()),piorou_erro=bool((b.erro>a.erro).any()),
            FP_antes=int(a.FP.sum()),FP_depois=int(b.FP.sum()))
    (OUT/'resultado.json').write_text(json.dumps(dict(rotas=resumo,bernard=detalhes,limiar=limiar,
        promovido=False,limite=protocolo['limites']),indent=2),encoding='utf-8')
    print(json.dumps(dict(rotas=resumo,bernard=detalhes),indent=2),flush=True)


if __name__=='__main__':main()
