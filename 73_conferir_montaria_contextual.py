"""Restringe veto semantico a cena com pessoa em posicao de montaria."""
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
from verificador_semantico import manter,candidatas_montaria

ROOT=Path(__file__).resolve().parent
FONTE=ROOT/'data/experimentos/verificador_clip_tp_v3'
OUT=ROOT/'data/experimentos/verificador_montaria_v4'


def imagem(origem,nome):
    bases={'rotulados_antigos':ROOT/'data/brasil_dataset_v2/images',
        'piloto_conhecido':ROOT/'data/experimentos/solo_piloto_separado_v1/teste/images',
        'val_positiva':ROOT/'data/experimentos/solo_denso_validacao_v2/validacao/images',
        'intervalos_conhecidos':ROOT/'data/teste_brasil',
        'bernard_conhecido':ROOT/'data/busca_teste_denso_independente'}
    return bases[origem]/nome


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    cfg=json.loads((FONTE/'protocolo_antes_reavaliacao.json').read_text())
    peso=ROOT/'data/modelos_candidatos/yolo11x.pt'
    protocolo=dict(limiar_semantico=cfg['limiar'],somente_solo_denso=True,pessoa_classe=0,
        pessoa_conf=.5,pessoa_imgsz=1280,regra_geometrica=1,peso_pessoa_sha256=hashlib.sha256(peso.read_bytes()).hexdigest(),
        criterio='Duas vistas indicam equino E pessoa acima/sobreposta em posicao de montaria.',
        avaliacao='Regressao conhecida. Sem novo teste independente.')
    (OUT/'protocolo_antes_pessoas.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    torch.set_num_threads(4);modelo=YOLO(str(peso));assert modelo.names[0]=='person'
    spec=spec_from_file_location('referencias',ROOT/'72_calibrar_clip_correspondencias.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    tabela=pd.read_csv(FONTE/'por_imagem.csv');tabela=tabela[~tabela.semantico]
    gab=pd.read_csv(ROOT/'data/teste_brasil/gabarito.csv').set_index('arquivo')
    rows=[];pessoas_log=[];bernard={}
    for r in tabela.itertuples():
        dados=np.load(FONTE/(r.rota+'__'+Path(r.arquivo).stem+'.npz'))
        b,s,mm=dados['caixas'],dados['scores'],dados['margens']
        rejeitar=~manter(mm,cfg['limiar']);pessoas=np.empty((0,4));ps=np.empty(0)
        if rejeitar.any() and r.rota=='solo_denso':
            im=cv2.imread(str(imagem(r.origem,r.arquivo)))
            pessoas,ps=ct.detectar_inteira(modelo,im,imgsz=1280,conf_min=.5,iou_nms=.5,
                classes=[0],augment=False,device=0)
        regiao=candidatas_montaria(b,pessoas)
        mask=~(rejeitar & regiao) if r.rota=='solo_denso' else np.ones(len(b),dtype=bool)
        if rejeitar.any():pessoas_log.append(dict(arquivo=r.arquivo,rota=r.rota,pessoas=pessoas.tolist(),scores=ps.tolist(),
            indices_semanticos=np.flatnonzero(rejeitar).tolist(),indices_montaria=np.flatnonzero(regiao).tolist()))
        ref=mod.referencia(r.origem,r.arquivo)
        for usado,bb,ss in ((False,b,s),(True,b[mask],s[mask])):
            linha=dict(arquivo=r.arquivo,rota=r.rota,origem=r.origem,montaria=usado,previsto=len(bb))
            if ref is not None:linha.update(real=len(ref),erro=abs(len(bb)-len(ref)),**comparar_caixas(ref,bb,ss))
            elif r.origem=='intervalos_conhecidos':
                g=gab.loc[r.arquivo];linha['erro']=max(g.minimo-len(bb),0,len(bb)-g.maximo)
            rows.append(linha)
        if r.origem=='bernard_conhecido':
            spec=spec_from_file_location('pontos',ROOT/'67_verificar_especies_no_denso.py')
            ptsmod=module_from_spec(spec);spec.loader.exec_module(ptsmod)
            pts=json.loads((ROOT/'data/busca_teste_denso_independente/avaliacao_minimo_bernard/gabarito_antes_inferencia.json').read_text())['pontos_xy']
            alvo=np.array([1400,1090,1820,1980],float)
            inter=np.maximum(np.minimum(b[:,2:],alvo[2:])-np.maximum(b[:,:2],alvo[:2]),0).prod(axis=1)
            iou=inter/((b[:,2:]-b[:,:2]).prod(axis=1)+np.prod(alvo[2:]-alvo[:2])-inter)
            bernard=dict(antes=len(b),depois=int(mask.sum()),montaria_antes=int((iou>.5).sum()),
                montaria_depois=int(((iou>.5)&mask).sum()),pontos_antes=ptsmod.correspondencias_pontos(pts,b),
                pontos_depois=ptsmod.correspondencias_pontos(pts,b[mask]),contagem_exata=False)
            cv2.imwrite(str(OUT/'bernard_depois.jpg'),ct.desenhar(im,b[mask]))
        np.savez(OUT/(r.rota+'__'+Path(r.arquivo).stem+'.npz'),caixas=b,scores=s,margens=mm,pessoas=pessoas,manter=mask)
    d=pd.DataFrame(rows);d.to_csv(OUT/'por_imagem.csv',index=False)
    resumo={}
    for rota,g in d[d.origem!='bernard_conhecido'].groupby('rota'):
        a=g[~g.montaria].set_index(['origem','arquivo']);b=g[g.montaria].set_index(['origem','arquivo'])
        resumo[rota]=dict(perdeu_tp=bool((b.TP<a.TP).any()),piorou_erro=bool((b.erro>a.erro).any()),
            FP_antes=int(a.FP.sum()),FP_depois=int(b.FP.sum()))
    (OUT/'pessoas.json').write_text(json.dumps(pessoas_log,indent=2),encoding='utf-8')
    resultado=dict(rotas=resumo,bernard=bernard,promovido=False,avaliacao=protocolo['avaliacao'])
    (OUT/'resultado.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    print(json.dumps(resultado,indent=2),flush=True)


if __name__=='__main__':main()
