"""Compara omissoes em val e regressao conhecida, sem treino nem promocao."""
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
from avaliacao import comparar_caixas,taxas_deteccao
from verificador_visual import carregar_binario,confirmar,mascara_confirmacao
from verificador_semantico import carregar,margens,manter,candidatas_montaria
from recuperacao_multiescala import selecionar

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/recuperacao_multiescala_v1'
FONTE=ROOT/'data/experimentos/verificador_montaria_v4'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    torch.set_num_threads(4)
    cfg=json.loads((ROOT/'data/modelos/verificador_solo_v1/config.json').read_text());mc=cfg['montaria']
    protocolo=dict(referencia='Pipeline atual completo de solo_denso, cache verificado da rodada anterior',
        candidato_tta=dict(imgsz=1280,conf=.20,iou=.7,augment=True),
        apoio_alta=dict(imgsz=1920,conf=.15,iou=.7,augment=False),
        consenso=dict(iou_acordo=.5,iou_existente=.3,ios_existente=.7,uma_confirmacao_tta_por_caixa=True),
        verificadores=cfg,detector_sha256=hashlib.sha256((ROOT/'data/processed/modelo_gado.pt').read_bytes()).hexdigest(),
        criterio='Melhorar omissoes e contagem sem piorar casos anteriores. Nenhuma caixa base removida pelo consenso.',
        avaliacao='Uma cena positiva de val e regressao conhecida; sem teste independente novo',promover=False)
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    detector=YOLO(str(ROOT/'data/processed/modelo_gado.pt'))
    visual=carregar_binario(ROOT/cfg['extrator'],ROOT/cfg['classificador'],cfg['classificador_sha256'])
    semantico=carregar(ROOT/mc['peso_clip']);pessoa=YOLO(str(ROOT/mc['peso_pessoas']))
    spec=spec_from_file_location('referencias',ROOT/'72_calibrar_clip_correspondencias.py')
    refmod=module_from_spec(spec);spec.loader.exec_module(refmod)
    spec=spec_from_file_location('caminhos',ROOT/'73_conferir_montaria_contextual.py')
    caminhos=module_from_spec(spec);spec.loader.exec_module(caminhos)
    def conferir(im,b,s):
        v=confirmar(visual,im,b);mask,_=mascara_confirmacao(b,v,cfg['limiar'],cfg.get('min_lado_px',0))
        b,s=b[mask],s[mask]
        mm,_=margens(semantico,im,b);rejeitar=~manter(mm,mc['limiar'])
        pessoas=np.empty((0,4))
        if rejeitar.any():
            pessoas,_=ct.detectar_inteira(pessoa,im,imgsz=1280,conf_min=.5,iou_nms=.5,classes=[0],augment=False,device=0)
        mask=~(rejeitar & candidatas_montaria(b,pessoas))
        return b[mask],s[mask]
    tabela=pd.read_csv(FONTE/'por_imagem.csv');tabela=tabela[(~tabela.montaria)&(tabela.rota=='solo_denso')]
    gab=pd.read_csv(ROOT/'data/teste_brasil/gabarito.csv').set_index('arquivo')
    rows=[];adicoes=[]
    for r in tabela.itertuples():
        im=cv2.imread(str(caminhos.imagem(r.origem,r.arquivo)))
        cache=np.load(FONTE/('solo_denso__'+Path(r.arquivo).stem+'.npz'))
        base,bs=cache['caixas'][cache['manter']],cache['scores'][cache['manter']]
        bruto_t,st=ct.detectar_inteira(detector,im,imgsz=1280,iou_nms=.7,classes=[0],augment=True,device=0)
        bruto_t,st=ct.filtrar(bruto_t,st,.20);bt,st=conferir(im,bruto_t,st)
        bruto_a,sa=ct.detectar_inteira(detector,im,imgsz=1920,iou_nms=.7,classes=[0],augment=False,device=0)
        bruto_a,sa=ct.filtrar(bruto_a,sa,.15);ba,sa=conferir(im,bruto_a,sa)
        ids,j=selecionar(base,ba,sa,bt,st)
        bc=np.concatenate([base,ba[ids]]);sc=np.concatenate([bs,sa[ids]])
        ref=refmod.referencia(r.origem,r.arquivo)
        for nome,b,s in [('referencia',base,bs),('tta',bt,st),('alta1920',ba,sa),('consenso',bc,sc)]:
            row=dict(arquivo=r.arquivo,origem=r.origem,estrategia=nome,previsto=len(b))
            if ref is not None:row.update(real=len(ref),erro=abs(len(b)-len(ref)),**comparar_caixas(ref,b,s))
            elif r.origem=='intervalos_conhecidos':
                g=gab.loc[r.arquivo];row['erro']=max(g.minimo-len(b),0,len(b)-g.maximo)
            rows.append(row)
        np.savez(OUT/(Path(r.arquivo).stem+'.npz'),base=base,scores_base=bs,tta=bt,scores_tta=st,
            alta=ba,scores_alta=sa,adicoes=ids,confirmacoes_tta=j,consenso=bc,scores_consenso=sc)
        adicoes.append(dict(arquivo=r.arquivo,n_adicoes=len(ids)))
        if r.origem=='val_positiva' or r.origem=='bernard_conhecido':
            cv2.imwrite(str(OUT/('consenso_'+r.arquivo)),ct.desenhar(im,bc))
        print(r.arquivo,'base',len(base),'TTA',len(bt),'alta',len(ba),'consenso',len(bc),flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'por_imagem.csv',index=False)
    resumo={};a=d[d.estrategia=='referencia'].set_index('arquivo')
    for nome in ('tta','alta1920','consenso'):
        b=d[d.estrategia==nome].set_index('arquivo')
        val=b[b.origem=='val_positiva'];tp,fp,fn=(int(val[k].sum()) for k in ('TP','FP','FN'))
        resumo[nome]=dict(val_TP=tp,val_FP=fp,val_FN=fn,val_MAE=float(val.erro.mean()),
            val_F1=taxas_deteccao(tp,fp,fn)['F1_caixas'],perdeu_tp=bool((b.TP<a.TP).any()),
            piorou_erro=bool((b.erro>a.erro).any()),FP_total=int(b.FP.sum()),
            mudou_contagens=int((b.previsto!=a.previsto).sum()))
    val=a[a.origem=='val_positiva']
    resultado=dict(referencia_val=dict(TP=int(val.TP.sum()),FP=int(val.FP.sum()),FN=int(val.FN.sum()),MAE=float(val.erro.mean())),
        estrategias=resumo,adicoes=adicoes,promovido=False,limite=protocolo['avaliacao'])
    (OUT/'resultado.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    print(json.dumps(resultado,indent=2),flush=True)


if __name__=='__main__':main()
