"""Escolhe checkpoint, tamanho e limiar na validacao; compara no teste depois."""
from pathlib import Path
import json,hashlib
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import contagem as ct
from avaliacao import metricas_contagem,comparar_caixas,taxas_deteccao

ROOT=Path(__file__).resolve().parent;EXP=ROOT/'data/experimentos/detector_denso_v3';OUT=EXP/'avaliacao'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultados.json').exists():print('Avaliacao ja concluida');return
    val=pd.read_csv(EXP/'catalogo_treino_validacao.csv');val=val[(val.conjunto=='val')&(val.n_bovinos>=5)].copy()
    assert len(val)>=10
    weights=EXP/'treino/weights'
    candidatos=[ROOT/'data/processed/modelo_gado.pt']+[p for p in [weights/'best.pt',weights/'epoch10.pt',weights/'last.pt'] if p.exists()]
    refs={};cache={};modelos={};todos=[];por_peso={}
    desenvolvimento=pd.read_csv(ROOT/'data/teste_brasil/gabarito.csv');cache_dev={}
    def deteccao(peso,r,sz,iou):
        chave=(str(peso),r.img,sz,iou)
        if chave not in cache:
            im=cv2.imread(str(r.img));assert im is not None;h,w=im.shape[:2]
            refs[r.img]=ct.yolo_para_pixels(Path(r.rot).read_text(encoding='utf-8').splitlines(),w,h)
            if str(peso) not in modelos:modelos[str(peso)]=YOLO(str(peso))
            cache[chave]=ct.detectar_inteira(modelos[str(peso)],im,imgsz=sz,iou_nms=iou,conf_min=.05,device=0,max_det=1000)
            if len(cache)%100==0:print(f'Inferencias distintas concluidas: {len(cache)}; {peso.name}, entrada {sz}',flush=True)
        return cache[chave]
    def avaliar(peso,df,cfg,exportar=False):
        real=[];prev=[];tp=fp=fn=0;grupos=[];linhas=[]
        for r in df.itertuples():
            b,s=deteccao(peso,r,cfg['imgsz'],cfg['iou']);b,s=ct.filtrar(b,s,cfg['conf'])
            ref=refs[r.img];met=comparar_caixas(ref,b,s);tp+=met['TP'];fp+=met['FP'];fn+=met['FN']
            real.append(len(ref));prev.append(len(b));grupos.append(r.grupo)
            if exportar:linhas.append(dict(stem=r.stem,img=r.img,grupo=r.grupo,real=len(ref),previsto=len(b),**met))
        m=dict(**metricas_contagem(real,prev),TP=tp,FP=fp,FN=fn,**taxas_deteccao(tp,fp,fn))
        e=pd.DataFrame(dict(g=grupos,e=np.abs(np.array(prev)-np.array(real))))
        m['MAE_macro_grupos']=float(e.groupby('g').e.mean().mean());m['score_validacao']=(m['MAE']+m['MAE_macro_grupos'])/2
        return m,linhas
    def erro_desenvolvimento(peso,sz,iou,conf):
        erros=[]
        for r in desenvolvimento.itertuples():
            chave=(str(peso),r.arquivo,sz,iou)
            if chave not in cache_dev:
                im=cv2.imread(str(ROOT/'data/teste_brasil'/r.arquivo))
                cache_dev[chave]=ct.detectar_inteira(modelos[str(peso)],im,imgsz=sz,iou_nms=iou,conf_min=.05,device=0,max_det=1000)
            n=len(ct.filtrar(*cache_dev[chave],conf)[0]);erros.append(max(r.minimo-n,0,n-r.maximo))
        return float(np.mean(erros))
    for peso in candidatos:
        linhas=[]
        for sz in (1280,1536):
            for iou in (.5,.7):
                for conf in (.05,.10,.15,.20,.25,.30,.35,.40,.50,.60):
                    cfg=dict(imgsz=sz,iou=iou,conf=conf);m,_=avaliar(peso,val,cfg)
                    erro_br=erro_desenvolvimento(peso,sz,iou,conf)
                    linhas.append(dict(pesos=str(peso),**cfg,**m,MAE_brasil_desenvolvimento=erro_br,score_escolha=m['score_validacao']+erro_br))
        linhas.sort(key=lambda r:(r['score_escolha'],-r['F1_caixas'],r['imgsz']))
        por_peso[str(peso)]=linhas[0];todos.extend(linhas)
        print('Validacao',peso.name,linhas[0]['MAE'],linhas[0]['F1_caixas'],flush=True)
    baseline=por_peso[str(candidatos[0])]
    novos=[por_peso[str(p)] for p in candidatos[1:]];novos.sort(key=lambda r:(r['score_escolha'],-r['F1_caixas']))
    escolhido=novos[0]
    congelado=dict(original=baseline,ajustado=escolhido,validacao_imagens=len(val),brasil_desenvolvimento_imagens=len(desenvolvimento),criterio='Media MAE por imagem e macro por grupo na validacao publicada, somada ao MAE dos intervalos das dez fotos brasileiras de desenvolvimento. F1 como desempate. As fotos brasileiras antigas foram retiradas da condicao de teste independente; as novas imagens de teste nao participam da escolha.')
    (OUT/'escolha_antes_teste.json').write_text(json.dumps(congelado,indent=2),encoding='utf-8');pd.DataFrame(todos).to_csv(OUT/'grade_validacao.csv',index=False)
    test=pd.read_csv(EXP/'teste_congelado.csv');results={};detalhes=[]
    for nome,r in [('original',baseline),('ajustado',escolhido)]:
        peso=Path(r['pesos']);cfg={k:r[k] for k in ('imgsz','iou','conf')}
        m,linhas=avaliar(peso,test,cfg,True);results[nome]=dict(pesos=str(peso),sha256=hashlib.sha256(peso.read_bytes()).hexdigest(),config=cfg,validacao=r,teste=m)
        detalhes.extend(dict(modelo=nome,**l) for l in linhas);print('Teste',nome,m,flush=True)
        denso=test[test.n_bovinos>=8]
        results[nome]['teste_8_ou_mais']=avaliar(peso,denso,cfg)[0] if len(denso) else None
        # Diagnosticos brasileiros separados e ja examinados em rodadas anteriores.
        br=pd.read_csv(ROOT/'data/brasil_dataset_v2/manifesto.csv');br=br[br.conjunto=='test'].copy()
        br['img']=br.arquivo.map(lambda x:str(ROOT/'data/brasil_dataset_v2/images'/x));br['rot']=br.arquivo.map(lambda x:str(ROOT/'data/brasil_dataset_v2/labels'/(Path(x).stem+'.txt')));br['stem']=br.arquivo
        results[nome]['brasil_13_reutilizado']=avaliar(peso,br,cfg)[0]
        gab=pd.read_csv(ROOT/'data/teste_brasil/gabarito.csv');diag=[]
        for g in gab.itertuples():
            img=cv2.imread(str(ROOT/'data/teste_brasil'/g.arquivo));b,s=ct.detectar_inteira(modelos[str(peso)],img,imgsz=cfg['imgsz'],iou_nms=cfg['iou'],device=0)
            n=len(ct.filtrar(b,s,cfg['conf'])[0]);erro=max(g.minimo-n,0,n-g.maximo)
            diag.append(dict(arquivo=g.arquivo,minimo=g.minimo,maximo=g.maximo,previsto=n,erro_intervalo=erro))
        results[nome]['brasil_antigo_exploratorio']=dict(dentro=sum(l['erro_intervalo']==0 for l in diag),MAE_intervalo=float(np.mean([l['erro_intervalo'] for l in diag])),imagens=diag)
    pd.DataFrame(detalhes).to_csv(OUT/'contagem_teste.csv',index=False)
    a,b=results['ajustado']['teste'],results['original']['teste']
    results['passou_teste_novo']=bool(a['MAE']<b['MAE'] and a['F1_caixas']>=b['F1_caixas'] and a['recall_caixas']>=b['recall_caixas'])
    ad=results['ajustado']['teste_8_ou_mais'];bd=results['original']['teste_8_ou_mais']
    results['passou_cenario_denso']=bool(ad is not None and ad['MAE']<bd['MAE'] and ad['F1_caixas']>=bd['F1_caixas'] and ad['recall_caixas']>=bd['recall_caixas'])
    results['protocolo']=dict(teste_imagens=len(test),teste_grupos=int(test.grupo.nunique()),teste_maximo=int(test.n_bovinos.max()),caixas_reais_publicadas=True,novas_fotos_nao_usadas_no_treino=True,brasil_10_usado_como_desenvolvimento=True,limitacao='Agrupamento conservador, sem certificado de fazendas independentes. As dez fotos brasileiras antigas sao desenvolvimento, nao teste; as treze brasileiras com caixas ja foram examinadas em rodadas anteriores.')
    (OUT/'resultados.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
