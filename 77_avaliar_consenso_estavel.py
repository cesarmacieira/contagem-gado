"""Reavalia caches conhecidos com consenso de quantidade e localizacao."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import json
import cv2
import numpy as np
import pandas as pd
from avaliacao import comparar_caixas,taxas_deteccao
from recuperacao_multiescala import selecionar_consenso_estavel
import contagem as ct

ROOT=Path(__file__).resolve().parent
FONTE=ROOT/'data/experimentos/recuperacao_multiescala_v1'
OUT=ROOT/'data/experimentos/recuperacao_consenso_v2'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    (OUT/'protocolo_antes_reavaliacao.json').write_text(json.dumps(dict(
        iou_acordo=.5,iou_existente=.7,ios_existente=.98,desacordo_maximo_totais=1,
        adicoes_maximas='Menor contagem das duas vistas menos contagem base',
        motivacao='Permitir sobreposicao compativel com NMS denso; exigir concordancia global e local.',
        avaliacao='Somente desenvolvimento e regressao conhecida, com mesmos caches.'),indent=2),encoding='utf-8')
    spec=spec_from_file_location('ref',ROOT/'72_calibrar_clip_correspondencias.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    spec=spec_from_file_location('img',ROOT/'73_conferir_montaria_contextual.py')
    imgmod=module_from_spec(spec);spec.loader.exec_module(imgmod)
    tabela=pd.read_csv(FONTE/'por_imagem.csv');tabela=tabela[tabela.estrategia=='referencia']
    gab=pd.read_csv(ROOT/'data/teste_brasil/gabarito.csv').set_index('arquivo')
    rows=[];adds=[]
    for r in tabela.itertuples():
        d=np.load(FONTE/(Path(r.arquivo).stem+'.npz'))
        ids,j=selecionar_consenso_estavel(d['base'],d['alta'],d['scores_alta'],d['tta'],d['scores_tta'])
        b=np.concatenate([d['base'],d['alta'][ids]]);s=np.concatenate([d['scores_base'],d['scores_alta'][ids]])
        ref=mod.referencia(r.origem,r.arquivo)
        for nome,bb,ss in [('referencia',d['base'],d['scores_base']),('consenso',b,s)]:
            row=dict(arquivo=r.arquivo,origem=r.origem,estrategia=nome,previsto=len(bb))
            if ref is not None:row.update(real=len(ref),erro=abs(len(bb)-len(ref)),**comparar_caixas(ref,bb,ss))
            elif r.origem=='intervalos_conhecidos':
                g=gab.loc[r.arquivo];row['erro']=max(g.minimo-len(bb),0,len(bb)-g.maximo)
            rows.append(row)
        adds.append(dict(arquivo=r.arquivo,n_adicoes=len(ids)))
        np.savez(OUT/(Path(r.arquivo).stem+'.npz'),base=d['base'],scores_base=d['scores_base'],
            alta=d['alta'],scores_alta=d['scores_alta'],tta=d['tta'],scores_tta=d['scores_tta'],
            adicoes=ids,confirmacoes=j,resultado=b,scores=s)
        if len(ids):
            im=cv2.imread(str(imgmod.imagem(r.origem,r.arquivo)));cv2.imwrite(str(OUT/r.arquivo),ct.desenhar(im,b,ref=ref))
    d=pd.DataFrame(rows);d.to_csv(OUT/'por_imagem.csv',index=False)
    a=d[d.estrategia=='referencia'].set_index('arquivo');b=d[d.estrategia=='consenso'].set_index('arquivo')
    val=b[b.origem=='val_positiva'];tp,fp,fn=(int(val[k].sum()) for k in ('TP','FP','FN'))
    resumo=dict(val_TP=tp,val_FP=fp,val_FN=fn,val_MAE=float(val.erro.mean()),val_F1=taxas_deteccao(tp,fp,fn)['F1_caixas'],
        perdeu_tp=bool((b.TP<a.TP).any()),piorou_erro=bool((b.erro>a.erro).any()),FP_antes=int(a.FP.sum()),
        FP_depois=int(b.FP.sum()),adicoes=adds,promovido=False,limite='Uma cena de val; restante e regressao conhecida.')
    (OUT/'resultado.json').write_text(json.dumps(resumo,indent=2),encoding='utf-8')
    print(json.dumps(resumo,indent=2),flush=True)


if __name__=='__main__':main()
