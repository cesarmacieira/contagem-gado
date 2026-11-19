"""Reusa features; calibra apenas as caixas corretas produzidas na validacao."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import contagem as ct
from avaliacao import comparar_caixas
from verificador_semantico import manter

ROOT=Path(__file__).resolve().parent
FONTE=ROOT/'data/experimentos/verificador_clip_calibrado_v2'
OUT=ROOT/'data/experimentos/verificador_clip_tp_v3'


def referencia(origem,nome):
    bases={'rotulados_antigos':ROOT/'data/brasil_dataset_v2',
        'piloto_conhecido':ROOT/'data/experimentos/solo_piloto_separado_v1/teste',
        'val_positiva':ROOT/'data/experimentos/solo_denso_validacao_v2/validacao'}
    if origem not in bases:return None
    p=bases[origem];im=cv2.imread(str(p/'images'/nome));h,w=im.shape[:2]
    return ct.yolo_para_pixels((p/'labels'/(Path(nome).stem+'.txt')).read_text().splitlines(),w,h)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    spec=spec_from_file_location('correspondencias',ROOT/'53_verificar_falsos_positivos.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    valores=[];registros=[]
    for nome in ('bdn_08_c1.jpg','bdn_08_c2.jpg'):
        p=FONTE/('solo_denso__'+Path(nome).stem+'.npz');d=np.load(p)
        ref=referencia('val_positiva',nome)
        ids=mod.indices_correspondentes(ref,d['caixas'],d['scores'])
        valores.extend(d['margens'][ids].min(axis=1).tolist())
        registros.append(dict(arquivo=nome,indices_tp=ids,cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assert valores
    limiar=float(max(0,max(valores)+.001))
    protocolo=json.loads((FONTE/'protocolo_antes_regressao.json').read_text())
    protocolo.update(limiar=limiar,criterio='Preservar caixas com correspondencia IoU>=.5 na validacao, folga .001, margem nao negativa.',
        positivos_calibracao=len(valores),calibracao=registros,
        nota='Corrige diferenca entre recorte do rotulo e recorte realmente produzido pelo detector. Toda regressao ja conhecida.')
    (OUT/'protocolo_antes_reavaliacao.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    tabela=pd.read_csv(FONTE/'por_imagem.csv');tabela=tabela[~tabela['clip']]
    gab=pd.read_csv(ROOT/'data/teste_brasil/gabarito.csv').set_index('arquivo')
    rows=[];bernard={}
    for r in tabela.itertuples():
        p=FONTE/(r.rota+'__'+Path(r.arquivo).stem+'.npz');cache=np.load(p)
        b,s,mm=cache['caixas'],cache['scores'],cache['margens'];mask=manter(mm,limiar)
        ref=referencia(r.origem,r.arquivo)
        for usado,bb,ss in ((False,b,s),(True,b[mask],s[mask])):
            linha=dict(arquivo=r.arquivo,rota=r.rota,origem=r.origem,semantico=usado,previsto=len(bb))
            if ref is not None:linha.update(real=len(ref),erro=abs(len(bb)-len(ref)),**comparar_caixas(ref,bb,ss))
            elif r.origem=='intervalos_conhecidos':
                g=gab.loc[r.arquivo];linha['erro']=max(g.minimo-len(bb),0,len(bb)-g.maximo)
            rows.append(linha)
        if r.origem=='bernard_conhecido':
            spec=spec_from_file_location('pontos',ROOT/'67_verificar_especies_no_denso.py')
            ptsmod=module_from_spec(spec);spec.loader.exec_module(ptsmod)
            base=ROOT/'data/busca_teste_denso_independente'
            pts=json.loads((base/'avaliacao_minimo_bernard/gabarito_antes_inferencia.json').read_text())['pontos_xy']
            alvo=np.array([1400,1090,1820,1980],float)
            inter=np.maximum(np.minimum(b[:,2:],alvo[2:])-np.maximum(b[:,:2],alvo[:2]),0).prod(axis=1)
            iou=inter/((b[:,2:]-b[:,:2]).prod(axis=1)+np.prod(alvo[2:]-alvo[:2])-inter)
            bernard=dict(antes=len(b),depois=int(mask.sum()),montaria_antes=int((iou>.5).sum()),
                montaria_depois=int(((iou>.5)&mask).sum()),pontos_antes=ptsmod.correspondencias_pontos(pts,b),
                pontos_depois=ptsmod.correspondencias_pontos(pts,b[mask]),contagem_exata=False)
            im=cv2.imread(str(base/r.arquivo));cv2.imwrite(str(OUT/'bernard_depois.jpg'),ct.desenhar(im,b[mask]))
        np.savez(OUT/(r.rota+'__'+Path(r.arquivo).stem+'.npz'),caixas=b,scores=s,margens=mm,manter=mask)
    d=pd.DataFrame(rows);d.to_csv(OUT/'por_imagem.csv',index=False)
    resumo={}
    for rota,g in d[d.origem!='bernard_conhecido'].groupby('rota'):
        a=g[~g.semantico].set_index(['origem','arquivo']);b=g[g.semantico].set_index(['origem','arquivo'])
        resumo[rota]=dict(perdeu_tp=bool((b.TP<a.TP).any()),piorou_erro=bool((b.erro>a.erro).any()),
            FP_antes=int(a.FP.sum()),FP_depois=int(b.FP.sum()))
    resultado=dict(limiar=limiar,positivos_calibracao=len(valores),rotas=resumo,bernard=bernard,promovido=False,
        limite='Somente calibracao e regressao conhecida. Minimo de vinte pontos nao e contagem total exata.')
    (OUT/'resultado.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    print(json.dumps(resultado,indent=2),flush=True)


if __name__=='__main__':main()
