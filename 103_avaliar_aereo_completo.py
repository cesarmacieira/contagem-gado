"""Compara fotos inteiras, congela escolha em val e so depois mede regressao."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib,json
import pandas as pd,torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/aereo_ampliado_v2'


def incluir_densas(metricas,linhas,avaliador):
    for conf,g in linhas.groupby('conf'):
        densas=g[g.real>=20]
        if len(densas):
            tp,fp,fn=[int(densas[k].sum()) for k in ['TP','FP','FN']]
            metricas[float(conf)]['vinte_ou_mais']=dict(**avaliador.metricas_contagem(densas.real,densas.previsto),
                **avaliador.taxas_deteccao(tp,fp,fn))


def main():
    if (OUT/'resultado_completo.json').exists():raise ValueError('Resultado preservado')
    torch.set_num_threads(4)
    catalogo_inicio=json.loads((ROOT/'data/modelos/especialistas.json').read_text(encoding='utf-8'))
    sp=spec_from_file_location('avaliador',ROOT/'48_especializar_drone.py');m=module_from_spec(sp);sp.loader.exec_module(m)
    referencia=ROOT/'data/experimentos/gabarito_aereo_revisado_v2/validacao_completa.csv'
    if not referencia.exists():raise ValueError('Referencias revisadas completas necessarias')
    val=pd.read_csv(referencia);confs=[.15,.25,.35,.45,.55,.65]
    protocol=dict(referencia=str(referencia),referencia_sha256=hashlib.sha256(referencia.read_bytes()).hexdigest(),
        rotulos_revisados_apos_inspecao=True,promocao_automatica=False,confs=confs,imagens_val=len(val),avaliacao='Fotos inteiras por tiles1024, overlap0.2, juncao0.5 sem supressao de contencao',
        escolha='Escolher nas 24 fotos; exigir queda de MAE >=10% e >=3 erros absolutos a menos, precisao/recall/F1 nao piores; gates >=8 e >=20 animais. Confirmar nos demais quadros de val sem recalibrar.',
        teste='Nao consultar nesta etapa. Confirmacao usa outros quadros historicos de val, mesmas fazendas/voos; nao e nova fazenda independente.')
    (OUT/'protocolo_fotos_inteiras.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');choices=[]
    weights=[('original',ROOT/'data/processed/modelo_gado.pt'),('best',OUT/'treino/weights/best.pt'),('last',OUT/'treino/weights/last.pt')]
    for name,w in weights:
        metrics,rows=m.avaliar(w,val,confs,False);incluir_densas(metrics,rows,m);rows.to_csv(OUT/('val_completa_'+name+'.csv'),index=False)
        conf=min(metrics,key=lambda c:(metrics[c]['score'],-metrics[c]['F1_caixas']))
        choices.append(dict(nome=name,peso=str(w),sha256=hashlib.sha256(w.read_bytes()).hexdigest(),conf=conf,metricas=metrics[conf]))
        print('VALIDACAO',name,conf,metrics[conf]['MAE'],metrics[conf]['F1_caixas'],flush=True)
    baseline=choices[0];chosen=min(choices[1:],key=lambda r:(r['metricas']['score'],-r['metricas']['F1_caixas']))
    (OUT/'escolha_fotos_inteiras_antes_regressao.json').write_text(json.dumps(dict(original=baseline,candidato=chosen,todas=choices),indent=2),encoding='utf-8')
    from criterios_qualidade import supera_referencia as supera
    valpass=supera(chosen['metricas'],baseline['metricas'])
    result=dict(validacao_original=baseline,validacao_candidato=chosen,passou_validacao=valpass,promovido=False)
    if valpass:
        all_rows=m.catalogo();remaining=all_rows[(all_rows.conjunto=='val')&~all_rows.arquivo.isin(val.arquivo)]
        assert len(remaining)>0 and not set(remaining.arquivo)&set(val.arquivo)
        remaining.to_csv(OUT/'quadros_confirmacao_antes_inferencia.csv',index=False);reg={}
        for name,w,c in [('original_calibrado',Path(baseline['peso']),baseline['conf']),('candidato',Path(chosen['peso']),chosen['conf'])]:
            mets,rows=m.avaliar(w,remaining,[c],False);incluir_densas(mets,rows,m);rows.to_csv(OUT/('confirmacao_completa_'+name+'.csv'),index=False);reg[name]=mets[c]
        result['confirmacao_outros_quadros_val']=reg
        passed=supera(reg['candidato'],reg['original_calibrado']);result['passou_confirmacao']=passed
        if passed:
            result['proposta_perfil']=dict(visao='drone_vertical',peso=chosen['peso'],sha256=chosen['sha256'],conf=chosen['conf'],legado=False)
            result['pendencia_incorporacao']='Conferir rotulos revisados e coerencia da rota; esta avaliacao nao reescreve o catalogo.'
    result['teste_consultado']=False
    result['limite']='Quadros historicos de val, mesmas fazendas e voos; gabarito de calibracao revisado depois de inspecao de previsoes. Nao e nova fazenda independente. Campanario excluido do treino e desta selecao.'
    (OUT/'resultado_completo.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
