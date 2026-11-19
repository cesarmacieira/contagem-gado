"""Avalia separacao de caixas fundidas nos caches, sem promover configuracao."""
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import json
import numpy as np
import pandas as pd
from avaliacao import comparar_caixas, taxas_deteccao
from recuperacao_multiescala import separar_fusoes

ROOT = Path(__file__).resolve().parent
FONTE = ROOT / 'data/experimentos/recuperacao_multiescala_v1'
OUT = ROOT / 'data/experimentos/separacao_fusoes_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'resultado.json').exists():
        raise ValueError('Resultado preservado')
    (OUT/'protocolo_antes_reavaliacao.json').write_text(json.dumps(dict(
        regra='Uma caixa substituida por duas; IoU entre partes <= .25; cobertura >= .7; '
              'cada parte <= .8 da area original e contida >= .8, centro interno; '
              'duas correspondencias TTA distintas IoU >= .5; sem duplicar outra base.',
        avaliacao='Caches de desenvolvimento e regressao conhecida; sem teste novo',
        promover=False), indent=2), encoding='utf-8')
    spec = spec_from_file_location('ref', ROOT/'72_calibrar_clip_correspondencias.py')
    mod = module_from_spec(spec); spec.loader.exec_module(mod)
    tabela = pd.read_csv(FONTE/'por_imagem.csv')
    tabela = tabela[tabela.estrategia == 'referencia']
    gab = pd.read_csv(ROOT/'data/teste_brasil/gabarito.csv').set_index('arquivo')
    rows, trocas = [], []
    for r in tabela.itertuples():
        d = np.load(FONTE/(Path(r.arquivo).stem+'.npz'))
        b, s, t = separar_fusoes(d['base'], d['scores_base'], d['alta'], d['scores_alta'], d['tta'], d['scores_tta'])
        trocas.append(dict(arquivo=r.arquivo, trocas=t))
        np.savez(OUT/(Path(r.arquivo).stem+'.npz'), caixas=b, scores=s)
        ref = mod.referencia(r.origem, r.arquivo)
        for nome, bb, ss in [('referencia',d['base'],d['scores_base']),('separacao',b,s)]:
            row = dict(arquivo=r.arquivo, origem=r.origem, estrategia=nome, previsto=len(bb))
            if ref is not None:
                row.update(real=len(ref), erro=abs(len(bb)-len(ref)), **comparar_caixas(ref, bb, ss))
            elif r.origem == 'intervalos_conhecidos':
                g = gab.loc[r.arquivo]; row['erro'] = max(g.minimo-len(bb), 0, len(bb)-g.maximo)
            rows.append(row)
    d = pd.DataFrame(rows); d.to_csv(OUT/'por_imagem.csv', index=False)
    a = d[d.estrategia=='referencia'].set_index('arquivo'); b = d[d.estrategia=='separacao'].set_index('arquivo')
    val = b[b.origem=='val_positiva']; tp, fp, fn = (int(val[k].sum()) for k in ('TP','FP','FN'))
    resumo = dict(val_TP=tp, val_FP=fp, val_FN=fn, val_MAE=float(val.erro.mean()),
        val_F1=taxas_deteccao(tp, fp, fn)['F1_caixas'], perdeu_tp=bool((b.TP<a.TP).any()),
        piorou_erro=bool((b.erro>a.erro).any()), trocas=trocas, promovido=False)
    (OUT/'resultado.json').write_text(json.dumps(resumo, indent=2), encoding='utf-8')
    print(json.dumps(resumo, indent=2))


if __name__ == '__main__':
    main()
