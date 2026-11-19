"""Versao nova de referencias discutidas; preserva dados e metricas originais."""
from pathlib import Path
import hashlib,json,re
import numpy as np,pandas as pd
import contagem as ct
from correcao_aerea import selecionar_instancias,complementar
from avaliacao import comparar_caixas

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/gabarito_aereo_revisado_v2'


def pos(nome):
    match=re.match(r'(.+)__x(\d+)_y(\d+)$',nome)
    return match.group(1),np.array([int(match[2]),int(match[3])]*2)


def main():
    if (OUT/'resultado.json').exists():raise ValueError('Revisao preservada')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'labels').mkdir(exist_ok=True)
    src=ROOT/'data/experimentos/pipeline_sombras_validacao_v1'
    cases=[('Jalogny__DJI_202309261339_059__dji_20230926135514_0016_v__x4095_y2457',2,'B1'),
        ('Jalogny__DJI_202309261339_059__dji_20230926135514_0016_v__x4095_y2457',4,'B2'),
        ('Jalogny__DJI_202309261339_059__dji_20230926135544_0022_v__x3276_y0',4,'C')]
    edits=[]
    for name,i,case in cases:
        d=np.load(src/(name+'.npz'));original,offset=pos(name)
        edits.append(dict(original=original,global_xyxy=(d['ref'][i]+offset).tolist(),caso=case,
            justificativa='Fragmento dentro de animal ja rotulado; conferencia visual Claude e segunda leitura do assistente.',
            confianca='media; sem revisao humana independente'))
    rows=[];changes=[]
    for f in sorted(src.glob('*.npz')):
        if f.name=='0359.npz':continue
        d=np.load(f)
        if 'ref' not in d:continue
        original,offset=pos(f.stem);refs=d['ref'];keep=np.ones(len(refs),bool)
        for e in edits:
            if e['original']!=original:continue
            distances=np.max(np.abs(refs+offset-np.array(e['global_xyxy'])),axis=1)
            ids=np.flatnonzero(distances<=2)
            for i in ids:keep[i]=False;changes.append(dict(arquivo=f.stem,indice=int(i),caso=e['caso'],caixa_original=refs[i].tolist()))
        corrected=refs[keep]
        lines=[]
        for x1,y1,x2,y2 in corrected:
            lines.append(f'0 {(x1+x2)/2048:.6f} {(y1+y2)/2048:.6f} {(x2-x1)/1024:.6f} {(y2-y1)/1024:.6f}')
        (OUT/'labels'/(f.stem+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        b,s,v=d['caixas'],d['scores'],d['verificador'];ix,_=selecionar_instancias(b,s,v,.3,.14485076425550358)
        high=np.load(ROOT/'data/experimentos/recuperacao_aerea_escala_v1'/f.name)
        add=complementar(b[ix],high['caixas'],high['scores'],high['visual'],.4,.5)
        boxes=np.concatenate([b[ix],high['caixas'][add]]);scores=np.r_[s[ix],high['scores'][add]]
        for version,ref in [('original',refs),('revisada',corrected)]:
            rows.append(dict(arquivo=f.stem,gabarito=version,real=len(ref),previsto=len(boxes),erro=abs(len(boxes)-len(ref)),**comparar_caixas(ref,boxes,scores)))
    assert len(changes)==5,changes
    table=pd.DataFrame(rows);table.to_csv(OUT/'comparacao_mesmas_previsoes.csv',index=False);summary=[]
    for version,g in table.groupby('gabarito'):
        summary.append(dict(gabarito=version,n_bovinos=int(g.real.sum()),TP=int(g.TP.sum()),FP=int(g.FP.sum()),FN=int(g.FN.sum()),MAE=float(g.erro.mean())))
    result=dict(edicoes=edits,alteracoes_por_tile=changes,mesmas_previsoes=summary,
        original_preservado=True,uso='Referencia de desenvolvimento revisada apos inspecao das previsoes; nao e teste independente e nao representa melhora do modelo. Sem mudanca no treino em curso.',
        auditoria_sha256=hashlib.sha256((ROOT/'data/revisao_estrategia_claude_v1/revisao_omissoes.csv').read_bytes()).hexdigest())
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
