"""Reproduz a verificacao da correcao integrada usando previsoes preservadas."""
from pathlib import Path
import ast,json
import cv2,numpy as np,pandas as pd
from avaliacao import comparar_caixas
from correcao_aerea import selecionar_instancias
import contagem as ct

ROOT=Path(__file__).resolve().parent


def main():
    fonte=ROOT/'data/experimentos/pipeline_sombras_validacao_v1'
    out=ROOT/'data/experimentos/correcao_aerea_instancias_v2'
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'data/modelos/especialistas.json').read_text(encoding='utf-8'))['visoes']['curral_externo_topo']
    conf,lim=cfg['config']['conf'],cfg['verificador_nucleo']['limiar']
    rows=[]
    for f in sorted(fonte.glob('*.npz')):
        d=np.load(f)
        if 'ref' not in d:continue
        b,s,v,ref=d['caixas'],d['scores'],d['verificador'],d['ref']
        ix,_=selecionar_instancias(b,s,v,conf,lim)
        for nome,k in [('anterior',np.flatnonzero((s>=.2)&(v>=lim))),('corrigida',ix)]:
            rows.append(dict(arquivo=f.name,versao=nome,erro=abs(len(k)-len(ref)),**comparar_caixas(ref,b[k],s[k])))
    df=pd.DataFrame(rows);df.to_csv(out/'validacao.csv',index=False)
    resumo={}
    for nome,g in df.groupby('versao'):
        resumo[nome]=dict(TP=int(g.TP.sum()),FP=int(g.FP.sum()),FN=int(g.FN.sum()),MAE=float(g.erro.mean()))
    d=np.load(fonte/'0359.npz');b,s,v=d['caixas'],d['scores'],d['verificador']
    ix,recovered=selecionar_instancias(b,s,v,conf,lim)
    # Execute the actual application branch, replacing only I/O and rendering.
    tree=ast.parse((ROOT/'app.py').read_text(encoding='utf-8'))
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='especialista and especialista.get(\'verificador_nucleo\')')
    class UI:
        def caption(self,*args):pass
    env=dict(especialista=cfg,PROJ=ROOT,conf=conf,caixas=b,scores=s,np=np,pd=pd,json=json,
        dados=b'',st=UI(),ler_verificador_nucleo=lambda *args:None,pontuar_nucleo_bytes=lambda *args:v)
    exec(compile(ast.Module(body=[branch],type_ignores=[]),'app_branch','exec'),env)
    assert np.array_equal(env['b'],b[ix])
    assert np.array_equal(env['s'],s[ix])
    assert np.array_equal(env['score_confirmado'],v[ix])
    assert env['detalhes_confirmacao']['mantida'].sum()==len(ix)
    env['detalhes_confirmacao'].to_csv(out/'auditoria_app.csv',index=False)
    base=ROOT/'data/fontes_pesquisa/nellore_inspecao/revisao_0359_recorte16'
    pts=np.array(json.loads((base/'gabarito_revisado_antes_detector.json').read_text())['pontos_xy'])
    from importlib.util import spec_from_file_location,module_from_spec
    sp=spec_from_file_location('pts',ROOT/'67_verificar_especies_no_denso.py');pm=module_from_spec(sp);sp.loader.exec_module(pm)
    hits=pm.correspondencias_pontos(pts,b[ix])
    cv2.imwrite(str(out/'resultado_0359.jpg'),ct.desenhar(cv2.imread(str(base/'recorte.png')),b[ix]))
    result=dict(validacao_96_imagens=resumo,cena_conhecida=dict(previsto=len(ix),pontos_cobertos=hits,
        recuperadas=int(recovered.sum()),indices=ix.tolist()),integracao_app_verificada=True,
        limite='Regras ajustadas apos observar a cena conhecida; nao e teste independente. Val aerea Jalogny, nao nova fazenda brasileira. Quatro omissoes adicionais na validacao.',
        perfil='curral_externo_topo',confianca=conf)
    (out/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
