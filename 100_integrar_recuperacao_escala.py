"""Confere a rotina real do app e regressao conhecida antes de ativar a escala."""
from pathlib import Path
import ast,json,hashlib
import cv2,numpy as np,pandas as pd,torch
from ultralytics import YOLO
from verificador_nucleo import carregar,pontuar
import contagem as ct

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/integracao_recuperacao_escala_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    catpath=ROOT/'data/modelos/especialistas.json';catalog=json.loads(catpath.read_text(encoding='utf-8'));antes=json.loads(json.dumps(catalog))
    profile=catalog['visoes']['curral_externo_topo'];v=profile['verificador_nucleo']
    evidence=json.loads((ROOT/'data/experimentos/recuperacao_aerea_escala_v1/resultado.json').read_text())
    assert evidence['selecionada']['politica']=='s0.4_v0.5'
    config=dict(versao=1,imgsz=1536,confianca_min=.4,visual_min=.5,
        validacao='data/experimentos/recuperacao_aerea_escala_v1/resultado.json')
    profile['recuperacao_escala']=config
    (OUT/'config_antes_regressao.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    tree=ast.parse((ROOT/'app.py').read_text(encoding='utf-8'));branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="especialista and especialista.get('verificador_nucleo')")
    torch.set_num_threads(4);model=YOLO(str(ROOT/profile['pesos']))
    vf=carregar(ROOT/v['extrator'],ROOT/v['classificador'],v['extrator_sha256'],v['classificador_sha256'],v['receita_sha256'])
    brazil=ROOT/'data/experimentos/avaliacao_curral_nova_v1'
    cases=[('0359',ROOT/'data/fontes_pesquisa/nellore_inspecao/revisao_0359_recorte16/recorte.png',ROOT/'data/experimentos/pipeline_sombras_validacao_v1/0359.npz',16)]
    for n,count in [('0354',8),('0356',7)]:cases.append((n,brazil/(n+'_recorte.png'),brazil/(n+'.npz'),count))
    results=[]
    class UI:
        def caption(self,*args):pass
    for name,path,cache,expected in cases:
        d=np.load(cache);im=cv2.imread(str(path));b,s=d['caixas'],d['scores'];pv=d['verificador']
        high,hs=ct.detectar_inteira(model,im,imgsz=1536,conf_min=.05,iou_nms=.7,device=0,classes=[0],augment=False)
        hv=pontuar(vf,im,high);np.savez(OUT/(name+'.npz'),caixas=high,scores=hs,visual=hv)
        env=dict(especialista=profile,PROJ=ROOT,conf=.3,caixas=b,scores=s,np=np,pd=pd,json=json,
            dados=path.read_bytes(),st=UI(),ler_verificador_nucleo=lambda *args:None,
            pontuar_nucleo_bytes=lambda dados,bb,*args:pv if bb is b else hv,
            detectar_bytes=lambda *args:(high,hs,im.shape[:2],'inteira'),tile_usado=1024,overlap_usado=.2,
            modelo=model,modelo_mt=0,modelo_chave='integracao',classes_modelo=[0],ct=ct)
        exec(compile(ast.Module(body=[branch],type_ignores=[]),'app_branch','exec'),env)
        assert len(env['b'])==expected,(name,len(env['b']))
        assert len(env['b'])==len(env['s'])==len(env['score_confirmado'])
        audit=env['detalhes_confirmacao'];assert audit.mantida.sum()==expected
        audit.to_csv(OUT/(name+'_auditoria.csv'),index=False)
        results.append(dict(id=name,contagem=expected,adicionadas_escala=int((audit.mantida&audit.segunda_resolucao).sum())))
    # Replay validation cache through the shared production implementation.
    from correcao_aerea import complementar,selecionar_instancias
    from avaliacao import comparar_caixas
    totals=np.zeros(3,int)
    for f in (ROOT/'data/experimentos/recuperacao_aerea_escala_v1').glob('*.npz'):
        a=np.load(f);d=np.load(ROOT/'data/experimentos/pipeline_sombras_validacao_v1'/f.name)
        ix,_=selecionar_instancias(d['caixas'],d['scores'],d['verificador'],.3,v['limiar']);b,s=d['caixas'][ix],d['scores'][ix]
        ids=complementar(b,a['caixas'],a['scores'],a['visual'],.4,.5)
        r=comparar_caixas(d['ref'],np.concatenate([b,a['caixas'][ids]]),np.r_[s,a['scores'][ids]])
        totals+=[r['TP'],r['FP'],r['FN']]
    assert totals.tolist()==[317,20,22]
    assert all(catalog['visoes'][k]==value for k,value in antes['visoes'].items() if k!='curral_externo_topo')
    (OUT/'catalogo_anterior.json').write_text(json.dumps(antes,indent=2),encoding='utf-8')
    catpath.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result=dict(ativada=True,perfil='curral_externo_topo',regressao_conhecida=results,validacao_TP_FP_FN=totals.tolist(),
        outros_perfis_preservados=True,teste_85_nao_consultado=True,limite='Ganho de um bovino na validacao conhecida. Nao demonstra ganho em nova fazenda.')
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
