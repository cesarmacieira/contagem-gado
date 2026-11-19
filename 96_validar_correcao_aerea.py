"""Valida deduplicacao e recuperacao confirmada por outra iluminacao."""
from pathlib import Path
import hashlib,json
import cv2,joblib,numpy as np,pandas as pd,torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas
from correcao_aerea import selecionar_correcao
from verificador_nucleo import carregar,pontuar

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/correcao_aerea_v1'
FONTE=ROOT/'data/experimentos/pipeline_sombras_validacao_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    cat=json.loads((ROOT/'data/modelos/especialistas.json').read_text());cfg=cat['visoes']['curral_externo_topo']['verificador_nucleo']
    protocolo=dict(confiancas=[.2,.3,.4],gamma=.5,recuperacao_conf_min=.1,iou_confirmacao=.5,
        duplicata=dict(iou_min=.4,contencao_min=.95),verificador=cfg,
        escolha='Menor FP+FN em val; desempate MAE, depois confianca maior. Inclui referencia atual no gate.',
        valida_nova_fazenda=False,promover=False)
    (OUT/'protocolo_antes_validacao.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    torch.set_num_threads(4);model=YOLO(str(ROOT/'data/processed/modelo_gado.pt'))
    lut=np.round(np.sqrt(np.arange(256)/255)*255).astype('uint8');rows=[]
    val=ROOT/'data/experimentos/especialista_drone_v1/dataset/images/val'
    for p in sorted(val.glob('*.jpg')):
        d=np.load(FONTE/(p.stem+'.npz'));b,s,v,ref=d['caixas'],d['scores'],d['verificador'],d['ref']
        im=cv2.imread(str(p));bg,sg=ct.detectar_inteira(model,cv2.LUT(im,lut),imgsz=1024,conf_min=.1,iou_nms=.7,device=0,classes=[0],augment=False)
        np.savez(OUT/(p.stem+'.npz'),gamma_caixas=bg,gamma_scores=sg)
        indices=[('atual',.2,np.flatnonzero((s>=.2)&(v>=cfg['limiar'])))]
        for conf in protocolo['confiancas']:
            idx,_=selecionar_correcao(b,s,v,bg,sg,conf,cfg['limiar']);indices.append(('correcao',conf,idx))
        for nome,conf,idx in indices:
            rows.append(dict(arquivo=p.name,estrategia=nome,conf=conf,erro=abs(len(idx)-len(ref)),
                **comparar_caixas(ref,b[idx],s[idx])))
    t=pd.DataFrame(rows);t.to_csv(OUT/'validacao.csv',index=False);summ=[]
    for (nome,conf),g in t.groupby(['estrategia','conf']):
        summ.append(dict(estrategia=nome,conf=float(conf),TP=int(g.TP.sum()),FP=int(g.FP.sum()),FN=int(g.FN.sum()),
            erros=int((g.FP+g.FN).sum()),MAE=float(g.erro.mean())))
    best=sorted([r for r in summ if r['estrategia']=='correcao'],key=lambda r:(r['erros'],r['MAE'],-r['conf']))[0]
    (OUT/'escolha_antes_regressao.json').write_text(json.dumps(dict(escolhida=best,todas=summ),indent=2),encoding='utf-8')
    base=ROOT/'data/fontes_pesquisa/nellore_inspecao/revisao_0359_recorte16';im=cv2.imread(str(base/'recorte.png'))
    d=np.load(FONTE/'0359.npz');b,s,v=d['caixas'],d['scores'],d['verificador']
    bg,sg=ct.detectar_inteira(model,cv2.LUT(im,lut),imgsz=1024,conf_min=.1,iou_nms=.7,device=0,classes=[0],augment=False)
    idx,recovered=selecionar_correcao(b,s,v,bg,sg,best['conf'],cfg['limiar'])
    pts=np.array(json.loads((base/'gabarito_revisado_antes_detector.json').read_text())['pontos_xy'])
    # Matching um-a-um por ponto, igual ao avaliador existente.
    from importlib.util import spec_from_file_location,module_from_spec
    sp=spec_from_file_location('pts',ROOT/'67_verificar_especies_no_denso.py');pm=module_from_spec(sp);sp.loader.exec_module(pm)
    hits=pm.correspondencias_pontos(pts,b[idx]);np.savez(OUT/'0359.npz',caixas=b,scores=s,verificador=v,
        gamma_caixas=bg,gamma_scores=sg,selecionadas=idx,recuperadas=recovered)
    cv2.imwrite(str(OUT/'0359_resultado.jpg'),ct.desenhar(im,b[idx]))
    result=dict(validacao=summ,selecionada=best,cena_conhecida=dict(real=16,previsto=len(idx),pontos_cobertos=hits,
        recuperadas=int(recovered.sum()),indices=idx.tolist()),promovido=False,
        limite='Configuracao escolhida em val aerea. 0359 e desenvolvimento conhecido, nao teste novo. Gamma e confirmacao do mesmo detector, nao evidencias independentes.')
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
