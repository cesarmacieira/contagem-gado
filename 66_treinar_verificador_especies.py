"""Treino com confundidores revisados; novos testes sao recortes, nao Brasil."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib
import json
import cv2
import numpy as np
import joblib
from verificador_visual import carregar_verificador

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/confundidores_especies_v1'
OUT=ROOT/'data/experimentos/verificador_especies_v4'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'teste_especies.json').exists():
        raise ValueError('Experimento concluido preservado')
    registros=json.loads((BASE/'selecao_antes_download.json').read_text())['registros']
    extras=[]
    for r in registros:
        r['caminho_absoluto']=str(BASE/'images'/r['arquivo'])
        r['sha256']=hashlib.sha256(Path(r['caminho_absoluto']).read_bytes()).hexdigest()
        r['revisado']=not r['id'].startswith('75489b3b')
        r['nota_revisao']='Recorte sem bovino visivel; outras especies aceitas como classe binaria zero.' if r['revisado'] else 'Excluido: bovino visivel junto ao cavalo no contexto.'
        if r['revisado']:
            extras.append(r)
    assert len(extras)==35 and sum(r['conjunto']=='train' for r in extras)==23
    assert len({r['grupo'] for r in registros})==36
    (BASE/'revisao_visual.json').write_text(json.dumps(registros,indent=2),encoding='utf-8')
    spec=spec_from_file_location('desfoque',ROOT/'57_ajustar_verificador_desfoque.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    mod.treino.OUT=OUT
    mod.treino.POS=ROOT/'data/experimentos/verificador_auxiliar_v3/positivos'
    mod.treino.features=mod.features
    mod.treino.main(extras)
    cfg=json.loads((OUT/'escolha_antes_teste.json').read_text())
    antiga=json.loads((ROOT/'data/modelos/verificador_solo_v1/config.json').read_text())
    protocolo=dict(licencas='CC BY 2.0 por imagem, metadados oficiais Open Images',
        classificacao='Bovino contra nao bovino em recortes anotados',treino_novo=23,val_novo=6,test_novo=6,
        autores_separados=True,modelo_novo=cfg['classificador_sha256'],limiar_novo=cfg['limiar'],
        modelo_anterior=antiga['classificador_sha256'],limiar_anterior=antiga['limiar'],
        teste_brasileiro_denso=False)
    (OUT/'protocolo_especies_antes_teste.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    encoder=carregar_verificador(mod.treino.PESO)
    novo=joblib.load(OUT/'classificador.joblib')
    antigo=joblib.load(ROOT/antiga['classificador'])
    linhas=[]
    for r in extras:
        if r['conjunto']=='train':
            continue
        im=cv2.imread(r['caminho_absoluto']);h,w=im.shape[:2]
        b=(np.asarray(r['bbox_rel'])*np.asarray([w,h,w,h])).reshape(1,4)
        feat=mod.features(encoder,im,b,False)
        sa=float(antigo.predict_proba(feat)[0,1]);sn=float(novo.predict_proba(feat)[0,1])
        linhas.append(dict(id=r['id'],classe_fonte=r['classe'],conjunto=r['conjunto'],
            score_antigo=sa,score_novo=sn,erro_antigo=sa>=antiga['limiar'],erro_novo=sn>=cfg['limiar']))
    resumo={split:dict(n=sum(r['conjunto']==split for r in linhas),
        aceitos_erradamente_antes=sum(r['erro_antigo'] for r in linhas if r['conjunto']==split),
        aceitos_erradamente_depois=sum(r['erro_novo'] for r in linhas if r['conjunto']==split)) for split in ('val','test')}
    (OUT/'teste_especies.json').write_text(json.dumps(dict(resumo=resumo,por_recorte=linhas,
        promove=False,limite='Recortes negativos de autores separados. Nao mede contagem densa brasileira nem recall positivo novo.'),indent=2),encoding='utf-8')
    for nome in ('escolha_antes_teste.json','resultado.json'):
        p=OUT/nome;d=json.loads(p.read_text());d['limites']='Validacao positiva de uma cena e regressao conhecida; novos negativos de outras especies avaliados separadamente em teste_especies.json.'
        if 'limite' in d:d['limite']=d['limites']
        p.write_text(json.dumps(d,indent=2),encoding='utf-8')
    print(json.dumps(resumo,indent=2),flush=True)


if __name__=='__main__':
    main()
