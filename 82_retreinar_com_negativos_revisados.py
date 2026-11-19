"""Acrescenta negativos de treino ja revisados ao detector; preserva holdouts."""
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import hashlib
import json
import shutil
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/especialista_solo_negativos_v3'
FONTE=ROOT/'data/experimentos/especialista_solo_revisado_v2'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spec=spec_from_file_location('treino',ROOT/'80_retreinar_solo_revisado.py')
    treino=module_from_spec(spec);spec.loader.exec_module(treino)
    spec=spec_from_file_location('normalizar',ROOT/'51_ajustar_especialista_solo.py')
    normal=module_from_spec(spec);spec.loader.exec_module(normal)
    if not (OUT/'protocolo_treino.json').exists():
        reserved=set()
        for base in ('brasil_dataset_v2','brasil_denso_novo'):
            m=pd.read_csv(ROOT/'data'/base/'manifesto.csv')
            reserved.update(normal.normalizar(x) for x in m[m.conjunto.isin(['val','test'])].autor)
        caminhos=(FONTE/'dataset/train.txt').read_text().splitlines()
        registros=[]
        def incluir(imagem,nome,autor,grupo,fonte,metadados):
            if normal.normalizar(autor) in reserved:
                return
            p=OUT/'dataset/images/train'/(nome+'.jpg');p.parent.mkdir(parents=True,exist_ok=True)
            if imagem.size==0:raise ValueError('Recorte vazio')
            if not cv2.imwrite(str(p),imagem):raise OSError(str(p))
            lab=OUT/'dataset/labels/train'/(nome+'.txt');lab.parent.mkdir(parents=True,exist_ok=True)
            lab.write_text('',encoding='utf-8')
            caminhos.extend([p.as_posix()]*2)
            registros.append(dict(arquivo=p.name,autor=autor,grupo=grupo,conjunto='train',
                n_bovinos=0,fonte=fonte,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),metadados=metadados))
        fonte=ROOT/'data/confundidores_especies_v1'
        for r in json.loads((fonte/'revisao_visual.json').read_text()):
            if r['conjunto']!='train' or not r['revisado']:continue
            p=fonte/'images'/r['arquivo']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
            im=cv2.imread(str(p));h,w=im.shape[:2]
            x1,y1,x2,y2=np.round(np.asarray(r['bbox_rel'])*[w,h,w,h]).astype(int)
            incluir(im[max(0,y1):min(h,y2),max(0,x1):min(w,x2)],'especie_'+r['id'],
                    r['autor'],r['grupo'],'recorte_especie_revisado',r)
        fonte=ROOT/'data/negativos_solo_v1'
        for r in pd.read_csv(fonte/'manifesto.csv').to_dict('records'):
            if r['conjunto']!='train' or r['status']!='aprovado_sem_bovinos':continue
            p=fonte/'images'/r['arquivo']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
            incluir(cv2.imread(str(p)),'negativo_'+p.stem,r['autor'],r['grupo'],'negativo_integral_revisado',r)
        if not registros:raise ValueError('Sem negativos aprovados de treino')
        (OUT/'dataset/train.txt').write_text('\n'.join(caminhos)+'\n',encoding='utf-8')
        shutil.copy2(FONTE/'dataset/val.txt',OUT/'dataset/val.txt')
        (OUT/'dataset.yaml').write_text(f'path: {OUT.as_posix()}/dataset\ntrain: train.txt\nval: val.txt\nnames:\n  0: vaca\n',encoding='utf-8')
        (OUT/'negativos_antes_treino.json').write_text(json.dumps(registros,indent=2,default=str),encoding='utf-8')
        d=json.loads((FONTE/'protocolo_treino.json').read_text())
        d.update(epochs=12,negativos_novos=len(registros),repeticoes_negativos=2,
            n_amostras_train_com_repeticao=len(caminhos),
            justificativa='Detector anterior usou poucos negativos. Acrescenta recortes de outras especies '
            'e fotos sem bovinos ja revisados, exclusivamente dos splits de treino; '
            'nenhum val/test negativo incorporado. Mantem positivos e parametros do experimento anterior.',
            selecao='Checkpoint best na validacao; mesma confianca fixa, sem busca apos regressao')
        (OUT/'protocolo_treino.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
        hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'dataset').rglob('*.txt')}
        # Inclui rotulos compartilhados dos positivos e da validacao congelada.
        hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (FONTE/'dataset/labels').rglob('*.txt')})
        (OUT/'hashes_antes_treino.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        print('Novos negativos de treino:',len(registros),flush=True)
    treino.OUT=OUT
    treino.main()
    spec=spec_from_file_location('relatorio',ROOT/'81_conferir_retreino_solo.py')
    rel=module_from_spec(spec);spec.loader.exec_module(rel)
    rel.EXP=OUT;rel.main()
    p=OUT/'RELATORIO.md';txt=p.read_text(encoding='utf-8')
    d=json.loads((OUT/'protocolo_treino.json').read_text())
    txt=txt.replace('Treino real de 16 epocas','Treino real de 12 epocas')
    txt+='\nForam acrescentados '+str(d['negativos_novos'])+' negativos revisados de treino, repetidos duas vezes. '
    txt+='Nenhum negativo reservado de validacao/teste foi usado no treino.\n'
    p.write_text(txt,encoding='utf-8')


if __name__=='__main__':main()
