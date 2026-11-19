"""Reduz influencia de vizinhos; todos os testes anteriores agora sao conhecidos."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import json
import numpy as np
from verificador_visual import extrair_features

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/verificador_contexto_v5'
CONTEXTOS=(1.0,1.05)


def features(extrator,im,b,blur=False):
    if not blur:
        return extrair_features(extrator,im,b,contextos=CONTEXTOS)
    return np.concatenate([extrair_features(extrator,im,b,contextos=CONTEXTOS,resolucao=n,raio=r)
        for n,r in ((24,0),(12,1),(8,2))])


def main():
    extras=json.loads((ROOT/'data/confundidores_especies_v1/revisao_visual.json').read_text())
    extras=[r for r in extras if r['revisado']]
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Experimento preservado')
    (OUT/'protocolo_contextos.json').write_text(json.dumps(dict(contextos=CONTEXTOS,
        motivacao='Recortes largos incluem bovinos ao redor da montaria; limitar ao objeto candidato.',
        treino_fonte='Mesmo treino de v4, sem imagem de Bernard',
        avaliacao='Validacao e regressao conhecida; nenhum teste independente novo'),indent=2),encoding='utf-8')
    spec=spec_from_file_location('treino_base',ROOT/'55_treinar_verificador_solo.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    mod.POS=ROOT/'data/experimentos/verificador_auxiliar_v3/positivos'
    mod.OUT=OUT;mod.features=features
    mod.main(extras,contextos=CONTEXTOS)
    for nome in ('escolha_antes_teste.json','resultado.json'):
        p=OUT/nome;d=json.loads(p.read_text());d['limites']='Somente validacao e regressao conhecida, inclusive negativos reservados de rodadas anteriores.'
        if 'limite' in d:d['limite']=d['limites']
        p.write_text(json.dumps(d,indent=2),encoding='utf-8')


if __name__=='__main__':main()
