"""Diagnostico de regressao conhecida, sem ajustar limiar aos resultados."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import json
import cv2
import numpy as np
import torch
import contagem as ct
from verificador_semantico import carregar,margens,manter,POSITIVOS,NEGATIVOS,SHA_PESO

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/verificador_clip_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'bernard_resultado.json').exists():raise ValueError('Diagnostico existente preservado')
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(dict(
        peso='ViT-B-32.pt',sha256=SHA_PESO,prompts_positivos=POSITIVOS,prompts_negativos=NEGATIVOS,
        vistas=['caixa_inteira','60_porcento_superior'],margem=.03,criterio='Rejeitar somente com ambas as vistas concordantes',
        avaliacao='Regressao conhecida. Nenhum teste independente novo.'),indent=2),encoding='utf-8')
    torch.set_num_threads(4)
    ver=carregar(ROOT/'data/modelos_candidatos/ViT-B-32.pt')
    base=ROOT/'data/busca_teste_denso_independente'
    im=cv2.imread(str(base/'bernard_28059238281.jpg'))
    d=np.load(ROOT/'data/experimentos/conferencia_especies_v1/bernard_caixas.npz')
    b=d['antes'];scores=d['scores']
    margem,sims=margens(ver,im,b);mask=manter(margem,.03)
    spec=spec_from_file_location('pontos',ROOT/'67_verificar_especies_no_denso.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    pontos=json.loads((base/'avaliacao_minimo_bernard/gabarito_antes_inferencia.json').read_text())['pontos_xy']
    ref=np.array([1400,1090,1820,1980],float)
    inter=np.maximum(np.minimum(b[:,2:],ref[2:])-np.maximum(b[:,:2],ref[:2]),0).prod(axis=1)
    iou=inter/((b[:,2:]-b[:,:2]).prod(axis=1)+np.prod(ref[2:]-ref[:2])-inter)
    alvo=iou>.5
    np.savez(OUT/'bernard_cache.npz',caixas=b,scores=scores,margens=margem,similaridades=sims,manter=mask)
    resumo=dict(antes=len(b),depois=int(mask.sum()),montaria_antes=int(alvo.sum()),montaria_depois=int((alvo&mask).sum()),
        pontos_antes=mod.correspondencias_pontos(pontos,b),pontos_depois=mod.correspondencias_pontos(pontos,b[mask]),
        margens_montaria=margem[alvo].tolist(),contagem_exata=False,promovido=False)
    (OUT/'bernard_resultado.json').write_text(json.dumps(resumo,indent=2),encoding='utf-8')
    cv2.imwrite(str(OUT/'bernard_depois.jpg'),ct.desenhar(im,b[mask]))
    print(json.dumps(resumo,indent=2),flush=True)


if __name__=='__main__':main()
