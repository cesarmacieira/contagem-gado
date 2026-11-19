"""Regressao solicitada pelo usuario; nao escolhe parametros pelo total esperado."""
from pathlib import Path
import hashlib,json,shutil
import cv2,numpy as np,torch
from ultralytics import YOLO
import contagem as ct
from iluminacao import detectar_passagens,combinar,variantes

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/iluminacao_nelore_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    source=Path('C:/Users/cesar.macieira/Downloads/nelore_1512.png')
    shutil.copy2(source,OUT/'original.png')
    protocol=dict(imagem_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),gabarito_usuario=7,
        conf=.3,iou=.5,imgsz=1024,suporte_min=2,variantes=['original','contraste_local','sombras_claras','luzes_reduzidas'],
        pesos_sha256=hashlib.sha256((ROOT/'data/processed/modelo_gado.pt').read_bytes()).hexdigest(),
        limite='Foto conhecida; total informado pelo usuario. Sem gabarito espacial independente. Contagem igual a 7 nao prova correspondencia correta.')
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.set_num_threads(4);model=YOLO(str(ROOT/'data/processed/modelo_gado.pt'));im=cv2.imread(str(source))
    passes=detectar_passagens(model,im,classes=[0]);summary={}
    for nome,(b,s) in passes.items():
        np.savez(OUT/(nome+'.npz'),caixas=b,scores=s);keep=s>=.3;summary[nome]=int(keep.sum())
        cv2.imwrite(str(OUT/(nome+'.jpg')),ct.desenhar(variantes(im)[nome],b[keep]))
    b,s,audit=combinar(passes);cv2.imwrite(str(OUT/'combinada.jpg'),ct.desenhar(im,b));np.savez(OUT/'combinada.npz',caixas=b,scores=s)
    result=dict(contagens=summary,combinada=len(b),recuperadas=audit,erro_contagem=abs(len(b)-7),
        api_chamada=False,pesos_alterados=False,limite=protocol['limite'])
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
