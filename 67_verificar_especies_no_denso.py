"""Regressao conhecida da foto densa, usando caixas ja armazenadas."""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from verificador_visual import carregar_binario,confirmar,mascara_confirmacao
import contagem as ct

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/verificador_especies_v4'


def correspondencias_pontos(pontos,caixas):
    adj=[[i for i,(x1,y1,x2,y2) in enumerate(caixas) if x1<=x<=x2 and y1<=y<=y2] for x,y in pontos]
    donos={}
    def tentar(p,usados):
        for b in adj[p]:
            if b in usados:continue
            usados.add(b)
            if b not in donos or tentar(donos[b],usados):
                donos[b]=p;return True
        return False
    return sum(tentar(p,set()) for p in range(len(pontos)))


def main():
    if (OUT/'bernard_regressao.json').exists():raise ValueError('Resultado preservado')
    torch.set_num_threads(4)
    cfg=json.loads((OUT/'escolha_antes_teste.json').read_text())
    peso=ROOT/'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth'
    modelo=carregar_binario(peso,OUT/'classificador.joblib',cfg['classificador_sha256'])
    d=np.load(ROOT/'data/experimentos/conferencia_especies_v1/bernard_caixas.npz')
    b,s=d['antes'],d['scores']
    base=ROOT/'data/busca_teste_denso_independente'
    im=cv2.imread(str(base/'bernard_28059238281.jpg'))
    scores=confirmar(modelo,im,b)
    manter,_=mascara_confirmacao(b,scores,cfg['limiar'])
    pts=json.loads((base/'avaliacao_minimo_bernard/gabarito_antes_inferencia.json').read_text())['pontos_xy']
    # Regiao da montaria revisada depois da primeira inferencia: regressao,
    # jamais novo teste independente nem rotulo utilizado no treino.
    ref=np.array([1400,1090,1820,1980],float)
    inter=np.maximum(np.minimum(b[:,2:],ref[2:])-np.maximum(b[:,:2],ref[:2]),0).prod(axis=1)
    iou=inter/((b[:,2:]-b[:,:2]).prod(axis=1)+np.prod(ref[2:]-ref[:2])-inter)
    alvos=iou>.5
    resumo=dict(antes=len(b),depois=int(manter.sum()),pontos_antes=correspondencias_pontos(pts,b),
        pontos_depois=correspondencias_pontos(pts,b[manter]),caixas_montaria_antes=int(alvos.sum()),
        caixas_montaria_depois=int((alvos & manter).sum()),contagem_exata=False,
        limite='Regressao conhecida; correspondencia de pontos nao equivale a IoU de caixas nem gabarito total.')
    (OUT/'bernard_regressao.json').write_text(json.dumps(resumo,indent=2),encoding='utf-8')
    np.savez(OUT/'bernard_verificacao.npz',caixas=b,scores_detector=s,scores_verificador=scores,manter=manter)
    vis=ct.desenhar(im,b[manter]);cv2.imwrite(str(OUT/'bernard_depois.jpg'),vis)
    print(json.dumps(resumo,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--binario',type=Path,default=OUT)
    args=ap.parse_args();OUT=args.binario.resolve();main()
