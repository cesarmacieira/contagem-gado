"""Conferencia experimental CLIP, com margem de similaridade (nao probabilidade)."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parent
SHA_PESO='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
POSITIVOS=['a photo of a cow.','a photo of cattle.','a photo of a bull.',
           'a photo of a calf.','a photo of a zebu bull.']
NEGATIVOS=['a photo of a horse.','a photo of a donkey.','a photo of a mule.',
           'a photo of a person riding a horse.']


def assinatura_receita():
    receita=dict(versao=1,positivos=POSITIVOS,negativos=NEGATIVOS,vistas=[1.0,.6],
        resize=[224,224],regra_geometrica=1)
    return hashlib.sha256(json.dumps(receita,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def carregar(peso,device='cuda'):
    peso=Path(peso)
    if hashlib.sha256(peso.read_bytes()).hexdigest()!=SHA_PESO:
        raise ValueError('Assinatura inesperada do CLIP oficial')
    sys.path.insert(0,str(ROOT/'data/ferramentas/deps'))
    sys.path.insert(0,str(ROOT/'data/ferramentas/openai_clip'))
    import clip
    modelo,preprocess=clip.load(str(peso),device=device,jit=False)
    modelo.eval()
    with torch.inference_mode():
        textos=modelo.encode_text(clip.tokenize(POSITIVOS+NEGATIVOS).to(device)).float()
        textos=textos/textos.norm(dim=1,keepdim=True)
    return modelo,preprocess,textos,device


def margens(verificador,imagem_bgr,caixas):
    b=np.asarray(caixas,dtype=float).reshape(-1,4)
    if not len(b):return np.empty((0,2)),np.empty((0,2,len(POSITIVOS)+len(NEGATIVOS)))
    if not np.isfinite(b).all() or (b[:,2:]<=b[:,:2]).any():
        raise ValueError('Caixas invalidas')
    modelo,preprocess,textos,device=verificador
    im=Image.fromarray(imagem_bgr[:,:,::-1])
    patches=[]
    for x1,y1,x2,y2 in b:
        for fracao in (1.0,.6):
            crop=im.crop((x1,y1,x2,y1+(y2-y1)*fracao))
            # Preserva toda a regiao candidata antes do CenterCrop do CLIP.
            patches.append(preprocess(crop.resize((224,224),Image.Resampling.BICUBIC)))
    saidas=[]
    with torch.inference_mode():
        for i in range(0,len(patches),32):
            f=modelo.encode_image(torch.stack(patches[i:i+32]).to(device)).float()
            f=f/f.norm(dim=1,keepdim=True)
            saidas.append((f@textos.T).cpu().numpy())
    sims=np.concatenate(saidas).reshape(len(b),2,-1)
    margem=sims[:,:,len(POSITIVOS):].max(axis=2)-sims[:,:,:len(POSITIVOS)].max(axis=2)
    return margem,sims


def manter(margem,limiar=.03):
    m=np.asarray(margem,dtype=float)
    if m.ndim!=2 or m.shape[1]!=2 or not np.isfinite(m).all() or limiar<0:
        raise ValueError('Margens ou limiar invalidos')
    # Exige duas vistas concordantes; uma so vista favoravel a bovino preserva.
    return ~(m>=limiar).all(axis=1)


def candidatas_montaria(caixas,pessoas):
    """Exige uma pessoa acima e sobreposta ao animal, sem afirmar sua especie."""
    b=np.asarray(caixas,dtype=float).reshape(-1,4)
    p=np.asarray(pessoas,dtype=float).reshape(-1,4)
    if not np.isfinite(b).all() or not np.isfinite(p).all():
        raise ValueError('Coordenadas invalidas')
    if len(b) and (b[:,2:]<=b[:,:2]).any() or len(p) and (p[:,2:]<=p[:,:2]).any():
        raise ValueError('Caixa sem area positiva')
    if not len(b) or not len(p):return np.zeros(len(b),dtype=bool)
    largura=b[:,2]-b[:,0];altura=b[:,3]-b[:,1]
    cx=(p[:,0]+p[:,2])/2;cy=(p[:,1]+p[:,3])/2
    horizontal=(cx[None,:]>=b[:,None,0]-.1*largura[:,None]) & (cx[None,:]<=b[:,None,2]+.1*largura[:,None])
    vertical=(p[None,:,1]<=b[:,None,1]) & (p[None,:,3]>=b[:,None,1]+.1*altura[:,None])
    vertical &= (p[None,:,3]<=b[:,None,3]+.1*altura[:,None]) & (cy[None,:]<(b[:,1]+b[:,3])[:,None]/2)
    return (horizontal & vertical).any(axis=1)
