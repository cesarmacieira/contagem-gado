"""Verificador de nucleo retangular; nunca amplia a caixa para o contexto."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from PIL import Image,ImageOps
from torchvision.transforms.functional import to_tensor,normalize
from verificador_visual import carregar_verificador


def assinatura_receita():
    return hashlib.sha256(json.dumps(dict(versao=1,recorte='retangulo_estrito',
        vistas=['RGB','cinza_RGB'],resize=[224,224],normalizacao='ImageNet1K_V1',
        features='EfficientNetB0_avgpool_L2_concatenacao'),sort_keys=True).encode()).hexdigest()


def recorte_estrito(pil,box):
    b=np.asarray(box,dtype=float)
    if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=b[:2]).any():
        raise ValueError('Caixa invalida')
    x1,y1,x2,y2=b
    rect=(max(0,x1),max(0,y1),min(pil.width,x2),min(pil.height,y2))
    if rect[2]<=rect[0] or rect[3]<=rect[1]:raise ValueError('Caixa fora da imagem')
    return pil.crop(rect)


def features(encoder,im,boxes,blur=False):
    if not len(boxes):return np.empty((0,2560),np.float32)
    model,transform,_=encoder;device=next(model.parameters()).device
    pil=Image.fromarray(im[:,:,::-1]);patches=[]
    for box in boxes:
        crop=recorte_estrito(pil,box)
        if blur:crop=crop.resize((24,24),Image.Resampling.BICUBIC)
        for view in (crop,ImageOps.grayscale(crop).convert('RGB')):
            patches.append(normalize(to_tensor(view.resize((224,224),Image.Resampling.BICUBIC)),transform.mean,transform.std))
    encoded=[]
    with torch.inference_mode():
        for j in range(0,len(patches),32):
            f=model.avgpool(model.features(torch.stack(patches[j:j+32]).to(device))).flatten(1)
            encoded.append(torch.nn.functional.normalize(f,dim=1).cpu().numpy())
    return np.concatenate(encoded).reshape(-1,2560)


def carregar(extrator,classificador,sha_extrator,sha_classificador,receita_sha,device='cuda'):
    import joblib
    if assinatura_receita()!=receita_sha:raise ValueError('Receita do nucleo alterada')
    if hashlib.sha256(Path(extrator).read_bytes()).hexdigest()!=sha_extrator:raise ValueError('Extrator alterado')
    if hashlib.sha256(Path(classificador).read_bytes()).hexdigest()!=sha_classificador:raise ValueError('Classificador alterado')
    clf=joblib.load(classificador)
    if (getattr(clf,'receita_extracao_',None)!='retangulo_rgb_cinza_v1'
            or clf.n_features_in_!=2560 or list(clf.classes_)!=[0,1]):
        raise ValueError('Classificador incompativel com recorte estrito')
    return carregar_verificador(extrator,device=device),clf


def pontuar(verificador,im,boxes):
    if not len(boxes):return np.empty(0)
    encoder,clf=verificador
    return clf.predict_proba(features(encoder,im,boxes))[:,1]


def mascara(scores,limiar):
    s=np.asarray(scores,dtype=float)
    if s.ndim!=1 or not np.isfinite(s).all() or ((s<0)|(s>1)).any() or not 0<=limiar<=1:
        raise ValueError('Scores ou limiar invalidos')
    return s>=limiar
