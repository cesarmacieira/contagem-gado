"""Segunda leitura visual das caixas; prototipo, sem modificar o detector."""
from pathlib import Path
import hashlib
import math
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def carregar_verificador(peso, device='cuda'):
    peso = Path(peso)
    digest = hashlib.sha256(peso.read_bytes()).hexdigest()
    if not digest.startswith('7f5810bc'):
        raise ValueError('Assinatura inesperada do peso oficial EfficientNet-B0')
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    modelo = efficientnet_b0(weights=None)
    modelo.load_state_dict(torch.load(peso,map_location='cpu',weights_only=True))
    modelo.eval().to(device)
    ids = [i for i,n in enumerate(weights.meta['categories']) if n in ('ox','water buffalo','bison')]
    assert len(ids)==3
    return modelo,weights.transforms(),ids


def recorte_quadrado(im, caixa, fator=1.3):
    x1,y1,x2,y2 = map(float,caixa)
    if not all(math.isfinite(x) for x in (x1,y1,x2,y2)) or x2<=x1 or y2<=y1 or fator<1:
        raise ValueError('Caixa ou fator de contexto invalido')
    lado = max(1,math.ceil(max(x2-x1,y2-y1)*fator))
    x,y = math.floor((x1+x2-lado)/2),math.floor((y1+y2-lado)/2)
    return im.crop((x,y,x+lado,y+lado))


def pontuar(verificador, imagem_bgr, caixas, contextos=(1.3,2.0), lote=32, device='cuda'):
    if not len(caixas):
        return np.empty(0)
    modelo,transform,ids = verificador
    im = Image.fromarray(imagem_bgr[:,:,::-1])
    recortes = [transform(recorte_quadrado(im,b,fator)) for b in caixas for fator in contextos]
    scores = []
    with torch.inference_mode():
        for i in range(0,len(recortes),lote):
            batch = torch.stack(recortes[i:i+lote]).to(device)
            prob = modelo(batch).softmax(dim=1)[:,ids].sum(dim=1)
            scores.extend(prob.cpu().numpy().tolist())
    return np.asarray(scores).reshape(len(caixas),len(contextos)).max(axis=1)


def extrair_features(extrator,imagem_bgr,caixas,blur=False,device='cuda',contextos=(1.3,2.0),resolucao=None,raio=0):
    """Mesmo encoder/contexto usado no treino do classificador binario v1."""
    if not len(caixas):
        return np.empty((0,2560),dtype=np.float32)
    if len(contextos)!=2 or any(not math.isfinite(f) or f<1 for f in contextos):
        raise ValueError('Dois fatores de contexto validos sao obrigatorios')
    modelo,transform,_ = extrator
    pil = Image.fromarray(imagem_bgr[:,:,::-1])
    patches = []
    for box in caixas:
        for fator in contextos:
            crop = recorte_quadrado(pil,box,fator)
            if blur or resolucao is not None:
                lado=resolucao or 24
                crop = crop.resize((lado,lado),Image.Resampling.BICUBIC)
            if raio:
                crop=crop.filter(ImageFilter.GaussianBlur(raio))
            patches.append(transform(crop))
    encoded = []
    with torch.inference_mode():
        for i in range(0,len(patches),32):
            x = torch.stack(patches[i:i+32]).to(device)
            x = modelo.avgpool(modelo.features(x)).flatten(1)
            x = torch.nn.functional.normalize(x,dim=1)
            encoded.append(x.cpu().numpy())
    return np.concatenate(encoded).reshape(len(caixas),2560)


def carregar_binario(extrator_peso,classificador_peso,sha_classificador,device='cuda'):
    import joblib
    if hashlib.sha256(Path(classificador_peso).read_bytes()).hexdigest()!=sha_classificador:
        raise ValueError('Classificador alterado desde a avaliacao')
    extrator = carregar_verificador(extrator_peso,device=device)
    clf = joblib.load(classificador_peso)
    if clf.n_features_in_!=2560 or list(clf.classes_)!=[0,1]:
        raise ValueError('Classificador incompativel com o extrator v1')
    return extrator,clf


def confirmar(verificador,imagem_bgr,caixas,device='cuda'):
    if not len(caixas):
        return np.empty(0)
    extrator,clf = verificador
    return clf.predict_proba(extrair_features(extrator,imagem_bgr,caixas,device=device,
        contextos=getattr(clf,'contextos_',(1.3,2.0))))[:,1]


def mascara_confirmacao(caixas,scores,limiar,min_lado_px=0):
    """Abstem-se em objetos pequenos, preservando a decisao do detector."""
    b = np.asarray(caixas,dtype=float).reshape(-1,4)
    s = np.asarray(scores,dtype=float)
    if len(b)!=len(s) or not np.isfinite(s).all() or min_lado_px<0:
        raise ValueError('Caixas/scores ou limite de tamanho invalidos')
    pequenos = np.minimum(b[:,2]-b[:,0],b[:,3]-b[:,1]) < min_lado_px
    return (s>=limiar) | pequenos,pequenos
