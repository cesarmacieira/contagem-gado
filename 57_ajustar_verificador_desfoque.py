"""Experimento de desfoque: somente treino existente; avaliacao ja conhecida."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image, ImageFilter
from verificador_visual import recorte_quadrado

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('treino_binario', ROOT / '55_treinar_verificador_solo.py')
treino = module_from_spec(spec)
spec.loader.exec_module(treino)
original_features = treino.features


def features(extrator, im, caixas, blur=False):
    if not blur or not len(caixas):
        return original_features(extrator, im, caixas, False)
    modelo, transform, _ = extrator
    pil = Image.fromarray(im[:, :, ::-1])
    resultados = []
    # Mesmas degradacoes para positivos e negativos, evitando que a qualidade
    # artificial da imagem se torne uma pista para a classe.
    for resolucao, raio in ((24, 0), (12, 1), (8, 2)):
        patches = []
        for caixa in caixas:
            for fator in (1.3, 2.0):
                crop = recorte_quadrado(pil, caixa, fator)
                crop = crop.resize((resolucao, resolucao), Image.Resampling.BICUBIC)
                if raio:
                    crop = crop.filter(ImageFilter.GaussianBlur(raio))
                patches.append(transform(crop))
        partes = []
        with torch.inference_mode():
            for i in range(0, len(patches), 32):
                batch = torch.stack(patches[i:i+32]).to('cuda')
                f = modelo.avgpool(modelo.features(batch)).flatten(1)
                partes.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        resultados.append(np.concatenate(partes).reshape(len(caixas), 2560))
    return np.concatenate(resultados)


if __name__ == '__main__':
    treino.OUT = ROOT / 'data/experimentos/verificador_desfoque_v2'
    treino.OUT.mkdir(parents=True, exist_ok=True)
    (treino.OUT / 'protocolo_ajuste.json').write_text(json.dumps({
        'aumento': [[24, 0], [12, 1], [8, 2]],
        'aplicado_a_ambas_classes': True,
        'origem_positivos': 'somente train do catalogo anterior',
        'avaliacao': 'validacao e regressao conhecidas; nenhum teste independente novo',
        'promocao_automatica': False
    }, indent=2), encoding='utf-8')
    treino.features = features
    treino.main()
    # A descricao herdada da primeira rodada nao se aplica: todos os negativos
    # reservados ja foram consumidos naquela rodada.
    for nome in ('escolha_antes_teste.json', 'resultado.json'):
        p = treino.OUT / nome
        dados = json.loads(p.read_text(encoding='utf-8'))
        dados['limites'] = 'Avaliacao de desenvolvimento e regressao conhecida. Sem teste novo independente.'
        if 'limite' in dados:
            dados['limite'] = dados['limites']
        p.write_text(json.dumps(dados, indent=2), encoding='utf-8')
