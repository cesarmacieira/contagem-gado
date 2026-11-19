"""Audita omissoes com correspondencia um-a-um; nao altera modelos nem rotulos."""
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
from recuperacao_multiescala import sobreposicoes

ROOT = Path(__file__).resolve().parent
FONTE = ROOT / 'data/experimentos/recuperacao_multiescala_v1'
OUT = ROOT / 'data/experimentos/diagnostico_omissoes_v1'


def correspondencias(ref, caixas, scores, limiar=.5):
    """Mesmo criterio de avaliacao.py; devolve o indice previsto por rotulo."""
    usados = {}
    for k in np.argsort(-np.asarray(scores), kind='stable'):
        io, _ = sobreposicoes(caixas[k], ref)
        if usados:
            io[list(usados)] = -1
        if len(io) and io.max() >= limiar:
            usados[int(io.argmax())] = int(k)
    return usados


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = spec_from_file_location('referencias', ROOT / '72_calibrar_clip_correspondencias.py')
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = []
    for nome in ('bdn_08_c1.jpg', 'bdn_08_c2.jpg'):
        p = FONTE / (Path(nome).stem + '.npz')
        d = np.load(p)
        ref = mod.referencia('val_positiva', nome)
        matches = {v: correspondencias(ref, d[v], d['scores_' + v]) for v in ('base', 'tta', 'alta')}
        image = cv2.imread(str(ROOT / 'data/experimentos/solo_denso_validacao_v2/validacao/images' / nome))
        for j, box in enumerate(ref):
            if j in matches['base']:
                continue
            io, _ = sobreposicoes(box, d['base'])
            row = dict(arquivo=nome, indice_rotulo=j, caixa_rotulo=box.tolist(),
                       iou_base_max=float(io.max()) if len(io) else 0,
                       recuperado_tta=j in matches['tta'], recuperado_alta=j in matches['alta'],
                       cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            rows.append(row)
            x1, y1, x2, y2 = box
            pad = max(20, int(max(x2-x1, y2-y1)))
            h, w = image.shape[:2]
            left, top = max(0, int(x1)-pad), max(0, int(y1)-pad)
            right, bottom = min(w, int(x2)+pad), min(h, int(y2)+pad)
            panels = []
            for v in ('base', 'tta', 'alta'):
                tile = image[top:bottom, left:right].copy()
                for b in d[v]:
                    if b[2] < left or b[0] > right or b[3] < top or b[1] > bottom:
                        continue
                    a = np.round(b - [left, top, left, top]).astype(int)
                    cv2.rectangle(tile, tuple(a[:2]), tuple(a[2:]), (255, 120, 0), 1)
                a = np.round(box - [left, top, left, top]).astype(int)
                cv2.rectangle(tile, tuple(a[:2]), tuple(a[2:]), (0, 255, 0), 1)
                tile = cv2.resize(tile, (420, 300), interpolation=cv2.INTER_NEAREST)
                label = v + (' / TP' if j in matches[v] else ' / FN')
                cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 3)
                cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
                panels.append(tile)
            cv2.imwrite(str(OUT / (Path(nome).stem + f'_rotulo_{j}.jpg')), np.concatenate(panels, axis=1))
    pd.DataFrame(rows).to_csv(OUT / 'omissoes.csv', index=False)
    (OUT / 'omissoes.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    texto = '''# Recuperacao de bovinos omitidos

Foram comparadas 15 entradas de desenvolvimento e regressao conhecida. Os dois recortes de validacao representam uma unica cena, com 32 bovinos. Nao houve novo teste independente, treino ou alteracao no aplicativo, nos pesos, na configuracao ativa ou nas anotacoes de origem.

| Estrategia | TP em val | FP em val | FN em val | MAE de contagem em val | Decisao |
|---|---:|---:|---:|---:|---|
| Referencia atual | 26 | 2 | 6 | 2,0 | Preservada |
| TTA 1280 | 28 | 2 | 4 | 1,0 | Rejeitada por regressao nas demais cenas |
| Resolucao 1920 | 27 | 1 | 5 | 2,0 | Rejeitada: perde correspondencias anteriores |
| Consenso conservador | 26 | 2 | 6 | 2,0 | Nao recupera omissoes em val |
| Consenso de quantidade | 26 | 2 | 6 | 2,0 | Rejeitado: brasil_05 sobe de 2 para 3 |
| Separacao de caixas fundidas | 26 | 2 | 6 | 2,0 | Nao houve substituicoes elegiveis |

TTA reduz duas omissoes na validacao, mas brasil_03 sobe de 33 para 45 (intervalo de referencia 24 a 32), brasil_04 de 20 para 25 (16 a 22) e brasil_05 de 2 para 3 (real 2). A melhora local nao justifica substituir a configuracao atual.

O CSV omissoes.csv identifica cada um dos seis rotulos sem correspondencia na referencia. As figuras mostram o mesmo recorte nas tres inferencias: verde e a caixa do gabarito; azul e a predicao. TP/FN considera correspondencia individual por IoU >= 0,5, em ordem de confianca, igual ao avaliador do projeto. Presenca de uma caixa proxima, sozinha, nao comprova recuperacao.

FN nao significa necessariamente animal sem nenhuma caixa: tambem inclui localizacao abaixo do IoU minimo e caixas fundidas. Na figura do rotulo 23 de bdn_08_c1, a caixa base abrange dois animais; isso nao autoriza somar automaticamente outra deteccao. A tentativa de separar fusoes exige duas partes confirmadas em ambas as inferencias e impede duplicar um animal ja correspondente a outra caixa base. Nenhum caso passou por todos esses criterios; esse candidato tambem nao foi promovido.

Os protocolos, caches e metricas completos ficam em recuperacao_multiescala_v1 e recuperacao_consenso_v2, na pasta de experimentos. A anotacao e os grupos de treino, validacao e teste foram preservados. Nenhum candidato foi promovido.
'''
    (OUT / 'RELATORIO.md').write_text(texto, encoding='utf-8')
    print(json.dumps(dict(omissoes=len(rows), recuperadas_tta=sum(r['recuperado_tta'] for r in rows),
                         recuperadas_alta=sum(r['recuperado_alta'] for r in rows), pasta=str(OUT)), indent=2))


if __name__ == '__main__':
    main()
