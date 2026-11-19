"""Snapshot privado do treino aprovado; nao escreve no lote da coletora."""
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import hashlib
import json
import shutil
import pandas as pd
from integracao_denso import conferir

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/experimentos/verificador_auxiliar_v3'
POS = OUT / 'positivos'
BASE = ROOT / 'data/brasil_denso_novo'
ANTIGO = ROOT / 'data/experimentos/especialista_solo_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    resumo = conferir(BASE)
    assert not resumo['erros'], resumo['erros']
    (OUT / 'conferencia_entrada.json').write_text(json.dumps(resumo, indent=2), encoding='utf-8')
    catalogo = pd.read_csv(ANTIGO / 'catalogo.csv')
    catalogo = catalogo[(catalogo.conjunto == 'train') & (catalogo.fonte == 'brasil_antigo_train')]
    linhas = []
    for r in catalogo.to_dict('records'):
        origem = list((ANTIGO / 'dataset/images/train').glob(r['nome'] + '.*'))
        assert len(origem) == 1
        assert hashlib.sha256(origem[0].read_bytes()).hexdigest() == r['sha256']
        linhas.append((r, origem[0], ANTIGO / 'dataset/labels/train' / (r['nome'] + '.txt')))
    for r in resumo['registros']:
        if r['conjunto'] != 'train':
            continue
        nome = 'novo__' + Path(r['arquivo']).stem
        linha = dict(nome=nome, original=str(BASE / 'images' / r['arquivo']), grupo=r['grupo'],
            conjunto='train', repeticoes=1, n_bovinos=r['n_bovinos'], fonte='brasil_denso_novo', sha256=r['sha256'])
        linhas.append((linha, BASE / 'images' / r['arquivo'], BASE / 'labels' / (Path(r['arquivo']).stem + '.txt')))
    assert sum(r['n_bovinos'] for r, _, _ in linhas if r['fonte'] == 'brasil_denso_novo') == 62
    for linha, imagem, rotulo in linhas:
        img_dest = POS / 'dataset/images/train' / (linha['nome'] + imagem.suffix)
        rot_dest = POS / 'dataset/labels/train' / (linha['nome'] + '.txt')
        img_dest.parent.mkdir(parents=True, exist_ok=True)
        rot_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(imagem, img_dest)
        shutil.copy2(rotulo, rot_dest)
    pd.DataFrame([r for r, _, _ in linhas]).to_csv(POS / 'catalogo.csv', index=False)
    spec = spec_from_file_location('desfoque', ROOT / '57_ajustar_verificador_desfoque.py')
    modulo = module_from_spec(spec)
    spec.loader.exec_module(modulo)
    modulo.treino.POS = POS
    modulo.treino.OUT = OUT
    modulo.treino.features = modulo.features
    modulo.treino.main()
    for nome in ('escolha_antes_teste.json', 'resultado.json'):
        p = OUT / nome
        d = json.loads(p.read_text())
        d['limites'] = 'Somente validacao e regressao conhecida; nenhum teste independente novo.'
        if 'limite' in d:
            d['limite'] = d['limites']
        p.write_text(json.dumps(d, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
