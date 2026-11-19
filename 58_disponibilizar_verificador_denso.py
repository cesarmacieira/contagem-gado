"""Disponibiliza opcao experimental, sem ativar por padrao ou trocar YOLO."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
EXP = ROOT / 'data/experimentos/verificador_desfoque_rotas_v2'
BIN = ROOT / 'data/experimentos/verificador_desfoque_v2'
DEST = ROOT / 'data/modelos/verificador_solo_v2'


def main():
    resultado = json.loads((EXP / 'resultado.json').read_text())
    assert resultado['rotas']['solo_denso']['disponivel_apenas_com_opt_in']
    assert not resultado['rotas']['solo_poucos']['disponivel_apenas_com_opt_in']
    cfg = resultado['config']
    cfg.update(versao=3, ativo_por_padrao=False, visoes_permitidas=['solo_denso'],
        extrator='data/modelos/verificador_solo_v2/extrator.pth',
        classificador='data/modelos/verificador_solo_v2/classificador.joblib',
        limite='Somente desenvolvimento e regressao conhecida. Sem teste brasileiro denso independente novo. Bloqueado para solo_poucos.',
        evidencia='data/experimentos/verificador_desfoque_rotas_v2/resultado.json')
    for origem, esperado in ((BIN / 'classificador.joblib', cfg['classificador_sha256']),
        (ROOT / 'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth', cfg['extrator_sha256'])):
        assert hashlib.sha256(origem.read_bytes()).hexdigest() == esperado
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BIN / 'classificador.joblib', DEST / 'classificador.joblib')
    shutil.copy2(ROOT / 'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth', DEST / 'extrator.pth')
    (DEST / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    # O aplicativo consulta este apontador; os pesos antigos permanecem intactos.
    (ROOT / 'data/modelos/verificador_solo_v1/config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print('Opcao experimental disponivel apenas para solo_denso; desligada por padrao.')


if __name__ == '__main__':
    main()
