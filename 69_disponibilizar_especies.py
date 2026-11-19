"""Atualiza somente a opcao experimental de solo denso apos conferencias."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent
BIN=ROOT/'data/experimentos/verificador_especies_v4'
DEST=ROOT/'data/modelos/verificador_solo_v4'


def main():
    evidencia=json.loads((ROOT/'data/experimentos/verificador_especies_rotas_v4/resultado.json').read_text())
    teste=json.loads((BIN/'teste_especies.json').read_text())['resumo']['test']
    assert evidencia['rotas']['solo_denso']['disponivel_apenas_com_opt_in']
    assert not evidencia['rotas']['solo_poucos']['disponivel_apenas_com_opt_in']
    assert teste['n']==6 and teste['aceitos_erradamente_antes']==5 and teste['aceitos_erradamente_depois']==0
    cfg=evidencia['config']
    cfg.update(versao=4,visoes_permitidas=['solo_denso'],ativo_por_padrao=False,
        extrator='data/modelos/verificador_solo_v4/extrator.pth',
        classificador='data/modelos/verificador_solo_v4/classificador.joblib',
        descricao='Conferencia experimental treinada tambem com cavalos e outros nao bovinos.',
        limite='Ainda aceita a montaria na cena Bernard. Teste negativo novo tem seis recortes, sem novo teste positivo brasileiro denso exato. Solo_poucos bloqueado.',
        evidencia='data/experimentos/verificador_especies_v4/teste_especies.json')
    ext=ROOT/'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth'
    clf=BIN/'classificador.joblib'
    assert hashlib.sha256(ext.read_bytes()).hexdigest()==cfg['extrator_sha256']
    assert hashlib.sha256(clf.read_bytes()).hexdigest()==cfg['classificador_sha256']
    assert hashlib.sha256((ROOT/'data/processed/modelo_gado.pt').read_bytes()).hexdigest()==cfg['detector_sha256']
    DEST.mkdir(parents=True,exist_ok=True)
    apontador=ROOT/'data/modelos/verificador_solo_v1/config.json'
    if not (DEST/'config_anterior.json').exists():
        shutil.copy2(apontador,DEST/'config_anterior.json')
    shutil.copy2(ext,DEST/'extrator.pth');shutil.copy2(clf,DEST/'classificador.joblib')
    texto=json.dumps(cfg,ensure_ascii=False,indent=2)
    (DEST/'config.json').write_text(texto,encoding='utf-8')
    apontador.write_text(texto,encoding='utf-8')
    print('Verificador v4 disponivel somente em solo_denso; opcao desligada por padrao.',flush=True)


if __name__=='__main__':main()
