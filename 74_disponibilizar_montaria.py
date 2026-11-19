"""Integra conferencia contextual somente na opcao experimental ja existente."""
from pathlib import Path
import hashlib
import json
from verificador_semantico import assinatura_receita,SHA_PESO

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'data/experimentos/verificador_montaria_v4'
DEST=ROOT/'data/modelos/conferencia_montaria_v1'


def main():
    resultado=json.loads((EXP/'resultado.json').read_text())
    assert all(not v['perdeu_tp'] and not v['piorou_erro'] for v in resultado['rotas'].values())
    b=resultado['bernard']
    assert (b['antes'],b['depois'],b['montaria_depois'],b['pontos_depois'])==(29,28,0,20)
    protocolo=json.loads((EXP/'protocolo_antes_pessoas.json').read_text())
    sem=json.loads((ROOT/'data/experimentos/verificador_clip_tp_v3/protocolo_antes_reavaliacao.json').read_text())
    clip=ROOT/'data/modelos_candidatos/ViT-B-32.pt';pessoas=ROOT/'data/modelos_candidatos/yolo11x.pt'
    assert hashlib.sha256(clip.read_bytes()).hexdigest()==SHA_PESO
    assert hashlib.sha256(pessoas.read_bytes()).hexdigest()==protocolo['peso_pessoa_sha256']
    montaria=dict(ativo=True,versao=1,peso_clip=clip.relative_to(ROOT).as_posix(),sha_clip=SHA_PESO,
        peso_pessoas=pessoas.relative_to(ROOT).as_posix(),sha_pessoas=protocolo['peso_pessoa_sha256'],
        receita_sha256=assinatura_receita(),limiar=protocolo['limiar_semantico'],
        pessoa_imgsz=1280,pessoa_conf=.5,regra_geometrica=1,
        evidencia='data/experimentos/verificador_montaria_v4/resultado.json')
    apontador=ROOT/'data/modelos/verificador_solo_v1/config.json'
    cfg=json.loads(apontador.read_text())
    assert cfg['visoes_permitidas']==['solo_denso'] and not cfg['ativo_por_padrao']
    assert cfg['classificador_sha256']==sem['verificador_anterior']['classificador_sha256']
    DEST.mkdir(parents=True,exist_ok=True)
    if not (DEST/'config_anterior.json').exists():
        (DEST/'config_anterior.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    cfg['montaria']=montaria
    cfg['descricao']='Conferência complementar experimental de objetos e montarias.'
    cfg['limite']='Montaria corrigida no caso conhecido Bernard; sem contagem total exata desse rebanho nem novo teste denso independente. Somente solo_denso, desligada por padrão.'
    (DEST/'config.json').write_text(json.dumps(montaria,ensure_ascii=False,indent=2),encoding='utf-8')
    apontador.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Conferencia de montaria integrada; somente solo_denso e opcao experimental desligada por padrao.',flush=True)


if __name__=='__main__':main()
