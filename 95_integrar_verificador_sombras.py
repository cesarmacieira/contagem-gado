"""Integra apenas a nova visao externa; verifica o bloco real do app sem servidor."""
from pathlib import Path
import ast,copy,hashlib,json,shutil
from types import SimpleNamespace
import cv2,numpy as np,pandas as pd,torch
from verificador_nucleo import assinatura_receita

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/integracao_sombras_app_v1'


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Integracao ja registrada')
    train=ROOT/'data/experimentos/verificador_sombras_nucleo_v1'
    val=ROOT/'data/experimentos/pipeline_sombras_validacao_v1'
    protocolo=json.loads((val/'protocolo_antes_validacao.json').read_text())
    escolha=json.loads((val/'resultado.json').read_text())
    assert escolha['selecionada']['conf']==.2
    source=train/'com_nucleos_retangulo.joblib'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==protocolo['classificador_sha256']
    dest=ROOT/'data/modelos/verificador_sombras_v1';dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest/'classificador_retangulo.joblib')
    encoder=ROOT/'data/modelos_candidatos/efficientnet_b0_rwightman-7f5810bc.pth'
    shutil.copy2(encoder,dest/'extrator.pth')
    cfg=dict(versao=1,extrator='data/modelos/verificador_sombras_v1/extrator.pth',
        classificador='data/modelos/verificador_sombras_v1/classificador_retangulo.joblib',
        extrator_sha256=hashlib.sha256(encoder.read_bytes()).hexdigest(),
        classificador_sha256=protocolo['classificador_sha256'],receita_sha256=assinatura_receita(),
        limiar=protocolo['limiar_verificador'])
    catalog_path=ROOT/'data/modelos/especialistas.json';catalog=json.loads(catalog_path.read_text())
    (OUT/'catalogo_antes.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
    perfil=copy.deepcopy(catalog['visoes']['curral_topo'])
    perfil.update(rotulo='Curral externo - vista de cima',dominio='aereo_externo',estado='experimental',
        descricao='Conferencia de sombras para vista aerea externa, em desenvolvimento. '
        'Ainda pode duplicar ou omitir bovinos; confira as caixas na imagem.',verificador_nucleo=cfg)
    # Executa as funcoes e o bloco de filtragem reais, retirando so os decorators Streamlit.
    tree=ast.parse((ROOT/'app.py').read_text(encoding='utf-8'))
    names={'ler_verificador_nucleo','pontuar_nucleo_bytes'}
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(funcs)==2
    for n in funcs:n.decorator_list=[]
    data=np.load(ROOT/'data/experimentos/avaliacao_0359_curral_v1/previsoes.npz')
    image=(ROOT/'data/fontes_pesquisa/nellore_inspecao/revisao_0359_recorte16/recorte.png').read_bytes()
    def erro(msg):raise AssertionError(msg)
    env=dict(PROJ=ROOT,Path=Path,hashlib=hashlib,json=json,cv2=cv2,np=np,pd=pd,
        st=SimpleNamespace(caption=lambda _:None,error=erro,stop=lambda:erro('st.stop')),
        log=SimpleNamespace(exception=erro),especialista=perfil,dados=image,b=data['caixas'].copy(),s=data['scores'].copy())
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'app_helpers','exec'),env)
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and
        ast.unparse(n.test)=='especialista and especialista.get(\'verificador_nucleo\')')
    exec(compile(ast.Module(body=[block],type_ignores=[]),'app_filter','exec'),env)
    expected=np.load(train/'com_nucleos_0359.npz')
    assert np.allclose(env['b'],data['caixas'][expected['manter']])
    assert np.allclose(env['s'],data['scores'][expected['manter']])
    assert np.allclose(env['score_confirmado'],expected['probabilidade'][expected['manter']],atol=1e-5)
    assert len(env['b'])==17 and np.flatnonzero(~env['detalhes_confirmacao'].mantida).tolist()==[16,18]
    assert len(env['detalhes_confirmacao'])==19
    # Publicacao da configuracao local somente depois da verificacao da implementacao real.
    catalog['visoes']['curral_externo_topo']=perfil
    original=json.loads((OUT/'catalogo_antes.json').read_text())
    assert all(catalog['visoes'][k]==v for k,v in original['visoes'].items())
    catalog_path.write_text(json.dumps(catalog,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    result=dict(visao_integrada='curral_externo_topo',estado='experimental',publicacao_externa=False,
        detector_original_preservado=True,perfis_anteriores_preservados=True,antes=19,depois=17,
        caixas_removidas=[16,18],CSV_auditoria_linhas=19,CSV_final_linhas=17,
        teste='Funcoes e bloco de filtragem reais do app executados; sem teste visual do servidor Streamlit.',
        limite='Duas sombras removidas; ainda ha duplicacoes e omissao do bovino escuro. Nao e entrega final.')
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
