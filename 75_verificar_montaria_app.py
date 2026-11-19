"""Executa as funcoes reais do app; nao e verificacao visual do Streamlit."""
from pathlib import Path
import ast
import hashlib
import json
import cv2
import numpy as np
import torch
import contagem as ct

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/conferencia_montaria_app_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'resultado.json').exists():raise ValueError('Resultado preservado')
    torch.set_num_threads(4)
    arvore=ast.parse((ROOT/'app.py').read_text(encoding='utf-8'))
    nomes={'ler_conferencia_montaria','conferir_montaria_bytes'}
    funcoes=[n for n in arvore.body if isinstance(n,ast.FunctionDef) and n.name in nomes]
    assert len(funcoes)==2
    for n in funcoes:n.decorator_list=[]
    ns=dict(Path=Path,hashlib=hashlib,np=np,cv2=cv2,ct=ct)
    exec(compile(ast.Module(body=funcoes,type_ignores=[]),'app.py','exec'),ns)
    cfg=json.loads((ROOT/'data/modelos/verificador_solo_v1/config.json').read_text())
    m=cfg['montaria'];pc=ROOT/m['peso_clip'];pp=ROOT/m['peso_pessoas']
    ver,pessoas,device=ns['ler_conferencia_montaria'](str(pc),pc.stat().st_mtime_ns,m['sha_clip'],m['receita_sha256'],
        str(pp),pp.stat().st_mtime_ns,m['sha_pessoas'])
    fonte=ROOT/'data/experimentos/verificador_montaria_v4'
    amostras=[('bernard_28059238281.jpg',ROOT/'data/busca_teste_denso_independente',28),
        ('bdn_08_c2.jpg',ROOT/'data/experimentos/solo_denso_validacao_v2/validacao/images',5),
        ('brasil_01.jpg',ROOT/'data/teste_brasil',8),('brasil_06.jpg',ROOT/'data/teste_brasil',22),
        ('bdn_19.jpg',ROOT/'data/experimentos/solo_piloto_separado_v1/teste/images',0)]
    linhas=[]
    for nome,pasta,esperado in amostras:
        d=np.load(fonte/('solo_denso__'+Path(nome).stem+'.npz'))
        dados=(pasta/nome).read_bytes()
        mask,mm,regiao,pessoas_detectadas=ns['conferir_montaria_bytes'](dados,d['caixas'],m,ver,pessoas,device)
        np.testing.assert_array_equal(mask,d['manter'])
        np.testing.assert_allclose(mm,d['margens'],atol=1e-5)
        assert int(mask.sum())==esperado
        im=cv2.imdecode(np.frombuffer(dados,np.uint8),cv2.IMREAD_COLOR)
        vis=ct.desenhar(im,d['caixas'][mask]);cv2.imwrite(str(OUT/nome),vis)
        linha=dict(arquivo=nome,antes=len(mask),depois=int(mask.sum()),paridade_mascara=True,
            pessoas=len(pessoas_detectadas),versao_montaria=m['versao'])
        linhas.append(linha);print(linha,flush=True)
    (OUT/'resultado.json').write_text(json.dumps(dict(amostras=linhas,interface_visual_verificada=False,
        modo='Funcoes reais do aplicativo, sem decoradores de cache ou servidor Streamlit'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
