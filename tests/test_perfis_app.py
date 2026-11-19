"""Exercita o ramo real do perfil experimental sem carregar pesos."""
import ast
from pathlib import Path
from types import SimpleNamespace

APP=Path(__file__).resolve().parents[1]/'app.py'

def test_perfil_experimental_usa_id_e_aplica_configuracao():
    tree=ast.parse(APP.read_text(encoding='utf-8'))
    blocos=[n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='perfil' and any(isinstance(c,ast.Constant) and c.value=='solo_experimental' for c in n.test.comparators)]
    assert len(blocos)==1
    mensagens=[]
    contexto=dict(perfil='solo_experimental',modo='tiles',imgsz=640,conf_padrao=.55,iou=.5,st=SimpleNamespace(info=mensagens.append))
    exec(compile(ast.fix_missing_locations(ast.Module(body=blocos,type_ignores=[])),str(APP),'exec'),contexto)
    assert tuple(contexto[k] for k in ['modo','imgsz','conf_padrao','iou'])==('inteira',1280,.15,.7)
    assert mensagens

def test_radio_usa_chaves_e_rotulos_separados():
    tree=ast.parse(APP.read_text(encoding='utf-8'))
    radios=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='radio' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='Perfil de contagem']
    assert len(radios)==1
    assert any(k.arg=='format_func' for k in radios[0].keywords)
    assert isinstance(radios[0].args[1],ast.Call) and radios[0].args[1].func.id=='list'
