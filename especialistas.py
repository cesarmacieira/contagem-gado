"""Modelos independentes por visao, sem promover candidatos automaticamente."""
from pathlib import Path
import hashlib
import json


def carregar_catalogo(projeto):
    arquivo = Path(projeto) / 'data/modelos/especialistas.json'
    if not arquivo.exists():
        return {}
    dados = json.loads(arquivo.read_text(encoding='utf-8-sig'))
    if dados.get('versao') != 1:
        raise ValueError('Versao do catalogo de especialistas nao suportada')
    return {k:v for k,v in dados['visoes'].items() if v.get('disponivel', False)}


def resolver_especialista(projeto, visao):
    raiz = Path(projeto).resolve()
    catalogo = carregar_catalogo(raiz)
    if visao not in catalogo:
        raise ValueError(f'Especialista nao disponivel para a visao: {visao}')
    perfil = catalogo[visao].copy()
    peso = (raiz / perfil['pesos']).resolve()
    if not peso.is_relative_to(raiz) or not peso.is_file():
        raise ValueError('Peso do especialista ausente ou fora do projeto')
    if perfil.get('arquitetura') not in ('yolo', 'rtdetr'):
        raise ValueError('Arquitetura deve ser yolo ou rtdetr')
    if perfil.get('bloqueado', False):
        raise ValueError('Candidato bloqueado pela avaliacao desta visao')
    digest = hashlib.sha256(peso.read_bytes()).hexdigest()
    if digest != perfil.get('sha256'):
        raise ValueError('Pesos alterados desde o registro do especialista; reavaliar antes de usar')
    cfg = perfil['config']
    if cfg['modo'] not in ('inteira', 'tiles') or cfg['imgsz'] <= 0 or cfg['tile'] <= 0:
        raise ValueError('Modo ou dimensoes invalidas no especialista')
    if not .05 <= cfg['conf'] <= .95 or not 0 < cfg['iou'] < 1 or not 0 <= cfg['overlap'] < 1:
        raise ValueError('Parametros invalidos no especialista')
    if not 0 < cfg.get('iou_juncao', .5) < 1:
        raise ValueError('IoU de juncao invalido')
    if perfil['arquitetura'] == 'rtdetr' and cfg.get('augment', False):
        raise ValueError('TTA nao habilitada para RT-DETR neste aplicativo')
    classes = tuple(perfil.get('classes', [0]))
    if not classes or any(not isinstance(c, int) or c < 0 for c in classes):
        raise ValueError('Classes bovinas invalidas')
    perfil.update(pesos_absolutos=str(peso), classes=classes, mt=peso.stat().st_mtime_ns,
                  chave_modelo=f"{visao}:{perfil['arquitetura']}:{digest}")
    return perfil
