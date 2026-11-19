"""Uma leitura de candidatos congelados no pequeno teste novo, sem promocao.

Se o teste so tem poucos bovinos, o resultado nao e prova de rebanho denso.
Arquivos e configuracoes sao congelados antes de qualquer inferencia de teste.
"""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

import cv2
import pandas as pd
import torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas, metricas_contagem, taxas_deteccao
from integracao_denso import conferir

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'data/brasil_denso_novo'
OUT = ROOT / 'data/experimentos/solo_piloto_separado_v1'


def main():
    torch.set_num_threads(4)
    if (OUT / 'protocolo_antes_teste.json').exists():
        raise ValueError('Leitura de teste ja iniciada; preservar o protocolo e os resultados')
    status = conferir(BASE)
    if not status['pronto']:
        raise ValueError(status.get('erros'))
    teste = [r for r in status['registros'] if r['conjunto']=='test']
    if not any(r['n_bovinos']>0 for r in teste):
        raise ValueError('Ainda sem teste positivo aprovado')
    treino = pd.read_csv(ROOT / 'data/experimentos/especialista_solo_v1/catalogo.csv')
    assert not set(r['grupo'] for r in teste) & set(treino.grupo)
    reservados = {r['sha256']:r['arquivo'] for r in teste}
    assert not set(reservados) & set(treino.sha256)
    # Auditoria de copias exatas tambem no acervo do treino original.
    n_auditadas = 0
    for p in (ROOT / 'data/yolo/images/train').glob('*'):
        if p.suffix.lower() not in ('.jpg','.jpeg','.png'):
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in reservados:
            raise ValueError(f'Teste coincide com treino antigo: {p} / {reservados[digest]}')
        n_auditadas += 1
    escolhido = json.loads((ROOT / 'data/experimentos/especialista_solo_v1/resultado_validacao.json').read_text(encoding='utf-8'))['candidato']
    candidatos = {
        'referencia_solo_denso':dict(pesos='data/processed/modelo_gado.pt',imgsz=1280,iou=.7,conf=.15,augment=False),
        'referencia_solo_poucos':dict(pesos='data/processed/modelo_gado.pt',imgsz=1024,iou=.5,conf=.30,augment=False),
        'original_tta_validacao':dict(pesos='data/processed/modelo_gado.pt',imgsz=1280,iou=.7,conf=.20,augment=True),
        'especialista_solo_validacao':dict(pesos=escolhido['peso'],imgsz=int(escolhido['imgsz']),iou=.7,
                                         conf=float(escolhido['conf']),augment=bool(escolhido['augment'])),
    }
    OUT.mkdir(parents=True,exist_ok=True)
    for r in teste:
        for pasta,nome,digest in [('images',r['arquivo'],r['sha256']),('labels',Path(r['arquivo']).stem+'.txt',r['sha256_label'])]:
            destino = OUT / 'teste' / pasta / nome
            destino.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(BASE / pasta / nome,destino)
            assert hashlib.sha256(destino.read_bytes()).hexdigest()==digest
    protocolo = dict(instante_utc=datetime.now(timezone.utc).isoformat(),candidatos=candidatos,
        pesos_sha256={k:hashlib.sha256((ROOT / c['pesos']).read_bytes()).hexdigest() for k,c in candidatos.items()},
        imagens_teste=teste,treino_original_auditado_hash_exato=n_auditadas,
        criterio='Candidatos fixados pela validacao anterior; nenhuma busca no teste e nenhuma promocao automatica',
        campos=None,promove_app=False,
        limite='Pequeno piloto, no maximo poucos bovinos. Nao mede cenas com 20 ou mais animais. Auditoria nao certifica dados de pretreino nem toda similaridade visual.')
    (OUT / 'protocolo_antes_teste.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    print('Protocolo congelado; treino auditado:',n_auditadas,'imagens; teste:',len(teste),flush=True)
    rows,resumo = [],{}
    modelos = {}
    for nome,c in candidatos.items():
        peso = str(ROOT / c['pesos'])
        if peso not in modelos:
            modelos[peso] = YOLO(peso)
        modelo = modelos[peso]
        pasta = OUT / nome
        pasta.mkdir(exist_ok=True)
        for r in teste:
            im = cv2.imread(str(OUT / 'teste/images' / r['arquivo']))
            h,w = im.shape[:2]
            ref = ct.yolo_para_pixels((OUT / 'teste/labels' / (Path(r['arquivo']).stem+'.txt')).read_text().splitlines(),w,h)
            b,s = ct.detectar_inteira(modelo,im,imgsz=c['imgsz'],iou_nms=c['iou'],device=0,
                                     classes=[0],augment=c['augment'])
            b,s = ct.filtrar(b,s,c['conf'])
            rows.append(dict(modelo=nome,arquivo=r['arquivo'],real=len(ref),previsto=len(b),**comparar_caixas(ref,b,s)))
            cv2.imwrite(str(pasta / r['arquivo']),ct.desenhar(im,b,ref=ref))
        d = pd.DataFrame([r for r in rows if r['modelo']==nome])
        tp,fp,fn = (int(d[k].sum()) for k in ('TP','FP','FN'))
        resumo[nome] = dict(**metricas_contagem(d.real,d.previsto),**taxas_deteccao(tp,fp,fn),TP=tp,FP=fp,FN=fn)
        pd.DataFrame(rows).to_csv(OUT / 'por_imagem.csv',index=False)
        print(nome,resumo[nome],flush=True)
    (OUT / 'resultado.json').write_text(json.dumps(dict(resultados=resumo,promovido=False,
        limitacao=protocolo['limite']),indent=2),encoding='utf-8')
    (ROOT / 'data/apoio_coleta_denso/TESTE_PILOTO_CONSUMIDO.json').write_text(json.dumps(dict(
        arquivos=[r['arquivo'] for r in teste],grupos=[r['grupo'] for r in teste],
        protocolo=str(OUT / 'protocolo_antes_teste.json'),
        nota='Estas imagens foram avaliadas apos congelar os candidatos. A partir daqui sao regressao conhecida; novas rodadas precisam de teste positivo novo.'),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
