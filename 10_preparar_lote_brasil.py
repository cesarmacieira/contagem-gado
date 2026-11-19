"""Exporta revisao visual para YOLO, COCO, gabarito e marcacoes auditaveis.

As anotacoes sao uma primeira revisao do assistente, nao consenso de especialistas.
Fotos ambiguas e duplicadas nunca entram no treino ou nas metricas exatas.
"""
from pathlib import Path
import csv
import json
import hashlib
import shutil
from datetime import datetime, timezone
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
LOTE = ROOT / 'data/brasil_lote_01'
DEST = ROOT / 'data/brasil_revisado'


def salvar_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    fontes = json.loads((LOTE / 'divisao_congelada.json').read_text(encoding='utf-8'))
    manual = json.loads((LOTE / 'rotulos_manuais.json').read_text(encoding='utf-8'))
    manual.update(json.loads((LOTE / 'rotulos_complementares.json').read_text(encoding='utf-8')))
    for d in ('images', 'labels', 'marcacoes'):
        (DEST / d).mkdir(parents=True, exist_ok=True)
    duplicadas = {25925109, 25925110, 25925113, 2675144, 37182671}
    registro, manifesto, gabarito, imagens, anotacoes = [], [], [], [], []
    for f in fontes:
        pid = f['pageid']; m = manual.get(str(pid))
        r = dict(f)
        if pid in duplicadas:
            r.update(status='excluida', motivo_exclusao='Duplicata perceptual ou mesmo autor de duplicata do material antigo')
        elif not m:
            r.update(status='excluida' if f['conjunto'] in ('excluir','excluida') else 'pendente', motivo_exclusao=f.get('motivo_exclusao') or 'Rebanho ambiguo: falta anotacao completa; nao usado')
        else:
            img = cv2.imread(str(LOTE / 'images' / f['arquivo']))
            review = cv2.imread(str(LOTE / 'revisao_limpa' / f'{pid}.jpg'))
            h, w = img.shape[:2]; rh, rw = review.shape[:2]
            caixas = np.array(m['caixas'], dtype=float).reshape(-1, 4)
            if m.get('coordenadas') == 'panel':
                panel = next(x for x in json.loads((LOTE / (m['painel']+'.json')).read_text()) if x['id'] == pid)
                caixas /= panel['scale']
            elif m.get('coordenadas') == 'crop':
                c = m['crop']; caixas /= c['scale']; caixas += [c['x'], c['y'], c['x'], c['y']]
            caixas *= [w/rw, h/rh, w/rw, h/rh]
            nome = Path(f['arquivo']).stem + '.jpg'
            crop = m.get('recorte_supervisionado')
            if crop:
                x1,y1,x2,y2 = np.rint(np.array(crop)*[w/rw,h/rh,w/rw,h/rh]).astype(int)
                img = img[y1:y2,x1:x2].copy(); caixas -= [x1,y1,x1,y1]
                r['recorte_original_xyxy'] = [int(x1),int(y1),int(x2),int(y2)]
                nome = Path(nome).stem+'_frente.jpg'
                h,w = img.shape[:2]
            caixas[:,[0,2]] = caixas[:,[0,2]].clip(0,w)
            caixas[:,[1,3]] = caixas[:,[1,3]].clip(0,h)
            assert ((caixas[:,2:] - caixas[:,:2]) > 0).all(), pid
            parcial = m.get('qualidade') == 'parcial'
            split = 'val' if pid == 48890227 else f['conjunto']
            r.update(anotacao=m, caixas_xyxy=caixas.tolist(), status='parcial' if parcial else 'completa', conjunto=split,
                     revisao='assistente_visual_v1', arquivo_exportado=nome)
            r['metodo'] = 'manual_sobre_imagem_limpa' if f['conjunto'] in ('val','test') else 'propostas_treino_corrigidas_visualmente'
            gabarito.append(dict(arquivo=nome,conjunto=split,grupo=f['grupo'],minimo=m.get('minimo',len(caixas)),
                                 maximo=m.get('maximo',len(caixas)),caixas_confirmadas=len(caixas),completo=not parcial,nota=m['nota']))
            if not parcial:
                cv2.imwrite(str(DEST/'images'/nome), img, [cv2.IMWRITE_JPEG_QUALITY,96])
                linhas = []
                for b in caixas:
                    x1,y1,x2,y2=b
                    linhas.append('0 '+ ' '.join(f'{v:.8f}' for v in ((x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h)))
                (DEST/'labels'/(Path(nome).stem+'.txt')).write_text('\n'.join(linhas)+ ('\n' if linhas else ''),encoding='utf-8')
                manifesto.append(dict(arquivo=nome,grupo=f['grupo'],conjunto=split,revisado=True,autor=f['autor'],licenca=f['licenca'],url=f['url'],n_bovinos=len(caixas),revisor='assistente_visual_v1'))
                imagens.append(dict(id=pid,file_name=nome,width=w,height=h,split=split,source_url=f['url'],license=f['licenca']))
                for b in caixas:
                    x1,y1,x2,y2=map(float,b)
                    anotacoes.append(dict(id=len(anotacoes)+1,image_id=pid,category_id=1,bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1),iscrowd=0,segmentation=[]))
                draw = img.copy()
                for i,b in enumerate(caixas,1):
                    x1,y1,x2,y2=np.rint(b).astype(int); cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(draw,str(i),(x1,max(15,y1)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
                s=min(1,1400/w); draw=cv2.resize(draw,None,fx=s,fy=s)
                cv2.imwrite(str(DEST/'marcacoes'/nome),draw)
        registro.append(r)
    assert len({x['arquivo'] for x in manifesto}) == len(manifesto)
    grupos={}
    for r in manifesto:
        assert grupos.setdefault(r['grupo'],r['conjunto']) == r['conjunto']
    for nome, rows in [('manifesto.csv',manifesto),('gabarito.csv',gabarito)]:
        with (DEST/nome).open('w',encoding='utf-8-sig',newline='') as out:
            wr=csv.DictWriter(out,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    salvar_json(DEST/'anotacoes_coco.json',dict(images=imagens,annotations=anotacoes,categories=[dict(id=1,name='bovino')]))
    salvar_json(DEST/'dicionario_classes.json',dict(yolo={'0':'bovino'},coco={'1':'bovino'},inclui=['vaca','touro','boi','bezerro','bovino parcialmente visivel individualizavel'],exclui=['cavalo','pessoa','ave','carne embalada','objetos'],politica='Uma caixa por individuo, reunindo cabeca e corpo visiveis. Caixa do trecho visivel, sem inventar extensao oculta. Contar animais cortados nas bordas. Cenas com individuos nao separaveis ficam parciais e fora do treino.',qualidade='Primeira revisao visual do assistente; recomenda-se segunda revisao humana antes de uso operacional.'))
    salvar_json(LOTE/'anotacoes.json',registro)
    resumo={s:dict(imagens=sum(r['conjunto']==s for r in manifesto),caixas=sum(r['n_bovinos'] for r in manifesto if r['conjunto']==s)) for s in ('train','val','test')}
    resumo.update(fotos_baixadas=len(fontes),parciais=sum(r['status']=='parcial' for r in registro),revisao='assistente_visual_v1')
    salvar_json(DEST/'resumo.json',resumo)
    freeze=LOTE/'divisao_final_pre_treino.json'
    payload=dict(criado_em=datetime.now(timezone.utc).isoformat(),criterio='Autor separado; excluir duplicatas e ambiguas. Autor da foto 48890227 movido de train para val antes de treino/inferencia quantitativa, para incluir densidade na validacao.',imagens=manifesto)
    if not freeze.exists(): salvar_json(freeze,payload)
    else: assert json.loads(freeze.read_text(encoding='utf-8'))['imagens']==manifesto
    print(json.dumps(resumo,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
