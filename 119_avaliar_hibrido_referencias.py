"""Compara detectores em referencias existentes; nao treina nem promove pesos."""
from pathlib import Path
import csv,gc,hashlib,json
import cv2,numpy as np,torch
from ultralytics import YOLO,RTDETR
import contagem as ct
from avaliacao import comparar_caixas,metricas_contagem

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/decisao_modelos_v1'


def casos():
    rs=[]
    base=ROOT/'data/experimentos/nano_internet_v1'
    for f in json.loads((base/'referencias_antes_api.json').read_text(encoding='utf-8'))['fontes']:
        if f['referencia']['contagem'] is not None:rs.append(dict(id='pexels_'+f['id'],origem='internet_revisada',imagem=str(base/f['arquivo']),real=f['referencia']['contagem'],imgsz=1024,conf=.3,iou=.5))
    base=ROOT/'data/brasil_dataset_v2'
    for r in csv.DictReader((base/'manifesto.csv').open(encoding='utf-8-sig')):
        if r['conjunto']=='test':rs.append(dict(id=r['arquivo'],origem='brasil_teste_conhecido',imagem=str(base/'images'/r['arquivo']),label=str(base/'labels'/Path(r['arquivo']).with_suffix('.txt')),real=int(r['n_bovinos']),imgsz=1024,conf=.3,iou=.5))
    base=ROOT/'data/experimentos/solo_denso_validacao_v1'
    for r in json.loads((base/'catalogo_val.json').read_text(encoding='utf-8')):
        rs.append(dict(id=r['arquivo'],origem='denso_validacao_conhecida_mesma_cena',imagem=str(base/'validacao/images'/r['arquivo']),label=str(base/'validacao/labels'/Path(r['arquivo']).with_suffix('.txt')),real=r['n_bovinos'],imgsz=1280,conf=.15,iou=.7))
    return rs


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocolo.json').exists():raise RuntimeError('Rodada preservada, nao repetir')
    lista=casos()
    for c in lista:c['sha256']=hashlib.sha256(Path(c['imagem']).read_bytes()).hexdigest()
    modelos=[dict(nome='atual',pesos='data/processed/modelo_gado.pt',classes=[0],arquitetura='yolo'),
        dict(nome='yolo11x_coco',pesos='data/modelos_candidatos/yolo11x.pt',classes=[19],arquitetura='yolo'),
        dict(nome='rtdetr_l_coco',pesos='data/modelos_candidatos/rtdetr-l.pt',classes=[19],arquitetura='rtdetr')]
    for m in modelos:m['sha256']=hashlib.sha256((ROOT/m['pesos']).read_bytes()).hexdigest()
    protocolo=dict(casos=lista,modelos=modelos,treino=False,promocao_automatica=False,
        criterio='Candidato: MAE total pelo menos 0,10 menor, soma de erros pelo menos 3 menor, nenhum aumento de MAE nos tres grupos e nenhum FP no negativo de cavalos. Nao e prova independente.',
        limites='Referencias conhecidas, amostra nao aleatoria, duas imagens densas da mesma cena, uma revisao visual. A foto do usuario fica separada do ranking.')
    (OUT/'protocolo.json').write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.set_num_threads(2);resultados=[];resumo={}
    for m in modelos:
        model=(RTDETR if m['arquitetura']=='rtdetr' else YOLO)(str(ROOT/m['pesos']))
        for c in lista:
            im=cv2.imread(c['imagem']);b,s=ct.detectar_inteira(model,im,imgsz=c['imgsz'],iou_nms=c['iou'],classes=m['classes'])
            b,s=ct.filtrar(b,s,c['conf']);r=dict(modelo=m['nome'],id=c['id'],origem=c['origem'],real=c['real'],previsto=len(b),erro=abs(len(b)-c['real']),caixas=b.tolist(),scores=s.tolist())
            if c.get('label'):
                linhas=Path(c['label']).read_text(encoding='utf-8').splitlines();ref=ct.yolo_para_pixels(linhas,im.shape[1],im.shape[0]);r.update(comparar_caixas(ref,b,s))
            resultados.append(r)
            (OUT/'por_imagem.json').write_text(json.dumps(resultados,ensure_ascii=False,indent=2),encoding='utf-8')
        por=[r for r in resultados if r['modelo']==m['nome']]
        resumo[m['nome']]=dict(total=metricas_contagem([r['real'] for r in por],[r['previsto'] for r in por]),grupos={g:metricas_contagem([r['real'] for r in por if r['origem']==g],[r['previsto'] for r in por if r['origem']==g]) for g in sorted({r['origem'] for r in por})},erro_somado=sum(r['erro'] for r in por),cavalos=next(r['previsto'] for r in por if r['id']=='pexels_15956981'))
        # Caso do usuario e separado para nao selecionar o modelo por uma foto.
        im=cv2.imread(str(ROOT/'data/experimentos/iluminacao_nelore_v1/original.png'))
        b,s=ct.detectar_inteira(model,im,imgsz=1024,iou_nms=.5,classes=m['classes']);b,s=ct.filtrar(b,s,.3)
        resumo[m['nome']]['foto_usuario']=len(b)
        cv2.imwrite(str(OUT/(m['nome']+'_usuario.jpg')),ct.desenhar(im,b))
        (OUT/'resumo.json').write_text(json.dumps(resumo,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({m['nome']:resumo[m['nome']]},ensure_ascii=True),flush=True)
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    base=resumo['atual'];aprovados=[]
    for nome,m in resumo.items():
        if nome=='atual':continue
        m['passou_criterio']=(m['total']['MAE']<=base['total']['MAE']-.1 and m['erro_somado']<=base['erro_somado']-3 and m['cavalos']==0 and all(m['grupos'][g]['MAE']<=base['grupos'][g]['MAE'] for g in base['grupos']))
        if m['passou_criterio']:aprovados.append(nome)
    (OUT/'resumo.json').write_text(json.dumps(resumo,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'decisao.json').write_text(json.dumps(dict(aprovados=aprovados,promovido=False,concluido=True),indent=2),encoding='utf-8')
    with (OUT/'por_imagem.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['modelo','id','origem','real','previsto','erro','TP','FP','FN'],extrasaction='ignore');w.writeheader();w.writerows(resultados)


if __name__=='__main__':main()
