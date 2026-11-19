"""Benchmark do fluxo hibrido real, sem enviar gabarito nem contagem do detector."""
from pathlib import Path
import base64,hashlib,importlib,json,os,time
import numpy as np
from PIL import Image,ImageOps
import hibrido_vlm as hv,revisao_vlm as rv
from avaliacao import metricas_contagem

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/decisao_modelos_v1/vlms'
MODELOS=[dict(id='gpt-6-luna',entrada=.1,saida=.5,effort='low'),
         dict(id='gpt-5.4-mini-2026-03-17',entrada=.75,saida=4.5,effort='none'),
         dict(id='gpt-6.2-astra',entrada=8.,saida=40.,effort='low')]
LIMITE=.50


def salvar(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)


def main():
    rv.carregar_chave_ambiente(ROOT);chave=os.environ['OPENAI_API_KEY'];OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocolo.json').exists():raise RuntimeError('Rodada ja iniciada; nao repetir chamadas')
    casos=importlib.import_module('119_avaliar_hibrido_referencias').casos()
    det=json.loads((OUT.parent/'por_imagem.json').read_text(encoding='utf-8'));det={r['id']:r for r in det if r['modelo']=='atual'}
    salvar(OUT/'protocolo.json',dict(modelos=MODELOS,casos=casos,pergunta=hv.PROMPT,schema=hv.SCHEMA,max_output_tokens=2048,
        limite_estimado_usd=LIMITE,referencias_no_prompt=False,promocao_automatica=False,
        criterio='MAE pelo menos 0,10 menor e 3 erros a menos que detector, nenhum grupo pior, zero bovinos no negativo de cavalos. Respostas incertas sao revisao, nao confirmacao. Nenhuma falha ou truncamento pode ser excluido silenciosamente.',
        nota='Regressao conhecida. Modelos e prompt adaptados apos erros anteriores; nao e novo teste independente nem garantia geral. Orcamento separado do limite diario do aplicativo.'))
    resultados=[];reservado=0.
    for m in MODELOS:
        for c in casos:
            d=det[c['id']];dados=Path(c['imagem']).read_bytes();im=ImageOps.exif_transpose(Image.open(c['imagem']));w,h=im.size
            sinais=hv.gatilhos(d['caixas'],d['scores'],w,h,c['conf'])
            imagens=hv.preparar(dados,d['caixas'],sinais['indices'])
            envelope=(len(imagens)*2500*m['entrada']+2048*m['saida'])/1e6
            r=dict(modelo=m['id'],id=c['id'],origem=c['origem'],real=c['real'],detector=d['previsto'],recomendada=sinais['recomendada'],motivos=sinais['motivos'])
            if reservado+envelope>LIMITE:
                r['status']='orcamento';resultados.append(r);salvar(OUT/'resultados.json',resultados);continue
            reservado+=envelope;r['status']='iniciado';resultados.append(r);salvar(OUT/'resultados.json',resultados)
            conteudo=[dict(type='input_text',text=hv.PROMPT)]+[dict(type='input_image',image_url='data:image/jpeg;base64,'+base64.b64encode(i).decode(),detail='high') for i in imagens]
            payload=dict(model=m['id'],store=False,service_tier='default',max_output_tokens=2048,reasoning=dict(effort=m['effort']),
                input=[dict(role='user',content=conteudo)],text={'format':dict(type='json_schema',name='revisao_contagem',strict=True,schema=hv.SCHEMA)})
            inicio=time.monotonic()
            try:
                resposta=rv.enviar(payload,chave);salvar(OUT/(m['id']+'_'+c['id']+'.resposta.json'),resposta)
                uso=resposta.get('usage',{});custo=(uso.get('input_tokens',0)*m['entrada']+uso.get('output_tokens',0)*m['saida'])/1e6
                reservado+=custo-envelope;r.update(status=resposta.get('status'),uso=uso,custo_estimado_usd=custo)
                if r['status']=='completed':
                    texto=''.join(x.get('text','') for i in resposta.get('output',[]) for x in i.get('content',[]) if x.get('type')=='output_text')
                    revisao=hv.validar(json.loads(texto));r.update(revisao)
                    r['erro']=abs(r['contagem']-r['real']);r['situacao']=hv.comparar(r['detector'],revisao)
                    r['erro_confirmado']=r['situacao']=='concordam' and r['erro']>0
            except Exception as erro:r.update(status='falha',erro_execucao=str(erro))
            r['segundos']=round(time.monotonic()-inicio,2);salvar(OUT/'resultados.json',resultados)
            print(json.dumps({k:r.get(k) for k in ['modelo','id','status','contagem','incerto','erro','custo_estimado_usd']}),flush=True)
    resumo={}
    for m in MODELOS:
        rs=[r for r in resultados if r['modelo']==m['id']];validos=[r for r in rs if r.get('contagem') is not None]
        resumo[m['id']]=dict(total_previsto=len(rs),avaliados=len(validos),falhas=len(rs)-len(validos),
            contagem=metricas_contagem([r['real'] for r in validos],[r['contagem'] for r in validos]) if validos else {},
            grupos={g:metricas_contagem([r['real'] for r in validos if r['origem']==g],[r['contagem'] for r in validos if r['origem']==g]) for g in sorted({r['origem'] for r in validos})},
            incertos=sum(r.get('incerto',False) for r in validos),erros_confirmados=sum(r.get('erro_confirmado',False) for r in validos),
            custo_estimado_usd=sum(r.get('custo_estimado_usd',0) for r in rs))
    salvar(OUT/'resumo.json',resumo)
    salvar(OUT/'estado.json',dict(concluido=True,reservado_usd=reservado,custo_estimado_usd=sum(r.get('custo_estimado_usd',0) for r in resultados)))
    print(json.dumps(resumo),flush=True)


if __name__=='__main__':main()
