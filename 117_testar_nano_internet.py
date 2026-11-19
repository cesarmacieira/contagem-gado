"""Uma consulta por foto nova; referencias visuais congeladas antes da API."""
from pathlib import Path
import base64,csv,hashlib,html,json,os,time
from contagem_resposta_vlm import extrair
import revisao_vlm as rv

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/nano_internet_v1'
MODELO='gpt-5-nano-2025-08-07'
PERGUNTA='Quantos bovinos ha nesta imagem, contando os parcialmente visiveis? Responda com um unico numero inteiro.'
LIMITE=.02


def salvar(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def main():
    rv.carregar_chave_ambiente(ROOT);chave=os.environ['OPENAI_API_KEY']
    if (OUT/'protocolo.json').exists():raise RuntimeError('Rodada ja iniciada; nenhuma consulta sera repetida')
    refs=json.loads((OUT/'referencias_antes_api.json').read_text(encoding='utf-8'))
    fontes=refs['fontes'];envelope=(10000*.05+1024*.4)/1e6
    if len(fontes)*envelope>LIMITE:raise RuntimeError('Orcamento insuficiente para lote')
    salvar(OUT/'protocolo.json',dict(modelo=MODELO,pergunta=PERGUNTA,detail='high',max_output_tokens=1024,
        reasoning_effort='minimal',repeticoes=1,limite_estimado_usd=LIMITE,
        precos_usd_milhao=dict(input=.05,output=.4),fonte_precos='https://developers.openai.com/api/docs/pricing',
        referencia_sha256=hashlib.sha256((OUT/'referencias_antes_api.json').read_bytes()).hexdigest(),
        referencias_no_prompt=False,tratamento='JPEG baixado com largura 1600; sem filtros e sem recortes adicionais',
        escopo='Teste exploratorio, nao avaliacao independente de treino desconhecido do provedor. Uma revisao visual de referencia. Sem treino ou promocao.'))
    resultados=[]
    for fonte in fontes:
        imagem=(OUT/fonte['arquivo']).read_bytes()
        if hashlib.sha256(imagem).hexdigest()!=fonte['sha256']:raise RuntimeError('Imagem alterada desde revisao')
        r=dict(id=fonte['id'],url=fonte['url'],referencia=fonte['referencia']['contagem'],status='iniciado')
        resultados.append(r);salvar(OUT/'resultados.json',resultados)
        payload=dict(model=MODELO,store=False,service_tier='default',max_output_tokens=1024,reasoning=dict(effort='minimal'),
            input=[dict(role='user',content=[dict(type='input_text',text=PERGUNTA),dict(type='input_image',
                image_url='data:image/jpeg;base64,'+base64.b64encode(imagem).decode(),detail='high')])])
        inicio=time.monotonic()
        try:
            resposta=rv.enviar(payload,chave);salvar(OUT/(fonte['id']+'.resposta.json'),resposta)
            uso=resposta.get('usage',{});texto=''.join(c.get('text','') for i in resposta.get('output',[]) for c in i.get('content',[]) if c.get('type')=='output_text').strip()
            n=extrair(texto) if resposta.get('status')=='completed' else None
            r.update(status=resposta.get('status'),resposta=texto,contagem=n,uso=uso,
                custo_estimado_usd=(uso.get('input_tokens',0)*.05+uso.get('output_tokens',0)*.4)/1e6,
                erro_absoluto=abs(n-r['referencia']) if n is not None and r['referencia'] is not None else None)
        except Exception as erro:r.update(status='falha',erro=str(erro))
        r['segundos']=round(time.monotonic()-inicio,2);salvar(OUT/'resultados.json',resultados)
        print(json.dumps(r,ensure_ascii=True),flush=True)
    avaliaveis=[r for r in resultados if r['referencia'] is not None]
    erros=[r['erro_absoluto'] for r in avaliaveis if r.get('erro_absoluto') is not None]
    metricas=dict(fotos=len(resultados),com_referencia=len(avaliaveis),respostas_avaliaveis=len(erros),
        acertos=sum(e==0 for e in erros),MAE=sum(erros)/len(erros) if erros else None,
        custo_estimado_usd=sum(r.get('custo_estimado_usd',0) for r in resultados),concluido=True)
    salvar(OUT/'metricas.json',metricas)
    with (OUT/'resultados.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['id','url','referencia','contagem','erro_absoluto','status','resposta','custo_estimado_usd','segundos'],extrasaction='ignore');w.writeheader();w.writerows(resultados)
    linhas=['# GPT-5 nano em novas fotos da internet','',f'Modelo: {MODELO}. Uma chamada por foto, sem respostas esperadas no prompt. Fotos e referencias congeladas antes da API.',
        '',f"Fotos consultadas: {metricas['fotos']}. Com referencia visual exata: {metricas['com_referencia']}. Acertos: {metricas['acertos']}. MAE: {metricas['MAE']}. Custo estimado total: US$ {metricas['custo_estimado_usd']:.6f}.",
        '', 'As cenas ambiguas ficam fora das metricas. Uma revisao visual do assistente, sem segundo anotador; amostra pequena e nao aleatoria, sem garantia de ausencia no pre-treino da OpenAI. Nao comprova desempenho geral ou em rebanho denso. Nada incorporado ao treino ou promovido no aplicativo.',
        '', '| Foto | Referencia | Nano | Erro absoluto |','|---|---:|---:|---:|']
    cards=[]
    for s,r in zip(fontes,resultados):
        ref=r['referencia'] if r['referencia'] is not None else 'ambigua'
        linhas.append(f"| [{r['id']}]({r['url']}) | {ref} | {r.get('contagem','')} | {r.get('erro_absoluto','')} |")
        cards.append('<section><h2>'+html.escape(r['id'])+'</h2><img src="'+s['arquivo']+'"><p>Referencia: '+str(ref)+' | Nano: '+str(r.get('contagem'))+'</p><p>'+html.escape(s['referencia']['nota'])+'</p><a href="'+s['url']+'">Fonte Pexels</a></section>')
    (OUT/'RELATORIO.md').write_text('\n'.join(linhas),encoding='utf-8')
    (OUT/'galeria.html').write_text('<meta charset="utf-8"><title>Teste GPT-5 nano</title><style>body{font-family:Arial;margin:30px}section{display:inline-block;vertical-align:top;width:430px;margin:15px}img{max-width:420px;max-height:440px}</style><h1>Fotos novas — GPT-5 nano</h1>'+''.join(cards),encoding='utf-8')
    print(json.dumps(metricas),flush=True)


if __name__=='__main__':main()
