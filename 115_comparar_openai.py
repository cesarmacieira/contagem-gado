"""Uma rodada autorizada: uma consulta por modelo visual, sem repeticao."""
from pathlib import Path
import base64,csv,hashlib,json,os,re,time,urllib.request,urllib.error
import revisao_vlm as rv

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/comparacao_openai_nelore_v1'
PERGUNTA='Quantos bovinos ha nesta imagem, contando os parcialmente visiveis? Responda com um unico numero inteiro.'
LIMITE=.50
# USD por milhao, tarifas Standard verificadas em 18/11/2026.
PRECOS={
 'gpt-4o':(2.5,10),'gpt-4o-mini':(.15,.6),'gpt-4.1':(2,8),'gpt-4.1-mini':(.4,1.6),
 'gpt-5':(1.25,10),'gpt-5-mini':(.25,2),'gpt-5-nano':(.05,.4),'gpt-5-pro':(15,120),
 'gpt-5.1':(1.25,10),'gpt-5.2':(1.75,14),'gpt-5.2-pro':(21,168),'gpt-5.3-codex':(1.75,14),
 'gpt-5.4':(2.5,15),'gpt-5.4-mini':(.75,4.5),'gpt-5.4-nano':(.2,1.25),'gpt-5.4-pro':(30,180),
 'gpt-5.5':(5,30),'gpt-5.5-pro':(30,180),'gpt-5.6-sol':(4,20),'gpt-5.6-terra':(2,12),'gpt-5.6-luna':(.2,1.2),
 'gpt-6-astra':(10,50),'gpt-6-sol':(2,10),'gpt-6-luna':(.1,.5),'gpt-6.1-sol':(2,10),'gpt-6.1-terra':(1,5),
 'gpt-6.2-astra':(8,40),'gpt-6.2-sol':(2,10),'chat-latest':(5,30),'o3':(2,8),'o3-pro':(20,80)}


def parametros(modelo):
    p=dict(model=modelo,store=False,max_output_tokens=1024)
    if modelo.startswith('gpt-4') or modelo=='chat-latest':p['max_output_tokens']=64
    elif modelo=='gpt-5-pro':p['reasoning']={'effort':'high'}
    elif modelo.endswith('-pro'):p['reasoning']={'effort':'medium'}
    elif modelo in ('gpt-5','gpt-5-mini','gpt-5-nano'):p['reasoning']={'effort':'minimal'}
    else:p['reasoning']={'effort':'low'}
    return p


def gravar(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def main():
    rv.carregar_chave_ambiente(ROOT);chave=os.environ['OPENAI_API_KEY']
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocolo.json').exists():raise RuntimeError('Rodada ja iniciada. Preservada; nenhuma chamada repetida.')
    req=urllib.request.Request('https://api.openai.com/v1/models',headers={'Authorization':'Bearer '+chave})
    with urllib.request.urlopen(req,timeout=30) as res:disponiveis=json.load(res)
    gravar(OUT/'modelos_disponiveis.json',disponiveis)
    ids={m['id'] for m in disponiveis['data']}
    ordem=sorted(set(PRECOS)&ids,key=lambda m:('-pro' in m,PRECOS[m][1],m))
    imagem=(ROOT/'data/experimentos/iluminacao_nelore_v1/original.png').read_bytes()
    gravar(OUT/'protocolo.json',dict(pergunta=PERGUNTA,imagem_sha256=hashlib.sha256(imagem).hexdigest(),
        modelos=ordem,precos=PRECOS,limite_estimado_usd=LIMITE,comparacao_posterior_usuario=7,
        repeticoes=1,metodo='Imagem PNG original, sem filtros; uma versao por modelo. Sem gabarito no prompt.',
        parametros={m:parametros(m) for m in ordem},fonte_precos='https://developers.openai.com/api/docs/pricing',
        ressalva='Configuracao economica de raciocinio, nao capacidade maxima. Truncamentos nao equivalem a erros de contagem.'))
    gravar(OUT/'excluidos.json',[dict(modelo=m,motivo='Versao datada/alias representada pelo modelo-base ou modalidade especializada fora da contagem visual em Responses') for m in sorted(ids-set(ordem))])
    image_url='data:image/png;base64,'+base64.b64encode(imagem).decode()
    resultados=[];reserva_total=0.;custo_total=0.
    for modelo in ordem:
        p=parametros(modelo);entrada,saida=PRECOS[modelo]
        # Imagem 487x588: margem para tiles/patches e prompt. 4o-mini usa
        # uma escala de tokens propria, maior, embora com tarifa baixa.
        max_input=33000 if modelo=='gpt-4o-mini' else 2000
        envelope=(max_input*entrada+p['max_output_tokens']*saida)/1e6
        r=dict(modelo=modelo,reserva_max_estimada_usd=envelope)
        if reserva_total+envelope>LIMITE:
            r['status']='nao_executado_orcamento';resultados.append(r);gravar(OUT/'resultados.json',resultados);continue
        reserva_total+=envelope
        r['status']='iniciado';resultados.append(r);gravar(OUT/'resultados.json',resultados)
        p.update(service_tier='default',input=[dict(role='user',content=[dict(type='input_text',text=PERGUNTA),dict(type='input_image',image_url=image_url,detail='high')])])
        inicio=time.monotonic()
        try:
            resposta=rv.enviar(p,chave)
            gravar(OUT/(modelo+'.resposta.json'),resposta)
            uso=resposta.get('usage',{});custo=(uso.get('input_tokens',0)*entrada+uso.get('output_tokens',0)*saida)/1e6
            texto=''.join(c.get('text','') for item in resposta.get('output',[]) for c in item.get('content',[]) if c.get('type')=='output_text').strip()
            n=int(texto) if re.fullmatch(r'\d{1,3}',texto) else None
            r.update(status=resposta.get('status'),resposta=texto,contagem=n,coincide_com_7=n==7 if n is not None else None,
                uso=uso,custo_estimado_usd=custo,modelo_retornado=resposta.get('model'),detalhes_incompleto=resposta.get('incomplete_details'))
            reserva_total+=custo-envelope;custo_total+=custo
        except Exception as erro:
            # A reserva permanece em falhas: nao assumir que nao houve cobranca.
            r.update(status='falha',erro=str(erro))
        r['segundos']=round(time.monotonic()-inicio,2)
        gravar(OUT/'resultados.json',resultados)
        gravar(OUT/'estado.json',dict(concluido=False,modelo=modelo,custo_estimado_usd=custo_total,reserva_consumida_usd=reserva_total))
        print(json.dumps({k:r.get(k) for k in ('modelo','status','contagem','custo_estimado_usd','erro')},ensure_ascii=True),flush=True)
    with (OUT/'comparacao.csv').open('w',newline='',encoding='utf-8-sig') as f:
        cols=['modelo','modelo_retornado','status','contagem','coincide_com_7','custo_estimado_usd','segundos','resposta','erro']
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(resultados)
    gravar(OUT/'estado.json',dict(concluido=True,custo_estimado_usd=custo_total,reserva_consumida_usd=reserva_total,limite_estimado_usd=LIMITE))
    linhas=['# Comparacao OpenAI — uma foto, uma tentativa por modelo','',
        'Referencia do usuario: 7 bovinos. Nao fornecida aos modelos. Acertar esta foto nao comprova precisao geral.',
        'Raciocinio economico e saida limitada; respostas incompletas nao sao contagens incorretas. Nenhum modelo promovido.',
        f'Custo calculado pelo uso retornado e tarifas Standard, sem desconto de cache: US$ {custo_total:.6f}.',
        '', '| Modelo | Estado | Contagem | Coincide com 7 | USD estimado |','|---|---|---:|---|---:|']
    for r in resultados:linhas.append(f"| {r['modelo']} | {r['status']} | {r.get('contagem','')} | {r.get('coincide_com_7','')} | {r.get('custo_estimado_usd',0):.6f} |")
    (OUT/'RELATORIO.md').write_text('\n'.join(linhas),encoding='utf-8')


if __name__=='__main__':main()
