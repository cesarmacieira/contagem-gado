"""Comparacao pareada economica nas seis referencias congeladas antes do Nano."""
from pathlib import Path
import base64,hashlib,json,os
from contagem_resposta_vlm import extrair
import revisao_vlm as rv

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/nano_internet_v1'
MODELO='gpt-6-luna'
PERGUNTA='Quantos bovinos ha nesta imagem, contando os parcialmente visiveis? Responda com um unico numero inteiro.'


def main():
    rv.carregar_chave_ambiente(ROOT)
    if (OUT/'protocolo_luna.json').exists():raise RuntimeError('Comparacao ja iniciada; nenhuma repeticao')
    fontes=[f for f in json.loads((OUT/'referencias_antes_api.json').read_text(encoding='utf-8'))['fontes'] if f['referencia']['contagem'] is not None]
    protocolo=dict(modelo=MODELO,pergunta=PERGUNTA,max_output_tokens=1024,reasoning_effort='low',repeticoes=1,
        limite_estimado_usd=.02,precos_usd_milhao=dict(input=.1,output=.5),
        ids=[f['id'] for f in fontes],nota='Comparacao adaptativa apos observar falha do Nano. Sem promover ou alterar o detector. Referencias congeladas antes de ambos.')
    (OUT/'protocolo_luna.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    resultados=[]
    for f in fontes:
        dados=(OUT/f['arquivo']).read_bytes()
        if hashlib.sha256(dados).hexdigest()!=f['sha256']:raise RuntimeError('Imagem alterada')
        r=dict(id=f['id'],referencia=f['referencia']['contagem'],status='iniciado')
        resultados.append(r);(OUT/'resultados_luna.json').write_text(json.dumps(resultados,indent=2),encoding='utf-8')
        payload=dict(model=MODELO,store=False,service_tier='default',max_output_tokens=1024,reasoning=dict(effort='low'),
            input=[dict(role='user',content=[dict(type='input_text',text=PERGUNTA),dict(type='input_image',
            image_url='data:image/jpeg;base64,'+base64.b64encode(dados).decode(),detail='high')])])
        try:
            resposta=rv.enviar(payload,os.environ['OPENAI_API_KEY'])
            (OUT/(f['id']+'.luna.resposta.json')).write_text(json.dumps(resposta,ensure_ascii=False,indent=2),encoding='utf-8')
            uso=resposta.get('usage',{});texto=''.join(c.get('text','') for i in resposta.get('output',[]) for c in i.get('content',[]) if c.get('type')=='output_text').strip()
            n=extrair(texto) if resposta.get('status')=='completed' else None
            r.update(status=resposta.get('status'),resposta=texto,contagem=n,uso=uso,
                erro_absoluto=abs(n-r['referencia']) if n is not None else None,
                custo_estimado_usd=(uso.get('input_tokens',0)*.1+uso.get('output_tokens',0)*.5)/1e6)
        except Exception as erro:r.update(status='falha',erro=str(erro))
        (OUT/'resultados_luna.json').write_text(json.dumps(resultados,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(r),flush=True)
    nano=json.loads((OUT/'resultados.json').read_text(encoding='utf-8'))
    nano_ids={r['id']:r for r in nano}
    linhas=['# Comparacao economica nas seis cenas com referencia','',
        'Referencias visuais do assistente congeladas antes das consultas; sem segundo anotador. Mesma foto e mesma pergunta nos dois modelos. Comparacao Luna iniciada apos observar as falhas do Nano; amostra pequena e exploratoria.',
        '', '| Foto | Referencia | GPT-5 nano | GPT-6 Luna |','|---|---:|---:|---:|']
    for r in resultados:linhas.append(f"| {r['id']} | {r['referencia']} | {nano_ids[r['id']].get('contagem')} | {r.get('contagem')} |")
    resumo={}
    for nome,rs in [('nano',[r for r in nano if r['referencia'] is not None]),('luna',resultados)]:
        erros=[r['erro_absoluto'] for r in rs if r.get('erro_absoluto') is not None]
        resumo[nome]=dict(fotos=len(rs),respostas_avaliaveis=len(erros),acertos=sum(e==0 for e in erros),
            MAE=sum(erros)/len(erros) if erros else None,custo_estimado_usd=sum(r.get('custo_estimado_usd',0) for r in rs))
    resumo['custo_total_rodada_usd']=sum(r.get('custo_estimado_usd',0) for r in nano+resultados)
    (OUT/'comparacao_metricas.json').write_text(json.dumps(resumo,indent=2),encoding='utf-8')
    linhas+=['','```json',json.dumps(resumo,indent=2),'```','','Nenhum modelo foi promovido. Nao comprova contagem de rebanho denso. Fotos e fontes em galeria.html e fontes.json. As oito fotos ambiguas consultadas apenas com Nano ficam fora da metrica.']
    (OUT/'COMPARACAO.md').write_text('\n'.join(linhas),encoding='utf-8')
    print(json.dumps(resumo),flush=True)


if __name__=='__main__':main()
