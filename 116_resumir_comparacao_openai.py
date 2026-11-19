"""Resumo offline das respostas preservadas. Nao faz chamadas de API."""
from pathlib import Path
import csv,json
from contagem_resposta_vlm import extrair

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/comparacao_openai_nelore_v1'


def main():
    estado=json.loads((OUT/'estado.json').read_text(encoding='utf-8'))
    if not estado['concluido']:raise RuntimeError('Aguardar conclusao da rodada antes de resumir')
    rs=json.loads((OUT/'resultados.json').read_text(encoding='utf-8'))
    for r in rs:
        n=extrair(r.get('resposta','')) if r['status']=='completed' else None
        r.update(contagem=n,coincide_com_7=n==7 if n is not None else None)
    (OUT/'resultados.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'comparacao.csv').open('w',newline='',encoding='utf-8-sig') as f:
        campos=['modelo','modelo_retornado','status','contagem','coincide_com_7','custo_estimado_usd','segundos','resposta','erro']
        w=csv.DictWriter(f,fieldnames=campos,extrasaction='ignore');w.writeheader();w.writerows(rs)
    linhas=['# Comparacao OpenAI — foto nelore_1512','',
        'Uma consulta por modelo-base disponivel na chave; aliases e snapshots duplicados nao foram cobrados novamente. Modelos especializados em audio, embeddings, moderacao, geracao de imagem ou busca ficam fora.',
        '', 'Imagem PNG original, sem filtros. Pergunta: “Quantos bovinos ha nesta imagem, contando os parcialmente visiveis? Responda com um unico numero inteiro.”',
        'Referencia fornecida pelo usuario: 7. Nao informada no prompt. O resumo extrai numeros explicitos, inclusive por extenso, sem alterar as respostas completas. Intervalos ambiguos ficam sem contagem.',
        '', 'Este e um teste de uma foto conhecida, nao uma medida de acuracia geral. Nao houve ajuste ou promocao de modelos. Raciocinio economico (low/minimal, medium/high quando exigido por Pro); maximo de 1024 tokens, ou 64 em GPT-4/chat-latest. Respostas truncadas sao inconclusivas, nao erros de contagem. Um resultado nao e uma estimativa de repetibilidade.',
        '',f"Custo estimado pelos tokens retornados e tarifas Standard: **US$ {estado['custo_estimado_usd']:.6f}**, sem aplicar descontos de cache. Reserva consumida, incluindo falhas com custo desconhecido: US$ {estado['reserva_consumida_usd']:.6f}. Limite local estimado da rodada: US$ 0,50. A primeira consulta VLM de antes desta rodada nao esta incluida.",
        '', 'Precos consultados: https://developers.openai.com/api/docs/pricing (18/11/2026), copia local em precos_oficiais.md.',
        '', '| Modelo | Estado | Contagem | Coincide com 7 | USD estimado |','|---|---|---:|---|---:|']
    for r in rs:
        n=r['contagem'];acerto='' if n is None else ('sim' if n==7 else 'nao')
        linhas.append(f"| {r['modelo']} | {r['status']} | {'' if n is None else n} | {acerto} | {r.get('custo_estimado_usd',0):.6f} |")
    (OUT/'RELATORIO.md').write_text('\n'.join(linhas),encoding='utf-8')
    print(json.dumps(dict(estado=estado,resultados=[{k:r.get(k) for k in ['modelo','status','contagem','resposta','custo_estimado_usd','erro']} for r in rs]),ensure_ascii=True))


if __name__=='__main__':main()
