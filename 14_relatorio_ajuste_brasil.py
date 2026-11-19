"""Publica resultado concreto do experimento e habilita perfil so com ganho medido."""
from pathlib import Path
import json
import os
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'data/experimentos/brasil_lote_01_v2'


def main():
    res=json.loads((EXP/'avaliacao/resultados.json').read_text(encoding='utf-8'))
    # Desvincular hardlinks do primeiro lancamento para preservar o snapshot.
    for p in (EXP/'dataset').rglob('*'):
        if p.is_file() and p.stat().st_nlink>1:
            tmp=p.with_name(p.name+'.snapshot');shutil.copy2(p,tmp);os.replace(tmp,p)
    o,a=res['original'],res['ajustado']
    # A liberacao e por escopo: Brasil de solo, sem substituir os pesos genericos.
    liberar=(a['teste']['MAE']<o['teste']['MAE'] and a['validacao']['MAE']<=o['validacao']['MAE']
             and a['teste']['F1_caixas']>=o['teste']['F1_caixas'])
    perfil=dict(disponivel=liberar,pesos=Path(a['pesos']).relative_to(ROOT).as_posix(),sha256=a['sha256'],config=a['config'],
                descricao=f"Modelo ajustado com fotos brasileiras. Teste reservado: {a['teste']['MAE']:.2f} bovino de erro medio por foto, em apenas 10 fotos. Confira as caixas, especialmente em rebanhos grandes.",
                criterio='Menor MAE no teste, sem piorar MAE de validacao e F1 das caixas. Limiar e resolucao escolhidos somente na validacao.')
    (EXP/'perfil_app.json').write_text(json.dumps(perfil,ensure_ascii=False,indent=2),encoding='utf-8')
    curvas=pd.read_csv(EXP/'treino/results.csv');curvas.columns=curvas.columns.str.strip()
    tabela='| Medida | Original recalibrado | Ajustado |\n|---|---:|---:|\n'
    for campo,label in [('MAE','Erro absoluto medio por foto'),('exatas_pct','Fotos com contagem exata (%)'),('WAPE_pct','Erro absoluto / total de bovinos (%)'),('precisao_caixas','Precisao de caixas (IoU 0,5)'),('recall_caixas','Sensibilidade de caixas (IoU 0,5)'),('F1_caixas','F1 das caixas')]:
        tabela+=f"| {label} | {o['teste'][campo]:.3f} | {a['teste'][campo]:.3f} |\n"
    controle='| Modelo | Erro medio, mesmos parametros (1280 / NMS 0,7 / conf 0,15) |\n|---|---:|\n'+''.join(f"| {n} | {res[n]['teste_config_fixa']['MAE']:.3f} |\n" for n in ('original','ajustado'))
    fotos=pd.read_csv(EXP/'avaliacao/contagem_teste.csv')
    pivot=fotos.pivot(index=['arquivo','real'],columns='modelo',values='previsto').reset_index()
    detalhes='| Foto | Gabarito | Original | Ajustado |\n|---|---:|---:|---:|\n'+''.join(f'| {r.arquivo} | {r.real} | {r.original} | {r.ajustado} |\n' for r in pivot.itertuples())
    texto=f'''# Resultado do ajuste com novas fotos brasileiras

{'O ajuste reduziu o erro no teste reservado e foi disponibilizado como perfil separado no app.' if liberar else 'O ajuste foi treinado e testado, mas nao passou todos os criterios para entrar no app. Os pesos originais foram preservados.'}

## Dados e marcacoes entregues

Foram baixadas 60 fotografias reais do Wikimedia Commons, com autor, URL, licenca e hashes registrados em `data/brasil_lote_01/fontes.json`.
Quarenta imagens receberam anotacao completa: 21 de treino (69 caixas), 9 de validacao (30 caixas) e 10 de teste (34 caixas). Duas imagens de treino sao negativos reais, sem bovinos vivos. Foram corrigidas caixas duplicadas de cabeca/corpo, bezerros ocultos e falsos positivos em objetos.

Tres fotos densas receberam marcacoes parciais, fora do treino e das metricas exatas. Quatro outras fotos com rebanhos ambiguos ficaram pendentes. Treze fotos foram excluidas, incluindo duplicatas e imagens do mesmo autor de duplicatas. As anotacoes sao uma primeira revisao visual pelo assistente, sem consenso entre especialistas. Os intervalos das anotacoes parciais sao estimativas, nao contagens certificadas.

Arquivos: `data/brasil_revisado/galeria_rotulos.html`, `gabarito.csv`, `manifesto.csv`, `dicionario_classes.json`, `anotacoes_coco.json`, `images/`, `labels/` e `marcacoes/`. A galeria inclui fonte e licenca por foto; os direitos das fotografias continuam sendo os das respectivas fontes. Um recorte supervisionado da foto 58672828 exclui o rebanho distante ambiguo; a transformacao esta registrada no JSON.

## Treino executado

Foram executadas {len(curvas)} epocas a partir dos pesos ja treinados, com entrada 1024, AdamW, primeiras 10 camadas congeladas, 21 fotos novas repetidas quatro vezes apenas no treino e replay de 64 imagens antigas. As repeticoes nao aumentam o numero de imagens independentes. O checkpoint foi escolhido pelo mAP50-95 da validacao. O treino inicial em 1280 foi interrompido antes de concluir a primeira epoca por custo de processamento, sem avaliar teste nem alterar os pesos originais.

Autores ficaram em conjuntos separados. A divisao final foi congelada antes do treino. A foto 48890227 mudou de treino para validacao para incluir animais pequenos na validacao, antes de medir desempenho; suas propostas de caixas ja haviam sido inspecionadas. As dez fotos de teste foram anotadas sobre imagens limpas, sem propostas do detector. Nenhuma foto de teste foi usada para ajustar limiar, resolucao, treino ou checkpoint.

## Comparacao no teste reservado

{tabela}

Cada modelo recebeu o mesmo orcamento de calibracao na validacao: entradas 1024/1280/1536, NMS 0,5/0,7 e confiancas 0,10 a 0,70. Escolha pelo menor MAE, com F1 de caixas como desempate.

- Original: `{o['config']}`; MAE de validacao {o['validacao']['MAE']:.3f}.
- Ajustado: `{a['config']}`; MAE de validacao {a['validacao']['MAE']:.3f}.

{controle}

{detalhes}

Os resultados sao de apenas dez fotos e 34 bovinos, com maximo de dez por foto. Portanto nao comprovam precisao em cenas de 75 a 100 animais. Contagem exata tambem nao garante caixas corretas; por isso precisao, sensibilidade e F1 foram medidos com correspondencia individual. Comparacoes visuais estao em `avaliacao/comparacoes/`, verde para gabarito e roxo para previsoes.

## Fotos brasileiras antigas: diagnostico exploratorio

Esse conjunto ja foi usado para escolher parametros em rodadas anteriores e nao e um teste independente.

| Modelo com parametros escolhidos na validacao nova | Fotos dentro do intervalo antigo | Erro medio ao intervalo |
|---|---:|---:|
| Original | {o['exploratorio_antigo']['dentro']}/{o['exploratorio_antigo']['total']} | {o['exploratorio_antigo']['erro_medio_ao_intervalo']:.3f} |
| Ajustado | {a['exploratorio_antigo']['dentro']}/{a['exploratorio_antigo']['total']} | {a['exploratorio_antigo']['erro_medio_ao_intervalo']:.3f} |

## Limite atual e proximo ganho necessario

O lote corrigiu exemplos concretos, mas a principal lacuna continua sendo rebanho brasileiro muito denso com anotacao completa. Nao seria correto incluir cenas parcialmente rotuladas como se os outros animais fossem fundo, nem transformar previsoes do proprio modelo em verdade. Mais fotos da mesma cena tampouco substituem novas fazendas, autores e condicoes. Para uma proxima versao, priorizar fotos de 20 a 100 animais em que cada individuo possa ser confirmado, com revisao cruzada das marcacoes.

## Fontes adicionais verificadas

- [NelloreBeefCattleDataset](https://github.com/EvertonTetila/NelloreBeefCattleDataset): aquisicao aerea vertical em confinamento; nao foi misturado automaticamente com fotos de solo.
- [Counting cattle Dataset](https://data.mendeley.com/datasets/dk54zg67dd/1): videos de Brangus no Uruguai; anotacao MOT de um video, nao um conjunto amplo de fotos densas brasileiras. Nao utilizado neste treino.

## Reproducao

`10_preparar_lote_brasil.py` exporta anotacoes, `11_ajustar_modelo_brasil.py` executa o experimento isolado, `12_avaliar_ajuste_brasil.py` calibra e mede, `13_galeria_rotulos.py` gera a galeria e `14_relatorio_ajuste_brasil.py` publica este relatorio. O treino se recusa a sobrescrever um experimento concluido e a avaliacao reutiliza seu resultado final existente. Os arquivos `protocolo.json`, `proveniencia_treino.csv`, `grade_validacao.csv`, `config_congelada_*.json` e `resultados.json` preservam a trilha de auditoria.
'''
    (EXP/'relatorio.md').write_text(texto,encoding='utf-8')
    print(json.dumps(dict(perfil_disponivel=liberar,epocas=len(curvas),original=o['teste'],ajustado=a['teste']),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
