# Retomada: experimento cruzado de madeira na GPU

Estado: lote e protocolo lidos; nenhum novo treino iniciado nesta retomada. Preservar modelo/catálogo ativos. Não repetir os verificadores logísticos de madeira: já falharam.

Fonte: data/melhorias_claude_v1/lote_madeira/LEIAME_E_PROTOCOLO.md. Há 6 tiles A e 6 B, com rótulos vazios, provenientes de quatro fotos de validação. Antes de treinar, conferir visualmente pilhas_amostra.jpg e os recortes; ausência de GT não basta para comprovar negativo.

Executar duas dobras do próprio detector: base modelo_gado.pt original (SHA afd71a25...), mesmo dataset/receita de aereo_ampliado_v2, acrescentando apenas a pilha da dobra com repetição x5. A→B e B→A. Usar 20 épocas, batch16, workers0, cache RAM, retomada por checkpoint. Não promover automaticamente.

Evitar vazamento: retirar as fotos que geraram negativos e TODOS os recortes derivados delas da avaliação da dobra. Para seleção durante o treino, usar só referências sem essas fotos; não escolher checkpoint/limiar pela pilha-alvo. Fixar last da época20 e confiança0,55 antes de avaliar; reportar também a grade prefixada .35/.45/.55/.65 sem escolher o melhor pelo alvo. Documentar que a base pré-treinada já pode conhecer a fazenda e as pilhas; é teste de ajuste incremental, não avaliação independente.

Comparar base e candidato nas MESMAS fotos completas da dobra. Critérios: FP na pilha-alvo -50% ou mais, FN total aumenta no máximo2, FN Mauron aumenta no máximo1, MAE não piora. Coordenadas das regiões apenas para métricas, nunca para mascarar a inferência. Não usar teste nem Campanário. Não incorporar automaticamente os rótulos373 da outra IA: contenção geométrica não prova fragmento e pode excluir bezerros.

Execução Windows: sessões exec anteriores pararam ao encerrar turno. Gerenciador105 e spawn destacado pelo node_repl funcionaram. Ambiente mínimo sem credenciais está em data/runtime/ambiente_pipeline.json. Logs, PID e heartbeat; conferir avanço real antes de afirmar que está rodando.

Preferências: respostas curtas e economia de tokens; nunca sugerir coleta própria; nunca enviar mensagens a pessoas; Claude somente CLI ou desktop, sem chamadas diretas à API. Não repetir experimentos já rejeitados.
