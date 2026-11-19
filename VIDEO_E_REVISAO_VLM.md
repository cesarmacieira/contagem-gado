# Imagens, videos e conferencia seletiva no Streamlit

Em **Contar rebanho**, escolha **Imagem** ou **Video**.

## Imagem e exemplos reais

Em Imagem, a origem pode ser um arquivo enviado ou **Exemplos reais da internet**. A galeria usa as 14 fotos Pexels ja baixadas, com fonte, licenca, referencia visual e resultados historicos de Nano/Luna quando disponiveis. Os resultados historicos sao consultas de VLM sozinho, claramente separados do detector atual que roda abaixo. Oito referencias continuam ambiguas e nao viram gabarito exato.

## Detector + VLM

O detector fornece a contagem principal e as caixas. Baixa confianca, proximidade ao limiar, sobreposicao, bordas, 15 ou mais caixas e ausencia de deteccoes recomendam conferencia. Esses gatilhos sao heuristicos, nao uma probabilidade calibrada de erro. Omissao com alta confianca pode escapar; a conferencia manual permanece disponivel.

O botao **Conferir esta imagem/quadro** envia a imagem limpa e ate tres ampliacoes das regioes suspeitas ao GPT-6 Luna. O prompt avisa que ampliacoes sao da mesma cena e nao devem ser somadas. Nao informa a resposta esperada nem o numero de caixas. Discordancia ou incerteza pede revisao: nenhuma caixa e removida ou acrescentada automaticamente. Concordancia de contagens nao prova correspondencia animal a animal. Nao existe filtro fotometrico ativo por padrao.

Cada clique exige o envio explicito no formulario. Sem chamada por troca de slider, por recarga ou automaticamente para todos os frames. A chave e lida do .env local; nunca e gravada no cache. Reserva conservadora estimada de US$ 0,005 por consulta e limite local de US$ 0,05 por dia UTC, compartilhado no projeto. Falhas tambem consomem reserva; sem retry pago automatico. O controle local nao configura teto de cobranca no provedor. Respostas brutas e uso sao preservados antes da validacao. Precos usados: GPT-6 Luna Standard US$ 0,10/M entrada e US$ 0,50/M saida, consultados em 18/11/2026.

## Video

Formatos aceitos: MP4, MOV, AVI, MKV e WebM, desde que o OpenCV consiga decodificar o codec. Previa no navegador disponivel para MP4/WebM compativeis. O tamanho de upload segue o limite configurado no Streamlit.

Escolha **Analisar todos os quadros** ou um intervalo de 0,25 a 10 segundos. A decodificacao percorre o arquivo ate o fim e inclui o ultimo quadro decodificado. Amostragem nao significa inspecao de todos os frames: eventos entre amostras podem ser perdidos. Intervalos e tempos sao derivados do FPS declarado; videos com taxa variavel podem ter tempos aproximados.

O detector e os pos-processamentos do perfil selecionado sao compartilhados com a inferencia de imagem. O resultado traz linha do tempo, minimo/maximo/media de bovinos simultaneamente visiveis, seletor de quadro anotado e exportacoes CSV/JSON com todas as caixas. O VLM so e consultado no quadro que o usuario escolher.

**Nao ha identificacao de individuos entre frames nem contagem unica de passagem.** Nao somamos as contagens dos quadros. Mudanca de camera, reaparecimento e oclusao exigiriam rastreamento e validacao adicionais para contar individuos unicos.

Limite de 5.000 quadros analisados por execucao para conter consumo de memoria/disco; se excedido, o app pede intervalo maior e nao declara resultado parcial como concluido. Falhas detectaveis de decodificacao tambem interrompem a analise. Resultado e frames ficam em data/cache_video, fora do Git, reutilizados para mesmo arquivo/configuracao. O cache ainda nao tem limpeza automatica; pode ocupar bastante disco em videos longos.

## Verificacao desta entrega

- 109 testes passaram na suite completa, incluindo preservacao do ultimo frame, modo todos os quadros, video invalido, limite de amostras, gatilhos que nao dependem so da confianca, cache, sigilo da chave e bloqueio de novas cobrancas apos falha.
- AppTest abriu a pagina de contagem, rodou o detector em uma foto real da galeria e abriu a interface de video sem excecoes.
- Teste de upload/processamento pela interface: video de demonstracao montado com fotos reais, 18 frames a 6 FPS, 3 segundos. Quatro frames analisados na amostragem de 1 segundo, incluindo o ultimo. Exportacao e seletor renderizados sem excecoes. Isso testa o fluxo, nao a precisao em filmagens reais.
- Consulta real do fluxo hibrido na foto nelore_1512: detector 5; VLM 7 com incerteza; contagem principal preservada. Custo estimado pelos tokens retornados: US$ 0,000353. Resposta em data/experimentos/hibrido_video_v1/smoke_api.json.
- Nenhum peso foi treinado ou promovido. A precisao do conjunto detector+VLM em novas cenas e videos ainda nao foi demonstrada.
