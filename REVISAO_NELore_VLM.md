# Revisão da foto nelore_1512

Referência fornecida pelo usuário: 7 bovinos, incluindo animais parcialmente visíveis. O total ainda não é um gabarito espacial revisado.

O detector ativo foi executado com os parâmetros do perfil Solo - poucos bovinos: imagem inteira, entrada 1024, confiança 0,30, IoU 0,50. Resultado: 5.

Foram testadas quatro passagens sem alteração de geometria: original, CLAHE na luminância, gamma 0,7 para clarear sombras e gamma 1,4 para reduzir luzes. Todas retornaram 5; a combinação conservadora também retornou 5. Não houve recuperação. Esta receita não foi promovida ao aplicativo. “Pass4” não foi identificado como filtro definido no projeto; estas são quatro transformações explicitamente documentadas, não uma implementação presumida desse nome.

Resultados e imagens: data/experimentos/iluminacao_nelore_v1/. Pesos e perfis existentes preservados. A foto não foi incorporada ao treino.

## Revisão visual opcional no Streamlit

Após processar uma foto, abra “Segunda leitura por IA visual — opcional”. Informe a chave no campo protegido ou configure OPENAI_API_KEY no processo do Streamlit. Marque o envio e clique em “Revisar imagem uma vez”. Nunca coloque credenciais no chat ou no código.

O cliente envia somente a imagem reduzida para até 1024 px e uma instrução fixa; não informa ao modelo a resposta esperada nem a contagem do detector. Solicita pontos, evidências e incertezas. A resposta aparece separada do detector e não substitui sua contagem.

Modelo fixado: gpt-5.4-mini-2026-03-17. Sem ferramentas, sem repetição automática, saída máxima de 800 tokens. Estimativa conservadora local: US$ 0,0081 para a consulta, com reserva de US$ 0,01. Isso não configura limite de cobrança no provedor. Há uma tentativa por chave neste projeto; falhas também consomem a tentativa, impedindo repetição de cobrança por recarga. A mesma resposta concluída é reutilizada do cache. O cache e o controle de orçamento não gravam a chave e ficam fora do Git.

Na implementação inicial não havia OPENAI_API_KEY disponível; uma consulta real foi executada posteriormente, conforme atualização abaixo. Testes automatizados usam respostas simuladas, sem cobrança. Pontos propostos por VLM podem duplicar ou omitir animais; precisam ser conferidos visualmente. Este modo limitado a 20 animais não é solução validada para rebanhos densos.

Verificação: 95 testes passaram na suíte completa; app.py e os três novos módulos/scripts passaram na compilação Python. Os testes novos verificam o cache sem nova chamada, bloqueio após falha, ausência da chave nos arquivos, rejeição de contagem inconsistente e suporte entre passagens distintas. Não foi realizada verificação interativa da interface nem avaliação real da precisão do VLM.

## Atualização — chave AMS e consulta real

A chave foi localizada no .env de ams-first-posso-ajudar e configurada no .env local, confirmado como ignorado pelo Git. Uma consulta real retornou resposta, mas foi rejeitada por intervalo incompatível com os indicadores de incerteza dos pontos. Nenhuma contagem VLM foi aplicada e não houve repetição paga. O uso exato dessa primeira resposta não foi preservado pelo cliente anterior; não é possível declarar seu custo real.

O cliente foi corrigido para preservar resposta e uso antes da validação, inclusive em falhas. Seis testes específicos passaram, incluindo preservação de resposta inválida e bloqueio de nova chamada. O limite de uma tentativa desta chave já foi consumido.
