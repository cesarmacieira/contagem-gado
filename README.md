# Contagem de gado (qualquer ângulo)

Detecção e contagem de bovinos com **dados reais públicos**, nas fontes:
- **Zenodo 11048412** (Institut de l'Elevage / ICAERUS, CC BY 4.0): drone vertical em pasto francês, 60–100 m. 1.385 imagens; o zip tem 4.904 caixas em 1.387 rótulos; 2 rótulos (157 caixas) não têm imagem, então o projeto usa 4.747 caixas em 1.385 imagens (o registro declara 4.941; os 37 restantes não foram explicados).
- **COLO** (Hugging Face, MIT): vacas leiteiras em instalação, vista de cima, lateral e externa.
- **Open Images V7** (anotações CC BY 4.0): fotos comuns com a classe Cattle/Bull.
- **CattleEyeView** (SUTD / AnimalEYEQ, download manual): vista de cima de vacas avermelhadas numa rampa de curral, quadros de 14 vídeos.

Nenhum dado sintético é usado. O erro de contagem é medido **por cenário** (drone em pasto, COLO topo, COLO lateral, COLO externo, foto comum). A página **Origem dos dados** do app detalha fontes e limites.

## Sequência completa (PowerShell, ambiente compartilhado)

```powershell
cd C:\Users\cesar.macieira\Desktop\Trabalho\Python_312\contagem-gado
..\.venv\Scripts\Activate.ps1
python -m pip install pyarrow
python 00_verificar_ambiente.py
python 01_obter_dados.py --seletivo            # etapa 1: Zenodo (só imagens com vaca + parte das sem vaca)
python 03_obter_dados_extras.py                # etapa 1b: Open Images + COLO (retoma de onde parou)
python 04_obter_cattleeyeview.py               # etapa 1c (opcional): CattleEyeView, lê os zips de .\CattleEyeView (retoma de onde parou)
jupyter lab 02_treinamento.ipynb               # etapa 2: Run All (divide, gera tiles, treina, mede a contagem, salva em data\processed)
streamlit run app.py                           # etapa 3: app em http://localhost:8501 (logs em logs\app.log)
```

Se você baixou o zip completo do Zenodo em outra pasta (por exemplo, Downloads), mova para `data\raw` e rode `python 01_obter_dados.py`: ele confere o MD5 e, se bater, não baixa de novo.

```powershell
Move-Item "$env:USERPROFILE\Downloads\Cattle_drone_images_042024.zip" .\data\raw\
python 01_obter_dados.py
```

Com o zip completo em `data\raw`, o notebook usa ele no lugar do zip do `--seletivo`.

Variantes:
- `python 01_obter_dados.py` baixa o zip completo (~16,6 GB; lento se o Zenodo limitar a velocidade). `--listar` só refaz o inventário.
- `python 03_obter_dados_extras.py --inventario` só reconta o que existe em `data\extras`. `--fonte openimages` ou `--fonte colo` baixa uma fonte só; `--max-train 0` baixa as 2.749 imagens utilizáveis do Open Images (padrão: 2.000).
- No notebook, `USAR_EXTRAS = False` treina só com o Zenodo (o app continua funcionando, com um único cenário).
- As etapas 1 e 1b podem ser interrompidas (Ctrl+C) e repetidas.

Espaço em disco: ~25 GB (zip 16,6 GB + imagens extraídas + tiles) mais ~1,2 GB dos extras. O notebook extrai só as imagens necessárias.

## Dados extras

| Fonte | Conteúdo | Observação |
|---|---|---|
| Open Images V7 (`Cattle` + `Bull`) | Fotos comuns, vários ângulos e cenários (inclui búfalos, fotos antigas) | Só imagens SEM caixa de grupo (`IsGroupOf`) e SEM desenho (`IsDepiction`). Divisão oficial train/validation/test. |
| COLO | `1_top`, `2_side`, `3_external` | Usa só o `bbox`; a `segmentation` do dataset está deslocada. A divisão train/test dos autores **não** é usada (vaza: a mesma cena aparece nos dois lados); o notebook refaz a divisão por cena ou sequência. |

Estado em `data/extras` (conferido em 13/11/2026, antes do CattleEyeView): Open Images 2.000 / 115 / 368 imagens (train / validation / test); COLO 1.254 imagens (11.803 caixas; o README do COLO declara 11.818).

**CattleEyeView (download manual, já feito):** o download é por formulário do Google, linkado em github.com/AnimalEyeQ/CattleEyeView; coloque os arquivos em `.\CattleEyeView`. O `04_obter_cattleeyeview.py` lê os `annotation-*.zip` direto, usa a pasta `detect` (caixas, classe única) e grava em `data\extras\cattleeyeview` no máximo 120 quadros espaçados por vídeo (`--max-por-video`; quadros vizinhos são quase iguais). Conferido: 14 vídeos, 1920×1080, caixas batem com as vacas em quadro visualizado; 989 rótulos ficaram sem imagem no download (não usados). A divisão dos autores é por vídeo (teste: 01, 05, 07, 09, 10); a pasta `val` deles repete o teste e é ignorada, e a validação sai de 2 vídeos de treino. `images.tar.gz`, `videos-*.zip` e `YOLO.ipynb` não são usados.

## Teste com imagens do Brasil (gabarito)

`data\teste_brasil` tem 10 fotos de gado no Brasil (Wikimedia Commons; licença e autor de cada uma em `fontes.csv`) e o `gabarito.csv`, com **mínimo e máximo** por imagem. A contagem é visual, feita com zoom: em cenas densas ou com animais ocultos não existe número exato, e a coluna `confianca` diz quanto confiar em cada linha. Três fotos ruins para contagem ficaram em `descartadas\`. Nenhuma dessas imagens foi usada no treino.

```powershell
cd C:\Users\cesar.macieira\Desktop\Trabalho\Python_312\contagem-gado
..\.venv\Scripts\Activate.ps1
python 05_avaliar_gabarito.py
python 05_avaliar_gabarito.py --conf 0.35       # outro limiar
```

O script imprime a contagem do modelo contra o intervalo (`dentro`, `abaixo` ou `acima`), grava `data\teste_brasil\resultados.csv` e as imagens anotadas em `data\teste_brasil\saida`. Para testar pelo app, use `streamlit run app.py` e envie as imagens de `data\teste_brasil`.

## Estrutura

```text
contagem.py              tiles, rótulos YOLO, NMS, detecção por tiles (detectar) e por imagem inteira (detectar_inteira)
01_obter_dados.py        download resumível + MD5 + inventário (Zenodo)
03_obter_dados_extras.py Open Images + COLO em YOLO, com inventário (data/extras)
04_obter_cattleeyeview.py CattleEyeView (zips em .\CattleEyeView) em YOLO, quadros espaçados
05_avaliar_gabarito.py   roda o modelo nas fotos de data/teste_brasil e compara com o gabarito
02_treinamento.ipynb     preparo, divisão sem vazamento, treino YOLO, contagem por cenário, métricas
app.py                   aplicação Streamlit (tema CAST)
.streamlit/config.toml   tema (cores e fonte)
data/raw                 zip original e inventário
data/extras              Open Images, COLO e CattleEyeView em YOLO (não versionado)
data/interim             imagens extraídas (não versionado)
data/yolo                tiles, extras e dataset.yaml (não versionado)
data/processed           modelo_gado.pt, métricas e exemplos lidos pelo app
runs/                    saídas do treino do YOLO (não versionado)
```

## Método

- **Drone (Zenodo):** imagens de 4000×3000 px ou mais, vaca pequena. Treino e inferência em tiles de 1024 px sem redimensionar; as detecções dos tiles são unidas com NMS.
- **COLO e Open Images:** imagens de até ~1.150 px e vaca grande. Entram inteiras, sem tiles.
- **Divisão sem vazamento:** por voo (Zenodo), por cena/sequência (COLO), por vídeo (CattleEyeView) e pela divisão oficial (Open Images). O notebook confere com `assert` que nenhum grupo cai em dois conjuntos.
- **Treino:** classe única `vaca`; `flipud = 0.15` (giro vertical só em parte, porque inverte as vacas das fotos de solo).
- **Tamanho de entrada e limiar:** nos cenários de foto inteira o notebook testa 1024, 1280 e 1536 px e vários limiares na validação e fica, por cenário, com a combinação de menor MAE (limiar global: menor WAPE médio). O teste é medido uma vez, com essas escolhas. As fotos do Brasil são uma conferência independente: rode `python 05_avaliar_gabarito.py` depois do notebook.
- **Métricas por cenário:** MAE, RMSE, viés, MAPE, WAPE, R² e erro no total. Compare cenários pelo WAPE, não pelo MAE. O mAP por tile é secundário.
- No app, o tipo de imagem escolhido define o modo (recortes ou imagem inteira), o tamanho de entrada e o limiar padrão.

## Limites

- **Nelore e pasto brasileiro:** nenhuma fonte aberta com caixas foi encontrada. O desempenho nesse gado **não está medido**; valide com imagens suas anotadas.
- **COLO:** há vacas visíveis sem caixa (conferido em imagem), então o modelo parece contar a mais do que o real nesse cenário.
- **Open Images:** não foi feito para contagem; a classe inclui búfalo; fotos de rebanho denso ficam fora (`IsGroupOf`).
- **CattleEyeView:** uma só vista (rampa, de cima), poucas vacas por quadro (1 a 3 nos quadros conferidos). Quadros do mesmo vídeo são muito parecidos, então o erro medido vale menos do que o número de imagens sugere. Não prova nada sobre Nelore.
- **Sem negativos de solo:** quase não há fotos de solo sem bovino no treino; falsos positivos em cenas vazias são possíveis.
- Os resultados dos cenários não são comparáveis com os do artigo do COLO (a divisão é outra).
- Vacas sob árvores ou sombra podem ser perdidas.
- Não há identificação individual; imagens sobrepostas da mesma cena ou missão não podem ser somadas sem contar duas vezes.
- Treino original confirmado: YOLO11s, 60 épocas. A recalibração usa os pesos reais, sem novo treino. Novos experimentos não sobrescrevem o modelo do app.


## Correções de contagem (versão 2)

- Deduplicação de caixas exatamente iguais após unir Cattle/Bull. Caixas apenas parecidas não são fundidas.
- Junção de tiles por IoU, sem eliminar bezerros por contenção; NMS local separado da junção e limite explícito de 1.000 detecções.
- Tiles com fragmento visível de animal sem rótulo são excluídos do preparo do próximo treino.
- Cache do app considera a versão dos pesos/arquivos. Fotos grandes de celular não são classificadas automaticamente como drone.
- App oferece perfil **Rebanho de solo — experimental** (1280/0,15/NMS 0,7) e ajuste manual. O perfil exploratório não substitui o validado.
- Avaliação mede TP/FP/FN por correspondência individual (IoU 0,5), exatidão de contagem e erro por densidade, além de MAE/WAPE.
- As dez fotos de `data/teste_brasil` são exploratórias: já foram usadas para procurar parâmetros. Não são um teste independente.

### Recalibrar os pesos existentes (não treina)

```powershell
python 07_revisar_rotulos.py                 # simula a limpeza
python 07_revisar_rotulos.py --aplicar       # backup, limpeza e remoção do cache de rótulos antigo
python 06_calibrar_modelo.py                 # candidato isolado em data/avaliacoes
python 06_calibrar_modelo.py --promover      # calibra na validação, mede o teste, faz backup e atualiza o app
python 05_avaliar_gabarito.py --perfil brasil-experimental
python 05_avaliar_gabarito.py --pasta data/brasil_novo_teste
python -m unittest discover -s tests -v
```

Cada avaliação do gabarito cria uma pasta com parâmetros, hash dos pesos, CSV e imagens anotadas. `--varrer` inclui NMS de 0,4 a 0,85 e salva contagens por imagem. Seu vencedor é exploratório e não é promovido automaticamente.

A recalibração escolhe resolução/confiança/NMS **apenas na validação**, por menor MAE, usando F1 de caixas para desempate. O teste é executado depois, somente na configuração fixada. Pesos permanecem os mesmos. mAP e curvas de treino exibidos continuam históricos, identificados como tal; não confundir com a nova avaliação de contagem.

### Novas imagens brasileiras

```powershell
python 08_obter_brasil.py --max-imagens 60
streamlit run 09_revisar_brasil.py
```

O coletor grava imagens e proveniência em `data/brasil_candidatos`, filtra licenças, evita cópias do teste por hash perceptual e permite retomar downloads. Precisa de acesso HTTPS ao Wikimedia Commons. Uma imagem obtida no navegador já está disponível. Não há geração de imagens sintéticas.

O revisor permite editar caixas em coordenadas da imagem, verificar a sobreposição visual e confirmar imagens negativas. Só imagens explicitamente revisadas entram no manifesto de `data/brasil_revisado`. O grupo da cena deve ser o mesmo em fotos da mesma sessão/vídeo; um grupo não pode estar em mais de um conjunto. Por padrão o coletor agrupa conservadoramente por autor: refine apenas quando tiver evidência de cenas independentes.

O notebook já importa esse manifesto, mas exige imagens/caixas suficientes de validação e teste antes de treinar o novo cenário. **Não copie as dez imagens exploratórias para treino.** `dados_locais.py` bloqueia cópias exatas de imagens reservadas para avaliação.

`data/brasil_novo_teste` contém um lote inicial separado: uma foto pública de sete bovinos, com contagem e caixas manuais aproximadas por inspeção visual do assistente. Foi anotada antes de executar o detector e fica fora da calibração e do treino. Uma foto não prova generalização e as caixas merecem revisão humana antes de uso científico.

### Próximo treino

O notebook inicia com `RAPIDO = False`, 100 épocas propostas e sem inversão vertical de fotos de solo. Cada execução grava dataset, pesos e artefatos em pastas próprias com identificador de experimento. Um teste rápido também fica isolado e não substitui `data/processed`.

Para revisar um experimento no app sem promovê-lo:

```powershell
$env:CONTAGEM_ARTEFATOS = "data/experimentos/IDENTIFICADOR/artefatos"
streamlit run app.py
```


### Resultado da recalibração realizada em 15/11/2026

A candidata da versão 2 foi medida em 622 imagens de validação e 1.226 de teste, com os pesos originais. MAE no mesmo teste e com gabaritos deduplicados: referência congelada **0,7488**; candidata **0,8051**. A candidata foi rejeitada por regressão e ficou em `data/avaliacoes/calibracao_20261115_200713_400039` com `liberacao.json`. O modelo e as configurações originais continuam como padrão do app. Não houve novo treino.

O perfil validado conserva a junção antiga de tiles para corresponder aos resultados históricos. O caminho novo, sem supressão por contenção e com NMS local separado, é usado por novos experimentos e ajustes manuais. Alterar uma regra plausível não garante ganho nos pesos já treinados; por isso a compatibilidade é explícita.

A promoção com `06_calibrar_modelo.py --promover` agora compara o MAE da candidata com previsões congeladas da referência usando o mesmo gabarito corrigido. Se houver regressão ou não existir comparação válida, os artefatos do app são preservados. Essa é uma porta de liberação, não uma busca de novos parâmetros olhando o teste. Não repita buscas adaptativas nesse teste como se ele continuasse independente.

Na nova foto reservada com sete animais, tanto a configuração original de foto comum quanto a experimental localizaram sete caixas correspondentes, sem duplicação ou omissão frente à anotação visual aproximada. O resultado de uma única foto não valida rebanho denso.

### Novas fotos, anotacoes e dois ajustes executados em 16/11/2026

Foram obtidas 64 fotografias publicas e preparadas 44 imagens com marcacoes completas: 22 treino, 9 validacao e 13 teste, totalizando 152 caixas. O pacote consolidado esta em `data/brasil_dataset_v2`, com galeria, gabarito, dicionario, YOLO, COCO, fontes e licencas. Quatro cenas parcialmente rotuladas ficam fora do treino e do teste exato. Sao anotacoes de primeira revisao visual do assistente.

Dois ajustes reais foram treinados e rejeitados: nao superaram o modelo original recalibrado. Os experimentos, pesos e resultados estao em `data/experimentos/brasil_lote_01_v2` e `brasil_lote_01_v3`. O primeiro teste de dez fotos ja havia sido examinado antes da segunda rodada; tres novas fotos foram reservadas para esta rodada.

O ganho confirmado veio de configuracao escolhida nas nove imagens de validacao: entrada 1024, NMS 0,5, confianca 0,30, com os pesos originais. Nas 13 fotos dos dois testes, o perfil antigo do app (confianca 0,45) fez 8 contagens exatas e MAE 0,538; o calibrado fez 11 exatas e MAE 0,231. Essa amostra tem no maximo dez bovinos por imagem: nao demonstra melhora em rebanhos de 75 a 100 animais.

O app inclui a pagina **Rotulos Brasil** e o perfil opcional **Brasil - fotos com poucos animais**. O perfil experimental para rebanhos densos permanece identificado separadamente. Os scripts 10 a 19 preservam exportacao, treinos, calibracao, auditoria, consolidacao e verificacao da entrega. Nenhum peso original foi sobrescrito.

### Retomada e avaliacao do detector de rebanhos em 16/11/2026

O treino `detector_denso_v3` foi retomado do checkpoint e concluiu 20 epocas. Foram preparados 4.366 exemplos publicos com 14.430 caixas, agrupando variantes, linhagens de videos e cenas geometricamente correspondentes antes da divisao. O treino final utilizou 512 dessas imagens, 22 brasileiras repetidas somente no treino e 256 imagens antigas de replay. Gabarito e dicionario estao em `data/detectors_cows/gabarito_final.csv` e `dicionario.json`.

No teste reservado de 873 imagens, o detector ajustado reduziu o MAE de 2,297 para 1,712 frente ao original recalibrado para rebanhos; F1 das caixas passou de 0,678 para 0,753. No recorte com oito ou mais animais, MAE 7,814 para 5,221. Ainda ha sobrecontagem e o resultado nao representa alta precisao.

**O novo peso nao foi liberado no app:** a transferencia para as fotos brasileiras antigas regrediu. Ficou em 5/10 dentro dos intervalos, frente a 8/10 do preset experimental anterior. Em `brasil_08`, contou 46 frente ao intervalo de 75 a 100 (o preset anterior contava 90). Essas fotos antigas sao desenvolvimento e nao teste independente. O perfil para poucos animais e os pesos originais foram preservados.

Relatorio: `data/experimentos/detector_denso_v3/relatorio_final.md`. Os scripts 34, 37, 38 e 39 registram escolha antes do teste, bloqueio de liberacao por regressao brasileira, verificacao da funcao de inferencia e comparacao adicional com o preset preexistente. Resultados e marcacoes ficam em `data/experimentos/detector_denso_v3/avaliacao`. O proximo treino para Nelore denso precisa de caixas reais e completas nessas cenas; a colecao externa e os pontos de contagem nao substituem esse material.

### Modelos independentes por visao

Em **Contar rebanho**, a opcao **Escolher modelo por visao** permite selecionar
solo com poucos bovinos, solo denso, drone vertical, curral de cima, curral
lateral ou rampa. Cada rota pode ter pesos e parametros independentes em
`data/modelos/especialistas.json`. A escolha da visao e explicita.

As rotas que ainda nao possuem uma candidata superior usam o peso de referencia;
isso nao significa que seis especialistas ja foram treinados. Um ganho em drone
nao precisa melhorar solo para ser aproveitado exclusivamente na rota aerea.
`48_especializar_drone.py` prepara recortes sem fragmentos sem rotulo, ajusta um
especialista aereo e compara contagem e caixas antes de alterar somente essa rota.
O teste aereo reutilizado e identificado como regressao historica.

Foram comparados tambem YOLO11s geral, YOLO26n, RT-DETR-L, YOLO11x e TTA.
Nas 13 fotos brasileiras de regressao, as candidatas TTA ficaram abaixo do
perfil atual de poucos animais. Relatorio em
`data/experimentos/alternativas_regressao_v1/relatorio.md`.

**Ressalva de qualidade dos dados:** foi confirmada rotulagem incompleta em uma
foto da validacao externa publicada. As contagens obtidas pelo numero de linhas
dessa fonte nao sao contagens exatas integralmente revisadas. Ver
`data/experimentos/recortes_escala_v1/LEIA_ANTES_DO_PROXIMO_TREINO.md`.

O lote novo de outra IA fica em `data/brasil_denso_novo`; a conferencia, que nao
executa detectores, e `conferir_novo_lote.py`. Intervalos com minimo diferente de
maximo nao sao aprovados como gabarito exato, mesmo quando n_bovinos esta presente.
Atualizacoes e o recorte proposto de vinte bovinos estao em `data/apoio_coleta_denso`.

### Conferencia contextual de montaria (experimental)

Na visao `solo_denso`, a opcao `Conferir deteccoes — experimental` acrescenta ao
verificador visual uma conferencia CLIP condicionada a pessoa em posicao de
montaria. A opcao permanece desligada por padrao. A regra retirou a montaria no
caso conhecido de Bernard (29 para 28 deteccoes), preservando vinte pontos
visuais e os 31 casos anteriores conferidos. O total exato da cena continua sem
gabarito aprovado; esses resultados sao regressao conhecida.

O CSV completo informa as caixas removidas e as duas diferencas de similaridade
semantica, que nao representam probabilidades. Pesos e receita sao conferidos por
SHA-256 e participam das chaves de cache. Relatorio e comparacao visual:
`data/experimentos/verificador_montaria_v4/RELATORIO.md`.

### Conferencia de sombras para vista externa

A nova visao **Curral externo - vista de cima** usa um verificador de recorte
retangular estrito para reduzir sombras. Na cena conhecida de 16 bovinos,
remove duas sombras e passa de 19 para 17 deteccoes, preservando os 15 pontos
antes cobertos. Ainda existem duplicacoes e a omissao do bovino escuro.
As seis visoes anteriores e o peso do detector foram preservados.

Na validacao de 96 tiles aereos: TP 320 preservados, FP 45 para 43, FN 19
preservados. A visao permanece em desenvolvimento; nao demonstra desempenho
final em confinamento brasileiro. Os modelos do verificador ficam em
`data/modelos/verificador_sombras_v1/`, com identidades no catalogo de visoes.
Esse classificador usa receita diferente do verificador de solo e nao deve
ser carregado pela API de contexto quadrado. Detalhes da integracao e seus
limites em `data/experimentos/integracao_sombras_app_v1/RELATORIO.md`.


### Correção de instâncias no curral externo

O perfil `curral_externo_topo` agora usa confiança padrão 0,30, suprime caixas redundantes com sobreposição e contenção altas e recupera propostas isoladas de confiança 0,15–0,30 quando o verificador visual as confirma (score mínimo 0,50). A confiança original permanece no CSV; a auditoria registra as recuperações. Os outros perfis permanecem iguais.

Na cena conhecida 0359, a versão anterior retornava 17 caixas para 15 animais correspondidos; a correção retorna 16 caixas para os 16 animais, incluindo o bovino escuro. Essa cena foi usada para desenvolver as regras e não constitui teste independente. Em 96 imagens de validação aérea, FP/FN passaram de 43/19 para 20/23, e MAE de 0,4375 para 0,28125. Houve quatro omissões adicionais; a validação não representa novas fazendas brasileiras. Resultado reproduzível: `97_verificar_correcao_instancias.py`, registro `data/experimentos/correcao_aerea_instancias_v2/resultado.json`.


### Recuperação em segunda resolução

O perfil `curral_externo_topo` também executa uma passagem em 1536 px para procurar bovinos omitidos. Só acrescenta caixas isoladas (IoU abaixo de 0,10 e contenção abaixo de 0,40 em relação às caixas existentes), com confiança mínima 0,40 e conferência visual mínima 0,50. As caixas anteriores permanecem. O CSV registra `segunda_resolucao` e `recuperada_por_conferencia`; propostas rejeitadas continuam na auditoria. A segunda passagem aumenta o processamento.

Nas 96 imagens de validação usadas para desenvolvimento: TP 316→317, FP 20→20, FN 23→22, MAE 0,28125→0,27083. Os três recortes brasileiros conhecidos mantiveram 16, 8 e 7 bovinos. Nenhum ganho em nova fazenda é demonstrado por esse resultado. O lote histórico de 85 recortes não foi consultado para ajustar esta etapa. Configuração e verificação: `99_recuperar_omissoes_aereas.py`, `100_integrar_recuperacao_escala.py`; registros em `data/experimentos/integracao_recuperacao_escala_v1/resultado.json`.


### Treino aéreo ampliado e revisão de referências

O treino candidato aéreo ampliado usa 1.200 recortes de 394 originais e 17 grupos; não modifica os pesos ativos durante o ajuste. O gerenciador 105 executa treino e avaliações em sequência, com trava e estado persistente. A avaliação completa 103 gera uma proposta de perfil; não promove pesos automaticamente.

A auditoria pelo Claude desktop identificou fragmentos duplicados em três rótulos de validação. A versão revisada está em `data/experimentos/gabarito_aereo_revisado_v2/`; os originais permanecem intactos e a revisão foi feita após inspecionar previsões, para desenvolvimento. As mesmas previsões passaram de FN 22 para 17 por mudança da referência: isso não é ganho do modelo. Há três correções em fotos completas e cinco ocorrências nos recortes sobrepostos. O treino candidato continua avaliado por rota, sem incorporar a Fazenda Campanário ao treino.
