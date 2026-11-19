# Fontes verificadas para ampliar o cenário brasileiro

Consulta: 15/11/2026. A existência de um download não significa que as caixas estejam completas ou que a divisão não tenha cenas repetidas.

| Fonte | Conteúdo verificado | Licença/acesso | Decisão atual |
|---|---|---|---|
| [Wikimedia — Nelores.png](https://commons.wikimedia.org/wiki/File:Nelores.png) | Foto de Vinicius Cruz, Serra do Tabuleiro, 03/01/2018; 2048 × 1501; sete bovinos identificáveis | CC BY-SA 4.0; original baixado | Reservada em `data/brasil_novo_teste`; contagem e caixas aproximadas por inspeção visual do assistente, antes da inferência. Não usar em treino/calibração. |
| [Nelore — henriques Workspace, v3](https://universe.roboflow.com/henriques-workspace-lhcrk/nelore-5bbhr/dataset/3) | 255 imagens; 186/36/33 na divisão do autor; versão com redimensionamento para 640 × 640 | Página declara CC BY 4.0. Exportação YOLO abriu exigência de login | Candidata. Não foi baixada nem admitida no treino. Conferir cobertura de corpo inteiro e cenas entre conjuntos. |
| [NelloreBeefCattleDataset — Everton Tetila](https://github.com/EvertonTetila/NelloreBeefCattleDataset) | Nelore de confinamento em Laguna Carapã/MS; drone vertical a 15 m; doze voos e imagens de 4864 × 3648 | Link de download no README; licença dos dados não foi explicitada na página consultada | Candidata aérea; não resolve sozinha o domínio de solo. Verificar condições de uso, rótulos e separar por voo/data antes de incorporar. |
| [cattleCounting — Soares et al.](https://github.com/vhasoares/cattleCounting) | Bases de fazendas brasileiras, imagens aéreas e caixas em XML; inclui links de treino e BR_set | Downloads Google Drive; licença dos dados não explicitada na página consultada | Candidata aérea. Não foi misturada com o treino; BR_set é apresentado como teste pelos autores. |
| [ACI — UFJF, versão 2](https://data.mendeley.com/datasets/rzyc5ssvzx/2) | 6.182 imagens, mas rótulos de rosto, olhos, orelhas e focinho | CC BY 4.0 | Rejeitada para contagem de corpo inteiro: mapear caixas de rosto para caixas de animal mudaria a tarefa. |
| [ILVO/XGain — 15688349](https://zenodo.org/records/15688349) | Imagens de pasto e caixas de comportamentos; arquivo de aproximadamente 946 MB | CC BY-NC 4.0 no aviso de direitos | Não incorporada. Não é uma fonte de Nelore brasileiro e tem restrição de uso não comercial. |

## Coleta e admissão

`08_obter_brasil.py` busca fotografias públicas no Commons, conserva a página de origem, autor, licença, URL da imagem e SHA-256. Arquivos vão para candidatos, nunca diretamente para treino. A consulta HTTPS por Python falhou por resolução DNS neste ambiente; um original foi obtido pelo navegador.

`09_revisar_brasil.py` permite revisar caixas em pixels e salvar um manifesto com grupo, conjunto e confirmação da revisão. A origem deve permitir rastrear fotos da mesma sessão. O importador bloqueia cenas em mais de um conjunto e cópias exatas dos conjuntos brasileiros reservados. O coletor também compara hashes perceptuais com esses testes para evitar versões redimensionadas da mesma foto.

Fotos de grupo não podem receber apenas uma caixa envolvendo o rebanho. Anote um bovino por caixa, incluindo animais parcialmente visíveis com evidência suficiente. Registre ambiguidades e descarte cenas sem gabarito confiável. Uma caixa predita, sem revisão, não é anotação verdadeira.

O lote novo contém somente uma imagem e não permite medir generalização. A prioridade da próxima coleta continua sendo rebanho denso de solo, animais distantes, bezerros junto às vacas e negativos reais.
