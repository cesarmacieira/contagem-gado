# Fotos e marcacoes de bovinos brasileiros

44 imagens completas e 152 caixas, em dois lotes. Primeira revisao visual do assistente; nao representa consenso de especialistas. Fontes, autores, licencas e transformacoes de recorte estao em fontes_e_marcacoes.json e na galeria. As fotografias conservam suas licencas de origem.

Uma caixa por bovino individualizavel, incluindo bezerros, animais ocultos e cortados. Cabeca e corpo do mesmo animal nao sao duas instancias. Cavalos, pessoas, aves e objetos ficam fora. As caixas cobrem partes visiveis, sem inferir corpos ocultos. Os quatro exemplos parciais ficam em anotacoes_parciais.json e nao entram no treino ou no teste exato.

Treino: 22 fotos/78 bovinos. Validacao: 9/30. Teste: 13/44, provenientes de duas rodadas (10+3). O teste da primeira rodada ja foi examinado antes da segunda; consultar os relatorios dos experimentos, sem alegar que todas as 13 fotos permaneceram ineditas para ambas as rodadas.

gabarito.csv contem contagens exatas conforme a revisao visual. labels/ usa YOLO (classe 0), anotacoes_coco.json usa categoria 1. marcacoes/ e a galeria mostram cada individuo numerado. dataset.yaml aponta para listas separadas por autor. O dicionario define a classe e a politica de anotacao.
