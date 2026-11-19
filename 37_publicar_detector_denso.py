"""Relatorio e perfil opcionais somente apos medir o teste reservado."""
from pathlib import Path
import json,hashlib,shutil
import pandas as pd

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'data/experimentos/detector_denso_v3'


def main():
    r=json.loads((EXP/'avaliacao/resultados.json').read_text(encoding='utf-8'))
    a,b=r['ajustado'],r['original']
    # Salvaguarda de transferencia: referencia brasileira ja conhecida antes deste treino.
    passou_brasil=a['brasil_antigo_exploratorio']['MAE_intervalo']<=0.2 and a['brasil_antigo_exploratorio']['dentro']>=8
    aprovado=bool(r['passou_cenario_denso'] and passou_brasil)
    pasta=ROOT/'data/modelos';pasta.mkdir(exist_ok=True)
    meta=dict(disponivel=aprovado,passou_benchmark_externo=bool(r['passou_cenario_denso']),passou_transferencia_brasil=passou_brasil,config=a['config'],escopo='Fotos de solo com varios bovinos',
              pesos_checkpoint=a['pesos'],sha256_checkpoint=a['sha256'],
              descricao=f"Detector ajustado para rebanhos de solo. Avaliacao em {r['protocolo']['teste_imagens']} imagens publicas, com ate {r['protocolo']['teste_maximo']} animais por foto. As fotos brasileiras antigas foram usadas no desenvolvimento; confira as caixas.")
    if aprovado:
        destino=pasta/'detector_solo_ajustado.pt'
        if not destino.exists():
            shutil.copy2(a['pesos'],destino)
            from ultralytics.utils.torch_utils import strip_optimizer
            strip_optimizer(str(destino))
        meta.update(pesos=destino.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(destino.read_bytes()).hexdigest())
    (pasta/'perfil_detector_denso.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    dados=pd.read_csv(ROOT/'data/detectors_cows/divisao_linhagens.csv')
    dados[['stem','grupo','linhagem','conjunto','n_bovinos','img','rot']].to_csv(ROOT/'data/detectors_cows/gabarito_final.csv',index=False)
    dic=dict(classes={'0':'bovino'},formato='YOLO: classe centro_x centro_y largura altura, normalizados',
             anotador='Autores do conjunto publico; auditoria estrutural, de sobreposicao e amostral pelo assistente',
             fonte='https://huggingface.co/datasets/AlexeyZhigalov/Detectors',
             licenca='Nao informada por imagem no arquivo; nao atribuir licenca CC inexistente',
             imagens=len(dados),caixas=int(dados.n_bovinos.sum()),
             transformacoes='Bordas artificiais retiradas e coordenadas transformadas; variantes e linhagens agrupadas; uma sobreposicao identificada com acervo anterior foi excluida')
    (ROOT/'data/detectors_cows/dicionario.json').write_text(json.dumps(dic,ensure_ascii=False,indent=2),encoding='utf-8')
    tabela='| Medida no teste novo | Original recalibrado | Novo detector |\n|---|---:|---:|\n'
    for campo,nome in [('MAE','Erro medio por imagem'),('exatas_pct','Contagens exatas (%)'),('WAPE_pct','Erro absoluto / bovinos (%)'),('F1_caixas','F1 das caixas'),('recall_caixas','Sensibilidade das caixas')]:
        tabela+=f"| {nome} | {b['teste'][campo]:.4f} | {a['teste'][campo]:.4f} |\n"
    ad,bd=a['teste_8_ou_mais'],b['teste_8_ou_mais']
    comparacao_path=EXP/'avaliacao/comparacao_preset_existente.json'
    comparacao=json.loads(comparacao_path.read_text(encoding='utf-8')) if comparacao_path.exists() else None
    adicional=(f"Comparacao adicional com o preset experimental preexistente (1280, NMS 0,7, confianca 0,15), sem busca de parametros: MAE {comparacao['teste']['MAE']:.4f} no teste completo e {comparacao['teste_8_ou_mais']['MAE']:.4f} no recorte denso. Essa comparacao foi feita depois da primeira leitura do teste e nao e uma nova confirmacao independente." if comparacao else '')
    treino=pd.read_csv(EXP/'treino/results.csv');protocolo=json.loads((EXP/'protocolo.json').read_text())
    texto=f'''# Resultado do novo detector para rebanhos

{'O detector passou os criterios externos e a salvaguarda brasileira e foi disponibilizado como perfil opcional no app.' if aprovado else 'O detector nao foi liberado no app: melhorou no teste externo, mas regrediu nas fotos brasileiras de desenvolvimento. Os pesos originais e o preset brasileiro anterior foram preservados.'}

## Dados e treino executados

Foram obtidas 850 imagens com 18.403 pontos dos autores e 4.367 familias nominais de imagens com caixas publicadas. Apos uma sobreposicao identificada com material anterior, restaram {len(dados)} imagens e {int(dados.n_bovinos.sum())} caixas no conjunto de deteccao. Os pontos nao foram transformados em caixas ficticias.

O treino final usou {protocolo['imagens_publicadas_train']} imagens publicadas com {protocolo['caixas_publicadas_train']} caixas, 22 fotos brasileiras revisadas e 256 imagens antigas de replay. As brasileiras foram repetidas seis vezes apenas dentro do treino. Foram registradas {len(treino)} epocas, incluindo a retomada do checkpoint apos a interrupcao. Os pesos originais foram preservados.

Foram removidas bordas artificiais, com transformacao das caixas, e agrupadas variantes, imagens com fundos geometricamente correspondentes e linhagens como simmental_new1. Duas preparacoes anteriores foram descartadas antes de avaliar o teste, ao encontrar sobreposicoes de grupos. A configuracao publicada que apontava val e test para a mesma pasta nao foi utilizada.

## Escolha e teste

Checkpoint, resolucao, NMS e confianca foram escolhidos na validacao publicada e nas dez fotos brasileiras antigas de desenvolvimento. Essas dez fotos ja haviam sido usadas em varreduras e nao sao teste independente. O novo teste de {r['protocolo']['teste_imagens']} imagens, com {r['protocolo']['teste_grupos']} grupos e ate {r['protocolo']['teste_maximo']} bovinos por foto, foi avaliado depois de salvar a escolha.

{tabela}

{adicional}

No recorte previamente definido de oito ou mais animais, com {ad['n_imagens'] if ad else 0} imagens, o MAE foi {bd['MAE'] if bd else None} no original e {ad['MAE'] if ad else None} no ajustado. O criterio externo exige menor MAE sem piora de F1 e sensibilidade das caixas nesse recorte. Passou: {r['passou_cenario_denso']}.

A salvaguarda brasileira adicional impede promover um modelo que piore a referencia exploratoria conhecida antes deste treino: 8/10 dentro dos intervalos e MAE ao intervalo 0,2, com pesos originais, 1280 px, NMS 0,7 e confianca 0,15. Esta salvaguarda foi explicitada ao verificar a regressao, sem escolher novos parametros no teste externo. O novo detector ficou em {a['brasil_antigo_exploratorio']['dentro']}/10 e MAE {a['brasil_antigo_exploratorio']['MAE_intervalo']}; portanto a liberacao final foi {aprovado}. O Brasil 08 caiu de 90 no preset anterior para 46 no ajustado, frente ao intervalo 75–100.

Configuracao original: {b['config']}. Nova: {a['config']}. Checkpoint escolhido: {Path(a['pesos']).name}.

As treze fotos brasileiras com caixas ja examinadas em rodadas anteriores tiveram MAE {b['brasil_13_reutilizado']['MAE']:.4f} no original e {a['brasil_13_reutilizado']['MAE']:.4f} no ajustado, usando as configuracoes para rebanhos. Elas nao sao um teste brasileiro inedito. Nas dez fotos antigas de desenvolvimento, ficaram dentro dos intervalos {b['brasil_antigo_exploratorio']['dentro']}/10 no original e {a['brasil_antigo_exploratorio']['dentro']}/10 no novo; isso e ajuste ao desenvolvimento, nao prova de generalizacao.

## Experimentos de densidade

A rede de pontos reduziu o MAE de 14,37 para 7,57 no teste externo de 181 imagens, mas transferiu mal para as fotos brasileiras: 1/10 dentro dos intervalos contra 8/10 do detector experimental anterior. Por isso nao foi promovida para uso brasileiro. A versao com mosaicos e as escalas adaptativas tambem nao superaram a primeira rede na validacao. Os pesos e resultados ficaram isolados.

## Limites e arquivos

O agrupamento nao certifica fazendas independentes. As anotacoes publicadas foram verificadas estruturalmente e por amostras, sem revisao humana integral de todas as caixas. O teste novo de caixas chega a {r['protocolo']['teste_maximo']} animais; nao comprova contagem exata de 75 a 100 Nelore em fotos brasileiras novas. Contagem e correspondencia das caixas sao medidas separadas, pois omissoes e duplicacoes podem se compensar.

Foi verificada a paridade em oito imagens entre as contagens da avaliacao e a funcao de inferencia extraida do app. A galeria `avaliacao/galeria_conferencia.html` sobrepoe gabarito e deteccoes em oito posicoes fixas da distribuicao de quantidades, sem escolher por acerto. Isso nao equivale a teste visual completo da interface.

## Prioridade para a proxima melhoria brasileira

O novo conjunto tem outras racas, fundos e distribuicao de tamanhos; o treino nao resolveu a transferencia para Nelore denso. A confianca baixa tambem deixa falsos positivos: o modelo ajustado ainda tem vies de +1,46 animal por foto no teste externo. A prioridade e obter e revisar caixas completas de rebanhos brasileiros densos, com animais pequenos e sobrepostos, incluindo negativos semelhantes a bovinos. Separar cenas e sessoes antes de qualquer novo treino. As fotos antigas podem ser usadas como desenvolvimento, mas precisam de caixas verificadas, nao apenas intervalos. O proximo teste brasileiro precisa ser inedito; repetir a busca nas mesmas dez fotos nao demonstra melhora real. Nenhum novo treino foi iniciado automaticamente apos esta rejeicao.

Gabarito: data/detectors_cows/gabarito_final.csv. Dicionario: data/detectors_cows/dicionario.json. Imagens e rotulos canonicos: data/detectors_cows/canonico. Pontos: data/denso_cattle/marcacoes_pontos.csv e galeria_pontos.html. Resultados completos e contagens por imagem: data/experimentos/detector_denso_v3/avaliacao. Fontes: https://huggingface.co/datasets/AlexeyZhigalov/Detectors e https://github.com/menjure/Cattle-dataset.
'''
    (EXP/'relatorio_final.md').write_text(texto,encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
