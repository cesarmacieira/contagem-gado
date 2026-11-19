"""Relatorio e comparacoes visuais do treino 80; sem alteracao do aplicativo."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import contagem as ct

ROOT = Path(__file__).resolve().parent
EXP = ROOT/'data/experimentos/especialista_solo_revisado_v2'


def main():
    d=json.loads((EXP/'resultado.json').read_text())
    t=pd.read_csv(EXP/'por_imagem.csv')
    a=t[t.modelo=='referencia'].set_index(['origem','arquivo'])
    b=t[t.modelo=='candidato'].set_index(['origem','arquivo'])
    cmp=pd.DataFrame(dict(contagem_atual=a.previsto,contagem_candidata=b.previsto,
                         erro_atual=a.erro,erro_candidato=b.erro,TP_atual=a.TP,TP_candidato=b.TP))
    cmp['piorou_contagem']=cmp.erro_candidato>cmp.erro_atual
    cmp.to_csv(EXP/'comparacao.csv')
    val=d['metricas']['val_densa']
    densos=cmp.loc[['val_densa','intervalos_densos']]
    ganho=(val['candidato']['MAE']<val['referencia']['MAE'] and
           val['candidato']['F1_caixas']>=val['referencia']['F1_caixas'] and
           not densos.piorou_contagem.any() and
           not (densos.TP_candidato<densos.TP_atual).any())
    decision=dict(candidato_supera_regressao_densa=bool(ganho),pronto_para_publicacao_final=False,
        promovido=False,teste_independente_novo=False,
        motivo='Exige novas cenas independentes de avaliacao' if ganho else
               'Candidato nao supera todos os criterios de contagem e correspondencia da regressao densa',
        rotas_independentes='Piora na rota de poucos animais nao obriga alterar essa rota; ela conserva o peso original.',
        sha256_peso_original=hashlib.sha256((ROOT/'data/processed/modelo_gado.pt').read_bytes()).hexdigest())
    (EXP/'decisao.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    for nome in ('bdn_08_c1.jpg','bdn_08_c2.jpg'):
        im=cv2.imread(str(ROOT/'data/experimentos/solo_denso_validacao_v2/validacao/images'/nome))
        h,w=im.shape[:2]
        rot=ROOT/'data/experimentos/solo_denso_validacao_v2/validacao/labels'/(Path(nome).stem+'.txt')
        ref=ct.yolo_para_pixels(rot.read_text().splitlines(),w,h)
        panels=[]
        for modelo in ('referencia','candidato'):
            cache=np.load(EXP/(modelo+'__'+Path(nome).stem+'.npz'))
            fig=ct.desenhar(im,cache['caixas'],ref=ref)
            scale=1000/fig.shape[1]
            fig=cv2.resize(fig,(1000,max(1,round(fig.shape[0]*scale))))
            fig=cv2.copyMakeBorder(fig,35,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            cv2.putText(fig,f'{modelo}: {len(cache["caixas"])} / real {len(ref)}',(12,24),
                        cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
            panels.append(fig)
        cv2.imwrite(str(EXP/('comparacao_'+nome)),np.concatenate(panels,axis=0))
    linhas=['# Retreino de solo com lote revisado completo','',
        'Treino real de 16 epocas, entrada 1280 e dez primeiras camadas congeladas. '
        'O lote novo passou de 53 para 62 caixas aprovadas. Inclui replay do treino anterior. '
        'Nao utiliza imagens reservadas de validacao/teste no treino.','',
        '| Conjunto | Modelo | MAE de contagem | TP | FP | FN |',
        '|---|---|---:|---:|---:|---:|']
    for origem,modelos in d['metricas'].items():
        for nome,m in modelos.items():
            linhas.append(f'| {origem} | {nome} | {m["MAE"]:.3f} | {m.get("TP","—")} | {m.get("FP","—")} | {m.get("FN","—")} |')
    linhas+=['','## Decisao','',decision['motivo']+'. Nenhum peso foi promovido.',
        '', 'Os parametros de inferencia foram fixados antes da comparacao: solo denso '
        '1280/conf 0,15/IoU 0,7; poucos animais 1024/conf 0,30/IoU 0,5; sem TTA. '
        'Ambos os pesos foram comparados pelo detector, sem os verificadores opcionais, '
        'para isolar o efeito do treino.',
        '', 'A validacao densa tem 32 bovinos em dois recortes da mesma cena. '
        'As 13 fotos de poucos animais e as cenas com intervalos ja foram consultadas '
        'em rodadas anteriores: sao regressao conhecida, nao teste independente novo. '
        'O resultado nao torna a aplicacao final pronta para publicar.',
        '', 'Detalhes reproduziveis: protocolo_treino.json, hashes_antes_treino.json, '
        'escolha_antes_regressao.json, resultado.json, comparacao.csv e treino/weights/best.pt.']
    (EXP/'RELATORIO.md').write_text('\n'.join(linhas)+'\n',encoding='utf-8')
    print(json.dumps(decision,indent=2))


if __name__=='__main__':
    main()
