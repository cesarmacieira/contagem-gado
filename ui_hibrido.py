"""Interface de exemplos reais, revisao seletiva e video; API so em clique explicito."""
from pathlib import Path
import hashlib,io,json,os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import contagem as ct
import hibrido_vlm as hv
from config_publicacao import chave_autorizada,publicada,vaga_video


def exemplos(root):
    base=Path(root)/'data/galeria_publica'
    if not base.exists():base=Path(root)/'data/experimentos/nano_internet_v1'
    arquivo=base/'referencias_antes_api.json'
    if not arquivo.exists():st.info('Galeria de exemplos ainda nao disponivel neste computador.');return None
    fontes=json.loads(arquivo.read_text(encoding='utf-8'))['fontes']
    ids={s['id']:s for s in fontes}
    escolhido=st.selectbox('Foto real para analisar',list(ids),format_func=lambda k:f"Pexels {k} — "+('referencia ambigua' if ids[k]['referencia']['contagem'] is None else f"{ids[k]['referencia']['contagem']} bovinos na referencia visual"))
    fonte=ids[escolhido];path=(base/fonte['arquivo']).resolve()
    if not path.is_relative_to(base.resolve()) or not path.exists():st.error('Imagem local indisponivel');return None
    st.image(str(path),caption=fonte['referencia']['nota'],width=650)
    st.markdown(f"[Fonte da foto]({fonte['url']}) · [Licenca Pexels](https://www.pexels.com/license/)")
    tabela=[]
    for nome,arq in [('GPT-5 nano','resultados.json'),('GPT-6 Luna','resultados_luna.json')]:
        if (base/arq).exists():
            for r in json.loads((base/arq).read_text(encoding='utf-8')):
                if r['id']==escolhido:tabela.append(dict(Modelo=nome,Contagem=r.get('contagem'),Referencia=r.get('referencia'),Estado=r['status']))
    if tabela:st.dataframe(pd.DataFrame(tabela),hide_index=True)
    st.caption('Consultas anteriores de VLM sozinho, sem nova cobranca. O detector do perfil selecionado roda abaixo. Referencia visual de um assistente, sem segundo anotador; nao e teste independente.')
    return path.read_bytes()


def painel_revisao(dados,caixas,scores,root,conf=.3,candidatos_scores=None,prefixo='imagem'):
    img=cv2.imdecode(np.frombuffer(dados,np.uint8),cv2.IMREAD_COLOR)
    if img is None:return
    h,w=img.shape[:2];sinais=hv.gatilhos(caixas,scores,w,h,conf,candidatos_scores)
    chave=hashlib.sha256(dados+np.asarray(caixas,float).tobytes()+np.asarray(scores,float).tobytes()+str(conf).encode()).hexdigest()
    state=f'hibrido_{prefixo}_{chave}'
    st.subheader('Detector + revisao visual')
    if sinais['recomendada']:st.warning('Conferencia recomendada: '+', '.join(sinais['motivos'])+'.')
    else:st.info('Sem gatilho heuristico neste quadro. Confianca alta nao exclui animais omitidos; a conferencia manual continua disponivel.')
    st.caption('Um detector local + GPT-6 Luna. O VLM confere a imagem e ate tres ampliacoes quando ha suspeita de erro. A revisao nao gera novas caixas; resultados incertos exigem conferencia.')
    with st.expander('Conferir com VLM',expanded=sinais['recomendada']):
        st.caption('GPT-6 Luna · reserva estimada de US$ 0,005 por revisao · limite local de US$ 0,05/dia. Sem chamada em recargas, sem tentativas automaticas. O controle local nao configura teto na conta OpenAI.')
        token='';codigo=''
        solo=st.checkbox('Conferir tambem com detectores de solo (poucos animais)',key=f'local_{state}')
        with st.form(f'form_{state}',clear_on_submit=True):
            if publicada():
                token=st.text_input('Sua chave OpenAI (opcional)',type='password')
                if os.environ.get('CONTAGEM_VLM_ACCESS_CODE'):codigo=st.text_input('Codigo de acesso ao VLM do aplicativo',type='password')
            elif not os.environ.get('OPENAI_API_KEY'):token=st.text_input('Chave OpenAI',type='password')
            consent=st.checkbox('Enviar esta imagem e os recortes a OpenAI para conferencia paga')
            enviar=st.form_submit_button('Conferir este quadro' if prefixo=='video' else 'Conferir esta imagem')
        if enviar:
            if not consent:st.info('Marque o envio da imagem para usar a conferencia.')
            else:
                try:
                    with st.spinner('Conferindo a imagem e as regioes suspeitas...'):
                        st.session_state[state]=hv.consultar(dados,caixas,scores,chave_autorizada(token,codigo),Path(root)/'data/cache_vlm/hibrido',conf,candidatos_scores)
                    if solo and len(caixas)<=14:
                        import consenso_modelos as cm
                        with st.spinner('Localizando animais com dois detectores adicionais...'):
                            alternativas=cm.inferir(dados,root)
                            st.session_state[state+'_local']=dict(alternativas=alternativas,decisao=cm.propor(caixas,alternativas,st.session_state[state]))
                except Exception as erro:st.error(f'Conferencia nao concluida: {erro}. A contagem do detector foi preservada.')
        r=st.session_state.get(state)
        if r:
            from decisao_hibrida import decidir
            decisao=decidir(len(caixas),r)
            a,b,c=st.columns(3);a.metric('Detector',len(caixas));b.metric('Leitura do VLM',r['contagem']);c.metric('Contagem revisada',decisao['contagem'] if decisao['contagem'] is not None else 'Pendente')
            st.caption(decisao['nota'])
            if r['situacao']=='divergencia':st.warning('Divergencia: revise as marcacoes. Nenhuma caixa foi acrescentada ou removida automaticamente.')
            elif r['situacao']=='incerto':st.warning('O VLM indicou incerteza. A contagem permanece pendente de revisao.')
            else:st.info('As contagens concordam. Isso nao comprova que todos os animais foram identificados corretamente.')
            st.write(r['observacao']);st.caption(f"Custo estimado da consulta salva: US$ {r['custo_estimado_usd']:.6f}. "+('Resposta reutilizada do cache.' if r.get('cache') else ''))
            local=st.session_state.get(state+'_local') if solo else None
            if local:
                d=local['decisao']
                if d.get('proposta') or d['corrigido']:
                    st.metric('Proposta com animais localizados',d['contagem'])
                    st.warning(d['motivo'])
                    st.image(cv2.cvtColor(ct.desenhar(img,np.asarray(d['caixas'],float).reshape(-1,4)),cv2.COLOR_BGR2RGB),caption='Marcacoes da proposta; confira cada animal',use_container_width=True)
            pacote=dict(localizacao=local,detector=len(caixas),caixas=np.asarray(caixas).tolist(),scores=np.asarray(scores).tolist(),gatilhos=sinais,revisao=r,decisao=decisao)
            st.download_button('Baixar detector e revisao (JSON)',json.dumps(pacote,ensure_ascii=False,indent=2),file_name='detector_vlm.json',mime='application/json',key=f'download_{state}')


def painel_video(root,inferir,config,conf):
    from video_contagem import analisar
    st.subheader('Analisar um video')
    st.caption('O detector analisa todos os quadros ou uma amostra ao longo de toda a duracao, incluindo o ultimo quadro decodificado. O VLM so roda no quadro escolhido e com envio explicito.')
    arquivo=st.file_uploader('Video (MP4, MOV, AVI, MKV ou WebM)',type=['mp4','mov','avi','mkv','webm'],key='arquivo_video')
    if arquivo is None:return
    dados=arquivo.getvalue()
    if not dados:st.error('Arquivo vazio');return
    if arquivo.name.lower().endswith(('.mp4','.webm')):st.video(dados)
    todos=st.checkbox('Analisar todos os quadros (mais demorado)',value=False)
    intervalo=0. if todos else st.select_slider('Intervalo entre quadros analisados (segundos)',options=[.25,.5,1.,2.,5.,10.],value=1.)
    assinatura=hashlib.sha256(dados+json.dumps(dict(config=config,intervalo=intervalo,conf=conf),sort_keys=True,default=str).encode()).hexdigest()
    pasta=Path(root)/'data/cache_video'/assinatura
    pasta.mkdir(parents=True,exist_ok=True)
    entrada=pasta/('original'+Path(arquivo.name).suffix.lower())
    resultado_path=pasta/'resultado.json'
    if st.button('Analisar video completo',type='primary'):
        if not entrada.exists():entrada.write_bytes(dados)
        (pasta/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        try:
            if not resultado_path.exists():
                barra=st.progress(0.,text='Decodificando o video...')
                def progresso(n,i,total):barra.progress(min(.99,(i+1)/total) if total>0 else 0.,text=f'{n} quadros analisados; posicao {i+1}/{total if total else "?"}')
                with vaga_video(),st.spinner('Analisando o video ate o fim...'):analisar(entrada,inferir,pasta,intervalo,progresso,max_amostras=1000 if publicada() else 5000)
                barra.progress(1.,text='Video decodificado ate o fim.')
        except Exception as erro:st.error(f'Analise nao concluida: {erro}');return
    if not resultado_path.exists():return
    r=json.loads(resultado_path.read_text(encoding='utf-8'));meta=r['meta'];quadros=r['quadros']
    st.success(f"Analise concluida: {meta['quadros_analisados']} de {meta['quadros_decodificados']} quadros analisados; duracao {meta['duracao_segundos']:.2f} s.")
    if meta['modo']=='amostragem_temporal':st.info(f"Amostragem a cada {meta['intervalo_efetivo']:.2f} s. Eventos entre amostras podem passar despercebidos; use todos os quadros se necessario.")
    k=st.columns(3);k[0].metric('Minimo simultaneo',meta['min_simultaneo']);k[1].metric('Maximo simultaneo',meta['max_simultaneo']);k[2].metric('Media nos quadros',f"{meta['media_simultanea']:.2f}")
    st.warning('Esses numeros sao contagens simultaneas. Nao somamos quadros: o mesmo animal pode aparecer varias vezes. Identificacao de individuos e contagem unica de passagem ainda nao estao implementadas.')
    df=pd.DataFrame([{k:q[k] for k in ['quadro','segundos','contagem','confianca_media']} for q in quadros])
    st.line_chart(df.set_index('segundos')[['contagem']]);st.download_button('Baixar contagens do video (CSV)',df.to_csv(index=False).encode('utf-8'),file_name='contagem_video.csv',mime='text/csv')
    st.download_button('Baixar analise e caixas (JSON)',resultado_path.read_bytes(),file_name='analise_video.json',mime='application/json')
    id_q=st.select_slider('Quadro para conferir',options=list(range(len(quadros))),format_func=lambda i:f"{quadros[i]['segundos']:.2f} s — {quadros[i]['contagem']} bovinos")
    q=quadros[id_q];foto_path=pasta/q['arquivo'];foto=foto_path.read_bytes();img=cv2.imdecode(np.frombuffer(foto,np.uint8),cv2.IMREAD_COLOR)
    b=np.asarray(q['caixas'],float).reshape(-1,4);s=np.asarray(q['scores'],float)
    st.image(cv2.cvtColor(ct.desenhar(img,b),cv2.COLOR_BGR2RGB),caption=f"Quadro {q['quadro']}; {q['segundos']:.2f} s",use_container_width=True)
    painel_revisao(foto,b,s,root,conf,prefixo='video')
