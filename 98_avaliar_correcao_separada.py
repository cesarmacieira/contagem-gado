"""Avaliacao congelada da correcao; nao ajusta parametros ou pesos."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import hashlib,json
import cv2,numpy as np,pandas as pd,torch
from ultralytics import YOLO
import contagem as ct
from avaliacao import comparar_caixas
from especialistas import resolver_especialista
from correcao_aerea import selecionar_instancias
from verificador_nucleo import carregar,pontuar

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/avaliacao_curral_nova_v1'


def module(name,file):
    sp=spec_from_file_location(name,ROOT/file);m=module_from_spec(sp);sp.loader.exec_module(m);return m


def main():
    if (OUT/'resultado.json').exists():raise ValueError('Avaliacao preservada')
    perfil=resolver_especialista(ROOT,'curral_externo_topo');cfg=perfil['config'];v=perfil['verificador_nucleo']
    gpath=OUT/'gabaritos_antes_detector.json';g=json.loads(gpath.read_text())
    origem=module('preparar_aereo','48_especializar_drone.py');df=origem.catalogo()
    fontes=origem.selecionar(df[df.conjunto=='test'],24,981)
    protocolo=dict(perfil=perfil,gabaritos_sha256=hashlib.sha256(gpath.read_bytes()).hexdigest(),
        fontes_test=fontes.to_dict('records'),seed=981,max_tiles=96,
        limite='Teste historico do projeto, separado da validacao da correcao; nao se afirma teste virgem do projeto. Fotos brasileiras mesma fazenda conhecida, sem novo dominio/fazenda. Sem ajuste apos resultados.')
    (OUT/'protocolo_antes_inferencia.json').write_text(json.dumps(protocolo,indent=2),encoding='utf-8')
    torch.set_num_threads(4);modelo=YOLO(perfil['pesos_absolutos'])
    vf=carregar(ROOT/v['extrator'],ROOT/v['classificador'],v['extrator_sha256'],v['classificador_sha256'],v['receita_sha256'])
    ptsmod=module('match_pts','67_verificar_especies_no_denso.py');rows=[];manifest=[]
    def avaliar(im,nome,ref=None,pts=None):
        b,s=ct.detectar_inteira(modelo,im,imgsz=cfg['imgsz'],conf_min=.05,iou_nms=cfg['iou'],device=0,classes=[0],augment=False)
        prob=pontuar(vf,im,b);ix,rec=selecionar_instancias(b,s,prob,cfg['conf'],v['limiar'])
        np.savez(OUT/(nome+'.npz'),caixas=b,scores=s,verificador=prob,selecionadas=ix)
        for versao,k in [('detector_original',np.flatnonzero(s>=.2)),('sombras_anterior',np.flatnonzero((s>=.2)&(prob>=v['limiar']))),('correcao_atual',ix)]:
            real=len(ref) if ref is not None else len(pts)
            r=dict(id=nome,lote='aereo_test' if ref is not None else 'brasil',versao=versao,real=real,previsto=len(k),erro=abs(len(k)-real))
            if ref is not None:r.update(comparar_caixas(ref,b[k],s[k]))
            else:
                hits=ptsmod.correspondencias_pontos(pts,b[k]);r.update(pontos_cobertos=hits,omissoes=real-hits,extras=len(k)-hits)
            rows.append(r)
        if pts is not None:cv2.imwrite(str(OUT/(nome+'_resultado.jpg')),ct.desenhar(im,b[ix]))
    for c in g['casos']:
        p=OUT/(c['id']+'_recorte.png');assert hashlib.sha256(p.read_bytes()).hexdigest()==c['recorte_sha256']
        avaliar(cv2.imread(str(p)),c['id'],pts=c['pontos_xy'])
    rng=np.random.default_rng(982)
    for r in fontes.itertuples():
        im=cv2.imread(r.img);h,w=im.shape[:2];bb=ct.yolo_para_pixels(Path(r.rot).read_text().splitlines(),w,h);op=[]
        for x,y in ct.grade_tiles(w,h,1024,.2):
            tw,th=min(1024,w-x),min(1024,h-y)
            if ct.tile_tem_fragmento_sem_rotulo(bb,x,y,tw,th):continue
            lines=ct.rotulos_do_tile(bb,x,y,tw,th);op.append((x,y,lines))
        positives=sorted([q for q in op if q[2]],key=lambda q:len(q[2]),reverse=True)[:6]
        neg=[q for q in op if not q[2]]
        if neg:neg=[neg[j] for j in rng.choice(len(neg),min(2,len(neg)),replace=False)]
        for x,y,lines in positives+neg:
            if len(manifest)>=96:break
            crop=im[y:y+1024,x:x+1024];th,tw=crop.shape[:2];ref=ct.yolo_para_pixels(lines,tw,th)
            nome=f'test_{len(manifest):03d}';manifest.append(dict(id=nome,origem=r.img,grupo=r.grupo,x=x,y=y,n=len(ref)))
            avaliar(crop,nome,ref=ref)
        print('Teste',len(manifest),'recortes',flush=True)
        if len(manifest)>=96:break
    tab=pd.DataFrame(rows);tab.to_csv(OUT/'por_imagem.csv',index=False);pd.DataFrame(manifest).to_csv(OUT/'manifesto_test.csv',index=False)
    resumo=[]
    for (lote,versao),t in tab.groupby(['lote','versao']):
        r=dict(lote=lote,versao=versao,n_imagens=len(t),bovinos=int(t.real.sum()),MAE=float(t.erro.mean()),contagens_exatas=int((t.erro==0).sum()))
        for k in (['TP','FP','FN'] if lote=='aereo_test' else ['pontos_cobertos','omissoes','extras']):r[k]=int(t[k].sum())
        resumo.append(r)
    result=dict(resultados=resumo,limites=protocolo['limite'],parametros_alterados=False,pesos_alterados=False)
    (OUT/'resultado.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
