"""Continua a avaliacao local depois do treino, sem precisar de novo comando."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import datetime,json,time
import pandas as pd,torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/aereo_ampliado_v2'


def estado(fase,**extra):
    (OUT/'estado_pipeline.json').write_text(json.dumps(dict(fase=fase,atualizado_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2),encoding='utf-8')


def executar(arquivo,nome):
    sp=spec_from_file_location(nome,ROOT/arquivo);m=module_from_spec(sp);sp.loader.exec_module(m);m.main()


def main():
    limite=time.monotonic()+90*60
    try:
        while True:
            try:
                t=pd.read_csv(OUT/'treino/results.csv');epoch=int(t.epoch.iloc[-1])
            except (OSError,pd.errors.EmptyDataError,IndexError):epoch=0
            estado('aguardando_treino',epoca=epoch,total=20)
            if epoch>=20:
                # Optimizer stripping marks the training checkpoint as finalized.
                try:
                    c=torch.load(OUT/'treino/weights/last.pt',map_location='cpu',weights_only=False)
                    finished=c.get('epoch')==-1 and c.get('optimizer') is None
                    del c
                    if finished:break
                except (OSError,EOFError,RuntimeError):pass
            if time.monotonic()>limite:raise TimeoutError('Treino nao terminou em 90 minutos; checkpoints preservados')
            time.sleep(20)
        estado('avaliando_fotos_inteiras')
        if not (OUT/'resultado_completo.json').exists():executar('103_avaliar_aereo_completo.py','avaliar_completas')
        estado('avaliando_recortes_diagnostico')
        if not (OUT/'escolha_validacao.json').exists():executar('102_avaliar_aereo_ampliado.py','avaliar_recortes')
        r=json.loads((OUT/'resultado_completo.json').read_text(encoding='utf-8'))
        estado('concluido',promovido=r['promovido'],resultado='resultado_completo.json')
    except Exception as e:
        estado('erro',erro=str(e));raise


if __name__=='__main__':main()
