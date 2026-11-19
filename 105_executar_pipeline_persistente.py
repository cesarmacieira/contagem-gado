"""Executa treino e avaliacoes no mesmo processo local, com estado e trava."""
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import datetime,json,msvcrt,os,threading,traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/experimentos/aereo_ampliado_v2'
# A detached process may start without the interactive shell's environment.
os.environ.setdefault('USERPROFILE',str(ROOT.parents[3]))
os.environ.setdefault('USERNAME',ROOT.parents[3].name)
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'data/runtime/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/runtime/matplotlib'))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'data/runtime/torchinductor'))


def executar(file,name):
    spec=spec_from_file_location(name,ROOT/file);m=module_from_spec(spec);spec.loader.exec_module(m);m.main()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'pipeline.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('Pipeline ja em execucao; nao iniciar outro treino')
        state=dict(pid=os.getpid(),fase='iniciando',total=20);guard=threading.Lock();stop=threading.Event()
        def salvar(**updates):
            with guard:
                state.update(updates)
                state['atualizado_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
                try:
                    lines=(OUT/'treino/results.csv').read_text().strip().splitlines()
                    state['epoca']=int(lines[-1].split(',')[0]) if len(lines)>1 else 0
                except (OSError,ValueError):pass
                (OUT/'estado_pipeline.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        def heartbeat():
            while not stop.wait(10):salvar()
        salvar();thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            salvar(fase='treinando')
            executar('101_treinar_aereo_ampliado.py','treino_ampliado')
            salvar(fase='avaliando_fotos_inteiras')
            if not (OUT/'resultado_completo.json').exists():executar('103_avaliar_aereo_completo.py','fotos_completas')
            salvar(fase='avaliando_recortes')
            if not (OUT/'escolha_validacao.json').exists():executar('102_avaliar_aereo_ampliado.py','recortes')
            result=json.loads((OUT/'resultado_completo.json').read_text(encoding='utf-8'))
            salvar(fase='concluido',promovido=result['promovido'])
        except BaseException as e:
            salvar(fase='erro',erro=str(e));traceback.print_exc();raise
        finally:
            stop.set();thread.join(timeout=2)


if __name__=='__main__':main()
