"""Configuracao de producao sem expor a chave compartilhada aos visitantes."""
import hmac,os,threading
from contextlib import contextmanager

_video=threading.BoundedSemaphore(1)


def publicada():return os.environ.get('CONTAGEM_PUBLICADA','0')=='1'


def carregar_segredos(secrets):
    for nome in ['OPENAI_API_KEY','CONTAGEM_VLM_ACCESS_CODE']:
        try:valor=secrets.get(nome)
        except Exception:continue
        if isinstance(valor,str) and valor.strip():os.environ.setdefault(nome,valor.strip())


def chave_autorizada(chave_propria='',codigo='',env=None):
    env=os.environ if env is None else env
    if chave_propria.strip():return chave_propria.strip()
    servidor=env.get('OPENAI_API_KEY','')
    if env.get('CONTAGEM_PUBLICADA','0')!='1':return servidor
    esperado=env.get('CONTAGEM_VLM_ACCESS_CODE','')
    if esperado and codigo and hmac.compare_digest(esperado.encode(),codigo.encode()):return servidor
    return ''


@contextmanager
def vaga_video():
    if not _video.acquire(blocking=False):raise RuntimeError('Ja existe um video em analise neste servidor. Tente novamente apos a conclusao.')
    try:yield
    finally:_video.release()
