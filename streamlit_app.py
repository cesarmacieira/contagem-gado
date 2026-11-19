"""Entrada de hospedagem: modo publico e chave compartilhada protegida."""
import os,runpy
from pathlib import Path
os.environ['CONTAGEM_PUBLICADA']='1'
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
runpy.run_path(str(Path(__file__).resolve().parent/'app.py'),run_name='__main__')
