"""Confere se o Python ativo tem o que o projeto precisa. Rode com o .venv de Python_312 ativado."""
import shutil
import sys
from importlib import metadata
from pathlib import Path

NECESSARIOS = ["streamlit", "pandas", "numpy", "plotly", "matplotlib", "opencv-python-headless", "scikit-learn",
               "ultralytics", "torch", "openpyxl", "jupyterlab", "nbformat"]
print(f"Python {sys.version.split()[0]}  ({sys.executable})")
venv = ".venv" in sys.prefix.replace("/", "\\")
print("Ambiente .venv ativo:", "SIM" if venv else "NÃO  <-- ative com: ..\\.venv\\Scripts\\Activate.ps1")
faltam = []
for p in NECESSARIOS:
    try:
        print(f"  {p:24s} {metadata.version(p)}")
    except metadata.PackageNotFoundError:
        print(f"  {p:24s} NÃO INSTALADO")
        faltam.append(p)

try:
    import torch
    gpu = torch.cuda.is_available()
    print("\nGPU CUDA:", f"SIM ({torch.cuda.get_device_name(0)})" if gpu else "NÃO (treino em CPU será muito lento)")
except Exception as e:  # noqa: BLE001
    print("\nGPU CUDA: não foi possível verificar:", e)

raiz = Path(__file__).resolve().parent
print("\nArquivos do projeto:")
for f in ["contagem.py", "01_obter_dados.py", "02_treinamento.ipynb", "app.py"]:
    print(f"  {f:24s}", "ok" if (raiz / f).exists() else "FALTA")
z = raiz / "data" / "raw" / "Cattle_drone_images_042024.zip"
print(f"  zip do dataset           ", "ok" if z.exists() else "ainda não (rode a etapa 1)")
print(f"  modelo treinado          ", "ok" if (raiz / "data" / "processed" / "modelo_gado.pt").exists() else "ainda não (rode a etapa 2)")
livre = shutil.disk_usage(raiz).free / 1e9
print(f"\nEspaço livre no disco do projeto: {livre:.0f} GB (precisa de ~25 GB: zip 16,6 GB + imagens extraídas + tiles)")
if faltam:
    print("\nFALTAM pacotes:", ", ".join(faltam))
    print("Instale no .venv ativo: pip install -r requirements.txt")
    print("(torch: instale a versão com CUDA conforme https://pytorch.org/get-started/locally/ antes do ultralytics)")
    sys.exit(1)
print("\nAmbiente OK.")
