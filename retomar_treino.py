from pathlib import Path
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    checkpoint = Path(__file__).resolve().parent / 'data/experimentos/detector_denso_v3/retomada/checkpoint.pt'
    YOLO(str(checkpoint)).train(resume=True)
