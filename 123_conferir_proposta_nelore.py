from pathlib import Path
import json,cv2,numpy as np
from ultralytics import YOLO
import consenso_modelos as cm
root=Path(__file__).resolve().parent
dados=(root/'data/experimentos/iluminacao_nelore_v1/original.png').read_bytes()
m=YOLO(str(root/'data/processed/modelo_gado.pt'))
p=m.predict(cv2.imdecode(np.frombuffer(dados,np.uint8),1),imgsz=1024,conf=.3,iou=.5,classes=[0],verbose=False)[0]
a=cm.inferir(dados,root)
r=cm.propor(p.boxes.xyxy.cpu().numpy(),a,dict(contagem=7,incerto=True))
(root/'data/experimentos/decisao_modelos_v1/proposta_nelore.json').write_text(json.dumps(dict(decisao=r,alternativas=a,nota='Usa leitura VLM anterior 7 incerto; nenhuma nova chamada API.'),indent=2),encoding='utf-8')
print(r)
