from pathlib import Path
import json,cv2,numpy as np
from ultralytics import YOLO
import consenso_modelos as cm
import hibrido_vlm as hv
from revisao_vlm import carregar_chave_ambiente
root=Path(__file__).resolve().parent
base=root/'data/experimentos/decisao_modelos_v1/novas_fotos'
ref=json.loads((base/'referencias_antes_modelos.json').read_text())['contagens']
out=base/'resultados.json'
if out.exists():raise RuntimeError('Nao repetir')
carregar_chave_ambiente(root)
import os
m=YOLO(str(root/'data/processed/modelo_gado.pt'))
rows=[]
for id,real in ref.items():
 dados=(base/(id+'.jpg')).read_bytes()
 img=cv2.imdecode(np.frombuffer(dados,np.uint8),1)
 p=m.predict(img,imgsz=1024,conf=.3,iou=.5,classes=[0],verbose=False)[0]
 b=p.boxes.xyxy.cpu().numpy();s=p.boxes.conf.cpu().numpy()
 a=cm.inferir(dados,root)
 v=hv.consultar(dados,b,s,os.environ['OPENAI_API_KEY'],root/'data/cache_vlm/hibrido')
 d=cm.decidir(b,a,v)
 rows.append(dict(id=id,real=real,original=len(b),alternativas=a,vlm=v,consenso=d))
 out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 print(id,real,len(b),d['contagem'],flush=True)
