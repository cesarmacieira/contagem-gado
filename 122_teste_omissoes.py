from pathlib import Path
import json,os,io
import cv2,numpy as np
from PIL import Image,ImageOps
from ultralytics import YOLO
import hibrido_vlm as hv
from revisao_vlm import carregar_chave_ambiente
root=Path(__file__).resolve().parent
out=root/'data/experimentos/decisao_modelos_v1/revisao_contexto_v2.json'
if out.exists():raise RuntimeError('Nao repetir teste pago')
carregar_chave_ambiente(root)
dados=(root/'data/experimentos/iluminacao_nelore_v1/original.png').read_bytes()
m=YOLO(str(root/'data/processed/modelo_gado.pt'))
p=m.predict(cv2.imdecode(np.frombuffer(dados,np.uint8),1),imgsz=1024,conf=.3,iou=.5,classes=[0],verbose=False)[0]
def preparar(dados,caixas,indices):
 im=ImageOps.exif_transpose(Image.open(io.BytesIO(dados))).convert('RGB');w,h=im.size
 regs=[im,im.crop((0,0,int(w*.65),h)),im.crop((int(w*.35),0,w,h)),im.crop((0,int(h*.3),w,h))]
 saida=[]
 for r in regs:
  r.thumbnail((1024,1024));f=io.BytesIO();r.save(f,'JPEG',quality=90);saida.append(f.getvalue())
 return saida
hv.preparar=preparar;hv.VERSAO=2
r=hv.consultar(dados,p.boxes.xyxy.cpu().numpy(),p.boxes.conf.cpu().numpy(),os.environ['OPENAI_API_KEY'],root/'data/cache_vlm/hibrido')
out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=False))
