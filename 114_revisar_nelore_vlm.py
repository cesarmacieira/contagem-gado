"""Uma consulta real limitada; sem resposta esperada no prompt e sem chave nos logs."""
from pathlib import Path
import json,os
import revisao_vlm as rv


def main():
    root=Path(__file__).resolve().parent
    rv.carregar_chave_ambiente(root)
    out=root/'data/experimentos/iluminacao_nelore_v1'
    resultado=rv.consultar((out/'original.png').read_bytes(),os.environ.get('OPENAI_API_KEY',''),root/'data/cache_vlm')
    (out/'revisao_vlm.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2),encoding='utf-8')
    from PIL import Image,ImageDraw
    import io
    im=Image.open(io.BytesIO(rv.preparar_imagem((out/'original.png').read_bytes()))).convert('RGB')
    d=ImageDraw.Draw(im)
    for i,a in enumerate(resultado['animais'],1):
        x,y=a['x']*(im.width-1),a['y']*(im.height-1)
        cor='orange' if a['incerto'] else 'lime'
        d.ellipse((x-7,y-7,x+7,y+7),outline=cor,width=3)
        d.text((max(0,x-5),max(0,y-22)),str(i),fill=cor,stroke_width=1,stroke_fill='black')
    im.save(out/'revisao_vlm_pontos.jpg')
    print(json.dumps(resultado,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
