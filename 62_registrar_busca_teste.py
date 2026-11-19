"""Registra arquivos efetivamente baixados, proveniencia e reserva sem inferencia."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/busca_teste_denso_independente'
FONTES = [
    ('bernard_28059238281.jpg','Bernard DUPONT','CC BY-SA 2.0','https://creativecommons.org/licenses/by-sa/2.0/','https://commons.wikimedia.org/wiki/File:Herd_of_Brahma_Cattle_led_by_Pantaneiro_Cowboy_..._-_Flickr_-_berniedup.jpg','https://live.staticflickr.com/7394/28059238281_90c358d41d_o.jpg','Transpantaneira, Pocone, Mato Grosso','autor_bernard_dupont_from_france','test','candidata_densa','Autor ja reservado para test em bdn_07; foto diferente; cabeças e corpos sobrepostos exigem revisao. Nao e autor independente novo.'),
    ('christopher_tocando_boiada.jpg','Christopher Borges','CC BY-SA 3.0','https://creativecommons.org/licenses/by-sa/3.0/','https://commons.wikimedia.org/wiki/File:Tocando_a_Boiada.jpg','https://upload.wikimedia.org/wikipedia/commons/e/e4/Tocando_a_Boiada.jpg','Mato Grosso','autor_christopher_borges','test','candidata_densa','Novo autor nao localizado nos manifestos; cavaleiro e montaria nao sao bovinos; sobreposicao e animais de borda impedem aprovacao automatica.'),
    ('carlos_ebert_nelore.jpg','Carlos Ebert','CC BY 2.0','https://creativecommons.org/licenses/by/2.0/','https://commons.wikimedia.org/wiki/File:Nelore_cattle_Brasil.jpg','https://upload.wikimedia.org/wikipedia/commons/1/1c/Nelore_cattle_Brasil.jpg','Campos Belos, Goias','autor_carlos_ebert','nao_integrado','fora_criterio_denso','Vista geral com cerca de uma duzia de bovinos; nao atinge vinte. Nao usado no treino.'),
    ('ac_moraes_para.jpg','A C Moraes','CC BY 2.0','https://creativecommons.org/licenses/by/2.0/','https://commons.wikimedia.org/wiki/File:Cattle_in_Par%C3%A1.jpg','https://upload.wikimedia.org/wikipedia/commons/a/aa/Cattle_in_Par%C3%A1.jpg','Para','autor_a_c_moraes_from_bras_lia_brasi','val','duplicata_fonte','Mesma fotografia de bdn_16; manter em val, nao contar como nova.'),
    ('daniel_julie_sunset.jpg','DANIEL JULIE','CC BY 2.0','https://creativecommons.org/licenses/by/2.0/','https://commons.wikimedia.org/wiki/File:Cattle_on_the_road_at_sunset_in_Brazil.jpg','https://upload.wikimedia.org/wikipedia/commons/f/fe/Cattle_on_the_road_at_sunset_in_Brazil.jpg','Pantanal, Brasil','autor_daniel_julie','test','candidata_densa','Novo autor nao localizado nos manifestos; animais escuros e sobrepostos; inspecao em resolucao original necessaria, sem contagem exata aprovada.'),
    ('adriana_cezar_pastando.jpg','Adriana Cezar','CC BY-SA 4.0','https://creativecommons.org/licenses/by-sa/4.0/','https://commons.wikimedia.org/wiki/File:Gado_pastando.jpg','https://upload.wikimedia.org/wikipedia/commons/c/c7/Gado_pastando.jpg','Macae de Cima, Nova Friburgo, Rio de Janeiro','autor_adriana_cezar','test','duplicata_fonte','Mesma fotografia de wc_70248877.jpg, teste historico; quatro bovinos, nao densa.'),
]


def main():
    linhas=[]
    for arquivo,autor,licenca,url_licenca,pagina,url,local,grupo,conjunto,status,nota in FONTES:
        p=OUT/arquivo
        assert p.is_file()
        with Image.open(p) as im:
            largura,altura=im.size
            im.verify()
        linhas.append(dict(arquivo=arquivo,autor=autor,licenca=licenca,url_licenca=url_licenca,
            pagina_origem=pagina,url_imagem=url,origem_brasileira=local,grupo=grupo,conjunto=conjunto,
            status=status,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
            largura=largura,altura=altura,nota=nota,detector_executado=False))
    with (OUT/'manifesto_busca.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(linhas[0]));w.writeheader();w.writerows(linhas)
    p=OUT/'reserva_antes_anotacao.json'
    if not p.exists():
        p.write_text(json.dumps(dict(data=datetime.now(timezone.utc).isoformat(),
            grupos_novos={'autor_christopher_borges':'test','autor_daniel_julie':'test'},
            grupos_existentes_preservados=True,anotacao_iniciada=False,detector_executado=False,
            observacao='Reserva complementar desta busca; nao altera divisao da coletora. Candidatos nao aprovados como gabarito exato.'),indent=2),encoding='utf-8')
    print(json.dumps(dict(baixadas=len(linhas),candidatas_densas=sum(r['status']=='candidata_densa' for r in linhas),
        duplicatas_fonte=sum(r['status']=='duplicata_fonte' for r in linhas),teste_exato_aprovado=0),indent=2))


if __name__=='__main__':
    main()
