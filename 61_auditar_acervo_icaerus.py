"""Confere inventario ja local antes de propor novo download ou treino."""
from pathlib import Path, PurePosixPath
import csv
import json
import zipfile
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/experimentos/auditoria_fontes_externas'


def main():
    with (ROOT / 'data/processed/divisao.csv').open(encoding='utf-8-sig', newline='') as f:
        divisao = [r for r in csv.DictReader(f) if r['dominio'] == 'aereo_pasto']
    indices = {(r['fazenda'].lower(), r['voo'].lower(), r['stem'].lower()): r for r in divisao}
    grupos = defaultdict(set)
    for r in divisao:
        grupos[(r['fazenda'].lower(), r['voo'].lower())].add(r['conjunto'])
    assert all(len(v) == 1 for v in grupos.values()), 'Voo em mais de um conjunto'
    linhas = []
    arquivo = ROOT / 'data/raw/Cattle_drone_images_042024.zip'
    with zipfile.ZipFile(arquivo) as z:
        imagens = {}
        labels = {}
        for nome in z.namelist():
            p = PurePosixPath(nome)
            partes = p.parts
            if len(partes) < 5:
                continue
            if partes[2] == 'JPGImages' and p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                imagens[(partes[1].lower(), partes[3].lower(), p.stem.lower())] = nome
            if partes[2].startswith('YOLO') and p.stem.lower().startswith('dji_') and p.suffix.lower() == '.txt':
                labels[(partes[1].lower(), partes[3].lower(), p.stem.lower())] = nome
        assert imagens and len(labels) >= len(imagens), 'Inventario incompleto de rotulos'
        por_nome = defaultdict(list)
        for chave_rot, nome_rot in labels.items():
            por_nome[(chave_rot[0], chave_rot[2])].append(nome_rot)
        for chave, nome in imagens.items():
            rot = labels.get(chave)
            if rot is None:
                candidatos = por_nome[(chave[0], chave[2])]
                assert len(candidatos) <= 1, 'Rotulos ambiguos entre voos'
                rot = candidatos[0] if candidatos else None
            n = len([l for l in z.read(rot).decode('utf-8-sig').splitlines() if l.strip()]) if rot else None
            r = indices.get(chave)
            linhas.append(dict(fazenda=chave[0], voo=chave[1], stem=chave[2],
                n_bovinos=n, arquivo_zip=nome, rotulo_zip=rot,
                conjunto=r['conjunto'] if r else next(iter(grupos.get(chave[:2], [])), 'nao_reservado'),
                usar_anterior=r['usar'] if r else 'nao_catalogado'))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'inventario_local.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader(); w.writerows(linhas)
    resumo = dict(imagens=len(linhas), caixas=sum(r['n_bovinos'] or 0 for r in linhas),
        imagens_sem_rotulo_correspondente=sum(r['n_bovinos'] is None for r in linhas),
        caixas_publicadas=4941,
        max_bovinos=max(r['n_bovinos'] or 0 for r in linhas),
        imagens_20_ou_mais=sum((r['n_bovinos'] or 0) >= 20 for r in linhas),
        uso_anterior=dict(Counter(r['usar_anterior'] for r in linhas)),
        positivos_nao_usados=sum((r['n_bovinos'] or 0) > 0 and r['usar_anterior'].lower() not in ('true','1') for r in linhas),
        zip_bytes=arquivo.stat().st_size,
        novo_download=False, brasileira=False,
        nota='Fonte ja usada pelo projeto. Inventario nao comprova completude dos rotulos nem substitui verificacao de checksum do zip.')
    (OUT / 'resultado.json').write_text(json.dumps(resumo, indent=2), encoding='utf-8')
    print(json.dumps(resumo, indent=2), flush=True)


if __name__ == '__main__':
    main()
