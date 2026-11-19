"""Deduplica rótulos existentes com backup e relatório; não reescreve métricas históricas."""
import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import contagem as ct

PROJ = Path(__file__).resolve().parent


def revisar(aplicar=False):
    backup = PROJ / "data/backups" / ("rotulos_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    linhas = []
    arquivos = list((PROJ / "data/extras").glob("*/labels/**/*.txt")) + list((PROJ / "data/yolo/labels").glob("**/*.txt"))
    for p in arquivos:
        antes = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        depois = ct.linhas_yolo_unicas(antes)
        if len(depois) == len(antes):
            continue
        linhas.append(dict(arquivo=str(p.relative_to(PROJ)), antes=len(antes), depois=len(depois), removidas=len(antes)-len(depois)))
        if aplicar:
            dst = backup / p.relative_to(PROJ)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            p.write_text("\n".join(depois) + ("\n" if depois else ""), encoding="utf-8")
    if aplicar and linhas:
        with (backup / "alteracoes.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
        # O cache do YOLO contém rótulos antigos: mover para backup, nunca deixá-lo ativo.
        for p in (PROJ / "data/yolo").rglob("*.cache"):
            dst = backup / p.relative_to(PROJ); dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(p), str(dst))
        print("Backup:", backup)
    print(f"{'Aplicado' if aplicar else 'Simulação'}: {len(linhas)} arquivos; {sum(r['removidas'] for r in linhas)} linhas duplicadas.")
    return linhas


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aplicar", action="store_true")
    revisar(ap.parse_args().aplicar)
