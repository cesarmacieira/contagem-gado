"""Gera prontidao do novo lote sem alterar os arquivos da IA coletora."""
from pathlib import Path
import argparse
import json
from integracao_denso import conferir

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=ROOT / 'data/brasil_denso_novo')
    ap.add_argument('--saida', type=Path, default=ROOT / 'data/apoio_coleta_denso/prontidao_integracao.json')
    a = ap.parse_args()
    resumo = conferir(a.base)
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    a.saida.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in resumo.items() if k != 'registros'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
