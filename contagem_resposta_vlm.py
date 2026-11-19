"""Extrai uma contagem explicita, sem completar respostas ambiguas."""
import re,unicodedata

PALAVRAS={'zero':0,'um':1,'uma':1,'dois':2,'duas':2,'tres':3,'quatro':4,'cinco':5,'seis':6,'sete':7,'oito':8,'nove':9,'dez':10,
          'onze':11,'doze':12,'treze':13,'catorze':14,'quatorze':14,'quinze':15,'dezesseis':16,'dezessete':17,'dezoito':18,'dezenove':19,'vinte':20,
          'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}


def extrair(texto):
    texto=''.join(c for c in unicodedata.normalize('NFKD',texto.lower()) if not unicodedata.combining(c))
    tokens=re.findall(r'\d+|[a-z]+',texto)
    valores=[int(t) if t.isdigit() else PALAVRAS[t] for t in tokens if t.isdigit() or t in PALAVRAS]
    if valores and len(set(valores))==1:return valores[0]
    return None
