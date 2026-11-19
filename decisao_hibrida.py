"""Decisao de um detector com uma revisao VLM; sem confundir contagem e caixas."""
def decidir(detector, revisao):
    n=revisao.get('contagem')
    if type(n) is not int or n<0 or revisao.get('incerto',True):
        return dict(contagem=None,estado='pendente',nota='Revisao incerta: nenhuma contagem final confirmada. Marcacoes continuam sendo do detector.')
    return dict(contagem=n,estado='concordancia' if n==detector else 'revisado_vlm',nota='Contagem revisada pelo VLM; pode conter erros. As caixas exibidas pertencem ao detector e podem nao representar esta contagem.')
