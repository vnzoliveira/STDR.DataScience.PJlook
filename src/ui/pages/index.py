# src/ui/pages/index.py
from dash import register_page, dcc

register_page(__name__, path="/", title="Início")

def layout():
    # redireciona imediatamente para /desafio1
    return dcc.Location(id="redir", href="/desafio1")
