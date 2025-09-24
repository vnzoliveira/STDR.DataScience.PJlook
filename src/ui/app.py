from __future__ import annotations
import os
from dash import Dash, dcc, html, page_container
import dash_bootstrap_components as dbc

# Tema dark consistente
THEME = dbc.themes.SLATE

app = Dash(
    __name__,
    use_pages=True,                 # habilita multipages
    external_stylesheets=[THEME],
    title="FIAP Datalab – Santander",
    suppress_callback_exceptions=True,
)

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("FIAP Datalab – Santander", className="ms-2"),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Desafio 1: Perfil & Estágio", href="/desafio1")),
                    dbc.NavItem(dbc.NavLink("Desafio 2: Relações & Risco", href="/desafio2")),
                ], className="ms-auto", navbar=True),
            ]),
            color="dark", dark=True, className="mb-3"
        ),
        page_container  # conteúdo da página atual
    ],
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8060"))
    app.run(debug=False, host="0.0.0.0", port=port)