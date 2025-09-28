from __future__ import annotations
import os
import time
from dash import Dash, dcc, html, page_container
import dash_bootstrap_components as dbc

# Tema dark consistente
THEME = dbc.themes.SLATE

# AGGRESSIVE CACHE BUSTING - Force asset refresh
asset_timestamp = str(int(time.time()))

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[THEME],
    title="FIAP Datalab – Santander",
    suppress_callback_exceptions=True,
    # FORCE ASSET REFRESH
    assets_ignore='',
    assets_external_path='',
    update_title=None,
    assets_folder='assets',
    assets_url_path=f'/assets/{asset_timestamp}/'
)

# Disable all caching
app.server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# AGGRESSIVE HTTP HEADERS TO PREVENT CACHING
@app.server.after_request
def after_request(response):
    # Universal no-cache headers
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Expires'] = '0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Last-Modified'] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())
    response.headers['ETag'] = f'"{asset_timestamp}"'
    response.headers['Vary'] = '*'
    
    # SPECIFIC JAVASCRIPT CACHE BUSTING
    content_type = response.headers.get('Content-Type', '').lower()
    request_path = str(response.headers.get('Content-Location', ''))
    
    if ('javascript' in content_type or 
        '.js' in request_path or 
        'plotly' in request_path or
        'dash' in request_path or
        'async-' in request_path):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0, s-maxage=0'
        response.headers['Expires'] = 'Thu, 01 Jan 1970 00:00:00 GMT'
        response.headers['Pragma'] = 'no-cache'
    
    return response

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