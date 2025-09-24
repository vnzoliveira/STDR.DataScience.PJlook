from __future__ import annotations
from dash import register_page, dcc, html, Input, Output, callback, no_update
import dash_bootstrap_components as dbc
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

register_page(__name__, path="/desafio1", title="Perfil & Estágio")

F_EMP_PATH = Path("reports/exports/f_empresa_mes.parquet")
EST_PATH   = Path("reports/exports/estagio.parquet")

# -------------------- helpers --------------------
def _load():
    f = pd.read_parquet(F_EMP_PATH) if F_EMP_PATH.exists() else pd.DataFrame()
    e = pd.read_parquet(EST_PATH)   if EST_PATH.exists()   else pd.DataFrame()
    if not f.empty:
        f["year_month"] = pd.to_datetime(f["year_month"])
        f["ym_str"] = f["year_month"].dt.strftime("%Y-%m")
    return f, e

def _fmt_money_br(x) -> str:
    try:
        return f"R$ {x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0"

def _fmt_num(x, nd=4) -> str:
    try:
        return f"{float(x):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"

def _build_table(df: pd.DataFrame) -> dbc.Table:
    header = html.Thead(html.Tr([
        html.Th("ID"),
        html.Th("Mês"),
        html.Th("Receita", className="num"),
        html.Th("Despesa", className="num"),
        html.Th("Fluxo", className="num"),
        html.Th("g_MoM", className="num"),
        html.Th("Vol_3m", className="num"),
    ]))
    rows = []
    for _, r in df.iterrows():
        rows.append(
            html.Tr([
                html.Td(r["ID"]),
                html.Td(r["Mês"]),
                html.Td(_fmt_money_br(r["Receita"]), className="num"),
                html.Td(_fmt_money_br(r["Despesa"]), className="num"),
                html.Td(_fmt_money_br(r["Fluxo"]),   className="num"),
                html.Td(_fmt_num(r.get("g_MoM", np.nan), nd=4), className="num"),
                html.Td(_fmt_num(r.get("Vol_3m", np.nan), nd=2), className="num"),
            ])
        )
    body = html.Tbody(rows)
    return dbc.Table([header, body], striped=True, bordered=True, hover=True, size="sm", className="table-compact")

def _make_series_figure(df_period: pd.DataFrame) -> go.Figure:
    """
    Gráfico robusto:
      - eixo X categórico (YYYY-MM) com ordem explícita
      - 1 mês  -> barras
      - 2+ meses -> linhas+marcadores
    """
    cat_sel = df_period["ym_str"].tolist()
    fig = go.Figure()

    if len(cat_sel) <= 1:
        # barras lado a lado
        r = df_period.iloc[0]
        fig.add_bar(name="receita_mensal", x=cat_sel, y=[float(r["receita_mensal"])])
        fig.add_bar(name="despesa_mensal", x=cat_sel, y=[float(r["despesa_mensal"])])
        fig.add_bar(name="fluxo_liquido",  x=cat_sel, y=[float(r["fluxo_liquido"])])
        fig.update_layout(barmode="group")
    else:
        # linhas + marcadores
        fig.add_scatter(name="receita_mensal", x=df_period["ym_str"], y=df_period["receita_mensal"],
                        mode="lines+markers")
        fig.add_scatter(name="despesa_mensal", x=df_period["ym_str"], y=df_period["despesa_mensal"],
                        mode="lines+markers")
        fig.add_scatter(name="fluxo_liquido",  x=df_period["ym_str"], y=df_period["fluxo_liquido"],
                        mode="lines+markers")

    fig.update_layout(
        template="plotly_dark",
        height=380,
        legend_title_text="",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=cat_sel,
        title_text="Mês"
    )
    fig.update_yaxes(title_text="Valor")
    return fig

# -------------------- layout --------------------
def layout():
    f, e = _load()
    if f.empty or e.empty:
        return dbc.Alert("Artefatos não encontrados. Rode o pipeline: `python -m src.cli.main build-all`.", color="warning")

    ids = sorted(f["ID"].unique().tolist())
    months_all = [m.strftime("%Y-%m") for m in sorted(f["year_month"].unique())]

    return dbc.Container(fluid=True, children=[
        html.H3("Desafio 1 – Perfil & Estágio"),

        dbc.Row([
            dbc.Col(dcc.Dropdown(ids, ids[0], id="d1-id", clearable=False, className="dash-dropdown"), md=4),
            dbc.Col(dcc.Dropdown(months_all, months_all, id="d1-months",
                                 clearable=False, multi=True, className="dash-dropdown",
                                 placeholder="Selecione 1+ meses (default: todos)"), md=6),
        ], className="mb-3"),

        dbc.Row(id="d1-kpis", className="mb-2"),

        dbc.Row([ dbc.Col(dcc.Graph(id="d1-series", config={"displayModeBar": False}), md=12) ], className="mb-3"),

        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(html.Div(id="d1-table", style={"maxHeight":"360px","overflowY":"auto"}))),
                md=12
            )
        ], className="mb-3"),

        dbc.Accordion([
            dbc.AccordionItem(
                title="Sugestões automatizadas (simples)",
                children=html.Ul(id="d1-hints")
            )
        ], start_collapsed=False)
    ])

# -------------------- callbacks --------------------
@callback(
    Output("d1-kpis", "children"),
    Output("d1-series", "figure"),
    Output("d1-table", "children"),
    Output("d1-hints", "children"),
    Input("d1-id", "value"),
    Input("d1-months", "value"),
)
def _update(id_sel: str, months_sel):
    f, e = _load()
    if f.empty or e.empty:
        return no_update, no_update, no_update, no_update

    if isinstance(months_sel, str) or months_sel is None:
        months_sel = [months_sel] if months_sel else []

    df_id = f[f["ID"] == id_sel].sort_values("year_month")
    if df_id.empty:
        return dbc.Alert("ID sem dados.", color="warning"), {}, "—", []

    # se nada selecionado -> todos
    if not months_sel:
        months_sel = df_id["ym_str"].unique().tolist()

    # mantém ordem cronológica explícita
    cat_all = df_id["ym_str"].tolist()
    cat_sel = [m for m in cat_all if m in months_sel]

    df_period = df_id[df_id["ym_str"].isin(cat_sel)].copy()
    df_period = df_period.sort_values("year_month")

    # gráfico com eixo categórico robusto
    fig = _make_series_figure(df_period)

    # KPIs
    est_row = e[e["ID"] == id_sel].head(1)
    estagio = est_row["estagio"].iloc[0] if not est_row.empty else "N/D"

    if len(df_period) == 1:
        receita = float(df_period["receita_mensal"].iloc[0])
        despesa = float(df_period["despesa_mensal"].iloc[0])
        fluxo   = float(df_period["fluxo_liquido"].iloc[0])
        subtitle = df_period["year_month"].dt.strftime("%Y-%m").iloc[0]
    else:
        receita = float(df_period["receita_mensal"].sum())
        despesa = float(df_period["despesa_mensal"].sum())
        fluxo   = float(df_period["fluxo_liquido"].sum())
        subtitle = f"{df_period['year_month'].min().strftime('%Y-%m')} → {df_period['year_month'].max().strftime('%Y-%m')}"

    badge = html.Span(f"Estágio: {estagio}", className="stage-badge")
    kpis = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([badge, html.Div(subtitle, style={"opacity":0.8,"marginTop":"4px"})])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.Div(className="kpi", children=[html.H3("Receita (período)"), html.H2(_fmt_money_br(receita))])])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.Div(className="kpi", children=[html.H3("Despesa (período)"), html.H2(_fmt_money_br(despesa))])])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.Div(className="kpi", children=[html.H3("Fluxo (período)"),   html.H2(_fmt_money_br(fluxo))])])), md=3),
    ], justify="start")

    # tabela
    tbl = df_period[["ID","year_month","receita_mensal","despesa_mensal","fluxo_liquido","g_receita_mom","vol_receita_3m"]].copy()
    tbl.columns = ["ID","Mês","Receita","Despesa","Fluxo","g_MoM","Vol_3m"]
    tbl["Mês"] = pd.to_datetime(tbl["Mês"]).dt.strftime("%Y-%m")
    table = _build_table(tbl)

    # sugestões
    hints = []
    if len(df_period) == 1:
        row = df_period.iloc[0]
        if row["fluxo_liquido"] < 0:
            hints.append(html.Li("Fluxo negativo no mês — avaliar custos fixos e prazos com fornecedores."))
        g_mom = row.get("g_receita_mom", np.nan)
        if pd.notna(g_mom) and g_mom < 0:
            hints.append(html.Li("Receita caiu vs mês anterior — revisar mix/preços e concentração de clientes."))
        if "vol_receita_3m" in df_id.columns:
            p70 = np.nanpercentile(df_id["vol_receita_3m"], 70)
            vol = row.get("vol_receita_3m", np.nan)
            if pd.notna(vol) and vol > p70:
                hints.append(html.Li("Volatilidade acima do P70 — atenção à previsibilidade do caixa."))
    else:
        if (df_period["fluxo_liquido"] < 0).sum() > 0:
            hints.append(html.Li("Existem meses com fluxo negativo no período — revisar estrutura de custos."))
        if df_period["receita_mensal"].diff().mean() < 0:
            hints.append(html.Li("Tendência de queda média de receita no período."))
        if "vol_receita_3m" in df_period.columns and df_period["vol_receita_3m"].mean() > np.nanpercentile(df_id["vol_receita_3m"], 70):
            hints.append(html.Li("Volatilidade média do período acima do P70 histórico."))
    if not hints:
        hints = [html.Li("Sem alertas para a seleção atual.")]

    return kpis, fig, table, hints
