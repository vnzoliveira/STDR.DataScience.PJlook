# src/ui/pages/desafio1.py
"""
Dashboard executivo do Desafio 1 (dark theme, corrigido, robusto e com explainability).
- Evita merges que duplicavam meses: trabalha f_empresa_mes "puro" e agrega por mês quando necessário
- KPIs e gráficos agregam corretamente por mês
- Visualizações resilientes a NaN/inf/colunas ausentes
- Benchmarking por percentil vs peers (setor+porte)
- Painel "Sobre o Modelo" (explainability leve por percentis e drivers)
- Tema dark padronizado (gráficos, rótulos, legendas, dropdowns)
"""
from __future__ import annotations

from dash import register_page, dcc, html, Input, Output, State, callback, no_update, callback_context
import dash_bootstrap_components as dbc
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import unicodedata

from src.ui.components.metrics_formatter import (
    format_currency_br, format_percentage, get_metric_color
)
from src.features.sector_analysis import SectorBenchmarkEngine
from src.features.business_insights import BusinessInsightEngine

register_page(__name__, path="/desafio1", title="Dashboard Executivo - Análise Empresarial")

# Caminhos
F_EMP_PATH = Path("reports/exports/f_empresa_mes.parquet")
EST_PATH   = Path("reports/exports/estagio.parquet")
B1_PATH    = Path("data/processed/base1/base1.parquet")
COMP_PATH  = Path("reports/exports/companies.parquet")  # novo artefato (resumo por empresa)

# Engines
sector_engine = SectorBenchmarkEngine()
insight_engine = BusinessInsightEngine()

# ===================== CARREGAMENTO & HELPERS =====================

def _safe_num(s, fill=0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill)

def _agg_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Garante 1 linha por mês/ID. (evita duplicações em merges anteriores)"""
    keep_cols = [c for c in ["ID", "year_month", "ym_str"] if c in df.columns]
    num_cols = [c for c in ["receita_mensal", "despesa_mensal", "fluxo_liquido",
                            "g_receita_mom", "vol_receita_3m"] if c in df.columns]
    if not keep_cols or not num_cols:
        return df.drop_duplicates(subset=keep_cols) if keep_cols else df

    out = (df[keep_cols + num_cols]
           .assign(**{c: _safe_num(df[c]) for c in num_cols})
           .groupby(keep_cols, as_index=False)
           .agg({c: "mean" if c in ["g_receita_mom","vol_receita_3m"] else "sum" for c in num_cols}))
    return out.sort_values("year_month")

def _company_summary(f: pd.DataFrame, e: pd.DataFrame, b1: pd.DataFrame) -> pd.DataFrame:
    """Resumo por empresa para benchmarking (fallback caso companies.parquet não exista)."""
    if f.empty:
        return pd.DataFrame()

    g = (f.groupby("ID")
           .agg(
               receita_media=("receita_mensal", "mean"),
               despesa_media=("despesa_mensal", "mean"),
               fluxo_medio=("fluxo_liquido", "mean"),
               receita_total=("receita_mensal", "sum"),
               crescimento_medio=("g_receita_mom", "mean"),
               volatilidade_receita=("vol_receita_3m", "mean"),
               consistencia_fluxo=("fluxo_liquido", lambda s: float((s > 0).mean()))
           )
           .reset_index())
    g["margem_media"] = np.where(g["receita_media"].abs() > 1e-9,
                                 g["fluxo_medio"] / g["receita_media"],
                                 0.0)

    # setor/porte
    if not b1.empty:
        cols = [c for c in ["ID", "DS_CNAE", "VL_FATU"] if c in b1.columns]
        b1u = b1[cols].drop_duplicates(subset=["ID"])
        g = g.merge(b1u, on="ID", how="left")

        def _porte(fat):
            fat = float(fat) if pd.notna(fat) else 0.0
            if fat < 360_000: return "MICRO"
            if fat < 4_800_000: return "PEQUENA"
            if fat < 300_000_000: return "MEDIA"
            return "GRANDE"

        g["porte"] = g["VL_FATU"].apply(_porte)
        g["setor"] = (g.get("DS_CNAE").astype(str).upper()
                        .str.extract(r"(^[^-;\|/]+)")[0].fillna("OUTROS"))
    if "estagio" in e.columns:
        g = g.merge(e[["ID", "estagio"]].drop_duplicates("ID"), on="ID", how="left")
    return g

def load_all_data():
    """Carrega datasets. NÃO faz merge em f_empresa_mes para não duplicar meses."""
    f = pd.read_parquet(F_EMP_PATH) if F_EMP_PATH.exists() else pd.DataFrame()
    e = pd.read_parquet(EST_PATH)   if EST_PATH.exists()   else pd.DataFrame()
    b1 = pd.read_parquet(B1_PATH)   if B1_PATH.exists()   else pd.DataFrame()

    if not f.empty:
        f["year_month"] = pd.to_datetime(f["year_month"], errors="coerce")
        f = f.dropna(subset=["year_month"]).copy()
        f["ym_str"] = f["year_month"].dt.strftime("%Y-%m")
        for col in ["receita_mensal", "despesa_mensal", "fluxo_liquido",
                    "g_receita_mom", "vol_receita_3m"]:
            if col in f.columns:
                f[col] = _safe_num(f[col])

    if COMP_PATH.exists():
        companies = pd.read_parquet(COMP_PATH)
    else:
        try:
            from src.features.company_summary import build_companies_summary
            companies = build_companies_summary(f, e, b1)
        except Exception:
            companies = _company_summary(f, e, b1)

    return f, e, companies

def _stage_key(s: str) -> str:
    s = str(s or "").lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s

# ===================== LAYOUT =====================

def layout():
    f, e, companies = load_all_data()
    if f.empty or e.empty:
        return dbc.Alert("Dados não encontrados. Execute: python -m src.cli.main build-all", color="warning")

    ids = sorted(f["ID"].unique().tolist())
    timestamp = int(time.time())

    return dbc.Container(fluid=True, children=[
        # Cache busting leve
        html.Script(f"""
        console.log('FIAP Dashboard - Cache busting: {timestamp}');
        if ('caches' in window) {{
            caches.keys().then(ns => ns.forEach(n => caches.delete(n)));
        }}
        """),

        # Header
        dbc.Row([
            dbc.Col([
                html.H2("Dashboard Executivo - Análise Empresarial",
                        style={"color": "#e5e7eb", "marginBottom": "6px", "fontWeight": "600"}),
                html.P(f"Análise completa com benchmarking setorial e ML avançado - Cache: {timestamp}",
                       style={"color": "#94a3b8", "fontSize": "14px"})
            ], md=8),
            dbc.Col([
                html.Div([
                    html.Small("Última atualização: ", style={"color": "#94a3b8"}),
                    html.Small(pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
                               style={"color": "#38bdf8", "fontWeight": "500"})
                ], className="text-end", style={"marginTop": "20px"})
            ], md=4)
        ], className="mb-4", style={"borderBottom": "1px solid #334155", "paddingBottom": "14px"}),

        # Filtros
        dbc.Row([
            dbc.Col([
                html.Label("Empresa", style={"color": "#94a3b8", "fontSize": "12px", "fontWeight": "500"}),
                dcc.Dropdown(id="d1-company", options=[{"label": i, "value": i} for i in ids],
                             value=ids[0], clearable=False, className="dash-dropdown-dark")
            ], md=3),
            dbc.Col([
                html.Label("Período", style={"color": "#94a3b8", "fontSize": "12px", "fontWeight": "500"}),
                dcc.Dropdown(id="d1-period", options=[], value=[], multi=True,
                             className="dash-dropdown-dark",
                             placeholder="Selecione meses...")
            ], md=5),
            dbc.Col([
                html.Label("Visualização", style={"color": "#94a3b8", "fontSize": "12px", "fontWeight": "500", "marginBottom": "8px"}),
                dbc.ButtonGroup([
                    dbc.Button("📈 Executivo", id="btn-executive", color="primary",
                              size="sm", className="view-button active-view"),
                    dbc.Button("🎯 Benchmarking", id="btn-benchmark", color="secondary",
                              size="sm", className="view-button"),
                    dbc.Button("🔬 Análise Avançada", id="btn-advanced", color="secondary",
                              size="sm", className="view-button")
                ], className="d-flex w-100"),
                dcc.Store(id="d1-view", data="executive")
            ], md=4)
        ], className="mb-3"),

        # KPIs
        html.Div(id="d1-kpis", className="mb-3"),

        # Área principal
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(html.Div(id="d1-main"))), md=8, className="card-dark"),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H6("Score Competitivo", style={"color": "#e5e7eb", "fontWeight": "600"}),
                    html.Div(id="d1-score")
                ]), className="card-dark mb-3"),
                dbc.Card(dbc.CardBody([
                    html.H6("Insights & Recomendações", style={"color": "#e5e7eb", "fontWeight": "600"}),
                    html.Div(id="d1-insights", style={"maxHeight": "320px", "overflowY": "auto"})
                ]), className="card-dark mb-3"),
                dbc.Card(dbc.CardBody([
                    html.H6("Sobre o Modelo (Explainability)", style={"color": "#e5e7eb", "fontWeight": "600"}),
                    html.Div(id="d1-explain", style={"maxHeight": "380px", "overflowY": "auto"})
                ]), className="card-dark")
            ], md=4)
        ], className="mb-3"),

        # Extras (só no modo advanced)
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Análise Temporal", style={"color": "#e5e7eb"}),
                dcc.Graph(id="d1-trend", config={'displayModeBar': False})
            ]), className="card-dark"), md=12)
        ], id="d1-extras"),

        dbc.Row([
            dbc.Col([html.Hr(style={"borderColor": "#334155"}),
                     html.Div(id="d1-model", style={"color": "#94a3b8", "fontSize": "12px"})], md=12)
        ], className="mt-3")
    ])

# Dropdown de período sincronizado por empresa
@callback(
    Output("d1-period", "options"),
    Output("d1-period", "value"),
    Input("d1-company", "value")
)
def _sync_period(company_id: str):
    f, _, _ = load_all_data()
    if f.empty or not company_id:
        return [], []
    df = (f[f["ID"] == company_id]
          .drop_duplicates(subset=["ym_str"])
          .sort_values("year_month"))
    periods = df["ym_str"].tolist()
    default = periods[-6:] if len(periods) > 6 else periods
    return [{"label": p, "value": p} for p in periods], default

# ===================== CALLBACKS: Botões de visualização =====================

@callback(
    [Output("d1-view", "data"),
     Output("btn-executive", "color"),
     Output("btn-executive", "className"),
     Output("btn-benchmark", "color"),
     Output("btn-benchmark", "className"),
     Output("btn-advanced", "color"),
     Output("btn-advanced", "className")],
    [Input("btn-executive", "n_clicks"),
     Input("btn-benchmark", "n_clicks"),
     Input("btn-advanced", "n_clicks")],
    prevent_initial_call=True
)
def update_view_buttons(exec_clicks, bench_clicks, adv_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return "executive", "primary", "view-button active-view", "secondary", "view-button", "secondary", "view-button"
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "btn-executive":
        return ("executive", "primary", "view-button active-view",
                "secondary", "view-button", "secondary", "view-button")
    if button_id == "btn-benchmark":
        return ("benchmark", "secondary", "view-button",
                "primary", "view-button active-view", "secondary", "view-button")
    if button_id == "btn-advanced":
        return ("advanced", "secondary", "view-button",
                "secondary", "view-button", "primary", "view-button active-view")
    return "executive", "primary", "view-button active-view", "secondary", "view-button", "secondary", "view-button"

# ===================== CALLBACK PRINCIPAL =====================

@callback(
    Output("d1-kpis", "children"),
    Output("d1-main", "children"),
    Output("d1-score", "children"),
    Output("d1-insights", "children"),
    Output("d1-explain", "children"),
    Output("d1-trend", "figure"),
    Output("d1-extras", "style"),
    Output("d1-model", "children"),
    Input("d1-company", "value"),
    Input("d1-period", "value"),
    Input("d1-view", "data")
)
def _update(company_id, periods, view):
    f, e, companies = load_all_data()
    if not company_id:
        return [], html.Div("Selecione uma empresa e período"), "", "", "", go.Figure(), {"display": "none"}, ""

    df_company = f[f["ID"] == company_id].copy()
    if isinstance(periods, str) or periods is None:
        periods = [periods] if periods else []
    if not periods:
        periods = df_company["ym_str"].drop_duplicates().tolist()

    df_period = df_company[df_company["ym_str"].isin(periods)].copy()
    if df_period.empty:
        return [], html.Div("Sem dados para o período"), "", "", "", go.Figure(), {"display": "none"}, ""

    # garante 1 linha por mês
    df_period = _agg_monthly(df_period)

    # ===== KPIs =====
    kpis = _kpi_cards(df_period, e, company_id)

    # ===== MAIN VIEW =====
    if view == "benchmark":
        main_view = _benchmark_view(company_id, companies)
    elif view == "advanced":
        main_view = _advanced_view(df_period)
    else:
        main_view = _executive_view(df_period)

    # ===== SCORE =====
    score_comp = _competitive_score(company_id, companies)

    # ===== ESTÁGIO/CONF =====
    est_row = e[e["ID"] == company_id].head(1)
    stage = est_row["estagio"].iloc[0] if not est_row.empty else "N/D"
    conf  = float(est_row["confianca"].iloc[0]) if not est_row.empty and "confianca" in est_row.columns else 0.5

    # ===== INSIGHTS =====
    insights = _insights(stage, conf, df_period)

    # ===== EXPLAINABILITY =====
    explain = _explainability_panel(company_id, companies, stage, conf)

    # ===== TREND =====
    trend_fig = _trend_fig(df_company)
    ts = int(time.time())
    if hasattr(trend_fig, 'layout'):
        trend_fig.layout.meta = {"timestamp": ts, "cache_bust": True}

    extras_style = {} if view == "advanced" else {"display": "none"}
    model_info = f"Modelo: Não-supervisionado/supervisionado híbrido | Estágio: {stage} | Confiança: {conf:.0%}"

    return kpis, main_view, score_comp, insights, explain, trend_fig, extras_style, model_info

# ===================== VIEWS & CARDS =====================

def _kpi_cards(df: pd.DataFrame, e: pd.DataFrame, company_id: str):
    receita_total = float(df["receita_mensal"].sum()) if "receita_mensal" in df.columns else 0.0
    despesa_total = float(df["despesa_mensal"].sum()) if "despesa_mensal" in df.columns else 0.0
    fluxo_total   = float(df["fluxo_liquido"].sum()) if "fluxo_liquido" in df.columns else 0.0
    margem_media  = (fluxo_total / receita_total * 100) if receita_total > 0 else 0.0
    crescimento   = float(df["g_receita_mom"].mean() * 100) if "g_receita_mom" in df.columns else 0.0
    consist       = float((df["fluxo_liquido"] > 0).mean() * 100) if "fluxo_liquido" in df.columns else 0.0
    vola          = float(df["vol_receita_3m"].mean()) if "vol_receita_3m" in df.columns else None

    est_row = e[e["ID"] == company_id].head(1)
    stage = est_row["estagio"].iloc[0] if not est_row.empty else "N/D"
    conf  = float(est_row["confianca"].iloc[0]) if not est_row.empty and "confianca" in est_row.columns else 0.5

    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Estágio", className="kpi-label", style={"color":"#94a3b8"}),
            html.H4(stage, className="kpi-value", style={"color": get_metric_color("estagio", conf)}),
            html.Small(f"Confiança: {conf:.0%}", className="kpi-sub", style={"color":"#94a3b8"})
        ]), className="kpi-card"), md=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Receita Total", className="kpi-label", style={"color":"#94a3b8"}),
            html.H4(format_currency_br(receita_total), className="kpi-value", style={"color": "#22c55e"}),
            html.Small(f"{df['year_month'].nunique()} meses", className="kpi-sub", style={"color":"#94a3b8"})
        ]), className="kpi-card"), md=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Fluxo Líquido", className="kpi-label", style={"color":"#94a3b8"}),
            html.H4(format_currency_br(fluxo_total), className="kpi-value",
                    style={"color": "#22c55e" if fluxo_total >= 0 else "#ef4444"}),
            html.Small(f"Margem: {margem_media:.1f}%", className="kpi-sub", style={"color":"#94a3b8"})
        ]), className="kpi-card"), md=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Crescimento Médio", className="kpi-label", style={"color":"#94a3b8"}),
            html.H4(f"{crescimento:+.1f}%", className="kpi-value",
                    style={"color": "#22c55e" if crescimento >= 0 else "#ef4444"}),
            html.Small("MoM", className="kpi-sub", style={"color":"#94a3b8"})
        ]), className="kpi-card"), md=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Consistência", className="kpi-label", style={"color":"#94a3b8"}),
            html.H4(f"{consist:.0f}%", className="kpi-value", style={"color": "#8b5cf6"}),
            html.Small("Fluxo positivo", className="kpi-sub", style={"color":"#94a3b8"})
        ]), className="kpi-card"), md=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Volatilidade", className="kpi-label", style={"color":"#94a3b8"}),
            html.H4(f"{vola:.1f}" if vola is not None else "N/D",
                    className="kpi-value", style={"color": "#f59e0b"}),
            html.Small("Média 3M", className="kpi-sub", style={"color":"#94a3b8"})
        ]), className="kpi-card"), md=2),
    ])

def _executive_view(df: pd.DataFrame):
    """Vista executiva com gráficos individuais (tema dark)."""
    timestamp = int(time.time())
    x = df["year_month"].dt.to_pydatetime().tolist()

    # 1) Evolução
    fig_evolution = go.Figure()
    if "receita_mensal" in df.columns:
        fig_evolution.add_scatter(x=x, y=df["receita_mensal"].tolist(),
                                  name='Receita', mode='lines+markers',
                                  line=dict(color='#22c55e', width=3), marker=dict(size=8))
    if "despesa_mensal" in df.columns:
        fig_evolution.add_scatter(x=x, y=df["despesa_mensal"].tolist(),
                                  name='Despesa', mode='lines+markers',
                                  line=dict(color='#ef4444', width=3), marker=dict(size=8))
    if "fluxo_liquido" in df.columns:
        fig_evolution.add_scatter(x=x, y=df["fluxo_liquido"].tolist(),
                                  name='Fluxo', mode='lines+markers',
                                  line=dict(color='#3b82f6', width=3), marker=dict(size=8))
    fig_evolution.update_layout(
        title="Evolução Financeira",
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e5e7eb", 'size': 12},
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig_evolution.update_xaxes(gridcolor='#334155')
    fig_evolution.update_yaxes(gridcolor='#334155')
    fig_evolution.layout.meta = {"timestamp": timestamp, "cache_bust": True}

    # 2) Composição
    rec = float(df.get("receita_mensal", pd.Series([0])).sum())
    desp = float(df.get("despesa_mensal", pd.Series([0])).sum())
    luc = max(0.0, rec - desp)
    fig_composition = go.Figure(data=[go.Pie(
        labels=["Despesas", "Lucro"], values=[desp, luc],
        hole=0.4, marker=dict(colors=["#ef4444", "#22c55e"]),
        textinfo='label+percent', textfont=dict(size=12, color="#e5e7eb")
    )])
    fig_composition.update_layout(
        title="Composição Financeira",
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e5e7eb", 'size': 12}, margin=dict(l=20, r=20, t=50, b=20)
    )
    fig_composition.layout.meta = {"timestamp": timestamp, "cache_bust": True}

    # 3) Variação do Fluxo
    fig_flow = go.Figure()
    if "fluxo_liquido" in df.columns and len(df) >= 2:
        deltas = [df["fluxo_liquido"].iloc[0]] + \
                 [df["fluxo_liquido"].iloc[i] - df["fluxo_liquido"].iloc[i-1] for i in range(1, len(df))]
        fig_flow.add_trace(go.Bar(
            x=df["ym_str"], y=deltas,
            marker=dict(color=["#22c55e" if v >= 0 else "#ef4444" for v in deltas]),
            text=[f"{v:+.0f}" for v in deltas], textposition='auto'
        ))
    fig_flow.update_layout(
        title="Variação do Fluxo",
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e5e7eb", 'size': 12}, showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig_flow.update_xaxes(gridcolor='#334155')
    fig_flow.update_yaxes(gridcolor='#334155')
    fig_flow.layout.meta = {"timestamp": timestamp, "cache_bust": True}

    # 4) Correlação Receita x Crescimento
    fig_correlation = go.Figure()
    if "g_receita_mom" in df.columns and "receita_mensal" in df.columns:
        fig_correlation.add_trace(go.Scatter(
            x=_safe_num(df["receita_mensal"]), y=_safe_num(df["g_receita_mom"]) * 100,
            mode="markers",
            marker=dict(size=12, color=_safe_num(df.get("fluxo_liquido", 0.0)),
                        colorscale="RdYlGn", showscale=True,
                        colorbar=dict(title="Fluxo", len=0.7, tickcolor="#e5e7eb", titlefont=dict(color="#e5e7eb"))),
            text=df["ym_str"],
            hovertemplate="Mês: %{text}<br>Receita: %{x:,.0f}<br>Crescimento: %{y:.1f}%<extra></extra>"
        ))
    fig_correlation.update_layout(
        title="Correlação Receita x Crescimento",
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e5e7eb", 'size': 12}, showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig_correlation.update_xaxes(gridcolor='#334155', title="Receita", tickfont=dict(color="#e5e7eb"), titlefont=dict(color="#e5e7eb"))
    fig_correlation.update_yaxes(gridcolor='#334155', title="Crescimento %", tickfont=dict(color="#e5e7eb"), titlefont=dict(color="#e5e7eb"))
    fig_correlation.layout.meta = {"timestamp": timestamp, "cache_bust": True}

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_evolution, config={'displayModeBar': False}), width=12)], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_composition, config={'displayModeBar': False}), width=4),
            dbc.Col(dcc.Graph(figure=fig_flow,        config={'displayModeBar': False}), width=8),
        ], className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_correlation, config={'displayModeBar': False}), width=12)])
    ])

def _benchmark_view(company_id: str, companies: pd.DataFrame):
    """Radar em percentis vs peers (setor+porte), com fallback quando peer<8."""
    if companies.empty or company_id not in companies["ID"].values:
        return html.Div([html.H5("Sem base para benchmarking.", className="text-center text-muted mt-4")],
                        className="chart-card")

    row = companies[companies["ID"] == company_id].iloc[0]
    setor = str(row.get("setor", "OUTROS"))
    porte = str(row.get("porte", "MICRO"))

    peers = companies[(companies["setor"] == setor) & (companies["porte"] == porte)].copy()
    if len(peers) < 8:
        peers = companies[companies["setor"] == setor].copy()
    if len(peers) < 3:
        return html.Div([html.H5("Peer group insuficiente para benchmarking.",
                                 className="text-center text-muted mt-4")],
                        className="chart-card")

    def pct_rank(series: pd.Series, v: float, invert: bool = False) -> float:
        s = pd.to_numeric(series, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
        if s.empty:
            return 50.0
        p = float((s <= v).mean()*100.0)
        return 100.0 - p if invert else p

    metrics = [
        ("Receita",       "receita_media",        False),
        ("Crescimento",   "crescimento_medio",    False),
        ("Margem",        "margem_media",         False),
        ("Consistência",  "consistencia_fluxo",   False),
        ("Eficiência",    "volatilidade_receita", True),  # menor é melhor
    ]

    cats, vals = [], []
    for label, col, invert in metrics:
        if col not in peers.columns:
            continue
        cats.append(label)
        vals.append(round(pct_rank(peers[col], float(row.get(col, 0.0)), invert), 1))

    ts = int(time.time())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        fillcolor='rgba(34,197,94,0.20)', line=dict(color='#22c55e', width=2),
        marker=dict(size=6), name='Empresa (percentil)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[50]*len(cats), theta=cats, line=dict(color='#94a3b8', width=1, dash='dash'),
        name='Referência (P50)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100], gridcolor="#334155", tickfont=dict(color="#e5e7eb"))),
        showlegend=True, height=450, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font={'color': "#e5e7eb", 'size': 12},
        margin=dict(l=40,r=40,t=40,b=40)
    )
    fig.layout.meta = {"timestamp": ts, "cache_bust": True}

    footer = html.Div(
        f"Peer group: {len(peers)}  •  Setor: {setor}  •  Porte: {porte}",
        style={"color": "#94a3b8", "fontSize": "12px", "marginTop": "6px"}
    )
    return html.Div([dcc.Graph(figure=fig, config={'displayModeBar': False}), footer], className="chart-card")

def _advanced_view(df: pd.DataFrame):
    """Heatmap de correlação minimalista (dark)."""
    cols = [c for c in ['receita_mensal','despesa_mensal','fluxo_liquido','g_receita_mom','vol_receita_3m'] if c in df.columns]
    if len(cols) < 2:
        return html.Div([html.H5("Dados insuficientes para correlação.", className="text-center text-muted mt-4")],
                        className="chart-card")

    corr = df[cols].corr().round(2)
    timestamp = int(time.time())

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmid=0,
        text=corr.values, texttemplate='%{text}', textfont={"size": 11, "color": "#ffffff"},
        colorbar=dict(title="Correlação", tickfont=dict(color="#e5e7eb"),
                      bgcolor="rgba(15,23,42,0.9)", bordercolor="#334155", borderwidth=1),
        hoverongaps=False
    ))
    fig.update_layout(
        title="Matriz de Correlação",
        height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e5e7eb", 'size': 12}, margin=dict(l=60, r=60, t=60, b=60)
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10, color="#e5e7eb"), gridcolor="#334155")
    fig.update_yaxes(tickfont=dict(size=10, color="#e5e7eb"), gridcolor="#334155")
    fig.layout.meta = {"timestamp": timestamp, "cache_bust": True}

    return html.Div([dcc.Graph(figure=fig, config={'displayModeBar': False})], className="chart-card")

def _gauge(value: float, title: str) -> go.Figure:
    """Gauge elegante e bem dimensionado (dark)."""
    timestamp = int(time.time())
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#e5e7eb'}},
        delta={'reference': 50, 'increasing': {'color': "#22c55e"}, 'decreasing': {'color': "#ef4444"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#e5e7eb"},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "rgba(15,23,42,0.8)",
            'borderwidth': 2, 'bordercolor': "#334155",
            'steps': [
                {'range': [0, 25], 'color': "rgba(239, 68, 68, 0.18)"},
                {'range': [25, 50], 'color': "rgba(245, 158, 11, 0.18)"},
                {'range': [50, 75], 'color': "rgba(34, 197, 94, 0.20)"},
                {'range': [75, 100], 'color': "rgba(34, 197, 94, 0.35)"}
            ],
            'threshold': {'line': {'color': "#ef4444", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={'color': "#e5e7eb", 'size': 14}, height=280, margin=dict(l=20, r=20, t=60, b=20))
    fig.layout.meta = {"timestamp": timestamp, "cache_bust": True}
    return fig

def _competitive_score(company_id: str, companies: pd.DataFrame):
    """Score simplificado (0–100)."""
    if companies.empty or company_id not in companies["ID"].values:
        return html.Div([dcc.Graph(figure=_gauge(50, "Score Competitivo"), config={'displayModeBar': False})],
                        className="chart-card")
    row = companies[companies["ID"] == company_id].iloc[0]
    parts = []
    parts.append(float(np.clip(float(row.get("receita_media", 0.0))/100_000 * 30, 0, 30)))
    parts.append(float(np.clip(15 + float(row.get("crescimento_medio", 0.0))*50, 0, 30)))
    parts.append(float(np.clip(float(row.get("consistencia_fluxo", 0.0))*40, 0, 40)))
    total = sum(parts)
    return html.Div([dcc.Graph(figure=_gauge(total, "Score Competitivo"), config={'displayModeBar': False})],
                    className="chart-card")

def _insights(stage: str, conf: float, df: pd.DataFrame):
    skey = _stage_key(stage)
    items = []
    items.append(html.Div([
        html.Strong(f"Estágio: {stage}", style={"color": "#60a5fa"}),
        html.Span(f" ({conf:.0%} confiança)", style={"color": "#94a3b8", "fontSize":"12px"})
    ], className="mb-2"))

    if "receita_mensal" in df.columns and len(df) > 1:
        trend = np.polyfit(range(len(df)), df["receita_mensal"], 1)[0]
        if trend > 1000:
            items.append(html.Li("📈 Tendência de crescimento forte na receita"))
        elif trend < -1000:
            items.append(html.Li("📉 Tendência de queda na receita"))
    else:
        items.append(html.Li("➡️ Receita relativamente estável"))

    if "fluxo_liquido" in df.columns:
        pos = float((df["fluxo_liquido"] > 0).mean())
        if pos >= 0.8: items.append(html.Li("✅ Excelente consistência de fluxo positivo"))
        elif pos < 0.5: items.append(html.Li("⚠️ Fluxo negativo em mais de 50% do período"))

        margem = float(df["fluxo_liquido"].sum()) / float(df.get("receita_mensal", 0).sum() + 1e-9)
        if margem > 0.2: items.append(html.Li(f"💎 Margem saudável de {margem:.1%}"))
        elif margem < 0: items.append(html.Li(f"🔴 Margem negativa de {margem:.1%}"))

    items.append(html.Hr(style={"borderColor": "#334155", "margin":"10px 0"}))
    items.append(html.Strong("Recomendações:", style={"color":"#e5e7eb"}))

    if skey == "inicio":
        items += [html.Li("Foque em validação de mercado"),
                  html.Li("Monitore burn rate mensalmente"),
                  html.Li("Busque investimento seed")]
    elif skey == "crescimento":
        items += [html.Li("Invista em escalabilidade"),
                  html.Li("Estruture processos internos"),
                  html.Li("Considere Series A/B")]
    elif skey == "maturidade":
        items += [html.Li("Otimize eficiência operacional"),
                  html.Li("Explore novos mercados"),
                  html.Li("Considere M&A estratégico")]
    elif skey == "declinio":
        items += [html.Li("Reestruture custos urgentemente"),
                  html.Li("Renegocie dívidas"),
                  html.Li("Foque no core business")]
    elif skey == "reestruturacao":
        items += [html.Li("Plano de 90 dias com metas objetivas"),
                  html.Li("Comunicação transparente das mudanças"),
                  html.Li("Foco no core rentável")]

    return html.Ul(items, style={"listStyle":"none","padding":"0"})

# -------- Explainability leve --------

def _explainability_panel(company_id: str, companies: pd.DataFrame, stage: str, conf: float):
    """Percentis vs peers e 'drivers' (maiores desvios do P50)."""
    if companies.empty or company_id not in companies["ID"].values:
        return html.Div("Sem base para explicabilidade.", style={"color": "#94a3b8"})

    row = companies[companies["ID"] == company_id].iloc[0]
    setor = str(row.get("setor", "OUTROS"))
    porte = str(row.get("porte", "MICRO"))

    peers = companies[(companies["setor"] == setor) & (companies["porte"] == porte)].copy()
    if len(peers) < 8:
        peers = companies[companies["setor"] == setor].copy()
    if len(peers) < 3:
        return html.Div("Peer group insuficiente para explicabilidade.", style={"color": "#94a3b8"})

    def pct(series: pd.Series, v: float, invert: bool = False) -> float:
        s = pd.to_numeric(series, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
        if s.empty:
            return 50.0
        p = float((s <= v).mean()*100.0)
        return 100.0 - p if invert else p

    spec = [
        ("Receita média",        "receita_media",        False),
        ("Crescimento médio",    "crescimento_medio",    False),
        ("Margem média",         "margem_media",         False),
        ("Consistência de fluxo","consistencia_fluxo",   False),
        ("Volatilidade receita", "volatilidade_receita", True),  # menor é melhor
    ]
    rows = []
    for label, col, inv in spec:
        if col not in peers.columns: continue
        pv = pct(peers[col], float(row.get(col, 0.0)), inv)
        rows.append((label, pv))

    labels = [r[0] for r in rows]
    diffs  = [round(r[1] - 50.0, 1) for r in rows]

    bar = go.Figure(go.Bar(
        x=diffs, y=labels, orientation="h",
        text=[f"{r[1]:.0f}º pct" for r in rows], textposition="auto",
        marker=dict(color=["#22c55e" if d >= 0 else "#ef4444" for d in diffs])
    ))
    bar.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e5e7eb", 'size': 12},
        xaxis=dict(title="Desvio do P50 (p.p.)", zeroline=True, zerolinewidth=1, zerolinecolor="#94a3b8", gridcolor="#334155", tickfont=dict(color="#e5e7eb")),
        yaxis=dict(title="", tickfont=dict(color="#e5e7eb"), gridcolor="#334155")
    )

    top = sorted(rows, key=lambda r: abs(r[1]-50.0), reverse=True)[:3]
    drivers = [html.Li(f"{name}: {pctv:.0f}º pct") for name, pctv in top]

    header = html.Div([
        html.Span("Estágio atual: ", style={"color": "#94a3b8"}),
        html.Strong(stage, style={"color": "#e2e8f0"}),
        html.Span(f" • Confiança: {conf:.0%}", style={"color": "#94a3b8", "marginLeft": "6px"}),
        html.Br(),
        html.Small(f"Peers: {len(peers)} • Setor: {setor} • Porte: {porte}", style={"color": "#94a3b8"})
    ], className="mb-2")

    return html.Div([
        header,
        dcc.Graph(figure=bar, config={'displayModeBar': False}),
        html.Div([
            html.Strong("Principais drivers (|P - 50|):", style={"color": "#e2e8f0"}),
            html.Ul(drivers, style={"color": "#cbd5e1", "marginBottom": 0})
        ], style={"marginTop": "6px"})
    ])

def _trend_fig(df_company: pd.DataFrame) -> go.Figure:
    if df_company.empty:
        return go.Figure()
    df = _agg_monthly(df_company.sort_values("year_month"))
    x = df["year_month"].dt.to_pydatetime().tolist()

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Receita", "Crescimento %", "Volatilidade", "Consistência"),
                        vertical_spacing=0.12, horizontal_spacing=0.1)

    if "receita_mensal" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=_safe_num(df["receita_mensal"]),
                                 mode="lines+markers", name="Receita",
                                 line=dict(color="#22c55e", width=2)), row=1, col=1)
        if len(df) > 1:
            z = np.polyfit(range(len(df)), _safe_num(df["receita_mensal"]), 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=x, y=p(range(len(df))),
                                     mode="lines", name="Tendência",
                                     line=dict(color="#60a5fa", width=1, dash="dash")),
                          row=1, col=1)

    if "g_receita_mom" in df.columns:
        fig.add_trace(go.Bar(x=x, y=_safe_num(df["g_receita_mom"])*100, name="Crescimento %"), row=1, col=2)

    if "vol_receita_3m" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=_safe_num(df["vol_receita_3m"]),
                                 mode="lines+markers", name="Volatilidade",
                                 line=dict(color="#f59e0b", width=2),
                                 fill="tozeroy", fillcolor="rgba(245,158,11,0.20)"),
                      row=2, col=1)

    if "fluxo_liquido" in df.columns:
        cons = (df["fluxo_liquido"] > 0).rolling(window=3, min_periods=1).mean()*100
        fig.add_trace(go.Scatter(x=x, y=_safe_num(cons),
                                 mode="lines+markers", name="Consistência %",
                                 line=dict(color="#8b5cf6", width=2)), row=2, col=2)

    fig.update_layout(
        height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e5e7eb", 'size': 11}, showlegend=False,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    fig.update_xaxes(gridcolor="#334155", tickfont=dict(color="#e5e7eb"))
    fig.update_yaxes(gridcolor="#334155", tickfont=dict(color="#e5e7eb"))

    timestamp = int(time.time())
    fig.layout.meta = {"timestamp": timestamp, "cache_bust": True}
    return fig
