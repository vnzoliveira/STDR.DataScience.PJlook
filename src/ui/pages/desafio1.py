# src/ui/pages/desafio1.py
"""
Dashboard executivo do Desafio 1 (corrigido e robusto).
- Mantém f_empresa_mes sem merges para evitar duplicação de meses
- KPIs e gráficos agregam corretamente por mês
- Visualizações resilientes a NaN/inf/colunas ausentes
"""
from __future__ import annotations
from dash import register_page, dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

from src.ui.components.metrics_formatter import (
    format_currency_br, format_percentage, get_standardized_label, get_metric_color
)
from src.features.sector_analysis import SectorBenchmarkEngine
from src.features.business_insights import BusinessInsightEngine

register_page(__name__, path="/desafio1", title="Dashboard Executivo - Análise Empresarial")

# Caminhos
F_EMP_PATH = Path("reports/exports/f_empresa_mes.parquet")
EST_PATH   = Path("reports/exports/estagio.parquet")
B1_PATH    = Path("data/processed/base1/base1.parquet")

# Engines (somente onde realmente usamos)
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
    """Resumo por empresa para benchmarking (sem replicar meses)."""
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

    # setor/porte (deduplicar base1 por ID se vier com várias linhas)
    if not b1.empty:
        cols = [c for c in ["ID", "DS_CNAE", "VL_FATU"] if c in b1.columns]
        b1u = b1[cols].drop_duplicates(subset=["ID"])
        g = g.merge(b1u, on="ID", how="left")
        # porte simples pelo faturamento anual
        def _porte(fat):
            fat = float(fat) if pd.notna(fat) else 0.0
            if fat < 360_000: return "MICRO"
            if fat < 4_800_000: return "PEQUENA"
            if fat < 300_000_000: return "MEDIA"
            return "GRANDE"
        g["porte"] = g["VL_FATU"].apply(_porte)
        g["setor"] = (g.get("DS_CNAE").astype(str).str.upper()
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

    companies = _company_summary(f, e, b1)
    return f, e, companies


def _gauge(value: float, title: str, vmax: float = 100) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, vmax]},
            'bar': {'color': "#22c55e" if value >= 70 else "#f59e0b" if value >= 40 else "#ef4444"},
            'bgcolor': "#0e1117",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 0.4*vmax], 'color': '#dc2626'},
                {'range': [0.4*vmax, 0.7*vmax], 'color': '#f59e0b'},
                {'range': [0.7*vmax, vmax], 'color': '#22c55e'}
            ]
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="#0e1117", font={'color': "#e6e6e6"})
    return fig


# ===================== LAYOUT =====================

def layout():
    f, e, companies = load_all_data()
    if f.empty or e.empty:
        return dbc.Alert("Dados não encontrados. Execute: python -m src.cli.main build-all", color="warning")

    ids = sorted(f["ID"].unique().tolist())

    # Generate unique timestamp for cache busting
    timestamp = int(time.time())
    
    return dbc.Container(fluid=True, children=[
        # AGGRESSIVE CLIENT-SIDE CACHE BUSTING
        html.Script(f"""
        // Force browser cache clear - timestamp: {timestamp}
        console.log('FIAP Dashboard - Cache busting active: {timestamp}');
        
        // Clear all caches
        if ('caches' in window) {{
            caches.keys().then(function(names) {{
                names.forEach(function(name) {{
                    caches.delete(name);
                }});
            }});
        }}
        
        // Force Plotly refresh
        if (window.Plotly) {{
            console.log('Clearing Plotly cache...');
            window.Plotly.purge();
            // Force re-download of Plotly assets
            if (window.Plotly.d3) {{
                delete window.Plotly.d3;
            }}
        }}
        
        // Force page reload on navigation if cached
        if (performance.navigation.type === 1) {{
            console.log('Page reloaded - forcing hard refresh');
            setTimeout(function() {{
                window.location.reload(true);
            }}, 100);
        }}
        
        // Prevent browser caching of AJAX requests
        if (window.jQuery) {{
            jQuery.ajaxSetup({{ cache: false }});
        }}
        """),
        
        dbc.Row([
            dbc.Col([
                html.H2("Dashboard Executivo - Análise Empresarial",
                        style={"color": "#e5e7eb", "marginBottom": "6px"}),
                html.P(f"Análise completa com benchmarking setorial e ML avançado - Cache: {timestamp}",
                       style={"color": "#9ca3af", "fontSize": "14px"})
            ], md=8),
            dbc.Col([
                html.Div([
                    html.Small("Última atualização: ", style={"color": "#9ca3af"}),
                    html.Small(pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
                               style={"color": "#60a5fa"})
                ], className="text-end", style={"marginTop": "20px"})
            ], md=4)
        ], className="mb-4", style={"borderBottom": "1px solid #374151", "paddingBottom": "14px"}),

        # Filtros
        dbc.Row([
            dbc.Col([
                html.Label("Empresa", style={"color": "#9ca3af", "fontSize": "12px"}),
                dcc.Dropdown(id="d1-company", options=[{"label": i, "value": i} for i in ids],
                             value=ids[0], clearable=False, className="dash-dropdown-dark")
            ], md=3),
            dbc.Col([
                html.Label("Período", style={"color": "#9ca3af", "fontSize": "12px"}),
                dcc.Dropdown(id="d1-period", options=[], value=[], multi=True,
                             className="dash-dropdown-dark",
                             placeholder="Selecione meses...")
            ], md=5),
            dbc.Col([
                html.Label("Visualização", style={"color": "#9ca3af", "fontSize": "12px"}),
                dcc.RadioItems(
                    id="d1-view",
                    options=[
                        {"label": "📈 Executivo", "value": "executive"},
                        {"label": "🎯 Benchmarking", "value": "benchmark"},
                        {"label": "🔬 Análise Avançada", "value": "advanced"}
                    ],
                    value="executive", inline=True, className="radio-dark",
                    style={"marginTop": "5px"}
                )
            ], md=4)
        ], className="mb-3"),

        # KPIs
        html.Div(id="d1-kpis", className="mb-3"),

        # Área principal
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(html.Div(id="d1-main"))), md=8, className="card-dark"),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H6("Score Competitivo", style={"color": "#e5e7eb"}),
                    html.Div(id="d1-score")
                ]), className="card-dark mb-3"),
                dbc.Card(dbc.CardBody([
                    html.H6("Insights & Recomendações", style={"color": "#e5e7eb"}),
                    html.Div(id="d1-insights", style={"maxHeight": "420px", "overflowY": "auto"})
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
            dbc.Col([html.Hr(style={"borderColor": "#374151"}),
                     html.Div(id="d1-model", style={"color": "#6b7280", "fontSize": "12px"})], md=12)
        ], className="mt-3")
    ])


# Dropdown de período sincronizado por empresa (sem duplicar)
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


# ===================== CALLBACK PRINCIPAL =====================

@callback(
    Output("d1-kpis", "children"),
    Output("d1-main", "children"),
    Output("d1-score", "children"),
    Output("d1-insights", "children"),
    Output("d1-trend", "figure"),
    Output("d1-extras", "style"),
    Output("d1-model", "children"),
    Input("d1-company", "value"),
    Input("d1-period", "value"),
    Input("d1-view", "value")
)
def _update(company_id, periods, view):
    f, e, companies = load_all_data()
    if not company_id:
        return [], html.Div("Selecione uma empresa e período"), "", "", go.Figure(), {"display": "none"}, ""

    df_company = f[f["ID"] == company_id].copy()
    if isinstance(periods, str) or periods is None:
        periods = [periods] if periods else []
    if not periods:
        periods = df_company["ym_str"].drop_duplicates().tolist()

    df_period = df_company[df_company["ym_str"].isin(periods)].copy()
    if df_period.empty:
        return [], html.Div("Sem dados para o período"), "", "", go.Figure(), {"display": "none"}, ""

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

    # ===== INSIGHTS =====
    est_row = e[e["ID"] == company_id].head(1)
    stage = est_row["estagio"].iloc[0] if not est_row.empty else "N/D"
    conf  = float(est_row["confianca"].iloc[0]) if not est_row.empty and "confianca" in est_row.columns else 0.5
    insights = _insights(stage, conf, df_period)

    # ===== TREND =====
    trend_fig = _trend_fig(df_company)
    
    # CACHE BUSTING - Add unique timestamp to figure metadata
    timestamp = int(time.time())
    if hasattr(trend_fig, 'layout'):
        trend_fig.layout.meta = {"timestamp": timestamp, "cache_bust": True}

    # ===== EXTRAS VISIBILITY =====
    extras_style = {} if view == "advanced" else {"display": "none"}

    # ===== MODEL INFO =====
    model_info = f"Modelo: Não-supervisionado/supervisionado híbrido | Estágio: {stage} | Confiança: {conf:.0%} | Cache: {timestamp}"

    return kpis, main_view, score_comp, insights, trend_fig, extras_style, model_info


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
            html.P("Estágio", className="kpi-label"),
            html.H4(stage, className="kpi-value", style={"color": get_metric_color("estagio", conf)}),
            html.Small(f"Confiança: {conf:.0%}", className="kpi-sub")
        ]), className="kpi-card"), md=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Receita Total", className="kpi-label"),
            html.H4(format_currency_br(receita_total), className="kpi-value", style={"color": "#22c55e"}),
            html.Small(f"{df['year_month'].nunique()} meses", className="kpi-sub")
        ]), className="kpi-card"), md=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Fluxo Líquido", className="kpi-label"),
            html.H4(format_currency_br(fluxo_total), className="kpi-value",
                    style={"color": "#22c55e" if fluxo_total >= 0 else "#ef4444"}),
            html.Small(f"Margem: {margem_media:.1f}%", className="kpi-sub")
        ]), className="kpi-card"), md=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Crescimento Médio", className="kpi-label"),
            html.H4(f"{crescimento:+.1f}%", className="kpi-value",
                    style={"color": "#22c55e" if crescimento >= 0 else "#ef4444"}),
            html.Small("MoM", className="kpi-sub")
        ]), className="kpi-card"), md=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Consistência", className="kpi-label"),
            html.H4(f"{consist:.0f}%", className="kpi-value", style={"color": "#8b5cf6"}),
            html.Small("Fluxo positivo", className="kpi-sub")
        ]), className="kpi-card"), md=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Volatilidade", className="kpi-label"),
            html.H4(f"{vola:.1f}" if vola is not None else "N/D",
                    className="kpi-value", style={"color": "#f59e0b"}),
            html.Small("Média 3M", className="kpi-sub")
        ]), className="kpi-card"), md=2),
    ])


def _executive_view(df: pd.DataFrame):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Evolução Financeira", "Composição",
                        "Variação do Fluxo", "Correlação Receita x Cresc."),
        specs=[[{"type": "scatter"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "scatter"}]],
        vertical_spacing=0.14, horizontal_spacing=0.12
    )
    x = df["year_month"].dt.to_pydatetime().tolist()

    if "receita_mensal" in df.columns:
        fig.add_scatter(
            x=x,
            y=df["receita_mensal"].tolist(),
            name='Receita',
            mode='lines+markers',
            line=dict(color='#22c55e', width=3),
            marker=dict(size=8)
        )
    if "despesa_mensal" in df.columns:
        fig.add_scatter(
            x=x,
            y=df["despesa_mensal"].tolist(),
            name='Despesa',
            mode='lines+markers',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=8)
        )
    if "fluxo_liquido" in df.columns:
        fig.add_scatter(
            x=x,
            y=df["fluxo_liquido"].tolist(),
            name='Fluxo',
            mode='lines+markers',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8)
        )

    # Pie (despesa vs lucro)
    rec = float(df.get("receita_mensal", pd.Series([0])).sum())
    desp = float(df.get("despesa_mensal", pd.Series([0])).sum())
    luc  = max(0.0, rec - desp)
    fig.add_trace(go.Pie(labels=["Despesas", "Lucro"],
                         values=[desp, luc],
                         hole=0.4,
                         marker=dict(colors=["#ef4444", "#22c55e"])),
                  row=1, col=2)

    # Variação do fluxo (delta mês a mês)
    if "fluxo_liquido" in df.columns and len(df) >= 2:
        deltas = [df["fluxo_liquido"].iloc[0]] + \
                 [df["fluxo_liquido"].iloc[i] - df["fluxo_liquido"].iloc[i-1] for i in range(1, len(df))]
        fig.add_trace(go.Bar(x=df["ym_str"], y=deltas,
                             marker=dict(color=["#22c55e" if v >= 0 else "#ef4444" for v in deltas])),
                      row=2, col=1)

    # Scatter correlação
    if "g_receita_mom" in df.columns and "receita_mensal" in df.columns:
        fig.add_trace(go.Scatter(
            x=_safe_num(df["receita_mensal"]), y=_safe_num(df["g_receita_mom"]) * 100,
            mode="markers",
            marker=dict(size=10,
                        color=_safe_num(df.get("fluxo_liquido", 0.0)),
                        colorscale="RdYlGn", showscale=True,
                        colorbar=dict(title="Fluxo", x=1.15)),
            text=df["ym_str"],
            hovertemplate="Mês: %{text}<br>Receita: %{x:,.0f}<br>Crescimento: %{y:.1f}%<extra></extra>"
        ), row=2, col=2)

    fig.update_layout(height=600, paper_bgcolor="#0e1117", plot_bgcolor="#1e293b",
                      font={'color': "#e6e6e6"}, showlegend=True,
                      legend=dict(bgcolor='rgba(30,41,59,0.8)', bordercolor='#475569', borderwidth=1))
    fig.update_xaxes(gridcolor='#374151'); fig.update_yaxes(gridcolor='#374151')
    
    # CACHE BUSTING - Add timestamp to figure metadata
    timestamp = int(time.time())
    fig.layout.meta = {"timestamp": timestamp, "cache_bust": True}
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _benchmark_view(company_id: str, companies: pd.DataFrame):
    """Radar simplificado vs 'média' (50) para visual rápido."""
    if companies.empty or company_id not in companies["ID"].values:
        return html.Div("Sem base para benchmarking.")
    row = companies[companies["ID"] == company_id].iloc[0]
    def _clip100(x): return float(np.clip(x, 0, 100))

    receita = _clip100((row.get("receita_media", 0) / 100000) * 100)
    cresc   = _clip100(50 + row.get("crescimento_medio", 0) * 100)
    margem  = _clip100(50 + row.get("margem_media", 0) * 100)
    consist = _clip100(row.get("consistencia_fluxo", 0) * 100)
    efic    = _clip100(100 - row.get("volatilidade_receita", 50))

    cats = ["Receita", "Crescimento", "Margem", "Consistência", "Eficiência"]
    vals = [receita, cresc, margem, consist, efic]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself',
                                  fillcolor='rgba(34,197,94,0.25)',
                                  line=dict(color='#22c55e', width=2),
                                  marker=dict(size=6), name='Empresa'))
    fig.add_trace(go.Scatterpolar(r=[50]*len(cats), theta=cats,
                                  line=dict(color='#6b7280', width=1, dash='dash'),
                                  name='Média Setor'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100], gridcolor="#374151")),
                      showlegend=True, height=420,
                      paper_bgcolor="#0e1117", font={'color': "#e6e6e6"})
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _advanced_view(df: pd.DataFrame):
    """Heatmap de correlação minimalista (quando houver colunas suficientes)."""
    cols = [c for c in ['receita_mensal','despesa_mensal','fluxo_liquido','g_receita_mom','vol_receita_3m'] if c in df.columns]
    if len(cols) < 2:
        return html.Div("Dados insuficientes para correlação.")
    corr = df[cols].corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='RdBu', zmid=0,
        text=corr.values, texttemplate='%{text}', textfont={"size":10},
        colorbar=dict(title="Correlação")))
    fig.update_layout(height=420, paper_bgcolor="#0e1117", font={'color': "#e6e6e6"})
    
    # CACHE BUSTING - Add timestamp to figure metadata
    timestamp = int(time.time())
    fig.layout.meta = {"timestamp": timestamp, "cache_bust": True}
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _competitive_score(company_id: str, companies: pd.DataFrame):
    if companies.empty or company_id not in companies["ID"].values:
        return dcc.Graph(figure=_gauge(50, "Score Competitivo"), config={'displayModeBar': False})
    # score simples por percentis das métricas-chave
    row = companies[companies["ID"] == company_id].iloc[0]
    parts = []
    for key, scale in [("receita_media", 100000), ("crescimento_medio", 1), ("consistencia_fluxo", 1)]:
        v = float(row.get(key, 0))
        if key == "receita_media":
            parts.append(np.clip((v/scale)*30, 0, 30))
        elif key == "crescimento_medio":
            parts.append(np.clip(15 + v*50, 0, 30))
        else:
            parts.append(np.clip(v*40, 0, 40))
    total = float(sum(parts))
    return dcc.Graph(figure=_gauge(total, "Score Competitivo"), config={'displayModeBar': False})


def _insights(stage: str, conf: float, df: pd.DataFrame):
    items = []
    items.append(html.Div([
        html.Strong(f"Estágio: {stage}", style={"color": "#60a5fa"}),
        html.Span(f" ({conf:.0%} confiança)", style={"color": "#9ca3af", "fontSize":"12px"})
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

        margem = float(df["fluxo_liquido"].sum()) / float(df["receita_mensal"].sum() + 1e-9) if "receita_mensal" in df.columns else 0.0
        if margem > 0.2: items.append(html.Li(f"💎 Margem saudável de {margem:.1%}"))
        elif margem < 0: items.append(html.Li(f"🔴 Margem negativa de {margem:.1%}"))

    items.append(html.Hr(style={"borderColor": "#374151", "margin":"10px 0"}))
    items.append(html.Strong("Recomendações:", style={"color":"#e5e7eb"}))
    if stage == "Inicio":
        items += [html.Li("Foque em validação de mercado"),
                  html.Li("Monitore burn rate mensalmente"),
                  html.Li("Busque investimento seed")]
    elif stage == "Crescimento":
        items += [html.Li("Invista em escalabilidade"),
                  html.Li("Estruture processos internos"),
                  html.Li("Considere Series A/B")]
    elif stage == "Maturidade":
        items += [html.Li("Otimize eficiência operacional"),
                  html.Li("Explore novos mercados"),
                  html.Li("Considere M&A estratégico")]
    elif stage == "Declinio":
        items += [html.Li("Reestruture custos urgentemente"),
                  html.Li("Renegocie dívidas"),
                  html.Li("Foque no core business")]
    return html.Ul(items, style={"listStyle":"none","padding":"0"})


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
        fig.add_trace(go.Bar(x=x, y=_safe_num(df["g_receita_mom"])*100,
                             name="Crescimento %"), row=1, col=2)

    if "vol_receita_3m" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=_safe_num(df["vol_receita_3m"]),
                                 mode="lines+markers", name="Volatilidade",
                                 line=dict(color="#f59e0b", width=2),
                                 fill="tozeroy", fillcolor="rgba(245,158,11,0.2)"),
                      row=2, col=1)

    if "fluxo_liquido" in df.columns:
        cons = (df["fluxo_liquido"] > 0).rolling(window=3, min_periods=1).mean()*100
        fig.add_trace(go.Scatter(x=x, y=_safe_num(cons),
                                 mode="lines+markers", name="Consistência %",
                                 line=dict(color="#8b5cf6", width=2)), row=2, col=2)

    fig.update_layout(height=600, paper_bgcolor="#0e1117", plot_bgcolor="#1e293b",
                      font={'color': "#e6e6e6", 'size': 11}, showlegend=False)
    fig.update_xaxes(gridcolor="#374151"); fig.update_yaxes(gridcolor="#374151")
    
    # CACHE BUSTING - Add timestamp to figure metadata
    timestamp = int(time.time())
    fig.layout.meta = {"timestamp": timestamp, "cache_bust": True}
    
    return fig
