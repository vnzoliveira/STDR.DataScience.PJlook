"""
ANÁLISE DE RISCO DE CADEIA DE VALOR - Interface Profissional (DARK + ajustes)
- Tema dark coerente com Desafio 1
- Dropdowns brancos (via CSS)
- Layout radial com anéis + espaçamento maior para “quem paga” (menos aglomeração)
- Zoom inicial e limites ajustados (sem “muito longe” / “muito perto”)
- Gauge dark
- Cenários de risco com porcentagem garantida e não invertida
"""
from __future__ import annotations

from typing import Dict, Any, List
from dash import register_page, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from pathlib import Path
import math
import pandas as pd
import plotly.graph_objects as go

from src.sna.graph import build_ego_network
from src.features.risk_analysis import ValueChainRiskAnalyzer
from src.ui.components.metrics_formatter import format_currency_br

register_page(__name__, path="/desafio2", title="Análise de Risco - Cadeia de Valor")
cyto.load_extra_layouts()

# Paths
REL_PATH = Path("reports/exports/relations_latest.parquet")
B2_PATH = Path("data/processed/base2/base2.parquet")
F_EMP_PATH = Path("reports/exports/f_empresa_mes.parquet")

# Risk analyzer
risk_analyzer = ValueChainRiskAnalyzer()

# Labels
DIRECTION_LABELS = {
    "in": "📥 Recebimentos (Quem me paga)",
    "out": "📤 Pagamentos (Quem eu pago)",
    "both": "🔄 Ambos os Fluxos",
}
CHANNEL_LABELS = {
    "PIX": "💳 PIX",
    "TED": "🏦 TED",
    "BOLETO": "📄 Boleto",
    "OUTROS": "📋 Outros",
    "TODOS": "🌐 Todos os Canais",
}


def load_sna_data():
    """Carrega dados para análise SNA."""
    try:
        df_edges = pd.read_parquet(B2_PATH) if B2_PATH.exists() else None
        if df_edges is not None:
            df_edges["year_month"] = (
                pd.to_datetime(df_edges["DT_REFE"]).dt.to_period("M").dt.to_timestamp()
            )
            if "DS_TRAN" in df_edges.columns:
                df_edges["DS_TRAN"] = df_edges["DS_TRAN"].astype(str).str.upper().str.strip()

        ids: List[str] = []
        if F_EMP_PATH.exists():
            f = pd.read_parquet(F_EMP_PATH)
            ids = sorted(f["ID"].unique().tolist())

        relations = pd.read_parquet(REL_PATH) if REL_PATH.exists() else None
        return df_edges, ids, relations
    except Exception:
        return None, [], None


def create_enhanced_cytoscape_stylesheet():
    """Stylesheet do Cytoscape (cores e estados) — DARK."""
    return [
        # NÓS
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "font-size": "12px",
                "font-weight": "bold",
                "color": "#e5e7eb",
                "text-valign": "center",
                "text-halign": "center",
                "background-color": "#475569",
                "width": "data(size)",
                "height": "data(size)",
                "border-width": "3px",
                "border-color": "#ef4444",
                "text-outline-width": "1px",
                "text-outline-color": "#0b1220",
                "overlay-opacity": 0.05,
            },
        },
        {"selector": ".ego", "style": {
            "background-color": "#dc2626", "border-color": "#fca5a5",
            "border-width": "6px", "font-size": "14px", "font-weight": "900",
            "text-outline-width": "2px"
        }},
        {"selector": ".payer_to_ego", "style": {   # QUEM PAGA → verde
            "background-color": "#16a34a", "border-color": "#86efac", "border-width": "3px"
        }},
        {"selector": ".receiver_from_ego", "style": {   # QUEM RECEBE → azul
            "background-color": "#2563eb", "border-color": "#93c5fd", "border-width": "3px"
        }},
        {"selector": ".neighbor", "style": {
            "background-color": "#7c3aed", "border-color": "#c4b5fd", "border-width": "2px"
        }},
        # ARESTAS
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "target-arrow-color": "data(color)",
                "line-color": "data(color)",
                "width": "data(width)",
                "label": "data(label)",
                "font-size": "9px",
                "font-weight": "bold",
                "color": "#e5e7eb",
                "text-outline-width": "1px",
                "text-outline-color": "#0b1220",
                "arrow-scale": 1.35,
                "opacity": 0.95,
            },
        },
        {"selector": ".most-important", "style": {
            "line-color": "#fbbf24", "target-arrow-color": "#fbbf24", "width": "8px", "opacity": 1,
        }},
        # Interações
        {"selector": "node:selected", "style": {"border-color": "#fbbf24", "border-width": "7px"}},
        {"selector": "edge:selected", "style": {"line-color": "#fbbf24", "target-arrow-color": "#fbbf24", "width": "6px"}},
    ]


def create_risk_gauge(risk_score: float, title: str) -> go.Figure:
    """Gauge 0-100 dark."""
    if risk_score > 70: color = "#dc2626"
    elif risk_score > 50: color = "#ea580c"
    elif risk_score > 30: color = "#ca8a04"
    else: color = "#16a34a"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 16, "color": "#e5e7eb"}},
        number={"font": {"size": 26, "color": "#e5e7eb"}},
        gauge={
            "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#0f172a",
            "borderwidth": 1, "bordercolor": "#1f2937",
            "steps": [
                {"range": [0, 30], "color": "rgba(34,197,94,0.15)"},
                {"range": [30, 50], "color": "rgba(202,138,4,0.15)"},
                {"range": [50, 70], "color": "rgba(234,88,12,0.15)"},
                {"range": [70, 100], "color": "rgba(220,38,38,0.18)"},
            ],
            "threshold": {"line": {"color": "#e5e7eb", "width": 2}, "thickness": 0.7, "value": risk_score},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def layout():
    """Layout principal do Desafio 2."""
    df_edges, ids, _ = load_sna_data()
    if df_edges is None or not ids:
        return dbc.Alert("Execute: python -m src.cli.main build-all", color="warning")

    months = sorted(df_edges["year_month"].unique())

    return dbc.Container(fluid=True, children=[
        # Header
        dbc.Row([
            dbc.Col([
                html.H2("🌐 Análise de Risco - Cadeia de Valor",
                        style={"color": "#e5e7eb", "marginBottom": "5px"}),
                html.P("Mapeamento de interdependências e análise de cenários de risco",
                       style={"color": "#94a3b8", "fontSize": "14px"})
            ], md=8),
            dbc.Col([dbc.Badge("SNA + Machine Learning", color="primary", className="float-end mt-3")], md=4)
        ], className="mb-4", style={"borderBottom": "1px solid #1f2937", "paddingBottom": "12px"}),

        # Painel de controle
        dbc.Card([dbc.CardBody([
            html.H5("⚙️ Configurações de Análise", style={"color": "#e5e7eb", "marginBottom": "12px"}),

            dbc.Row([
                dbc.Col([
                    html.Label("🏢 Empresa para Análise"),
                    dcc.Dropdown(
                        id="d2-company",
                        options=[{"label": f"🏢 {i}", "value": i} for i in ids],
                        value=ids[0], clearable=False, className="dash-dropdown-dark"
                    )
                ], md=4),
                dbc.Col([
                    html.Label("💸 Tipo de Fluxo Financeiro"),
                    dcc.RadioItems(
                        id="d2-direction",
                        options=[
                            {"label": DIRECTION_LABELS["in"], "value": "in"},
                            {"label": DIRECTION_LABELS["out"], "value": "out"},
                            {"label": DIRECTION_LABELS["both"], "value": "both"},
                        ],
                        value="both", className="radio-modern", style={"marginTop": "8px"}
                    )
                ], md=4),
                dbc.Col([
                    html.Label("📅 Período de Análise"),
                    dcc.Dropdown(
                        id="d2-month",
                        options=[{"label": f"📅 {m.strftime('%m/%Y')}", "value": m.strftime("%Y-%m")} for m in months],
                        value=months[-1].strftime("%Y-%m"), clearable=False, className="dash-dropdown-dark"
                    )
                ], md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("⏱️ Janela Temporal"),
                    dcc.Slider(
                        id="d2-window", min=1, max=12, step=1, value=3,
                        marks={1: {"label": "1 mês"}, 3: {"label": "3 meses"}, 6: {"label": "6 meses"}, 12: {"label": "1 ano"}},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=4),
                dbc.Col([
                    html.Label("🎯 Principais Relacionamentos"),
                    dcc.Slider(
                        id="d2-topn", min=5, max=50, step=5, value=25,
                        marks={5: {"label": "5"}, 15: {"label": "15"}, 25: {"label": "25"}, 50: {"label": "50"}},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=4),
                dbc.Col([
                    html.Label("💳 Canais de Pagamento"),
                    dcc.Dropdown(
                        id="d2-channels",
                        options=[{"label": CHANNEL_LABELS[ch], "value": ch} for ch in ["TODOS", "PIX", "TED", "BOLETO", "OUTROS"]],
                        value=["TODOS"], multi=True, className="dash-dropdown-dark"
                    )
                ], md=4),
            ])
        ])], className="mb-4"),

        # Métricas de risco
        html.Div(id="d2-risk-metrics", className="mb-4"),

        # Área principal
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🕸️ Mapa da Cadeia de Valor", style={"color": "#e5e7eb", "margin": "0"}),
                        html.Small("Clique nos nós para mais informações", style={"color": "#94a3b8"})
                    ]),
                    dbc.CardBody([
                        # Legenda
                        dbc.Row([dbc.Col(html.Div([
                            html.Span("🏢 "), html.Small("Empresa Focal", style={"color": "#fca5a5"}),
                            html.Span(" | ", style={"color": "#9ca3af"}),
                            html.Span("💰 "), html.Small("Quem Paga", style={"color": "#16a34a"}),
                            html.Span(" | ", style={"color": "#9ca3af"}),
                            html.Span("🎯 "), html.Small("Quem Recebe", style={"color": "#60a5fa"}),
                            html.Span(" | ", style={"color": "#9ca3af"}),
                            html.Span("⭐ "), html.Small("Mais Importante", style={"color": "#f59e0b"}),
                        ], style={"textAlign": "center", "marginBottom": "10px"}))]),

                        cyto.Cytoscape(
                            id="d2-network-graph", elements=[],
                            layout={"name": "preset", "fit": True, "padding": 80, "animate": False},
                            style={"width": "100%", "height": "600px"},
                            minZoom=0.5, maxZoom=1.8, zoom=0.9,  # evita extremos
                            stylesheet=create_enhanced_cytoscape_stylesheet(),
                            responsive=True
                        ),

                        html.Div(id="d2-graph-info",
                                 style={"marginTop": "10px", "fontSize": "12px", "color": "#94a3b8"})
                    ])
                ])
            ], md=8),

            dbc.Col([
                dbc.Card([dbc.CardBody([
                    html.H6("📊 Score de Risco", style={"color": "#e5e7eb", "marginBottom": "10px"}),
                    html.Div(id="d2-risk-gauge")
                ])], className="mb-3"),

                dbc.Card([dbc.CardBody([
                    html.H6("🎯 Relacionamento Crítico", style={"color": "#e5e7eb", "marginBottom": "10px"}),
                    html.Div(id="d2-critical-relationship")
                ])], className="mb-3"),

                dbc.Card([dbc.CardBody([
                    html.H6("⚠️ Cenários de Risco", style={"color": "#e5e7eb", "marginBottom": "10px"}),
                    html.Div(id="d2-risk-scenarios", style={"maxHeight": "300px", "overflowY": "auto"})
                ])])
            ], md=4)
        ], className="mb-4"),

        # Ranking de relacionamentos
        dbc.Card([
            dbc.CardHeader([html.H5("📋 Ranking de Relacionamentos", style={"color": "#e5e7eb", "margin": "0"})]),
            dbc.CardBody([html.Div(id="d2-relationships-table", style={"maxHeight": "400px", "overflowY": "auto"})])
        ], className="mb-4"),

        html.Div(id="d2-diversification-suggestions")
    ])


# ========================= CALLBACK PRINCIPAL =========================

@callback(
    Output("d2-risk-metrics", "children"),
    Output("d2-network-graph", "elements"),
    Output("d2-graph-info", "children"),
    Output("d2-risk-gauge", "children"),
    Output("d2-critical-relationship", "children"),
    Output("d2-risk-scenarios", "children"),
    Output("d2-relationships-table", "children"),
    Output("d2-diversification-suggestions", "children"),
    Input("d2-company", "value"),
    Input("d2-direction", "value"),
    Input("d2-month", "value"),
    Input("d2-window", "value"),
    Input("d2-topn", "value"),
    Input("d2-channels", "value"),
)
def update_network_analysis(company_id, direction, month, window, topn, channels):
    """Callback principal para análise de rede."""
    df_edges, _, _ = load_sna_data()
    if df_edges is None or not company_id:
        empty_returns = [html.Div("Sem dados")] * 8
        return empty_returns

    # 1) Grafo ego
    nodes_df, edges_df, summary = build_ego_network(
        df_edges=df_edges, company_id=company_id, year_month=month,
        months_window=window, direction=direction, top_n=topn,
        channels=channels or ["TODOS"]
    )

    # 2) Elements (preset radial anti-aglomeração)
    cyto_elements = create_enhanced_cyto_elements(nodes_df, edges_df, company_id)

    # 3) Relatório de risco (respeita janela)
    risk_report = risk_analyzer.generate_monthly_risk_report(
        df_edges, company_id, months_window=window
    )

    # 4) Cards / infos
    risk_metrics = create_risk_metrics_cards(risk_report)
    graph_info = f"🔢 {summary['n_nodes']} empresas conectadas | 🔗 {summary['n_edges']} relacionamentos | 📊 Densidade: {summary['density']:.2%} | ⚡ Centralidade calculada"

    risk_gauge = dcc.Graph(
        figure=create_risk_gauge(risk_report["overall_risk_score"], "Risco Geral"),
        config={"displayModeBar": False},
    )
    critical_rel = create_critical_relationship_display(edges_df, company_id)
    scenarios_display = create_risk_scenarios_display(
        risk_report["risk_scenarios"], risk_report.get("concentration_metrics", {})
    )
    relationships_table = create_relationships_table(edges_df, nodes_df, company_id)
    diversification = create_diversification_suggestions(risk_report["diversification_suggestions"])

    return (risk_metrics, cyto_elements, graph_info, risk_gauge,
            critical_rel, scenarios_display, relationships_table, diversification)


# ========================= HELPERS DE UI =========================

def _ring_positions(nodes_df: pd.DataFrame, ego_id: str) -> dict[str, dict]:
    """
    Posiciona nós em anéis: payers (in) num anel; receivers (out) em outro;
    neighbors em um anel externo. Mais raio = mais espaço (menos aglomeração).
    """
    if nodes_df.empty:
        return {}

    pos: dict[str, dict] = {}
    pos[ego_id] = {"x": 0.0, "y": 0.0}

    groups = {
        "payer_to_ego": {"r": 520.0, "offset": math.pi / 14},       # QUEM PAGA (espalha mais)
        "receiver_from_ego": {"r": 380.0, "offset": 0.0},
        "neighbor": {"r": 700.0, "offset": math.pi / 8},
    }

    for role, conf in groups.items():
        ring = nodes_df[nodes_df["role"] == role].sort_values("strength", ascending=False)
        n = len(ring)
        if n == 0:
            continue
        # deixa um “respiro”: usa n+1 pra dar gap
        step = 2 * math.pi / (n + 1)
        for i, (_, row) in enumerate(ring.iterrows()):
            angle = conf["offset"] + i * step
            x = conf["r"] * math.cos(angle)
            y = conf["r"] * math.sin(angle)
            pos[row["node_id"]] = {"x": float(x), "y": float(y)}

    return pos


def create_enhanced_cyto_elements(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, ego_id: str):
    """Elementos Cytoscape com **layout radial preset** para evitar 'bolo'."""
    if nodes_df.empty:
        return []

    # Tamanhos
    s_min = nodes_df["strength"].min()
    s_max = nodes_df["strength"].max()

    def scale_size(v, lo=36, hi=84):
        if s_max <= s_min:
            return (lo + hi) / 2
        return lo + (float(v) - s_min) / (s_max - s_min) * (hi - lo)

    # Posições radiais
    preset = _ring_positions(nodes_df, ego_id)

    # Nós
    nodes = []
    for _, row in nodes_df.iterrows():
        nodes.append({
            "data": {
                "id": row["node_id"],
                "label": f"🏢 {row['node_id']}" if row["node_id"] == ego_id else row["node_id"],
                "size": scale_size(row["strength"]),
                "strength": float(row["strength"]),
                "pagerank": float(row["pagerank"]),
                "betweenness": float(row["betweenness"]),
                "role": row["role"],
            },
            "classes": row["role"] if row["node_id"] != ego_id else "ego",
            "position": preset.get(row["node_id"], {"x": 0.0, "y": 0.0}),
        })

    # Arestas
    edges = []
    if not edges_df.empty:
        e_min = edges_df["weight"].min()
        e_max = edges_df["weight"].max()
        top_idx = edges_df["weight"].idxmax()

        for idx, row in edges_df.iterrows():
            width = 2 + ((row["weight"] - e_min) / (e_max - e_min)) * 6 if e_max > e_min else 4
            if row["src"] == ego_id:
                color = "#2563eb"  # empresa PAGA → saída
                label = f"💸 {format_currency_br(row['weight'])}"
            else:
                color = "#16a34a"  # empresa RECEBE → entrada
                label = f"💰 {format_currency_br(row['weight'])}"

            edges.append({
                "data": {
                    "source": row["src"], "target": row["dst"], "weight": float(row["weight"]),
                    "width": width, "color": color, "label": label
                },
                "classes": "most-important" if idx == top_idx else "edge"
            })

    return nodes + edges


def create_risk_metrics_cards(risk_report: Dict[str, Any]):
    conc_in = risk_report["concentration_metrics"]["recebimentos"]
    conc_out = risk_report["concentration_metrics"]["pagamentos"]

    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("📥 Concentração Recebimentos", className="kpi-label"),
            html.H4(f"{(conc_in.get('top_partner_share', 0)*100):.1f}%", className="kpi-value", style={"color": "#22c55e"}),
            html.Small(f"HHI: {conc_in.get('hhi', 0):.3f}", className="kpi-sub")
        ]), className="kpi-card"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("📤 Concentração Pagamentos", className="kpi-label"),
            html.H4(f"{(conc_out.get('top_partner_share', 0)*100):.1f}%", className="kpi-value", style={"color": "#ef4444"}),
            html.Small(f"HHI: {conc_out.get('hhi', 0):.3f}", className="kpi-sub")
        ]), className="kpi-card"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("🤝 Total de Parceiros", className="kpi-label"),
            html.H4(f"{risk_report['current_partners_count']}", className="kpi-value", style={"color": "#a78bfa"}),
            html.Small("empresas conectadas", className="kpi-sub")
        ]), className="kpi-card"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("⚠️ Nível de Risco", className="kpi-label"),
            html.H4(risk_report["risk_classification"], className="kpi-value",
                    style={"color": {"BAIXO":"#22c55e","MÉDIO":"#ca8a04","ALTO":"#ea580c","CRÍTICO":"#dc2626"}.get(risk_report["risk_classification"], "#94a3b8")}),
            html.Small(f"Score: {risk_report['overall_risk_score']:.0f}/100", className="kpi-sub")
        ]), className="kpi-card"), md=3),
    ])


def create_critical_relationship_display(edges_df: pd.DataFrame, company_id: str):
    if edges_df.empty:
        return html.Div("Sem relacionamentos", style={"color": "#94a3b8"})
    top = edges_df.loc[edges_df["weight"].idxmax()]
    if top["src"] == company_id:
        icon, text, partner = "📤", "PAGA PARA", top["dst"]
    else:
        icon, text, partner = "📥", "RECEBE DE", top["src"]
    return html.Div([
        html.Div([html.Span(icon, style={"fontSize": "20px", "marginRight": "8px"}),
                  html.Strong(text, style={"color": "#e5e7eb"})], className="mb-2"),
        html.Div([html.Strong("Parceiro: ", style={"color": "#94a3b8"}), html.Span(partner, style={"color": "#60a5fa"})], className="mb-1"),
        html.Div([html.Strong("Valor: ", style={"color": "#94a3b8"}),
                  html.Span(format_currency_br(top["weight"]), style={"color": "#22c55e", "fontSize": "18px", "fontWeight": "bold"})],
                 className="mb-2"),
        html.Hr(style={"borderColor": "#1f2937"}),
        html.Div([html.Strong("💡 Impacto da Perda:", style={"color": "#f59e0b"}),
                  html.P("Relacionamento crítico - desenvolva alternativas",
                         style={"color": "#cbd5e1", "fontSize": "13px", "marginTop": "5px"})])
    ])


def _pct(v) -> float:
    """Converte para % (0–100) e protege contra inversão/escala."""
    try:
        x = float(v)
    except Exception:
        return 0.0
    if 0 <= x <= 1:
        return x * 100.0
    if x < 0:
        return 0.0
    if x > 100:
        return 100.0
    return x


def create_risk_scenarios_display(scenarios: List, conc_metrics: Dict[str, Any]):
    """Garante % correta (sem inversão) usando concentração como fonte primária."""
    if not scenarios:
        return html.Div("Nenhum cenário de risco identificado", style={"color": "#94a3b8"})

    # fontes de verdade para % (evita rollback que invertia)
    in_share = _pct((conc_metrics.get("recebimentos", {}) or {}).get("top_partner_share", 0))
    out_share = _pct((conc_metrics.get("pagamentos", {}) or {}).get("top_partner_share", 0))
    top3_share = _pct((conc_metrics.get("recebimentos", {}) or {}).get("top3_share", 0))

    risk_color = {"BAIXO": "#22c55e", "MÉDIO": "#ca8a04", "ALTO": "#ea580c", "CRÍTICO": "#dc2626"}
    items = []
    for sc in scenarios[:3]:
        # escolhe com base no nome, mas dá fallback pro valor do cenário
        name = sc.scenario_name or ""
        if "Cliente" in name:
            impact = in_share or _pct(sc.impact_percentage)
        elif "Fornecedor" in name:
            impact = out_share or _pct(sc.impact_percentage)
        elif "Top 3" in name:
            impact = top3_share or _pct(sc.impact_percentage)
        else:
            impact = _pct(sc.impact_percentage)

        items.append(html.Div([
            html.Div([
                html.Strong(name, style={"color": "#e5e7eb"}),
                dbc.Badge(sc.risk_level, color=("danger" if sc.risk_level == "CRÍTICO" else "warning"),
                          className="float-end")
            ], className="mb-2"),
            html.P(sc.description, style={"color": "#94a3b8", "fontSize": "12px"}),
            html.Div([html.Strong("Impacto: ", style={"color": "#94a3b8", "fontSize": "11px"}),
                      html.Span(f"{impact:.1f}%", style={"color": risk_color.get(sc.risk_level, "#94a3b8"),
                                                         "fontWeight": "bold"})]),
            html.Hr(style={"borderColor": "#1f2937", "margin": "10px 0"})
        ], style={"marginBottom": "15px"}))
    return html.Div(items)


def create_relationships_table(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, company_id: str):
    if edges_df.empty:
        return html.Div("Sem relacionamentos", style={"color": "#94a3b8"})

    rows = []
    df = edges_df.copy()
    for _, row in df.head(15).iterrows():
        outflow = row["src"] == company_id
        partner = row["dst"] if outflow else row["src"]
        rows.append(html.Tr([
            html.Td("📤" if outflow else "📥", style={"textAlign": "center", "fontSize": "16px"}),
            html.Td(partner, style={"fontFamily": "monospace", "color": "#e5e7eb"}),
            html.Td(format_currency_br(row["weight"]), style={"textAlign": "right", "fontWeight": "bold", "color": "#e5e7eb"}),
            html.Td(dbc.Badge("Alto Risco" if row["weight"] > df["weight"].quantile(0.8) else "Normal",
                              color="warning" if row["weight"] > df["weight"].quantile(0.8) else "secondary"))
        ]))

    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Fluxo", style={"color": "#94a3b8"}),
            html.Th("Parceiro", style={"color": "#94a3b8"}),
            html.Th("Valor", style={"color": "#94a3b8", "textAlign": "right"}),
            html.Th("Risco", style={"color": "#94a3b8"}),
        ])),
        html.Tbody(rows)
    ], striped=True, hover=True, size="sm")


def create_diversification_suggestions(suggestions: List[Dict[str, Any]]):
    if not suggestions:
        return html.Div()

    cards = []
    for s in suggestions:
        cards.append(dbc.Col(dbc.Card(dbc.CardBody([
            html.H6(f"🤝 {s['partner_id']}", style={"color": "#60a5fa"}),
            html.P(s["recommendation_reason"], style={"fontSize": "12px", "color": "#94a3b8"}),
            html.Small(f"Vol. médio: {format_currency_br(s['avg_transaction_volume'])}", style={"color": "#22c55e"})
        ])), md=4))
    return dbc.Card([
        dbc.CardHeader([html.H5("🎯 Sugestões de Diversificação", style={"color": "#e5e7eb", "margin": "0"})]),
        dbc.CardBody([html.P("Novos parceiros recomendados para reduzir concentração de risco:",
                             style={"color": "#94a3b8", "marginBottom": "15px"}),
                      dbc.Row(cards)])
    ], className="mt-4")
