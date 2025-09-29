# src/ui/pages/desafio2_enhanced.py
"""
ANÁLISE DE RISCO DE CADEIA DE VALOR - Interface Profissional
Sistema completo de SNA com análise de interdependência e cenários de risco
"""
from __future__ import annotations
from dash import register_page, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.sna.graph import build_ego_network, to_cyto_elements
from src.features.risk_analysis import ValueChainRiskAnalyzer
from src.ui.components.metrics_formatter import format_currency_br, format_percentage

register_page(__name__, path="/desafio2", title="Análise de Risco - Cadeia de Valor")
cyto.load_extra_layouts()

# Paths
REL_PATH = Path("reports/exports/relations_latest.parquet")
B2_PATH = Path("data/processed/base2/base2.parquet")
F_EMP_PATH = Path("reports/exports/f_empresa_mes.parquet")

# Risk analyzer
risk_analyzer = ValueChainRiskAnalyzer()

# Mapeamento de terminologia para stakeholders
DIRECTION_LABELS = {
    "in": "📥 Recebimentos (Quem me paga)",
    "out": "📤 Pagamentos (Quem eu pago)", 
    "both": "🔄 Ambos os Fluxos"
}

ROLE_LABELS = {
    "ego": "🏢 Empresa Focal",
    "payer_to_ego": "💰 Fornecedores/Clientes",
    "receiver_from_ego": "🎯 Beneficiários/Parceiros",
    "neighbor": "🔗 Conectores da Rede"
}

CHANNEL_LABELS = {
    "PIX": "💳 PIX",
    "TED": "🏦 TED",
    "BOLETO": "📄 Boleto",
    "OUTROS": "📋 Outros",
    "TODOS": "🌐 Todos os Canais"
}

def load_sna_data():
    """Carrega dados para análise SNA."""
    try:
        df_edges = pd.read_parquet(B2_PATH) if B2_PATH.exists() else None
        if df_edges is not None:
            df_edges["year_month"] = (
                pd.to_datetime(df_edges["DT_REFE"])
          .dt.to_period("M")
          .dt.to_timestamp()
    )
            df_edges["DS_TRAN"] = df_edges["DS_TRAN"].astype(str).str.upper().str.strip()
        
        ids = []
        if F_EMP_PATH.exists():
            f = pd.read_parquet(F_EMP_PATH)
            ids = sorted(f["ID"].unique().tolist())
        
        relations = pd.read_parquet(REL_PATH) if REL_PATH.exists() else None
        
        return df_edges, ids, relations
    except Exception:
        return None, [], None

def create_enhanced_cytoscape_stylesheet():
    """Cria stylesheet aprimorado para o Cytoscape."""
    return [
        # === ESTILOS DE NÓS ===
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "font-size": "12px",
                "font-weight": "bold",
                "color": "#1a202c",
                "text-valign": "center",
                "text-halign": "center",
                "background-color": "#64748b",
                "width": "data(size)",
                "height": "data(size)",
                "border-width": "3px",
                "border-color": "#e53e3e",
                "text-outline-width": "1px",
                "text-outline-color": "#ffffff",
                "overlay-opacity": 0.1
            }
        },
        
        # Empresa focal (centro da análise)
        {
            "selector": ".ego",
            "style": {
                "background-color": "#dc2626",  # Vermelho vibrante
                "border-color": "#fca5a5",
                "border-width": "5px",
                "font-size": "14px",
                "font-weight": "900",
                "text-outline-width": "2px",
                "box-shadow": "0px 0px 20px #dc2626"
            }
        },
        
        # Quem paga para a empresa (fornecedores/clientes)
        {
            "selector": ".payer_to_ego",
            "style": {
                "background-color": "#16a34a",  # Verde - entrada de dinheiro
                "border-color": "#86efac",
                "border-width": "3px",
                "font-size": "11px"
            }
        },
        
        # Quem recebe da empresa (beneficiários)
        {
            "selector": ".receiver_from_ego",
            "style": {
                "background-color": "#2563eb",  # Azul - saída de dinheiro
                "border-color": "#93c5fd", 
                "border-width": "3px",
                "font-size": "11px"
            }
        },
        
        # Conexões indiretas
        {
            "selector": ".neighbor",
            "style": {
                "background-color": "#7c3aed",  # Roxo - rede estendida
                "border-color": "#c4b5fd",
                "border-width": "2px",
                "font-size": "10px"
            }
        },
        
        # === ESTILOS DE ARESTAS ===
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
                "color": "#1a202c",
                "text-outline-width": "1px",
                "text-outline-color": "#ffffff",
                "arrow-scale": 1.5,
                "opacity": 0.8
            }
        },
        
        # Destaque para relacionamento mais importante
        {
            "selector": ".most-important",
            "style": {
                "line-color": "#fbbf24",  # Amarelo/dourado
                "target-arrow-color": "#fbbf24",
                "width": "8px",
                "opacity": 1,
                "z-index": 10,
                "overlay-opacity": 0.3,
                "overlay-color": "#fbbf24"
            }
        },
        
        # === INTERAÇÕES ===
        {
            "selector": "node:selected",
            "style": {
                "border-color": "#fbbf24",
                "border-width": "6px",
                "box-shadow": "0px 0px 30px #fbbf24"
            }
        },
        
        {
            "selector": "edge:selected",
            "style": {
                "line-color": "#fbbf24",
                "target-arrow-color": "#fbbf24",
                "width": "6px",
                "opacity": 1
            }
        }
    ]

def create_risk_gauge(risk_score: float, title: str) -> go.Figure:
    """Cria gauge de risco (0-100)."""
    # Definir cor baseada no score
    if risk_score > 70:
        color = "#dc2626"  # Vermelho
    elif risk_score > 50:
        color = "#ea580c"  # Laranja
    elif risk_score > 30:
        color = "#ca8a04"  # Amarelo
    else:
        color = "#16a34a"  # Verde
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#e5e7eb'}},
        number={'font': {'size': 24, 'color': '#ffffff'}},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "#e5e7eb",
                'tickfont': {'color': '#e5e7eb'}
            },
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "#1e293b",
            'borderwidth': 2,
            'bordercolor': "#475569",
            'steps': [
                {'range': [0, 30], 'color': '#065f46'},    # Verde escuro
                {'range': [30, 50], 'color': '#a16207'},   # Amarelo escuro  
                {'range': [50, 70], 'color': '#c2410c'},   # Laranja escuro
                {'range': [70, 100], 'color': '#7f1d1d'}   # Vermelho escuro
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 4},
                'thickness': 0.8,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(255,255,255,0)",
        font={'color': "#1a202c"}
    )
    
    return fig

def layout():
    """Layout principal do Desafio 2."""
    df_edges, ids, relations = load_sna_data()
    
    if df_edges is None or not ids:
        return dbc.Alert("Execute: python -m src.cli.main build-all", color="warning")

    months = sorted(df_edges["year_month"].unique())

    return dbc.Container(fluid=True, children=[
        # === HEADER PROFISSIONAL ===
        dbc.Row([
            dbc.Col([
                html.H2("🌐 Análise de Risco - Cadeia de Valor", 
                       style={"color": "#e5e7eb", "marginBottom": "5px"}),
                html.P("Mapeamento de interdependências e análise de cenários de risco", 
                      style={"color": "#9ca3af", "fontSize": "14px"})
            ], md=8),
            dbc.Col([
                dbc.Badge("SNA + Machine Learning", color="info", className="float-end mt-3")
            ], md=4)
        ], className="mb-4", style={"borderBottom": "2px solid #374151", "paddingBottom": "15px"}),
        
        # === PAINEL DE CONTROLE ===
        dbc.Card([
            dbc.CardBody([
                html.H5("⚙️ Configurações de Análise", style={"color": "#e5e7eb", "marginBottom": "15px"}),
                
                # Linha 1: Seletores principais
                dbc.Row([
                    dbc.Col([
                        html.Label("🏢 Empresa para Análise", style={"color": "#9ca3af", "fontSize": "12px"}),
                        dcc.Dropdown(
                            id="d2-company",
                            options=[{"label": f"🏢 {id_val}", "value": id_val} for id_val in ids],
                            value=ids[0],
                            clearable=False,
                            className="dash-dropdown-dark"
                        )
                    ], md=4),
                    
                    dbc.Col([
                        html.Label("💸 Tipo de Fluxo Financeiro", style={"color": "#9ca3af", "fontSize": "12px"}),
                        dcc.RadioItems(
                            id="d2-direction",
                            options=[
                                {"label": DIRECTION_LABELS["in"], "value": "in"},
                                {"label": DIRECTION_LABELS["out"], "value": "out"},
                                {"label": DIRECTION_LABELS["both"], "value": "both"}
                            ],
                            value="both",
                            className="radio-modern",
                            style={"marginTop": "8px"}
                        )
                    ], md=4),
                    
                    dbc.Col([
                        html.Label("📅 Período de Análise", style={"color": "#9ca3af", "fontSize": "12px"}),
                        dcc.Dropdown(
                            id="d2-month",
                            options=[{"label": f"📅 {m.strftime('%m/%Y')}", "value": m.strftime("%Y-%m")} 
                                   for m in months],
                            value=months[-1].strftime("%Y-%m"),
                            clearable=False,
                            className="dash-dropdown-dark"
                        )
                    ], md=4)
                ], className="mb-3"),
                
                # Linha 2: Filtros avançados
                dbc.Row([
                    dbc.Col([
                        html.Label("⏱️ Janela Temporal", style={"color": "#9ca3af", "fontSize": "12px"}),
                        dcc.Slider(
                            id="d2-window",
                            min=1, max=12, step=1, value=3,
                            marks={
                                1: {"label": "1 mês", "style": {"color": "#9ca3af"}},
                                3: {"label": "3 meses", "style": {"color": "#9ca3af"}},
                                6: {"label": "6 meses", "style": {"color": "#9ca3af"}},
                                12: {"label": "1 ano", "style": {"color": "#9ca3af"}}
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], md=4),
                    
                    dbc.Col([
                        html.Label("🎯 Principais Relacionamentos", style={"color": "#9ca3af", "fontSize": "12px"}),
                        dcc.Slider(
                            id="d2-topn",
                            min=5, max=50, step=5, value=25,
                            marks={
                                5: {"label": "5", "style": {"color": "#9ca3af"}},
                                15: {"label": "15", "style": {"color": "#9ca3af"}},
                                25: {"label": "25", "style": {"color": "#9ca3af"}},
                                50: {"label": "50", "style": {"color": "#9ca3af"}}
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], md=4),
                    
                    dbc.Col([
                        html.Label("💳 Canais de Pagamento", style={"color": "#9ca3af", "fontSize": "12px"}),
                        dcc.Dropdown(
                            id="d2-channels",
                            options=[{"label": CHANNEL_LABELS[ch], "value": ch} 
                                   for ch in ["TODOS", "PIX", "TED", "BOLETO", "OUTROS"]],
                            value=["TODOS"],
                            multi=True,
                            className="dash-dropdown-dark"
                        )
                    ], md=4)
                ], className="mb-2")
            ])
        ], className="mb-4"),
        
        # === MÉTRICAS DE RISCO ===
        html.Div(id="d2-risk-metrics", className="mb-4"),
        
        # === ÁREA PRINCIPAL ===
        dbc.Row([
            # Gráfico de rede
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🕸️ Mapa da Cadeia de Valor", 
                               style={"color": "#e5e7eb", "margin": "0"}),
                        html.Small("Clique nos nós para mais informações", 
                                 style={"color": "#9ca3af"})
                    ]),
                    dbc.CardBody([
                        # Legenda do gráfico
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Span("🏢", style={"fontSize": "16px", "marginRight": "5px"}),
                                    html.Small("Empresa Focal", style={"color": "#dc2626"}),
                                    html.Span(" | ", style={"margin": "0 8px", "color": "#6b7280"}),
                                    html.Span("💰", style={"fontSize": "16px", "marginRight": "5px"}),
                                    html.Small("Quem Paga", style={"color": "#16a34a"}),
                                    html.Span(" | ", style={"margin": "0 8px", "color": "#6b7280"}),
                                    html.Span("🎯", style={"fontSize": "16px", "marginRight": "5px"}),
                                    html.Small("Quem Recebe", style={"color": "#2563eb"}),
                                    html.Span(" | ", style={"margin": "0 8px", "color": "#6b7280"}),
                                    html.Span("⭐", style={"fontSize": "16px", "marginRight": "5px"}),
                                    html.Small("Mais Importante", style={"color": "#fbbf24"})
                                ], style={"textAlign": "center", "marginBottom": "10px"})
                            ], md=12)
                        ]),
                        
                        # Gráfico Cytoscape aprimorado
        cyto.Cytoscape(
                            id="d2-network-graph",
            elements=[],
                            layout={
                                "name": "cose-bilkent",
                                "idealEdgeLength": 100,
                                "nodeOverlap": 20,
                                "refresh": 20,
                                "fit": True,
                                "padding": 30,
                                "randomize": False,
                                "componentSpacing": 100,
                                "nodeRepulsion": 400000,
                                "edgeElasticity": 100,
                                "nestingFactor": 5,
                                "gravity": 80,
                                "numIter": 1000,
                                "initialTemp": 200,
                                "coolingFactor": 0.95,
                                "minTemp": 1.0
                            },
                            style={
                                "width": "100%",
                                "height": "600px",
                                "backgroundColor": "#ffffff",
                                "border": "2px solid #e53e3e",
                                "borderRadius": "12px"
                            },
                            stylesheet=create_enhanced_cytoscape_stylesheet(),
                            responsive=True
                        ),
                        
                        # Informações do grafo
                        html.Div(id="d2-graph-info", 
                               style={"marginTop": "10px", "fontSize": "12px", "color": "#9ca3af"})
                    ])
                ])
            ], md=8),
            
            # Painel lateral de análise
            dbc.Col([
                # Score de risco
                dbc.Card([
                    dbc.CardBody([
                        html.H6("📊 Score de Risco", style={"color": "#e5e7eb", "marginBottom": "10px"}),
                        html.Div(id="d2-risk-gauge")
                    ])
                ], className="mb-3"),
                
                # Relacionamento principal
                dbc.Card([
                    dbc.CardBody([
                        html.H6("🎯 Relacionamento Crítico", style={"color": "#e5e7eb", "marginBottom": "10px"}),
                        html.Div(id="d2-critical-relationship")
                    ])
                ], className="mb-3"),
                
                # Cenários de risco
                dbc.Card([
                    dbc.CardBody([
                        html.H6("⚠️ Cenários de Risco", style={"color": "#e5e7eb", "marginBottom": "10px"}),
                        html.Div(id="d2-risk-scenarios", style={"maxHeight": "300px", "overflowY": "auto"})
                    ])
                ])
            ], md=4)
        ], className="mb-4"),
        
        # === TABELA DE RELACIONAMENTOS ===
        dbc.Card([
            dbc.CardHeader([
                html.H5("📋 Ranking de Relacionamentos", style={"color": "#e5e7eb", "margin": "0"})
            ]),
            dbc.CardBody([
                html.Div(id="d2-relationships-table", style={"maxHeight": "400px", "overflowY": "auto"})
            ])
        ], className="mb-4"),
        
        # === SUGESTÕES DE DIVERSIFICAÇÃO ===
        html.Div(id="d2-diversification-suggestions")
    ])

# === CALLBACKS ===

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
    Input("d2-channels", "value")
)
def update_network_analysis(company_id, direction, month, window, topn, channels):
    """Callback principal para análise de rede."""
    df_edges, _, relations = load_sna_data()
    
    if df_edges is None or not company_id:
        empty_returns = [html.Div("Sem dados")] * 8
        return empty_returns
    
    try:
        # === 1. CONSTRUIR REDE ===
        nodes_df, edges_df, summary = build_ego_network(
        df_edges=df_edges,
            company_id=company_id,
        year_month=month,
            months_window=window,
        direction=direction,
            top_n=topn,
        channels=channels or ["TODOS"]
    )
        
        # === 2. ELEMENTOS CYTOSCAPE COM MELHORIAS ===
        cyto_elements = create_enhanced_cyto_elements(nodes_df, edges_df, company_id)
        
        # === 3. ANÁLISE DE RISCO ===
        risk_report = risk_analyzer.generate_monthly_risk_report(
            df_edges, company_id, window
        )
        
        # === 4. MÉTRICAS DE RISCO ===
        risk_metrics = create_risk_metrics_cards(risk_report)
        
        # === 5. INFORMAÇÕES DO GRAFO ===
        graph_info = f"""
        🔢 {summary['n_nodes']} empresas conectadas | 
        🔗 {summary['n_edges']} relacionamentos | 
        📊 Densidade: {summary['density']:.2%} |
        ⚡ Centralidade calculada
        """
        
        # === 6. GAUGE DE RISCO ===
        risk_gauge = dcc.Graph(
            figure=create_risk_gauge(risk_report["overall_risk_score"], "Risco Geral"),
            config={'displayModeBar': False}
        )
        
        # === 7. RELACIONAMENTO CRÍTICO ===
        critical_rel = create_critical_relationship_display(edges_df, company_id)
        
        # === 8. CENÁRIOS DE RISCO ===
        scenarios_display = create_risk_scenarios_display(risk_report["risk_scenarios"])
        
        # === 9. TABELA DE RELACIONAMENTOS ===
        relationships_table = create_relationships_table(edges_df, nodes_df, company_id)
        
        # === 10. SUGESTÕES DE DIVERSIFICAÇÃO ===
        diversification = create_diversification_suggestions(risk_report["diversification_suggestions"])
        
        return (
            risk_metrics,
            cyto_elements,
            graph_info,
            risk_gauge,
            critical_rel,
            scenarios_display,
            relationships_table,
            diversification
        )
        
    except Exception as e:
        error_msg = html.Div(f"Erro na análise: {str(e)}", style={"color": "#ef4444"})
        empty_returns = [error_msg] + [html.Div("Erro")] * 7
        return empty_returns

def create_enhanced_cyto_elements(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, ego_id: str):
    """Cria elementos Cytoscape com melhorias visuais."""
    if nodes_df.empty:
        return []
    
    # Escalar tamanhos dos nós
    s_min = nodes_df["strength"].min()
    s_max = nodes_df["strength"].max()
    
    def scale_size(strength, min_size=30, max_size=80):
        if s_max <= s_min:
            return (min_size + max_size) / 2
        normalized = (strength - s_min) / (s_max - s_min)
        return min_size + normalized * (max_size - min_size)
    
    # Criar nós
    nodes = []
    for _, row in nodes_df.iterrows():
        size = scale_size(row["strength"])
        
        # Label mais informativo
        if row["node_id"] == ego_id:
            label = f"🏢 {row['node_id']}"
        else:
            label = row["node_id"]
        
        nodes.append({
            "data": {
                "id": row["node_id"],
                "label": label,
                "size": size,
                "strength": float(row["strength"]),
                "pagerank": float(row["pagerank"]),
                "betweenness": float(row["betweenness"]),
                "role": row["role"]
            },
            "classes": row["role"]
        })
    
    # Criar arestas com cores baseadas no valor
    edges = []
    if not edges_df.empty:
        e_min = edges_df["weight"].min()
        e_max = edges_df["weight"].max()
        
        # Identificar aresta mais importante
        most_important_idx = edges_df["weight"].idxmax()
        
        for idx, row in edges_df.iterrows():
            # Escalar largura
            if e_max > e_min:
                width = 2 + ((row["weight"] - e_min) / (e_max - e_min)) * 6
            else:
                width = 4
            
            # Cor baseada na direção do fluxo
            if row["src"] == ego_id:
                color = "#2563eb"  # Azul - empresa paga
                arrow_label = f"💸 {format_currency_br(row['weight'])}"
            else:
                color = "#16a34a"  # Verde - empresa recebe
                arrow_label = f"💰 {format_currency_br(row['weight'])}"
            
            # Classe especial para relacionamento mais importante
            classes = "most-important" if idx == most_important_idx else "edge"
            
            edges.append({
                "data": {
                    "source": row["src"],
                    "target": row["dst"],
                    "weight": float(row["weight"]),
                    "width": width,
                    "color": color,
                    "label": arrow_label
                },
                "classes": classes
            })
    
    return nodes + edges

def create_risk_metrics_cards(risk_report: Dict[str, Any]):
    """Cria cards de métricas de risco."""
    conc_in = risk_report["concentration_metrics"]["recebimentos"]
    conc_out = risk_report["concentration_metrics"]["pagamentos"]
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("📥 Concentração Recebimentos", className="kpi-label"),
                    html.H4(f"{conc_in.get('top_partner_share', 0):.1%}", className="kpi-value",
                           style={"color": "#16a34a"}),
                    html.Small(f"HHI: {conc_in.get('hhi', 0):.3f}", className="kpi-sub")
                ])
            ], className="kpi-card")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("📤 Concentração Pagamentos", className="kpi-label"),
                    html.H4(f"{conc_out.get('top_partner_share', 0):.1%}", className="kpi-value",
                           style={"color": "#dc2626"}),
                    html.Small(f"HHI: {conc_out.get('hhi', 0):.3f}", className="kpi-sub")
                ])
            ], className="kpi-card")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("🤝 Total de Parceiros", className="kpi-label"),
                    html.H4(f"{risk_report['current_partners_count']}", className="kpi-value",
                           style={"color": "#7c3aed"}),
                    html.Small("empresas conectadas", className="kpi-sub")
                ])
            ], className="kpi-card")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("⚠️ Nível de Risco", className="kpi-label"),
                    html.H4(risk_report["risk_classification"], className="kpi-value",
                           style={"color": get_risk_color(risk_report["risk_classification"])}),
                    html.Small(f"Score: {risk_report['overall_risk_score']:.0f}/100", className="kpi-sub")
                ])
            ], className="kpi-card")
        ], md=3)
    ])

def get_risk_color(risk_level: str) -> str:
    """Retorna cor baseada no nível de risco."""
    colors = {
        "BAIXO": "#16a34a",
        "MÉDIO": "#ca8a04", 
        "ALTO": "#ea580c",
        "CRÍTICO": "#dc2626"
    }
    return colors.get(risk_level, "#6b7280")

def create_critical_relationship_display(edges_df: pd.DataFrame, company_id: str):
    """Cria display do relacionamento mais crítico."""
    if edges_df.empty:
        return html.Div("Sem relacionamentos", style={"color": "#9ca3af"})
    
    # Relacionamento mais importante por valor
    top_relationship = edges_df.loc[edges_df["weight"].idxmax()]
    
    # Determinar direção do fluxo
    if top_relationship["src"] == company_id:
        direction_icon = "📤"
        direction_text = "PAGA PARA"
        partner = top_relationship["dst"]
    else:
        direction_icon = "📥"
        direction_text = "RECEBE DE"
        partner = top_relationship["src"]
    
    return html.Div([
        html.Div([
            html.Span(direction_icon, style={"fontSize": "20px", "marginRight": "8px"}),
            html.Strong(direction_text, style={"color": "#e5e7eb"})
        ], className="mb-2"),
        
        html.Div([
            html.Strong("Parceiro: ", style={"color": "#9ca3af"}),
            html.Span(partner, style={"color": "#60a5fa"})
        ], className="mb-1"),
        
        html.Div([
            html.Strong("Valor: ", style={"color": "#9ca3af"}),
            html.Span(format_currency_br(top_relationship["weight"]), 
                     style={"color": "#22c55e", "fontSize": "18px", "fontWeight": "bold"})
        ], className="mb-2"),
        
        html.Hr(style={"borderColor": "#374151"}),
        
        html.Div([
            html.Strong("💡 Impacto da Perda:", style={"color": "#fbbf24"}),
            html.P("Relacionamento crítico - desenvolva alternativas", 
                  style={"color": "#e5e7eb", "fontSize": "13px", "marginTop": "5px"})
        ])
    ])

def create_risk_scenarios_display(scenarios: List):
    """Cria display dos cenários de risco."""
    if not scenarios:
        return html.Div("Nenhum cenário de risco identificado", style={"color": "#9ca3af"})
    
    scenario_items = []
    
    for scenario in scenarios[:3]:  # Top 3 cenários
        risk_color = get_risk_color(scenario.risk_level)
        
        scenario_items.append(
            html.Div([
                html.Div([
                    html.Strong(scenario.scenario_name, style={"color": "#e5e7eb"}),
                    dbc.Badge(scenario.risk_level, color="danger" if scenario.risk_level == "CRÍTICO" else "warning",
                             className="float-end")
                ], className="mb-2"),
                
                html.P(scenario.description, style={"color": "#9ca3af", "fontSize": "12px"}),
                
                html.Div([
                    html.Strong("Impacto: ", style={"color": "#9ca3af", "fontSize": "11px"}),
                    html.Span(f"{scenario.impact_percentage:.1f}%", 
                             style={"color": risk_color, "fontWeight": "bold"})
                ]),
                
                html.Hr(style={"borderColor": "#374151", "margin": "10px 0"})
            ], style={"marginBottom": "15px"})
        )
    
    return html.Div(scenario_items)

def create_relationships_table(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, company_id: str):
    """Cria tabela melhorada de relacionamentos."""
    if edges_df.empty:
        return html.Div("Sem relacionamentos", style={"color": "#9ca3af"})
    
    # Enriquecer dados da tabela
    table_data = edges_df.copy()
    table_data["direction_icon"] = table_data.apply(
        lambda row: "📤" if row["src"] != "ego" else "📥", axis=1
    )
    table_data["formatted_value"] = table_data["weight"].apply(format_currency_br)
    
    # Criar tabela HTML customizada
    rows = []
    for _, row in table_data.head(15).iterrows():  # Top 15
        direction_icon = "📤" if row["src"] != company_id else "📥"
        partner = row["dst"] if row["src"] == company_id else row["src"]
        
        rows.append(
            html.Tr([
                html.Td(direction_icon, style={"textAlign": "center", "fontSize": "16px"}),
                html.Td(partner, style={"fontFamily": "monospace"}),
                html.Td(format_currency_br(row["weight"]), 
                       style={"textAlign": "right", "fontWeight": "bold", "color": "#22c55e"}),
                html.Td(
                    dbc.Badge("Alto Risco" if row["weight"] > edges_df["weight"].quantile(0.8) else "Normal",
                             color="warning" if row["weight"] > edges_df["weight"].quantile(0.8) else "secondary")
                )
            ])
        )
    
    table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Fluxo", style={"color": "#9ca3af"}),
                html.Th("Parceiro", style={"color": "#9ca3af"}),
                html.Th("Valor", style={"color": "#9ca3af", "textAlign": "right"}),
                html.Th("Risco", style={"color": "#9ca3af"})
            ])
        ]),
        html.Tbody(rows)
    ], striped=True, hover=True, size="sm")
    
    return table

def create_diversification_suggestions(suggestions: List[Dict[str, Any]]):
    """Cria painel de sugestões de diversificação."""
    if not suggestions:
        return html.Div()
    
    suggestion_items = []
    
    for suggestion in suggestions:
        suggestion_items.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6(f"🤝 {suggestion['partner_id']}", style={"color": "#60a5fa"}),
                        html.P(suggestion["recommendation_reason"], 
                              style={"fontSize": "12px", "color": "#9ca3af"}),
                        html.Small(f"Vol. médio: {format_currency_br(suggestion['avg_transaction_volume'])}", 
                                 style={"color": "#22c55e"})
                    ])
                ])
            ], md=4)
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("🎯 Sugestões de Diversificação", style={"color": "#e5e7eb", "margin": "0"})
        ]),
        dbc.CardBody([
            html.P("Novos parceiros recomendados para reduzir concentração de risco:",
                  style={"color": "#9ca3af", "marginBottom": "15px"}),
            dbc.Row(suggestion_items)
        ])
    ], className="mt-4")
