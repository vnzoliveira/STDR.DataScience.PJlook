from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx

from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_cytoscape as cyto

cyto.load_extra_layouts()  # habilita layouts adicionais (ex.: cose-bilkent)

DATA_PARQUET = Path("data/processed/base2/base2.parquet")  # já gerado no build-all

def ensure_df_edges() -> pd.DataFrame:
    if not DATA_PARQUET.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {DATA_PARQUET}")
    df = pd.read_parquet(str(DATA_PARQUET))
    # normaliza campo de mês
    df["year_month"] = pd.to_datetime(df["DT_REFE"]).dt.to_period("M").dt.to_timestamp()
    # normaliza canal, se existir
    if "DS_TRAN" in df.columns:
        df["DS_TRAN"] = df["DS_TRAN"].astype(str).str.upper().str.strip()
    else:
        df["DS_TRAN"] = "OUTROS"
    return df

DF_EDGES = ensure_df_edges()

def build_ego(company_id: str, year_month: str, months_window: int,
              direction: str, top_n: int, channels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ym_last = pd.to_datetime(year_month).to_period("M").to_timestamp()
    ym_first = (ym_last - pd.offsets.MonthBegin(months_window-1)).to_period("M").to_timestamp()

    df = DF_EDGES[(DF_EDGES["year_month"]>=ym_first) & (DF_EDGES["year_month"]<=ym_last)]
    if channels and "TODOS" not in channels:
        df = df[df["DS_TRAN"].isin(channels)]

    if direction == "in":
        eg = df[df["ID_RCBE"]==company_id]
    elif direction == "out":
        eg = df[df["ID_PGTO"]==company_id]
    else:
        eg = df[(df["ID_PGTO"]==company_id) | (df["ID_RCBE"]==company_id)]

    if eg.empty:
        return (pd.DataFrame(columns=["node_id","role","strength","in_w","out_w","pagerank","betweenness"]),
                pd.DataFrame(columns=["src","dst","weight"]), {"n_nodes":0,"n_edges":0,"density":0.0})

    eg = eg.assign(src=eg["ID_PGTO"], dst=eg["ID_RCBE"])
    agg = eg.groupby(["src","dst"], as_index=False)["VL"].sum().rename(columns={"VL":"weight"})
    agg = agg.sort_values("weight", ascending=False).head(top_n)

    # Grafo dirigido ponderado
    G = nx.DiGraph()
    for _, r in agg.iterrows():
        G.add_edge(r["src"], r["dst"], weight=float(r["weight"]))

    in_w  = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))
    strength = {n: in_w.get(n,0.0) + out_w.get(n,0.0) for n in G.nodes()}

    # PageRank (NumPy, sem SciPy)
    try:
        pagerank = nx.pagerank_numpy(G, alpha=0.85, weight="weight")
    except Exception:
        pagerank = {n: 1.0/len(G) for n in G} if len(G) else {}

    # Betweenness adaptativa
    N = G.number_of_nodes()
    if N <= 80:
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    else:
        k = min(32, max(2, N-1))
        betweenness = nx.betweenness_centrality(G, k=k, weight="weight", normalized=True, seed=42)

    def role(n: str) -> str:
        if n == company_id:
            return "ego"
        if G.has_edge(n, company_id):
            return "payer_to_ego"
        if G.has_edge(company_id, n):
            return "receiver_from_ego"
        return "neighbor"

    nodes = []
    for n in G.nodes():
        nodes.append({
            "node_id": n,
            "role": role(n),
            "in_w": float(in_w.get(n,0.0)),
            "out_w": float(out_w.get(n,0.0)),
            "strength": float(strength.get(n,0.0)),
            "pagerank": float(pagerank.get(n,0.0)),
            "betweenness": float(betweenness.get(n,0.0)),
        })
    nodes_df = pd.DataFrame(nodes).sort_values("strength", ascending=False)
    edges_df = agg.copy()

    # KPIs do subgrafo
    n = len(G.nodes())
    m = len(G.edges())
    possible = n*(n-1) if n>1 else 1
    density = m/possible if possible else 0.0
    summary = {"n_nodes": int(n), "n_edges": int(m), "density": float(density)}

    return nodes_df, edges_df, summary

def to_cyto_elements(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, ego_id: str):
    # escala (robusta) de tamanho/espessura
    s_min, s_max = nodes_df["strength"].min() if len(nodes_df) else 0, nodes_df["strength"].max() if len(nodes_df) else 1
    e_min, e_max = edges_df["weight"].min() if len(edges_df) else 0, edges_df["weight"].max() if len(edges_df) else 1
    def scale(v, a, b, out_min, out_max):
        if b <= a: return (out_min + out_max)/2
        x = (v - a) / (b - a)
        return float(out_min + x*(out_max - out_min))

    # nodes
    nodes = []
    for _, r in nodes_df.iterrows():
        nodes.append({
            "data": {
                "id": r["node_id"],
                "label": r["node_id"],
                "role": r["role"],
                "strength": r["strength"],
                "pagerank": r["pagerank"],
                "betweenness": r["betweenness"],
                "size": scale(r["strength"], s_min, s_max, 18, 60),
            },
            "classes": r["role"]
        })

    # edges
    edges = []
    for _, r in edges_df.iterrows():
        edges.append({
            "data": {
                "source": r["src"], "target": r["dst"],
                "weight": float(r["weight"]),
                "width": scale(float(r["weight"]), e_min, e_max, 1.5, 9.0),
                "label": f"{int(r['weight']):,}".replace(",", ".")
            },
            "classes": "edge"
        })
    return nodes + edges

def serve_app():
    app = Dash(__name__, title="Grafo PJ – Cytoscape", update_title=None)
    app.layout = html.Div(
        style={"backgroundColor":"#0f172a", "color":"#e5e7eb", "height":"100vh", "padding":"12px"},
        children=[
            html.Div([
                html.Div("Grafo Interativo – Cadeia de Valor", style={"fontWeight":"700","fontSize":"20px"}),
                html.Div(id="kpis", style={"marginTop":"4px", "fontSize":"13px","opacity":0.9})
            ]),
            html.Div(style={"display":"flex","gap":"10px","margin":"10px 0"}, children=[
                dcc.Dropdown(
                    id="company-id", clearable=False,
                    options=[{"label":i, "value":i} for i in sorted(DF_EDGES["ID_PGTO"].append(DF_EDGES["ID_RCBE"]).unique())],
                    value=sorted(DF_EDGES["ID_PGTO"].append(DF_EDGES["ID_RCBE"]).unique())[0],
                    style={"width":"340px"}
                ),
                dcc.Dropdown(
                    id="direction", clearable=False,
                    options=[{"label":l, "value":l} for l in ["both","in","out"]],
                    value="both", style={"width":"140px"}
                ),
                dcc.Dropdown(
                    id="channels", multi=True,
                    options=[{"label":c, "value":c} for c in ["TODOS","PIX","TED","BOLETO","OUTROS"]],
                    value=["TODOS"], style={"width":"260px"}
                ),
                dcc.Slider(id="months-window", min=1, max=12, step=1, value=3,
                           marks={1:"1m",3:"3m",6:"6m",12:"12m"},
                           tooltip={"placement":"bottom","always_visible":False},
                           updatemode="mouseup", style={"width":"220px"}),
                dcc.Slider(id="top-n", min=5, max=50, step=1, value=25,
                           marks={5:"5",15:"15",25:"25",40:"40",50:"50"},
                           tooltip={"placement":"bottom"}, updatemode="mouseup", style={"width":"260px"}),
                dcc.Dropdown(
                    id="anchor-month", clearable=False,
                    options=[{"label":d.strftime("%Y-%m"), "value":d.strftime("%Y-%m")}
                             for d in sorted(DF_EDGES["year_month"].unique())],
                    value=max(DF_EDGES["year_month"]).strftime("%Y-%m"),
                    style={"width":"160px"}
                )
            ]),
            cyto.Cytoscape(
                id="graph",
                elements=[],
                layout={"name": "cose-bilkent"},
                style={"width":"100%","height":"78vh","backgroundColor":"#0b1220"},
                stylesheet=[
                    {"selector":"node",
                     "style":{
                        "label":"data(label)",
                        "font-size":"10px",
                        "color":"#e5e7eb",
                        "background-color":"#64748b",
                        "width":"data(size)",
                        "height":"data(size)",
                        "border-width":2,
                        "border-color":"#0f172a"
                     }},
                    {"selector":".ego", "style":{"background-color":"#ef4444","border-color":"#fca5a5","border-width":3}},
                    {"selector":".payer_to_ego", "style":{"background-color":"#22c55e"}},
                    {"selector":".receiver_from_ego", "style":{"background-color":"#60a5fa"}},
                    {"selector":".neighbor", "style":{"background-color":"#a78bfa"}},
                    {"selector":"edge",
                     "style":{
                        "curve-style":"bezier",
                        "target-arrow-shape":"triangle",
                        "line-color":"#94a3b8",
                        "target-arrow-color":"#94a3b8",
                        "width":"data(width)"
                     }}
                ]
            ),
            html.Div(id="hover-info", style={"marginTop":"6px","fontSize":"12px","opacity":0.85})
        ]
    )

    @app.callback(
        Output("graph","elements"),
        Output("kpis","children"),
        Input("company-id","value"),
        Input("anchor-month","value"),
        Input("months-window","value"),
        Input("direction","value"),
        Input("top-n","value"),
        Input("channels","value")
    )
    def update_graph(company_id, anchor_month, months_window, direction, top_n, channels):
        nodes_df, edges_df, summary = build_ego(company_id, anchor_month, int(months_window),
                                                direction, int(top_n), channels or ["TODOS"])
        els = to_cyto_elements(nodes_df, edges_df, company_id)
        kpis = f"Nós: {summary['n_nodes']} | Arestas: {summary['n_edges']} | Densidade: {summary['density']:.3f}"
        return els, kpis

    @app.callback(
        Output("hover-info","children"),
        Input("graph","mouseoverNodeData"),
        Input("graph","mouseoverEdgeData"),
        State("company-id","value")
    )
    def hover(node, edge, ego):
        if node:
            return f"NODE {node.get('id')} | role={node.get('role')} | strength={node.get('strength'):.0f} | PR={node.get('pagerank'):.4f} | btw={node.get('betweenness'):.4f}"
        if edge:
            return f"EDGE {edge.get('source')} → {edge.get('target')} | weight={edge.get('weight'):.0f}"
        return "Passe o mouse para ver detalhes…"

    return app

if __name__ == "__main__":
    app = serve_app()
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)