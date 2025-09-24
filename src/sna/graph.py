# src/sna/graph.py
from __future__ import annotations
import pandas as pd
import networkx as nx

def _ensure_year_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year_month" not in out.columns:
        out["year_month"] = pd.to_datetime(out["DT_REFE"]).dt.to_period("M").dt.to_timestamp()
    return out

def build_ego_network(
    df_edges: pd.DataFrame,
    company_id: str,
    year_month: str,
    months_window: int = 3,
    direction: str = "both",          # "in" | "out" | "both"
    top_n: int = 25,
    channels: list[str] | None = None # e.g., ["PIX","TED"] ou ["TODOS"]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Retorna (nodes_df, edges_df, summary) do ego-network.
      nodes_df: [node_id, role, in_w, out_w, strength, pagerank, betweenness]
      edges_df: [src, dst, weight]
      summary : {"n_nodes","n_edges","density"}
    """
    df = _ensure_year_month(df_edges)

    ym_last = pd.to_datetime(year_month).to_period("M").to_timestamp()
    ym_first = (ym_last - pd.offsets.MonthBegin(months_window-1)).to_period("M").to_timestamp()

    win = df[(df["year_month"]>=ym_first) & (df["year_month"]<=ym_last)].copy()

    # filtro por canal
    if channels and "TODOS" not in channels:
        if "DS_TRAN" in win.columns:
            win["DS_TRAN"] = win["DS_TRAN"].astype(str).str.upper().str.strip()
            win = win[win["DS_TRAN"].isin([c.upper() for c in channels])]

    # direcionamento
    if direction == "in":
        ego = win[win["ID_RCBE"]==company_id]
    elif direction == "out":
        ego = win[win["ID_PGTO"]==company_id]
    else:
        ego = win[(win["ID_PGTO"]==company_id) | (win["ID_RCBE"]==company_id)]

    if ego.empty:
        return (
            pd.DataFrame(columns=["node_id","role","in_w","out_w","strength","pagerank","betweenness"]),
            pd.DataFrame(columns=["src","dst","weight"]),
            {"n_nodes":0,"n_edges":0,"density":0.0}
        )

    ego = ego.assign(src=ego["ID_PGTO"], dst=ego["ID_RCBE"])
    agg = (ego.groupby(["src","dst"], as_index=False)["VL"]
              .sum().rename(columns={"VL":"weight"})
              .sort_values("weight", ascending=False).head(top_n))

    # Grafo dirigido ponderado
    G = nx.DiGraph()
    for _, r in agg.iterrows():
        G.add_edge(r["src"], r["dst"], weight=float(r["weight"]))

    in_w  = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))
    strength = {n: in_w.get(n,0.0) + out_w.get(n,0.0) for n in G.nodes()}

    # PageRank (NumPy) — sem SciPy
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
        betweenness = nx.betweenness_centrality(G, k=k, weight="weight",
                                                normalized=True, seed=42)

    def role(n: str) -> str:
        if n == company_id: return "ego"
        if G.has_edge(n, company_id): return "payer_to_ego"
        if G.has_edge(company_id, n): return "receiver_from_ego"
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
    n = len(G.nodes()); m = len(G.edges())
    possible = n*(n-1) if n>1 else 1
    density = m/possible if possible else 0.0
    summary = {"n_nodes":int(n), "n_edges":int(m), "density":float(density)}

    return nodes_df, edges_df, summary

def _scale(v, a, b, lo, hi) -> float:
    if b <= a: return (lo+hi)/2
    x = (v-a)/(b-a)
    return float(lo + x*(hi-lo))

def to_cyto_elements(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, ego_id: str):
    # escalas
    s_min = nodes_df["strength"].min() if len(nodes_df) else 0
    s_max = nodes_df["strength"].max() if len(nodes_df) else 1
    e_min = edges_df["weight"].min() if len(edges_df) else 0
    e_max = edges_df["weight"].max() if len(edges_df) else 1

    nodes = [{
        "data":{
            "id": r["node_id"], "label": r["node_id"],
            "strength": float(r["strength"]),
            "pagerank": float(r["pagerank"]),
            "betweenness": float(r["betweenness"]),
            "size": _scale(float(r["strength"]), s_min, s_max, 18, 60)
        },
        "classes": r["role"]
    } for _, r in nodes_df.iterrows()]

    edges = [{
        "data":{
            "source": e["src"], "target": e["dst"],
            "weight": float(e["weight"]),
            "width": _scale(float(e["weight"]), e_min, e_max, 1.5, 9.0),
            "label": f"{int(e['weight']):,}".replace(",", ".")
        },
        "classes":"edge"
    } for _, e in edges_df.iterrows()]

    return nodes + edges
