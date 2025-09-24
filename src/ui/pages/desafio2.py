# src/ui/pages/desafio2.py
from __future__ import annotations
from dash import register_page, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from pathlib import Path
import pandas as pd
import numpy as np

from src.sna.graph import build_ego_network, to_cyto_elements

register_page(__name__, path="/desafio2", title="Relações & Risco")
cyto.load_extra_layouts()

REL_PATH = Path("reports/exports/relations_latest.parquet")
B2_PATH  = Path("data/processed/base2/base2.parquet")
F_EMP_PATH = Path("reports/exports/f_empresa_mes.parquet")

def _load_edges():
    if not B2_PATH.exists():
        return None
    df = pd.read_parquet(B2_PATH)
    # -> corrigido: usa .dt em ambas as etapas
    df["year_month"] = (
        pd.to_datetime(df["DT_REFE"])
          .dt.to_period("M")
          .dt.to_timestamp()
    )
    df["DS_TRAN"] = (
        df["DS_TRAN"].astype(str).str.upper().str.strip()
        if "DS_TRAN" in df.columns else "OUTROS"
    )
    return df

def _load_ids():
    if not F_EMP_PATH.exists(): return []
    f = pd.read_parquet(F_EMP_PATH)
    return sorted(f["ID"].unique().tolist())

def _load_rel():
    return pd.read_parquet(REL_PATH) if REL_PATH.exists() else None

def _fmt_pct(x) -> str:
    try: return f"{100.0*float(x):.1f}%"
    except Exception: return "0%"

def _hhi(df_edges: pd.DataFrame, cid: str, ym: str, win: int):
    ym_last = pd.to_datetime(ym).to_period("M").to_timestamp()
    ym_first = (ym_last - pd.offsets.MonthBegin(win-1)).to_period("M").to_timestamp()
    w = df_edges[(df_edges["year_month"]>=ym_first) & (df_edges["year_month"]<=ym_last)]
    tin = w[w["ID_RCBE"]==cid].groupby("ID_PGTO")["VL"].sum()
    tout = w[w["ID_PGTO"]==cid].groupby("ID_RCBE")["VL"].sum()
    hin = float(((tin/tin.sum())**2).sum()) if len(tin)>0 and tin.sum()>0 else np.nan
    hout = float(((tout/tout.sum())**2).sum()) if len(tout)>0 and tout.sum()>0 else np.nan
    return hin, hout

def layout():
    df_edges = _load_edges()
    ids = _load_ids()
    if df_edges is None or not ids:
        return dbc.Alert("Artefatos não encontrados. Rode o pipeline (build-all).", color="warning")

    months = sorted(df_edges["year_month"].unique())

    # linha 1 – filtros principais
    filters_row1 = dbc.Row([
        dbc.Col(dcc.Dropdown(ids, ids[0], id="d2-id", clearable=False, className="dash-dropdown"), md=4),
        dbc.Col(dcc.RadioItems(["in","out","both"], "both", id="d2-dir", inline=True,
                               inputClassName="form-check-input", labelClassName="form-check-label"), md=3),
        dbc.Col(dcc.Dropdown(["TODOS","PIX","TED","BOLETO","OUTROS"], ["TODOS"], id="d2-ch", multi=True, className="dash-dropdown"), md=5),
    ], className="mb-2", justify="start")

    # linha 2 – sliders/mes
    filters_row2 = dbc.Row([
        dbc.Col(dcc.Slider(1, 12, 1, value=3, id="d2-win",  marks={1:"1m",3:"3m",6:"6m",12:"12m"}), md=4),
        dbc.Col(dcc.Slider(5, 50, 1, value=25, id="d2-topn", marks={5:"5",15:"15",25:"25",40:"40",50:"50"}), md=4),
        dbc.Col(dcc.Dropdown([m.strftime("%Y-%m") for m in months],
                             months[-1].strftime("%Y-%m"),
                             id="d2-month", clearable=False, className="dash-dropdown"), md=4),
    ], className="mb-3")

    return dbc.Container(fluid=True, children=[
        html.H3("Relações & Risco"),

        filters_row1,
        filters_row2,

        # KPIs Top relação
        html.Div(id="d2-top-kpis", className="mb-2"),

        # Tabela com scroll (não ocupa a página toda)
        dbc.Card(dbc.CardBody(html.Div(id="d2-table",
                                       style={"maxHeight":"340px","overflowY":"auto"})),
                 className="mb-3"),

        html.H5("Grafo Interativo (Cytoscape)"),
        cyto.Cytoscape(
            id="d2-graph",
            elements=[],
            layout={"name": "cose-bilkent"},
            style={"width":"100%","height":"65vh","backgroundColor":"#0b1220"},  # menor
            stylesheet=[
                {"selector":"node","style":{
                    "label":"data(label)","font-size":"10px","color":"#e5e7eb",
                    "background-color":"#64748b","width":"data(size)","height":"data(size)",
                    "border-width":2,"border-color":"#0f172a"}},
                {"selector":".ego","style":{"background-color":"#ef4444","border-color":"#fca5a5","border-width":3}},
                {"selector":".payer_to_ego","style":{"background-color":"#22c55e"}},
                {"selector":".receiver_from_ego","style":{"background-color":"#60a5fa"}},
                {"selector":".neighbor","style":{"background-color":"#a78bfa"}},
                {"selector":"edge","style":{
                    "curve-style":"bezier","target-arrow-shape":"triangle",
                    "line-color":"#94a3b8","target-arrow-color":"#94a3b8","width":"data(width)"}}
            ]
        ),
        html.Small(id="d2-kpis", style={"opacity":0.85})
    ])

@callback(
    Output("d2-top-kpis","children"),
    Output("d2-table","children"),
    Input("d2-id","value")
)
def _update_top_rel(cid: str):
    rel = _load_rel()
    if rel is None or rel.empty:
        return dbc.Alert("`relations_latest.parquet` não encontrado.", color="secondary"), html.Div()

    r = rel[rel["company_id"] == cid].sort_values("score", ascending=False)
    if r.empty:
        return dbc.Alert("Sem relações para o ID.", color="secondary"), html.Div()

    top = r.iloc[0]
    kpis = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Contraparte"), html.H2(top["counterparty"])])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Direção"),    html.H2(top["direction"])])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Share"),      html.H2(_fmt_pct(top["share"]))])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Score"),      html.H2(f"{float(top['score']):.4f}")])), md=2),
    ], className="mb-2")
    table = dbc.Table.from_dataframe(
        r[["counterparty","valor","meses","share","freq_mensal","score","direction"]]
         .rename(columns={"valor":"valor_total"}),
        striped=True, bordered=True, hover=True, size="sm"
    )
    return kpis, table

@callback(
    Output("d2-graph","elements"),
    Output("d2-kpis","children"),
    Input("d2-id","value"),
    Input("d2-month","value"),
    Input("d2-win","value"),
    Input("d2-dir","value"),
    Input("d2-topn","value"),
    Input("d2-ch","value"),
)
def _update_graph(cid: str, month: str, win: int, direction: str, topn: int, channels: list[str]):
    df_edges = _load_edges()
    if df_edges is None:
        return [], "Sem dados."

    nodes_df, edges_df, summary = build_ego_network(
        df_edges=df_edges,
        company_id=cid,
        year_month=month,
        months_window=int(win),
        direction=direction,
        top_n=int(topn),
        channels=channels or ["TODOS"]
    )
    els = to_cyto_elements(nodes_df, edges_df, cid)
    hin, hout = _hhi(df_edges, cid, month, int(win))
    kpis = f"Nós: {summary['n_nodes']} | Arestas: {summary['n_edges']} | Densidade: {summary['density']:.3f} | HHI in/out: {(hin or 0):.3f}/{(hout or 0):.3f}"
    return els, kpis
