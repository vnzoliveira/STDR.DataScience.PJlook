import pandas as pd
from pathlib import Path
from ..etl.ingest import load_config
import numpy as np


def relations_monthly():
    cfg = load_config()
    b2 = pd.read_parquet(Path(cfg["data"]["processed_base2"]) / "base2.parquet").copy()
    b2["year_month"] = b2["DT_REFE"]
    return b2

def relation_importance(df, company_id, direction="in", months_window=12):
    last = df["year_month"].max()
    first = (last - pd.offsets.MonthBegin(months_window-1)).to_period("M").to_timestamp()
    win = df[(df["year_month"]>=first) & (df["year_month"]<=last)].copy()

    if direction=="in":
        ego = win[win["ID_RCBE"]==company_id]
        key = "ID_PGTO"
    else:
        ego = win[win["ID_PGTO"]==company_id]
        key = "ID_RCBE"

    if ego.empty:
        return pd.DataFrame(columns=["counterparty","share","freq_mensal","score"])

    tot = ego["VL"].sum()
    grp = ego.groupby(key).agg(valor=("VL","sum"), meses=("year_month","nunique")).reset_index()
    grp = grp.rename(columns={key:"counterparty"})
    n_meses = max(1, ego["year_month"].nunique())
    grp["share"] = grp["valor"]/tot
    grp["freq_mensal"] = grp["meses"]/n_meses
    grp["score"] = 0.6*grp["share"] + 0.4*grp["freq_mensal"]
    return grp.sort_values("score", ascending=False).reset_index(drop=True)

def concentration(df, company_id, months_window=12):
    last = df["year_month"].max()
    first = (last - pd.offsets.MonthBegin(months_window-1)).to_period("M").to_timestamp()
    win = df[(df["year_month"]>=first) & (df["year_month"]<=last)].copy()
    ego = win[(win["ID_PGTO"]==company_id) | (win["ID_RCBE"]==company_id)]

    # HHI entrada
    in_e = ego[ego["ID_RCBE"]==company_id]
    hhi_in = None
    if not in_e.empty:
        tot = in_e["VL"].sum()
        shares = in_e.groupby("ID_PGTO")["VL"].sum()/tot
        hhi_in = float((shares**2).sum())

    # HHI saída
    out_e = ego[ego["ID_PGTO"]==company_id]
    hhi_out = None
    if not out_e.empty:
        tot = out_e["VL"].sum()
        shares = out_e.groupby("ID_RCBE")["VL"].sum()/tot
        hhi_out = float((shares**2).sum())

    return {"hhi_in": hhi_in, "hhi_out": hhi_out}

def ego_edges_by_month(df, company_id, year_month, direction="both", top_n=15):
    """
     Retorna arestas (src, dst, peso) do ego-network da company_id no mês selecionado.
    direction: "in" (quem paga -> company), "out" (company -> recebedor), "both"
    top_n: limita as contrapartes por maior valor no mês para não poluir o gráfico
    """
    
    ym = pd.to_datetime(year_month).to_period("M").to_timestamp()
    m = df[df["year_month"]==ym].copy()
    if m.empty:
        return pd.DataFrame(columns=["src","dst","valor"])
    
    if direction == "in":
        ego = m[m["ID_RCBE"]==company_id].copy()
    elif direction == "out":
        ego = m[m["ID_PGTO"]==company_id].copy()
    else:
        ego = m[(m["ID_PGTO"]==company_id) | (m["ID_RCBE"]==company_id)].copy()
        
    
    if ego.empty:
        return pd.DataFrame(columns=["src","dst","valor"])
    
    ego["src"] = ego["ID_PGTO"]
    ego["dst"] = ego["ID_RCBE"]
    agg = ego.groupby(["src","dst"], as_index=False)["VL"].sum().rename(columns={"VL":"valor"})
    
    def other(u,v):
        return v if  u==company_id else u
    
    agg["counterparty"] = agg.apply(lambda r: other(r["src"], r["dst"]), axis=1)
    agg = agg.sort_values("valor", ascending=False).head(top_n)
    return agg[["src","dst", "valor"]]
