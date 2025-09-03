import pandas as pd
from pathlib import Path
from ..etl.ingest import load_config

def monthly_flows():
    cfg = load_config()
    b2 = pd.read_parquet(Path(cfg["data"]["processed_base2"]) / "base2.parquet")
    b2["year_month"] = b2["DT_REFE"]

    rec = b2.groupby(["ID_RCBE","year_month"])["VL"].sum().rename("receita_mensal")
    des = b2.groupby(["ID_PGTO","year_month"])["VL"].sum().rename("despesa_mensal")

    rec = rec.reset_index().rename(columns={"ID_RCBE":"ID"})
    des = des.reset_index().rename(columns={"ID_PGTO":"ID"})

    m = pd.merge(rec, des, on=["ID","year_month"], how="outer").fillna(0)
    m["fluxo_liquido"] = m["receita_mensal"] - m["despesa_mensal"]

    
    m = m.sort_values(["ID","year_month"])
    m["g_receita_mom"] = m.groupby("ID")["receita_mensal"].pct_change()
    
    m["vol_receita_3m"] = m.groupby("ID")["receita_mensal"].rolling(3).std().reset_index(level=0, drop=True)
    return m

def build_f_empresa_mes():
    cfg = load_config()
    b1 = pd.read_parquet(Path(cfg["data"]["processed_base1"]) / "base1.parquet")
    b1 = b1.rename(columns={"DT_REFE":"year_month"})
    m = monthly_flows()
    f = pd.merge(b1, m, on=["ID","year_month"], how="left").fillna(0)
    return f