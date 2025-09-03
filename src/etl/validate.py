import pandas as pd
from pathlib import Path
from .ingest import load_config

def run_checks():
    cfg = load_config()
    b1 = pd.read_parquet(Path(cfg["data"]["processed_base1"]) / "base1.parquet")
    b2 = pd.read_parquet(Path(cfg["data"]["processed_base2"]) / "base2.parquet")

    issues = []

    # Base 1: chave única (ID, DT_REFE)
    if b1.duplicated(subset=["ID", "DT_REFE"]).any():
        issues.append("Duplicidade em Base1 por (ID, DT_REFE).")

    # Base 2: valores > 0 e IDs distintos
    if (b2["VL"] <= 0).any():
        issues.append("Base2 contém VL <= 0.")
    if (b2["ID_PGTO"] == b2["ID_RCBE"]).any():
        issues.append("Base2 possui transações ID_PGTO == ID_RCBE.")

    # Domínios DS_TRAN
    ok_dom = {"PIX","TED","BOLETO","OUTROS"}
    if ~b2["DS_TRAN"].isin(ok_dom).all():
        issues.append("Base2 com DS_TRAN fora do domínio esperado.")

    # Janelas de datas plausíveis
    if b2["DT_REFE"].min() < pd.Timestamp("2025-03-01"):
        issues.append("Base2 com datas anteriores a 2025-03.")
    if b2["DT_REFE"].max() > b1["DT_REFE"].max():
        issues.append("Base2 com mês além do perfil em Base1.")

    return {"issues": issues, "ok": len(issues) == 0}