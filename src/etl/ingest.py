from pathlib import Path
import pandas as pd
import yaml

def _to_month(dt_series):
    dt = pd.to_datetime(dt_series)
    return dt.dt.to_period('M').dt.to_timestamp()

def load_config(path="configs/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ingest_bases():
    cfg = load_config()
    raw = cfg["data"]["raw_excel"]

    # ---- Base 1 (perfil/ID) ----
    b1 = pd.read_excel(raw, sheet_name="Base 1 - ID", dtype={"ID": str})
    b1["DT_REFE"] = _to_month(b1["DT_REFE"])
    for col in ["VL_FATU", "VL_SLDO"]:
        b1[col] = pd.to_numeric(b1[col], errors="coerce")
    out1 = Path(cfg["data"]["processed_base1"])
    b1.to_parquet(out1 / "base1.parquet", index=False)

    # ---- Base 2 (transações) ----
    b2 = pd.read_excel(raw, sheet_name="Base 2 - Transações",
                       dtype={"ID_PGTO": str, "ID_RCBE": str})
    b2["DT_REFE"] = _to_month(b2["DT_REFE"])
    b2["VL"] = pd.to_numeric(b2["VL"], errors="coerce").fillna(0)
    # normalizar DS_TRAN para domínios
    dom = cfg["domains"]["ds_tran_map"]
    b2["DS_TRAN"] = b2["DS_TRAN"].astype(str).str.upper().str.strip()
    b2["DS_TRAN"] = b2["DS_TRAN"].map(dom).fillna(dom.get("default", "OUTROS"))

    out2 = Path(cfg["data"]["processed_base2"])
    b2.to_parquet(out2 / "base2.parquet", index=False)

    return {"base1_rows": len(b1), "base2_rows": len(b2)}