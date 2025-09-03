import pandas as pd
from pathlib import Path
from ..etl.ingest import load_config
import numpy as np


def relations_monthly():
    cfg = load_config()
    b2 = pd.read_parquet(Path(cfg["data"]["processed_base2"]) / "base2.parquet").copy()
    b2["year_month"] = b2["DT_REFE"]
    return b2