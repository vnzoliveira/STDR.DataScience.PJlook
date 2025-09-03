import duckdb, pandas as pd
from pathlib import Path

def query_parquet(sql: str, paths: list[str]) -> pd.DataFrame:
    con = duckdb.connect()
    for i, p in enumerate(paths):
        con.register(f"t{i}", str(Path(p)))
    return con.execute(sql).df()