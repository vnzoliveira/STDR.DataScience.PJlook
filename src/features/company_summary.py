# src/features/company_summary.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _safe_num(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill)

def _porte_by_fat(fat: float) -> str:
    fat = float(fat) if pd.notna(fat) else 0.0
    if fat < 360_000: return "MICRO"
    if fat < 4_800_000: return "PEQUENA"
    if fat < 300_000_000: return "MEDIA"
    return "GRANDE"

def _first_token_upper(x: str) -> str:
    s = str(x).upper()
    return (pd.Series([s]).str.extract(r"(^[^;\|/,-]+)")[0].iloc[0] or "OUTROS").strip()

def build_companies_summary(f: pd.DataFrame,
                            est: pd.DataFrame | None = None,
                            b1: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    1 linha por empresa com métricas estáveis p/ benchmarking setorial.
    """
    if f is None or f.empty:
        return pd.DataFrame()

    for c in ["receita_mensal","despesa_mensal","fluxo_liquido","g_receita_mom","vol_receita_3m"]:
        if c in f.columns:
            f[c] = _safe_num(f[c])

    g = (f.groupby("ID")
           .agg(
               receita_media=("receita_mensal", "mean"),
               despesa_media=("despesa_mensal", "mean"),
               fluxo_medio=("fluxo_liquido", "mean"),
               receita_total=("receita_mensal", "sum"),
               crescimento_medio=("g_receita_mom", "mean"),
               volatilidade_receita=("vol_receita_3m", "mean"),
               consistencia_fluxo=("fluxo_liquido", lambda s: float((s > 0).mean())),
               meses_ativos=("year_month","nunique")
           )
           .reset_index())

    g["margem_media"] = np.where(g["receita_media"].abs() > 1e-9,
                                 g["fluxo_medio"] / g["receita_media"], 0.0)

    if b1 is not None and not b1.empty:
        cols = [c for c in ["ID","DS_CNAE","VL_FATU","DT_ABRT"] if c in b1.columns]
        b1u = b1[cols].drop_duplicates("ID")
        g = g.merge(b1u, on="ID", how="left")
        g["porte"] = g["VL_FATU"].apply(_porte_by_fat)
        g["setor"] = g["DS_CNAE"].apply(_first_token_upper)
    else:
        g["porte"] = "MICRO"
        g["setor"] = "OUTROS"

    if est is not None and not est.empty and "estagio" in est.columns:
        g = g.merge(est[["ID","estagio"]].drop_duplicates("ID"), on="ID", how="left")

    num_cols = g.select_dtypes(include=[np.number]).columns
    g[num_cols] = g[num_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return g
