import numpy as np
import pandas as pd

def classify_stage(f_empresa_mes, months_window=6):
    f = f_empresa_mes.sort_values(["ID","year_month"]).groupby("ID").tail(months_window)
    g = f.groupby("ID").agg(
        g_receita_mom_med=("g_receita_mom","mean"),
        fluxo_liq_med=("fluxo_liquido","mean"),
        vol_receita_med=("vol_receita_3m","mean"),
        receita_med=("receita_mensal","mean"),
        meses_fluxo_neg=("fluxo_liquido", lambda s: (s<0).sum())
    ).reset_index()

    
    pct = lambda x: x.rank(pct=True)
    g["pct_g"]   = pct(g["g_receita_mom_med"].fillna(g["g_receita_mom_med"].median()))
    g["pct_vol"] = pct(g["vol_receita_med"].fillna(g["vol_receita_med"].median()))
    g["pct_rec"] = pct(g["receita_med"].fillna(g["receita_med"].median()))
    g["score_fluxo"] = (g["fluxo_liq_med"] >= 0).astype(int)

    cond_exp = (g["pct_g"]>=0.6) & (g["score_fluxo"]==1)
    cond_mat = (g["pct_g"].between(0.4,0.6, inclusive="left")) & (g["score_fluxo"]==1) & ((1-g["pct_vol"])>=0.5)
    cond_ini = (g["pct_rec"]<=0.3) & (g["pct_vol"]>=0.6)
    cond_dec = (g["pct_g"]<=0.3) | (g["meses_fluxo_neg"]>=2)

    g["estagio"] = np.select(
        [cond_dec, cond_exp, cond_mat, cond_ini],
        ["Declínio","Expansão","Maturidade","Início"],
        default="Maturidade"
    )
    return g[["ID","estagio","g_receita_mom_med","fluxo_liq_med","vol_receita_med","receita_med","meses_fluxo_neg"]]