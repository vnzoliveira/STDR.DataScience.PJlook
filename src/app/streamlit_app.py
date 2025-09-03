import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="PJlook Santander", layout="wide")

@st.cache_data
def load_table(path):
    return pd.read_parquet(path)

tab1, tab2 = st.tabs(["Desafio 1 - Perfil & Estágio", "Desafio 2 - Relações & Risco"])

with tab1:
    st.header("Perfil & Estágio")
    f = load_table("reports/exports/f_empresa_mes.parquet")
    est = load_table("reports/exports/estagio.parquet")
    ids = sorted(f["ID"].unique().tolist())
    sel = st.selectbox("ID", ids)
    df_id = f[f["ID"]==sel].sort_values("year_month")
    st.line_chart(df_id.set_index("year_month")[["receita_mensal","despesa_mensal","fluxo_liquido"]])
    st.dataframe(est[est["ID"]==sel])

with tab2:
    st.header("Relações & Risco")
    rel = load_table("reports/exports/relations_latest.parquet")
    ids = sorted(rel["company_id"].unique().tolist())
    sel = st.selectbox("ID", ids, key="rel")
    st.dataframe(rel[rel["company_id"]==sel].sort_values("score", ascending=False))