import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.sna.graph import build_ego_network, to_pyvis_html
from src.sna.relations import concentration  # já existente
import streamlit.components.v1 as components

# Para o grafo (Desafio 2)
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="FIAP Datalab - Santander", layout="wide")

@st.cache_data
def load_table(path):
    return pd.read_parquet(path)

def fmt_money(x):
    try:
        return f"R$ {x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def stage_badge(estagio: str):
    # badge simples destacando estágio
    color = {
        "Expansão": "#22c55e",
        "Maturidade": "#06b6d4",
        "Declínio": "#ef4444",
        "Início": "#f59e0b"
    }.get(estagio, "#a3a3a3")
    st.markdown(
        f"""
        <div style="display:inline-block;padding:10px 16px;border-radius:12px;background:{color};color:white;font-weight:700;font-size:20px;">
            Estágio: {estagio}
        </div>
        """,
        unsafe_allow_html=True
    )

tab1, tab2 = st.tabs(["Desafio 1 - Perfil & Estágio", "Desafio 2 - Relações & Risco"])

# -------------------------
# DESAFIO 1
# -------------------------
with tab1:
    st.header("Perfil & Estágio")

    f = load_table("reports/exports/f_empresa_mes.parquet")
    est = load_table("reports/exports/estagio.parquet")

    ids = sorted(f["ID"].unique().tolist())
    sel = st.selectbox("ID", ids)

    df_id = f[f["ID"]==sel].sort_values("year_month")
    meses = df_id["year_month"].dt.strftime("%Y-%m").tolist()
    mes_sel = st.selectbox("Mês (filtro para KPIs e tabela)", meses, index=len(meses)-1 if meses else 0)

    # Série temporal (receita/despesa/fluxo)
    st.line_chart(
        df_id.set_index("year_month")[["receita_mensal","despesa_mensal","fluxo_liquido"]]
    )

    # KPIs
    # estágio (badge) + KPIs do mês filtrado
    est_row = est[est["ID"]==sel]
    estagio = est_row["estagio"].iloc[0] if not est_row.empty else "N/D"
    stage_badge(str(estagio))

    ym = pd.to_datetime(mes_sel)
    rowm = df_id[df_id["year_month"]==ym]
    if not rowm.empty:
        r = rowm.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Receita (mês)", fmt_money(r["receita_mensal"]))
        c2.metric("Despesa (mês)", fmt_money(r["despesa_mensal"]))
        c3.metric("Fluxo (mês)", fmt_money(r["fluxo_liquido"]))
        flag = "SIM" if r["fluxo_liquido"] < 0 else "NÃO"
        c4.metric("Gastos > Receitas?", flag)

    # Tabela do mês selecionado (além do gráfico)
    st.subheader("Detalhe do mês selecionado")
    if not rowm.empty:
        tbl = rowm[["ID","year_month","receita_mensal","despesa_mensal","fluxo_liquido","g_receita_mom","vol_receita_3m"]].copy()
        tbl.columns = ["ID","Mês","Receita","Despesa","Fluxo Líquido","Crescimento MoM","Vol(3m)"]
        st.dataframe(tbl)

    # Sugestão rápida: alerta textual
    with st.expander("Sugestões automatizadas (simples)"):
        if not rowm.empty:
            tips = []
            if r["fluxo_liquido"] < 0:
                tips.append("Fluxo negativo no mês — avaliar custos fixos e prazos com fornecedores.")
            if r["g_receita_mom"] is not None and r["g_receita_mom"] < 0:
                tips.append("Receita caiu vs mês anterior — revisar mix/preços e concentração de clientes.")
            if r["vol_receita_3m"] is not None and r["vol_receita_3m"] > np.nanpercentile(df_id["vol_receita_3m"], 70):
                tips.append("Volatilidade acima do P70 — atenção a previsibilidade do caixa.")
            st.write("- " + "\n- ".join(tips) if tips else "Sem alertas para este mês.")

# -------------------------
# DESAFIO 2
# -------------------------
with tab2:
    st.header("Relações & Risco")

    rel_path = Path("reports/exports/relations_latest.parquet")
    if not rel_path.exists():
        st.warning("relations_latest.parquet não encontrado. Rode o build-all novamente.")
    else:
        rel = load_table(str(rel_path))
        ids_rel = sorted(rel["company_id"].unique().tolist())
        sel_rel = st.selectbox("ID", ids_rel, key="rel_id")

        # Direção e mês p/ ego-graph
        direction = st.radio("Direção", ["in","out","both"], horizontal=True, index=2)
        df_f = load_table("reports/exports/f_empresa_mes.parquet")
        meses_rel = df_f[df_f["ID"]==sel_rel]["year_month"].dt.strftime("%Y-%m").tolist()
        mes_sel_rel = st.selectbox("Mês (grafo)", meses_rel, index=len(meses_rel)-1 if meses_rel else 0)

        # Destaque: relação mais importante (top score) para o ID
        top = rel[rel["company_id"]==sel_rel].sort_values("score", ascending=False).head(1)
        if not top.empty:
            t = top.iloc[0]
            st.markdown("### Relação mais importante (Top Score)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Contraparte", t["counterparty"])
            c2.metric("Direção", t["direction"])
            c3.metric("Share", f"{t['share']:.1%}")
            c4.metric("Score", f"{t['score']:.4f}")

        # Tabela completa (já existia)
        st.subheader("Top relações (tabela)")
        st.dataframe(rel[rel["company_id"]==sel_rel].sort_values("score", ascending=False))
        st.subheader("Grafo Interativo (Dash – Cytoscape)")
        st.info("Se o grafo não carregar, rode o micro-app:  `python -m src.graph.dash_app`")

        company_id = sel_rel  # seu ID já selecionado no Streamlit
        # Monte uma URL com parâmetros default (opcional)
        embed_url = f"http://localhost:8050"
        components.iframe(embed_url, height=820, scrolling=True)