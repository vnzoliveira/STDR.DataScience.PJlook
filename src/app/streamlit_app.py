import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

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

        # ---- Grafo (ego-network) simples ----
        # carrega base transacional mensal para montar as arestas do mês
        b2p = Path("data/processed/base2/base2.parquet")
        if b2p.exists():
            df_edges = pd.read_parquet(str(b2p))
            df_edges["year_month"] = df_edges["DT_REFE"]

            # import local para evitar dependência circular
            from src.sna.relations import ego_edges_by_month

            edges = ego_edges_by_month(df_edges, sel_rel, mes_sel_rel, direction=direction, top_n=15)
            st.subheader("Grafo do mês (ego-network)")
            if edges.empty:
                st.info("Sem arestas para o mês/direção selecionados.")
            else:
                # monta e desenha grafo
                G = nx.DiGraph()
                for _, r in edges.iterrows():
                    G.add_edge(r["src"], r["dst"], weight=float(r["valor"]))

                pos = nx.spring_layout(G, seed=42, k=None)  # layout estável
                plt.figure(figsize=(8,6))
                # nós — destaca a empresa selecionada
                node_colors = ["#ef4444" if n==sel_rel else "#60a5fa" for n in G.nodes()]
                nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, alpha=0.9)
                # arestas com largura proporcional ao valor
                weights = [max(1.0, d["weight"]/edges["valor"].max()*6.0) for _,_,d in G.edges(data=True)]
                nx.draw_networkx_edges(G, pos, width=weights, arrows=True, arrowstyle="-|>", alpha=0.6)
                nx.draw_networkx_labels(G, pos, font_size=9)

                st.pyplot(plt.gcf())
                plt.close()
        else:
            st.info("Arquivo de transações processadas não encontrado (data/processed/base2/base2.parquet). Rode o build-all.")
