# src/features/advanced_stage.py
"""
Modelo não supervisionado (avançado) para classificar estágios empresariais
com base em métricas financeiras/temporais agregadas.
Sem emojis; rótulos padronizados (com acentuação).
"""
from __future__ import annotations

from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# Mantemos os nomes em PT-BR com acentuação, sem emojis (compatível com a UI e engine de insights)
STAGES = ["Início", "Crescimento", "Maturidade", "Declínio", "Reestruturação"]

STAGE_DESCRIPTIONS: Dict[str, str] = {
    "Início": "Empresa em fase inicial com receitas baixas e alta incerteza.",
    "Crescimento": "Empresa em expansão com crescimento consistente de receita.",
    "Maturidade": "Empresa estável com fluxos previsíveis e baixa volatilidade.",
    "Declínio": "Queda de performance e fluxos negativos recorrentes.",
    "Reestruturação": "Processo de recuperação após dificuldades.",
}


class AdvancedStageModel:
    """
    Pipeline de clustering:
      extract_features -> scale/PCA -> k-means -> mapping -> classify
    """

    # Lista-alvo de features numéricas (selecionaremos apenas as disponíveis)
    numeric_features = [
        "receita_mensal_media",
        "crescimento_mom_medio",
        "volatilidade_receita",
        "fluxo_liquido_medio",
        "consistencia_fluxo_positivo",
        "margem_media",
        "tendencia_receita",
        "coef_variacao_receita",
        "taxa_recuperacao",
    ]

    def extract_features(self, f_empresa_mes: pd.DataFrame, months_window: int = 12) -> pd.DataFrame:
        """
        Extrai métricas por empresa, considerando a janela (últimos N meses).
        Requisitos: colunas 'ID', 'year_month' (datetime), e colunas financeiras básicas.
        """
        f = f_empresa_mes.copy()

        if "year_month" not in f.columns:
            raise ValueError("f_empresa_mes precisa conter 'year_month' (datetime).")

        f["year_month"] = pd.to_datetime(f["year_month"], errors="coerce")
        f = f.dropna(subset=["year_month"])

        f = (
            f.sort_values(["ID", "year_month"])
            .groupby("ID", group_keys=False)
            .tail(months_window)
        )

        feats: List[dict] = []

        for comp, g in f.groupby("ID"):
            g = g.sort_values("year_month")
            if len(g) < 3:
                continue

            receita_mensal = pd.to_numeric(g.get("receita_mensal", 0.0), errors="coerce").fillna(0.0)
            despesa_mensal = pd.to_numeric(g.get("despesa_mensal", 0.0), errors="coerce").fillna(0.0)
            fluxo_liq = pd.to_numeric(g.get("fluxo_liquido", receita_mensal - despesa_mensal), errors="coerce").fillna(0.0)
            g_mom = pd.to_numeric(g.get("g_receita_mom", np.nan), errors="coerce")
            vol3 = pd.to_numeric(g.get("vol_receita_3m", np.nan), errors="coerce")

            # tendência linear da receita
            x = np.arange(len(g))
            receita_trend = np.polyfit(x, receita_mensal.to_numpy(), 1)[0] if len(g) > 1 else 0.0

            # consistência de fluxo positivo
            consist_fluxo_pos = float((fluxo_liq > 0).mean())

            # contagens de recuperação após queda forte de receita
            rec = 0
            if g_mom.notna().any():
                for i in range(1, len(g)):
                    prev_drop = bool(g_mom.iloc[i - 1] < -0.10) if not pd.isna(g_mom.iloc[i - 1]) else False
                    now_up = bool(g_mom.iloc[i] > 0) if not pd.isna(g_mom.iloc[i]) else False
                    if prev_drop and now_up:
                        rec += 1
                quedas = int((g_mom < -0.10).sum())
            else:
                quedas = 0
            taxa_rec = rec / (quedas + 1e-6)

            # volatilidade (usa vol_receita_3m se disponível; senão, std da série)
            if vol3.notna().any():
                vol_receita = float(vol3.fillna(0.0).mean())
            else:
                vol_receita = float(receita_mensal.std())

            # coeficiente de variação
            rec_mean = float(receita_mensal.mean())
            coef_var = float(receita_mensal.std() / (rec_mean + 1e-6) if rec_mean > 0 else 0.0)

            feats.append(
                {
                    "ID": comp,
                    "receita_mensal_media": rec_mean,
                    "receita_total_periodo": float(receita_mensal.sum()),
                    "crescimento_mom_medio": float(
                        g_mom.replace([np.inf, -np.inf], np.nan).mean(skipna=True) if g_mom.notna().any() else 0.0
                    ),
                    "volatilidade_receita": vol_receita,
                    "fluxo_liquido_medio": float(fluxo_liq.mean()),
                    "margem_media": float((fluxo_liq / (receita_mensal + 1e-6)).mean()),
                    "tendencia_receita": receita_trend,
                    "coef_variacao_receita": coef_var,
                    "taxa_recuperacao": float(taxa_rec),
                    "consistencia_fluxo_positivo": consist_fluxo_pos,
                    "num_meses": int(len(g)),
                }
            )

        return pd.DataFrame(feats)

    @staticmethod
    def _determine_k(features_scaled: np.ndarray, max_k: int = 8) -> int:
        """Escolhe k por cotovelo + silhouette, limitado entre 2 e 5."""
        n = len(features_scaled)
        k_range = range(2, max(3, min(max_k + 1, n)))
        inertias, sils = [], []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(features_scaled)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(features_scaled, labels) if k < n else 0.0)

        # cotovelo
        if len(inertias) >= 3:
            d1 = np.diff(inertias)
            d2 = np.diff(d1)
            elbow_idx = int(np.argmax(np.abs(d2))) + 2
            elbow_k = list(k_range)[min(elbow_idx, len(list(k_range)) - 1)]
        else:
            elbow_k = list(k_range)[0]

        best_sil_k = list(k_range)[int(np.argmax(sils))] if sils else elbow_k
        optimal = best_sil_k if abs(best_sil_k - elbow_k) <= 1 else min(5, elbow_k)
        return int(max(2, min(optimal, 5)))

    @staticmethod
    def _map_clusters(features_df: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
        """Regras de mapeamento de perfil do cluster -> estágio de negócio."""
        profiles = {}
        for c in np.unique(labels):
            mask = labels == c
            sub = features_df.loc[mask]
            profiles[c] = {
                "receita_media": sub["receita_mensal_media"].mean(),
                "crescimento": sub["crescimento_mom_medio"].mean(),
                "fluxo": sub["fluxo_liquido_medio"].mean(),
                "consist": sub.get("consistencia_fluxo_positivo", pd.Series([0.0] * len(sub))).mean(),
                "vol": sub["volatilidade_receita"].mean(),
            }

        rec_vals = [p["receita_media"] for p in profiles.values()] or [0.0]
        fluxo_vals = [p["fluxo"] for p in profiles.values()] or [0.0]
        vol_vals = [p["vol"] for p in profiles.values()] or [0.0]
        p25 = np.percentile(rec_vals, 25)
        p50_fluxo = np.percentile(fluxo_vals, 50)
        p50_vol = np.percentile(vol_vals, 50)

        mapping = {}
        for c, p in profiles.items():
            if p["receita_media"] <= p25 and p["vol"] > p50_vol:
                mapping[c] = "Início"
            elif p["fluxo"] > p50_fluxo and p["crescimento"] > 0.02:
                mapping[c] = "Crescimento"
            elif p["fluxo"] > 0 and p["vol"] < p50_vol and abs(p["crescimento"]) < 0.05:
                mapping[c] = "Maturidade"
            elif p["fluxo"] < -1000 or p["crescimento"] < -0.05:
                mapping[c] = "Declínio"
            else:
                mapping[c] = "Reestruturação"
        return mapping

    def classify(self, f_empresa_mes: pd.DataFrame, months_window: int = 12) -> pd.DataFrame:
        """Executa todo o pipeline e retorna ID/estágio/confianca/cluster_id."""
        feats = self.extract_features(f_empresa_mes, months_window)
        if feats.empty:
            return pd.DataFrame(columns=["ID", "estagio", "confianca", "cluster_id"])

        if len(feats) < 5:
            out = feats[["ID"]].copy()
            out["estagio"] = "Maturidade"
            out["confianca"] = 0.5
            out["cluster_id"] = 0
            return out

        # Seleciona apenas as features disponíveis
        avail = [c for c in self.numeric_features if c in feats.columns]
        X = feats[avail].copy()

        # Sanitização
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))

        # Clipping leve
        for col in X.columns:
            q01, q99 = X[col].quantile(0.01), X[col].quantile(0.99)
            X[col] = X[col].clip(q01, q99)

        # Escalonamento e PCA opcional
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)

        if Xs.shape[1] > 6:
            pca = PCA(n_components=min(6, Xs.shape[0] - 1))
            Xs = pca.fit_transform(Xs)

        # Clustering
        k = self._determine_k(Xs)
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(Xs)

        # Confiança = 1 - distância_normalizada ao centróide
        dists = km.transform(Xs)
        min_d = np.min(dists, axis=1)
        conf = 1.0 - (min_d / (min_d.max() + 1e-6))

        # Mapeamento cluster -> estágio
        mapping = self._map_clusters(feats, labels)

        result = feats[["ID"]].copy()
        result["estagio"] = [mapping[l] for l in labels]
        result["confianca"] = conf
        result["cluster_id"] = labels
        return result


# --- Wrapper público esperado pelo main.py (compat) ---
def classify_advanced_stage(f_empresa_mes: pd.DataFrame, months_window: int = 12) -> pd.DataFrame:
    """
    Função de compatibilidade para o pipeline não supervisionado.
    """
    model = AdvancedStageModel()
    return model.classify(f_empresa_mes, months_window=months_window)
