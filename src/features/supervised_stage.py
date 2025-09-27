"""
Classificador supervisionado para estágios empresariais.
Integra Base1 (contexto) + Base2 (comportamento). Sem emojis.
"""
from __future__ import annotations
from typing import Dict, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# Perfis setoriais simplificados
SETOR_PERFIS: Dict[str, Dict[str, Any]] = {
    "MINERACAO": {"ciclo_anos": 10, "capital_intensivo": True},
    "ENERGIA": {"ciclo_anos": 15, "capital_intensivo": True},
    "PETROLEO": {"ciclo_anos": 20, "capital_intensivo": True},
    "SIDERURGIA": {"ciclo_anos": 12, "capital_intensivo": True},
    "CONSTRUCAO": {"ciclo_anos": 7, "capital_intensivo": True},
    "MANUFATURA": {"ciclo_anos": 8, "capital_intensivo": True},
    "QUIMICA": {"ciclo_anos": 9, "capital_intensivo": False},
    "FARMACEUTICA": {"ciclo_anos": 6, "capital_intensivo": False},
    "TECNOLOGIA": {"ciclo_anos": 3, "capital_intensivo": False},
    "VAREJO": {"ciclo_anos": 4, "capital_intensivo": False},
    "SERVICOS": {"ciclo_anos": 3, "capital_intensivo": False},
    "ALIMENTACAO": {"ciclo_anos": 5, "capital_intensivo": False},
    "OUTROS": {"ciclo_anos": 6, "capital_intensivo": False},
}

# Porte
PORTE_FATURAMENTO = {
    "MICRO": (0, 360_000),
    "PEQUENA": (360_000, 4_800_000),
    "MEDIA": (4_800_000, 300_000_000),
    "GRANDE": (300_000_000, float("inf")),
}

# ——— Helpers ———
def _classify_cnae(cnae_desc: str) -> str:
    s = str(cnae_desc).upper()
    if any(t in s for t in ["MINERIO", "MINERA", "EXTRAÇÃO", "EXTRACAO"]):
        return "MINERACAO"
    if any(t in s for t in ["ENERGIA", "ELETRICA", "GERAÇÃO", "GERACAO"]):
        return "ENERGIA"
    if any(t in s for t in ["PETROLEO", "GAS", "COMBUSTIVEL"]):
        return "PETROLEO"
    if any(t in s for t in ["SIDERUR", "METALUR", "AÇO", "ACO", "FERRO"]):
        return "SIDERURGIA"
    if any(t in s for t in ["CONSTRUÇÃO", "CONSTRUCAO", "OBRAS"]):
        return "CONSTRUCAO"
    if any(t in s for t in ["MANUFATURA", "FABRICA", "INDUSTRI"]):
        return "MANUFATURA"
    if "QUIMIC" in s:
        return "QUIMICA"
    if any(t in s for t in ["FARMACEUTICA", "MEDICAMENTO", "FARMACIA"]):
        return "FARMACEUTICA"
    if any(t in s for t in ["TECNOLOGIA", "SOFTWARE", "TI", "DIGITAL"]):
        return "TECNOLOGIA"
    if any(t in s for t in ["VAREJO", "COMERCIO", "LOJA", "VENDA"]):
        return "VAREJO"
    if any(t in s for t in ["SERVICO", "CONSULTORIA", "ASSESSORIA"]):
        return "SERVICOS"
    if any(t in s for t in ["ALIMENTO", "ALIMENTAR", "RESTAURANTE", "COMIDA"]):
        return "ALIMENTACAO"
    return "OUTROS"

def _classify_porte(fat: float) -> str:
    for porte, (lo, hi) in PORTE_FATURAMENTO.items():
        if lo <= fat < hi:
            return porte
    return "MICRO"

def _age_years(dt_open) -> float:
    dt = pd.to_datetime(dt_open, errors="coerce")
    if pd.isna(dt):
        return 5.0
    return max(0.0, (pd.Timestamp.now() - dt).days / 365.25)

# ——— Modelo ———
class SupervisedStageClassifier:
    """
    RandomForest supervisionado usando regras de negócio para gerar os rótulos de treino.
    Estágios padronizados (sem emojis): Início, Crescimento, Maturidade, Declínio, Reestruturação.
    """

    numeric_feature_set = [
        "idade_anos", "VL_FATU", "VL_SLDO", "faturamento_normalizado",
        "ciclo_setor_anos", "capital_intensivo", "saude_saldo",
        "receita_trend", "fluxo_trend", "receita_cv", "fluxo_cv",
        "consistencia_receita", "consistencia_fluxo", "diversificacao_contrapartes",
        "canais_utilizados", "receita_media", "fluxo_medio", "num_meses_ativo",
        "maturidade_relativa", "performance_vs_porte", "eficiencia_capital",
    ]

    def __init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.feature_names: list[str] = []
        self.fitted = False

    # ——— Feature engineering ———
    def extract_features(self, base1: pd.DataFrame, base2: pd.DataFrame) -> pd.DataFrame:
        b1 = base1.copy()
        b1["idade_anos"] = b1["DT_ABRT"].apply(_age_years)
        b1["setor"] = b1["DS_CNAE"].apply(_classify_cnae)
        b1["porte"] = b1["VL_FATU"].apply(_classify_porte)
        b1["saude_saldo"] = (b1["VL_SLDO"] > 0).astype(int)
        b1["faturamento_normalizado"] = np.log1p(pd.to_numeric(b1["VL_FATU"], errors="coerce").fillna(0.0))
        b1["ciclo_setor_anos"] = b1["setor"].map(lambda s: SETOR_PERFIS.get(s, SETOR_PERFIS["OUTROS"])["ciclo_anos"])
        b1["capital_intensivo"] = (
            b1["setor"].map(lambda s: SETOR_PERFIS.get(s, SETOR_PERFIS["OUTROS"])["capital_intensivo"]).astype(int)
        )

        b2 = base2.copy()
        # receitas por recebedor
        rec = b2.groupby(["ID_RCBE", "DT_REFE"]).agg(VL_sum=("VL", "sum")).reset_index()
        rec.columns = ["ID", "DT_REFE", "receita_total"]
        # despesas por pagador
        desp = b2.groupby(["ID_PGTO", "DT_REFE"]).agg(VL_sum=("VL", "sum")).reset_index()
        desp.columns = ["ID", "DT_REFE", "despesa_total"]
        # merge
        m = rec.merge(desp, on=["ID", "DT_REFE"], how="outer").fillna(0.0)
        m["fluxo_liquido"] = m["receita_total"] - m["despesa_total"]

        feats = []
        for cid, g in m.groupby("ID"):
            g = g.sort_values("DT_REFE")
            if len(g) < 2:
                continue
            x = np.arange(len(g))
            receita_trend = np.polyfit(x, g["receita_total"].to_numpy(), 1)[0] if len(g) > 1 else 0.0
            fluxo_trend = np.polyfit(x, g["fluxo_liquido"].to_numpy(), 1)[0] if len(g) > 1 else 0.0
            receita_cv = float(g["receita_total"].std() / (g["receita_total"].mean() + 1e-6))
            fluxo_cv = float(g["fluxo_liquido"].std() / (abs(g["fluxo_liquido"].mean()) + 1e-6))
            cons_rec = float((g["receita_total"] > 0).mean())
            cons_fluxo = float((g["fluxo_liquido"] > 0).mean())

            trans_cid = b2[(b2["ID_RCBE"] == cid) | (b2["ID_PGTO"] == cid)]
            div_in = trans_cid[trans_cid["ID_RCBE"] == cid]["ID_PGTO"].nunique()
            div_out = trans_cid[trans_cid["ID_PGTO"] == cid]["ID_RCBE"].nunique()
            divers = int(div_in + div_out)
            canais = int(trans_cid["DS_TRAN"].nunique())

            feats.append(
                {
                    "ID": cid,
                    "receita_trend": receita_trend,
                    "fluxo_trend": fluxo_trend,
                    "receita_cv": receita_cv,
                    "fluxo_cv": fluxo_cv,
                    "consistencia_receita": cons_rec,
                    "consistencia_fluxo": cons_fluxo,
                    "diversificacao_contrapartes": divers,
                    "canais_utilizados": canais,
                    "receita_media": float(g["receita_total"].mean()),
                    "fluxo_medio": float(g["fluxo_liquido"].mean()),
                    "num_meses_ativo": int(len(g)),
                }
            )

        bh = pd.DataFrame(feats)
        final = b1.merge(bh, left_on="ID", right_on="ID", how="inner")

        # features híbridas
        final["maturidade_relativa"] = final["idade_anos"] / final["ciclo_setor_anos"].replace(0, 1)
        final["performance_vs_porte"] = final["receita_media"] / (final["VL_FATU"] / 12 + 1e-6)
        final["eficiencia_capital"] = final["fluxo_medio"] / (final["VL_SLDO"].abs() + 1e-6)

        # limpeza
        final = final.replace([np.inf, -np.inf], np.nan)
        num_cols = final.select_dtypes(include=[np.number]).columns
        final[num_cols] = final[num_cols].fillna(final[num_cols].median())
        cat_cols = final.select_dtypes(exclude=[np.number]).columns
        for c in cat_cols:
            if c not in ["ID", "DS_CNAE"]:
                if final[c].isna().any():
                    mode = final[c].mode()
                    final[c] = final[c].fillna(mode.iloc[0] if len(mode) else "OUTROS")

        return final

    # ——— Regras de negócio -> labels ———
    @staticmethod
    def _business_labels(df: pd.DataFrame) -> pd.Series:
        labels = []
        for _, r in df.iterrows():
            idade = r["idade_anos"]
            mat_rel = r["maturidade_relativa"]
            porte = r["porte"]
            cons_fluxo = r["consistencia_fluxo"]
            rec_trend = r["receita_trend"]
            fluxo_med = r["fluxo_medio"]
            divers = r["diversificacao_contrapartes"]

            if (idade < 2) or (mat_rel < 0.3) or (porte == "MICRO" and idade < 3):
                s = "Início"
            elif (rec_trend > 0 and cons_fluxo > 0.6 and fluxo_med > 0 and 0.3 <= mat_rel < 0.7 and divers >= 3):
                s = "Crescimento"
            elif (mat_rel >= 0.7 and cons_fluxo > 0.7 and abs(rec_trend) < 1000 and divers >= 5 and fluxo_med > 0):
                s = "Maturidade"
            elif (fluxo_med < 0) or (rec_trend < -1000) or (cons_fluxo < 0.3):
                s = "Declínio"
            else:
                s = "Reestruturação"
            labels.append(s)
        return pd.Series(labels, index=df.index)

    # ——— Treino/Predict ———
    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        feats = [c for c in self.numeric_feature_set if c in df.columns]
        self.feature_names = feats
        X = df[feats].fillna(0.0).to_numpy()
        Xs = self.scaler.fit_transform(X)
        return Xs

    def fit(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        y = self._business_labels(features_df)
        X = self._prepare_X(features_df)
        y_enc = self.le.fit_transform(y)

        Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
        self.model.fit(Xtr, ytr)
        yp = self.model.predict(Xte)
        acc = float((yp == yte).mean())
        rpt = classification_report(yte, yp, target_names=list(self.le.classes_), output_dict=True)
        imp = dict(zip(self.feature_names, self.model.feature_importances_))
        self.fitted = True
        return {
            "accuracy": acc,
            "classification_report": rpt,
            "feature_importance": imp,
            "stage_distribution": dict(pd.Series(self.le.inverse_transform(y_enc)).value_counts()),
        }

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        X = self._prepare_X(features_df)
        y_enc = self.model.predict(X)
        proba = self.model.predict_proba(X)
        y_lbl = self.le.inverse_transform(y_enc)
        conf = np.max(proba, axis=1)

        out = pd.DataFrame(
            {
                "ID": features_df["ID"],
                "estagio": y_lbl,
                "confianca": conf,
                "idade_anos": features_df["idade_anos"],
                "setor": features_df["setor"],
                "porte": features_df["porte"],
                "receita_media": features_df.get("receita_media", 0.0),
                "fluxo_medio": features_df.get("fluxo_medio", 0.0),
                "consistencia_fluxo": features_df.get("consistencia_fluxo", 0.0),
            }
        )
        return out

    def train_and_classify(self, base1: pd.DataFrame, base2: pd.DataFrame) -> pd.DataFrame:
        feats = self.extract_features(base1, base2)
        self.fit(feats)
        return self.predict(feats)

# ——— Função wrapper esperada pelo main.py ———
def train_and_classify_stages(base1_data: pd.DataFrame, base2_data: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper compatível com o pipeline:
        from ..features.supervised_stage import train_and_classify_stages
    """
    clf = SupervisedStageClassifier()
    return clf.train_and_classify(base1_data, base2_data)