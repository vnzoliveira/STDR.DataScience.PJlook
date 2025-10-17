"""
Análise de Risco de Interdependência e Cenários de Falência
Corrigido: definição de cliente/fornecedor e uso de janela temporal.
"""
from __future__ import annotations
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass


@dataclass
class RiskScenario:
    scenario_name: str
    affected_companies: List[str]
    impact_percentage: float
    risk_level: str  # "BAIXO", "MÉDIO", "ALTO", "CRÍTICO"
    description: str
    mitigation_strategies: List[str]


class ValueChainRiskAnalyzer:
    def __init__(self):
        self.risk_thresholds = {
            "concentracao_critica": 0.5,
            "concentracao_alta": 0.3,
            "dependencia_minima": 0.05,
        }

    # ------------------------- utils -------------------------

    @staticmethod
    def _ensure_year_month(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "year_month" not in out.columns:
            out["year_month"] = pd.to_datetime(out["DT_REFE"]).dt.to_period("M").dt.to_timestamp()
        return out

    def _filter_last_months(self, df: pd.DataFrame, months_window: int) -> pd.DataFrame:
        df = self._ensure_year_month(df)
        if df.empty:
            return df
        last = df["year_month"].max()
        first = (last - pd.offsets.MonthBegin(months_window - 1)).to_period("M").to_timestamp()
        return df[(df["year_month"] >= first) & (df["year_month"] <= last)].copy()

    def _classify_risk_level(self, impact_pct: float) -> str:
        if impact_pct > 0.5:
            return "CRÍTICO"
        elif impact_pct > 0.3:
            return "ALTO"
        elif impact_pct > 0.15:
            return "MÉDIO"
        else:
            return "BAIXO"

    # ---------------------- métricas básicas ----------------------

    def calculate_concentration_metrics(self, transactions_df: pd.DataFrame, company_id: str, direction: str = "both") -> Dict[str, float]:
        """
        direction:
          - "in"  -> quem paga para a empresa (receita)
          - "out" -> quem recebe da empresa (pagamentos)
          - "both"
        """
        df = transactions_df
        if direction == "in":
            company_data = df[df["ID_RCBE"] == company_id]
            partner_col = "ID_PGTO"
        elif direction == "out":
            company_data = df[df["ID_PGTO"] == company_id]
            partner_col = "ID_RCBE"
        else:
            in_data = df[df["ID_RCBE"] == company_id].copy()
            in_data["partner"] = in_data["ID_PGTO"]
            out_data = df[df["ID_PGTO"] == company_id].copy()
            out_data["partner"] = out_data["ID_RCBE"]
            company_data = pd.concat([in_data, out_data], ignore_index=True)
            partner_col = "partner"

        if company_data.empty:
            return {"hhi": 0, "top_partner_share": 0, "top_3_share": 0, "num_partners": 0, "concentration_risk": "BAIXO", "total_volume": 0}

        partner_volumes = company_data.groupby(partner_col)["VL"].sum().sort_values(ascending=False)
        total_volume = partner_volumes.sum()
        if total_volume == 0:
            return {"hhi": 0, "top_partner_share": 0, "top_3_share": 0, "num_partners": 0, "concentration_risk": "BAIXO", "total_volume": 0}

        shares = partner_volumes / total_volume
        hhi = float((shares ** 2).sum())
        top_3_share = float(shares.head(3).sum())
        num_partners = int(len(partner_volumes))

        if shares.iloc[0] > self.risk_thresholds["concentracao_critica"]:
            risk_level = "CRÍTICO"
        elif top_3_share > 0.8:
            risk_level = "ALTO"
        elif top_3_share > 0.6:
            risk_level = "MÉDIO"
        else:
            risk_level = "BAIXO"

        return {
            "hhi": hhi,
            "top_partner_share": float(shares.iloc[0]),
            "top_3_share": top_3_share,
            "num_partners": num_partners,
            "concentration_risk": risk_level,
            "total_volume": float(total_volume),
        }

    def simulate_partner_failure(self, transactions_df: pd.DataFrame, company_id: str, failed_partner_id: str) -> Dict[str, Any]:
        """Impactos relativos em receita e pagamentos."""
        partner_volume_in = transactions_df[
            (transactions_df["ID_RCBE"] == company_id) & (transactions_df["ID_PGTO"] == failed_partner_id)
        ]["VL"].sum()

        partner_volume_out = transactions_df[
            (transactions_df["ID_PGTO"] == company_id) & (transactions_df["ID_RCBE"] == failed_partner_id)
        ]["VL"].sum()

        total_in = transactions_df[transactions_df["ID_RCBE"] == company_id]["VL"].sum()
        total_out = transactions_df[transactions_df["ID_PGTO"] == company_id]["VL"].sum()

        impact_receita = (partner_volume_in / total_in) if total_in > 0 else 0.0
        impact_pagamentos = (partner_volume_out / total_out) if total_out > 0 else 0.0
        impact_total = max(impact_receita, impact_pagamentos)

        if impact_total > 0.5:
            severity, recovery = "CATASTRÓFICO", "12+ meses"
        elif impact_total > 0.3:
            severity, recovery = "SEVERO", "6-12 meses"
        elif impact_total > 0.15:
            severity, recovery = "MODERADO", "3-6 meses"
        else:
            severity, recovery = "LEVE", "1-3 meses"

        return {
            "failed_partner": failed_partner_id,
            "impact_receita_pct": float(impact_receita),
            "impact_pagamentos_pct": float(impact_pagamentos),
            "impact_total_pct": float(impact_total),
            "severity": severity,
            "recovery_time_estimate": recovery,
            "volume_at_risk_in": float(partner_volume_in),
            "volume_at_risk_out": float(partner_volume_out),
        }

    # ---------------------- cenários & relatório ----------------------

    def generate_risk_scenarios(self, transactions_df: pd.DataFrame, company_id: str, top_n: int = 5) -> List[RiskScenario]:
        """
        Corrigido:
          - **Fornecedor** = quem recebe da empresa (outflow).
          - **Cliente**    = quem paga para a empresa (inflow).
        """
        df = transactions_df

        # Top CLIENTES (receita)
        top_clients = (df[df["ID_RCBE"] == company_id]
                       .groupby("ID_PGTO")["VL"].sum().sort_values(ascending=False).head(top_n))

        # Top FORNECEDORES (pagamentos)
        top_suppliers = (df[df["ID_PGTO"] == company_id]
                         .groupby("ID_RCBE")["VL"].sum().sort_values(ascending=False).head(top_n))

        scenarios: List[RiskScenario] = []

        # Cenário 1: Perda do principal cliente (impacta RECEITA)
        if len(top_clients) > 0:
            top_client = top_clients.index[0]
            imp = self.simulate_partner_failure(df, company_id, top_client)
            scenarios.append(RiskScenario(
                scenario_name="Perda do Principal Cliente",
                affected_companies=[top_client],
                impact_percentage=imp["impact_receita_pct"] * 100.0,
                risk_level=self._classify_risk_level(imp["impact_receita_pct"]),
                description=f"Perda do cliente {top_client} causaria {imp['impact_receita_pct']:.1%} de impacto na receita",
                mitigation_strategies=["Diversificar carteira de clientes", "Novos canais de venda", "Fidelização"]
            ))

        # Cenário 2: Falência do principal fornecedor (impacta PAGAMENTOS/OPERAÇÃO)
        if len(top_suppliers) > 0:
            top_supplier = top_suppliers.index[0]
            imp = self.simulate_partner_failure(df, company_id, top_supplier)
            scenarios.append(RiskScenario(
                scenario_name="Falência do Principal Fornecedor",
                affected_companies=[top_supplier],
                impact_percentage=imp["impact_pagamentos_pct"] * 100.0,
                risk_level=self._classify_risk_level(imp["impact_pagamentos_pct"]),
                description=f"Perda do fornecedor {top_supplier} causaria {imp['impact_pagamentos_pct']:.1%} de impacto nos pagamentos",
                mitigation_strategies=["Diversificar fornecedores", "Contratos alternativos", "Estoque de segurança"]
            ))

        # Cenário 3: Crise setorial (top 3 combinados)
        top3 = list(set(list(top_clients.head(3).index) + list(top_suppliers.head(3).index)))
        if len(top3) >= 2:
            impacts = [self.simulate_partner_failure(df, company_id, p) for p in top3[:3]]
            combined = min(1.0, sum(max(i["impact_receita_pct"], i["impact_pagamentos_pct"]) for i in impacts))
            scenarios.append(RiskScenario(
                scenario_name="Crise Setorial (Top 3 Parceiros)",
                affected_companies=top3[:3],
                impact_percentage=combined * 100.0,
                risk_level=self._classify_risk_level(combined),
                description=f"Crise simultânea dos 3 principais parceiros causaria {combined:.1%} de impacto",
                mitigation_strategies=["Diversificação geográfica e setorial", "Contratos de longo prazo", "Reservas de emergência"]
            ))

        return scenarios

    def generate_monthly_risk_report(self, transactions_df: pd.DataFrame, company_id: str, months_window: int = 6) -> Dict[str, Any]:
        """Relatório consolidado respeitando a janela temporal."""
        df = self._filter_last_months(transactions_df, months_window)

        metrics_in = self.calculate_concentration_metrics(df, company_id, "in")
        metrics_out = self.calculate_concentration_metrics(df, company_id, "out")
        scenarios = self.generate_risk_scenarios(df, company_id)

        current_partners = list(set(df[df["ID_RCBE"] == company_id]["ID_PGTO"].unique()) |
                                set(df[df["ID_PGTO"] == company_id]["ID_RCBE"].unique()))

        diversification_suggestions = self.suggest_diversification_targets(df, company_id, current_partners)

        overall_risk_score = (
            metrics_in["hhi"] * 30
            + metrics_out["hhi"] * 30
            + len([s for s in scenarios if s.risk_level in ["ALTO", "CRÍTICO"]]) * 10
            + (1 / max(1, len(current_partners))) * 30
        )
        overall_risk_score = float(min(100.0, overall_risk_score))

        return {
            "company_id": company_id,
            "period_months": months_window,
            "concentration_metrics": {"recebimentos": metrics_in, "pagamentos": metrics_out},
            "risk_scenarios": scenarios,
            "diversification_suggestions": diversification_suggestions,
            "overall_risk_score": overall_risk_score,
            "risk_classification": self._classify_overall_risk(overall_risk_score),
            "current_partners_count": len(current_partners),
        }

    def _classify_overall_risk(self, score: float) -> str:
        if score > 70:
            return "CRÍTICO"
        elif score > 50:
            return "ALTO"
        elif score > 30:
            return "MÉDIO"
        else:
            return "BAIXO"

    # ---------------------- sugestões ----------------------

    def suggest_diversification_targets(self, transactions_df: pd.DataFrame, company_id: str, current_partners: List[str]) -> List[Dict[str, Any]]:
        potential = set()
        for p in current_partners:
            a = set(transactions_df[transactions_df["ID_PGTO"] == p]["ID_RCBE"].unique())
            b = set(transactions_df[transactions_df["ID_RCBE"] == p]["ID_PGTO"].unique())
            potential |= (a | b)

        potential.discard(company_id)
        potential -= set(current_partners)

        suggestions = []
        for partner in list(potential)[:10]:
            tx = transactions_df[(transactions_df["ID_PGTO"] == partner) | (transactions_df["ID_RCBE"] == partner)]
            avg_volume = float(tx["VL"].mean()) if not tx.empty else 0.0
            num_partners = len(set(tx["ID_PGTO"].unique()) | set(tx["ID_RCBE"].unique()))
            suggestions.append({
                "partner_id": partner,
                "avg_transaction_volume": avg_volume,
                "diversification_level": num_partners,
                "risk_score": self._calculate_partner_risk_score(transactions_df, partner),
                "recommendation_reason": self._get_recommendation_reason(avg_volume, num_partners),
            })

        suggestions.sort(key=lambda x: (x["risk_score"], -x["avg_transaction_volume"]))
        return suggestions[:5]

    def _calculate_partner_risk_score(self, transactions_df: pd.DataFrame, partner_id: str) -> float:
        partner_transactions = transactions_df[(transactions_df["ID_PGTO"] == partner_id) | (transactions_df["ID_RCBE"] == partner_id)]
        if len(partner_transactions) < 2:
            return 0.8
        monthly = partner_transactions.groupby("year_month")["VL"].sum()
        volatility = float(monthly.std() / (monthly.mean() + 1e-6))
        conc = self.calculate_concentration_metrics(transactions_df, partner_id, "both")
        risk = min(1.0, min(0.5, volatility / 10) + conc["top_partner_share"] * 0.3 + (1 / max(1, conc["num_partners"])) * 0.2)
        return float(risk)

    def _get_recommendation_reason(self, avg_volume: float, num_partners: int) -> str:
        if avg_volume > 100000 and num_partners > 10:
            return "Alto volume + bem diversificado"
        elif avg_volume > 100000:
            return "Alto potencial de volume"
        elif num_partners > 15:
            return "Muito bem conectado na rede"
        else:
            return "Oportunidade de diversificação"
