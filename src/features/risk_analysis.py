# src/features/risk_analysis.py
"""
Análise de Risco de Interdependência e Cenários de Falência
Sistema para avaliar impacto de quebra de contrapartes na cadeia de valor
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass

@dataclass
class RiskScenario:
    """Cenário de risco para análise de impacto."""
    scenario_name: str
    affected_companies: List[str]
    impact_percentage: float
    risk_level: str  # "BAIXO", "MEDIO", "ALTO", "CRITICO"
    description: str
    mitigation_strategies: List[str]

class ValueChainRiskAnalyzer:
    """
    Analisador de risco da cadeia de valor.
    Calcula cenários de interdependência e impacto de falências.
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'concentracao_critica': 0.5,    # > 50% com um parceiro = crítico
            'concentracao_alta': 0.3,       # > 30% = alto risco
            'dependencia_minima': 0.05      # < 5% = baixo risco
        }
    
    def calculate_concentration_metrics(self, 
                                      transactions_df: pd.DataFrame,
                                      company_id: str,
                                      direction: str = "both") -> Dict[str, float]:
        """
        Calcula métricas de concentração para análise de risco.
        """
        # Filtrar transações da empresa
        if direction == "in":  # Recebimentos
            company_data = transactions_df[transactions_df["ID_RCBE"] == company_id]
            partner_col = "ID_PGTO"
        elif direction == "out":  # Pagamentos
            company_data = transactions_df[transactions_df["ID_PGTO"] == company_id]
            partner_col = "ID_RCBE"
        else:  # Ambos
            in_data = transactions_df[transactions_df["ID_RCBE"] == company_id].copy()
            in_data["partner"] = in_data["ID_PGTO"]
            out_data = transactions_df[transactions_df["ID_PGTO"] == company_id].copy()
            out_data["partner"] = out_data["ID_RCBE"]
            company_data = pd.concat([in_data, out_data])
            partner_col = "partner"
        
        if company_data.empty:
            return {"hhi": 0, "top_3_share": 0, "num_partners": 0, "concentration_risk": "BAIXO"}
        
        # Calcular concentração por parceiro
        partner_volumes = company_data.groupby(partner_col)["VL"].sum().sort_values(ascending=False)
        total_volume = partner_volumes.sum()
        
        if total_volume == 0:
            return {
                "hhi": 0, 
                "top_partner_share": 0, 
                "top_3_share": 0, 
                "num_partners": 0, 
                "concentration_risk": "BAIXO",
                "total_volume": 0
            }
        
        # Métricas de concentração
        shares = partner_volumes / total_volume
        hhi = float((shares ** 2).sum())  # Herfindahl-Hirschman Index
        top_3_share = float(shares.head(3).sum())
        num_partners = len(partner_volumes)
        
        # Classificação de risco
        if shares.iloc[0] > self.risk_thresholds['concentracao_critica']:
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
            "total_volume": float(total_volume)
        }
    
    def simulate_partner_failure(self,
                                transactions_df: pd.DataFrame,
                                company_id: str,
                                failed_partner_id: str) -> Dict[str, Any]:
        """
        Simula o impacto da falência de um parceiro específico.
        """
        # Calcular volume atual com o parceiro
        partner_volume_in = transactions_df[
            (transactions_df["ID_RCBE"] == company_id) & 
            (transactions_df["ID_PGTO"] == failed_partner_id)
        ]["VL"].sum()
        
        partner_volume_out = transactions_df[
            (transactions_df["ID_PGTO"] == company_id) & 
            (transactions_df["ID_RCBE"] == failed_partner_id)
        ]["VL"].sum()
        
        # Volume total da empresa
        total_in = transactions_df[transactions_df["ID_RCBE"] == company_id]["VL"].sum()
        total_out = transactions_df[transactions_df["ID_PGTO"] == company_id]["VL"].sum()
        
        # Calcular impactos
        impact_receita = (partner_volume_in / total_in) if total_in > 0 else 0
        impact_pagamentos = (partner_volume_out / total_out) if total_out > 0 else 0
        impact_total = max(impact_receita, impact_pagamentos)
        
        # Classificar severidade
        if impact_total > 0.5:
            severity = "CATASTRÓFICO"
            recovery_time = "12+ meses"
        elif impact_total > 0.3:
            severity = "SEVERO"
            recovery_time = "6-12 meses"
        elif impact_total > 0.15:
            severity = "MODERADO"
            recovery_time = "3-6 meses"
        else:
            severity = "LEVE"
            recovery_time = "1-3 meses"
        
        return {
            "failed_partner": failed_partner_id,
            "impact_receita_pct": impact_receita,
            "impact_pagamentos_pct": impact_pagamentos,
            "impact_total_pct": impact_total,
            "severity": severity,
            "recovery_time_estimate": recovery_time,
            "volume_at_risk_in": float(partner_volume_in),
            "volume_at_risk_out": float(partner_volume_out)
        }
    
    def generate_risk_scenarios(self,
                               transactions_df: pd.DataFrame,
                               company_id: str,
                               top_n: int = 5) -> List[RiskScenario]:
        """
        Gera cenários de risco baseados nos principais parceiros.
        """
        scenarios = []
        
        # Identificar top parceiros (receitas)
        top_suppliers = (transactions_df[transactions_df["ID_RCBE"] == company_id]
                        .groupby("ID_PGTO")["VL"].sum()
                        .sort_values(ascending=False)
                        .head(top_n))
        
        # Identificar top clientes (pagamentos)
        top_customers = (transactions_df[transactions_df["ID_PGTO"] == company_id]
                        .groupby("ID_RCBE")["VL"].sum()
                        .sort_values(ascending=False)
                        .head(top_n))
        
        # Cenário 1: Falência do maior fornecedor
        if len(top_suppliers) > 0:
            top_supplier = top_suppliers.index[0]
            supplier_impact = self.simulate_partner_failure(transactions_df, company_id, top_supplier)
            
            scenarios.append(RiskScenario(
                scenario_name="Falência do Principal Fornecedor",
                affected_companies=[top_supplier],
                impact_percentage=supplier_impact["impact_receita_pct"] * 100,
                risk_level=self._classify_risk_level(supplier_impact["impact_receita_pct"]),
                description=f"Perda do fornecedor {top_supplier} causaria {supplier_impact['impact_receita_pct']:.1%} de impacto na receita",
                mitigation_strategies=[
                    "Diversificar base de fornecedores",
                    "Contratos de exclusividade alternativos",
                    "Manter estoque de segurança estratégico"
                ]
            ))
        
        # Cenário 2: Falência do maior cliente
        if len(top_customers) > 0:
            top_customer = top_customers.index[0]
            customer_impact = self.simulate_partner_failure(transactions_df, company_id, top_customer)
            
            scenarios.append(RiskScenario(
                scenario_name="Perda do Principal Cliente",
                affected_companies=[top_customer],
                impact_percentage=customer_impact["impact_pagamentos_pct"] * 100,
                risk_level=self._classify_risk_level(customer_impact["impact_pagamentos_pct"]),
                description=f"Perda do cliente {top_customer} causaria {customer_impact['impact_pagamentos_pct']:.1%} de impacto nos pagamentos",
                mitigation_strategies=[
                    "Diversificar carteira de clientes",
                    "Desenvolver novos canais de venda",
                    "Fidelizar clientes existentes"
                ]
            ))
        
        # Cenário 3: Crise setorial (top 3 parceiros)
        top_3_partners = list(top_suppliers.head(3).index) + list(top_customers.head(3).index)
        top_3_partners = list(set(top_3_partners))  # Remover duplicatas
        
        if len(top_3_partners) >= 2:
            combined_impact = sum([
                self.simulate_partner_failure(transactions_df, company_id, partner)["impact_total_pct"]
                for partner in top_3_partners[:3]
            ])
            
            scenarios.append(RiskScenario(
                scenario_name="Crise Setorial (Top 3 Parceiros)",
                affected_companies=top_3_partners[:3],
                impact_percentage=min(100, combined_impact * 100),
                risk_level=self._classify_risk_level(combined_impact),
                description=f"Crise simultânea dos 3 principais parceiros causaria {combined_impact:.1%} de impacto",
                mitigation_strategies=[
                    "Diversificação geográfica e setorial",
                    "Contratos de longo prazo",
                    "Reservas de emergência",
                    "Plano de contingência operacional"
                ]
            ))
        
        return scenarios
    
    def _classify_risk_level(self, impact_pct: float) -> str:
        """Classifica nível de risco baseado no percentual de impacto."""
        if impact_pct > 0.5:
            return "CRÍTICO"
        elif impact_pct > 0.3:
            return "ALTO"
        elif impact_pct > 0.15:
            return "MÉDIO"
        else:
            return "BAIXO"
    
    def calculate_network_resilience(self,
                                   graph: nx.DiGraph,
                                   company_id: str) -> Dict[str, float]:
        """
        Calcula métricas de resiliência da rede.
        """
        if len(graph.nodes()) == 0:
            return {"resilience_score": 0, "redundancy": 0, "connectivity": 0}
        
        # Conectividade
        try:
            connectivity = nx.node_connectivity(graph.to_undirected())
        except:
            connectivity = 0
        
        # Redundância (múltiplos caminhos)
        redundancy = 0
        if company_id in graph.nodes():
            paths_count = 0
            for node in graph.nodes():
                if node != company_id:
                    try:
                        paths = list(nx.all_simple_paths(graph, company_id, node, cutoff=2))
                        paths_count += len(paths)
                    except:
                        pass
            redundancy = min(1.0, paths_count / max(1, len(graph.nodes()) - 1))
        
        # Score de resiliência (combinação de métricas)
        degree_centrality = nx.degree_centrality(graph).get(company_id, 0)
        clustering_coeff = nx.clustering(graph.to_undirected()).get(company_id, 0)
        
        resilience_score = (
            connectivity * 0.3 +
            redundancy * 0.3 +
            (1 - degree_centrality) * 0.2 +  # Menos centralização = mais resiliente
            clustering_coeff * 0.2
        )
        
        return {
            "resilience_score": float(resilience_score),
            "redundancy": float(redundancy),
            "connectivity": float(connectivity),
            "clustering": float(clustering_coeff)
        }
    
    def suggest_diversification_targets(self,
                                      transactions_df: pd.DataFrame,
                                      company_id: str,
                                      current_partners: List[str]) -> List[Dict[str, Any]]:
        """
        Sugere novos parceiros para diversificação de risco.
        """
        # Encontrar empresas que transacionam com os parceiros atuais
        # mas não diretamente com a empresa focal
        potential_partners = set()
        
        for partner in current_partners:
            # Quem mais transaciona com este parceiro?
            partner_connections = set(
                transactions_df[transactions_df["ID_PGTO"] == partner]["ID_RCBE"].unique()
            ) | set(
                transactions_df[transactions_df["ID_RCBE"] == partner]["ID_PGTO"].unique()
            )
            potential_partners.update(partner_connections)
        
        # Remover empresa focal e parceiros atuais
        potential_partners.discard(company_id)
        potential_partners -= set(current_partners)
        
        # Calcular métricas para cada potencial parceiro
        suggestions = []
        for potential in list(potential_partners)[:10]:  # Top 10
            # Volume médio de transações
            avg_volume = transactions_df[
                (transactions_df["ID_PGTO"] == potential) | 
                (transactions_df["ID_RCBE"] == potential)
            ]["VL"].mean()
            
            # Número de parceiros (diversificação)
            num_partners = len(set(
                transactions_df[transactions_df["ID_PGTO"] == potential]["ID_RCBE"].unique()
            ) | set(
                transactions_df[transactions_df["ID_RCBE"] == potential]["ID_PGTO"].unique()
            ))
            
            suggestions.append({
                "partner_id": potential,
                "avg_transaction_volume": float(avg_volume),
                "diversification_level": num_partners,
                "risk_score": self._calculate_partner_risk_score(transactions_df, potential),
                "recommendation_reason": self._get_recommendation_reason(avg_volume, num_partners)
            })
        
        # Ordenar por menor risco e maior volume
        suggestions.sort(key=lambda x: (x["risk_score"], -x["avg_transaction_volume"]))
        
        return suggestions[:5]  # Top 5 sugestões
    
    def _calculate_partner_risk_score(self, transactions_df: pd.DataFrame, partner_id: str) -> float:
        """Calcula score de risco de um parceiro (0-1, menor = melhor)."""
        # Volatilidade das transações
        partner_transactions = transactions_df[
            (transactions_df["ID_PGTO"] == partner_id) | 
            (transactions_df["ID_RCBE"] == partner_id)
        ]
        
        if len(partner_transactions) < 2:
            return 0.8  # Alto risco por falta de dados
        
        monthly_volumes = partner_transactions.groupby("year_month")["VL"].sum()
        volatility = monthly_volumes.std() / (monthly_volumes.mean() + 1e-6)
        
        # Concentração (se ele também é muito concentrado = mais risco)
        partner_concentration = self.calculate_concentration_metrics(
            transactions_df, partner_id, "both"
        )
        
        # Score combinado (0-1)
        risk_score = min(1.0, (
            min(0.5, volatility / 10) +  # Volatilidade normalizada
            partner_concentration["top_partner_share"] * 0.3 +  # Concentração dele
            (1 / max(1, partner_concentration["num_partners"])) * 0.2  # Diversificação
        ))
        
        return risk_score
    
    def _get_recommendation_reason(self, avg_volume: float, num_partners: int) -> str:
        """Gera razão para recomendação."""
        if avg_volume > 50000 and num_partners > 10:
            return "Alto volume + bem diversificado"
        elif avg_volume > 100000:
            return "Alto potencial de volume"
        elif num_partners > 15:
            return "Muito bem conectado na rede"
        else:
            return "Oportunidade de diversificação"
    
    def generate_monthly_risk_report(self,
                                   transactions_df: pd.DataFrame,
                                   company_id: str,
                                   months_window: int = 6) -> Dict[str, Any]:
        """
        Gera relatório mensal de risco consolidado.
        """
        # Métricas por direção
        metrics_in = self.calculate_concentration_metrics(transactions_df, company_id, "in")
        metrics_out = self.calculate_concentration_metrics(transactions_df, company_id, "out")
        
        # Cenários de risco
        risk_scenarios = self.generate_risk_scenarios(transactions_df, company_id)
        
        # Sugestões de diversificação
        current_partners = list(set(
            transactions_df[transactions_df["ID_RCBE"] == company_id]["ID_PGTO"].unique()
        ) | set(
            transactions_df[transactions_df["ID_PGTO"] == company_id]["ID_RCBE"].unique()
        ))
        
        diversification_suggestions = self.suggest_diversification_targets(
            transactions_df, company_id, current_partners
        )
        
        # Score geral de risco (0-100, menor = melhor)
        overall_risk_score = (
            metrics_in["hhi"] * 30 +
            metrics_out["hhi"] * 30 +
            len([s for s in risk_scenarios if s.risk_level in ["ALTO", "CRÍTICO"]]) * 10 +
            (1 / max(1, len(current_partners))) * 30
        )
        
        return {
            "company_id": company_id,
            "period_months": months_window,
            "concentration_metrics": {
                "recebimentos": metrics_in,
                "pagamentos": metrics_out
            },
            "risk_scenarios": risk_scenarios,
            "diversification_suggestions": diversification_suggestions,
            "overall_risk_score": min(100, overall_risk_score),
            "risk_classification": self._classify_overall_risk(overall_risk_score),
            "current_partners_count": len(current_partners)
        }
    
    def _classify_overall_risk(self, score: float) -> str:
        """Classifica risco geral da empresa."""
        if score > 70:
            return "CRÍTICO"
        elif score > 50:
            return "ALTO"
        elif score > 30:
            return "MÉDIO"
        else:
            return "BAIXO"
