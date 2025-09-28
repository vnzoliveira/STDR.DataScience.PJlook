# src/features/business_insights.py
"""
Motor de insights de negócio (sem emojis).
Gera mensagens contextuais por estágio, benchmarks e recomendações.
"""
from __future__ import annotations
from typing import Dict, Any, List

class BusinessInsightEngine:
    def contextual_insights(self, stage: str, company_data: Dict[str, Any]) -> List[str]:
        idade = float(company_data.get("idade_anos", 0))
        porte = str(company_data.get("porte", "MICRO"))
        receita_media = float(company_data.get("receita_media", 0))
        fluxo_medio = float(company_data.get("fluxo_medio", 0))
        consist_fluxo = float(company_data.get("consistencia_fluxo", 0))
        confianca = float(company_data.get("confianca", 0))

        out: List[str] = []

        if stage == "Inicio":
            out.append("Foco em validação e construção de base de clientes.")
            if idade < 2:
                out.append("Empresa jovem: priorize sobrevivência e validação do modelo.")
            if porte == "MICRO":
                out.append("Avalie programas de aceleração e microcrédito.")
            if fluxo_medio < 0:
                out.append("Fluxo negativo é comum no início: monitore queima de caixa.")

        elif stage == "Crescimento":
            out.append("Foco em escalabilidade: processos, pessoas e tecnologia.")
            if consist_fluxo > 0.8:
                out.append("Consistência de fluxo alta: bom momento para expandir.")
            elif consist_fluxo < 0.6:
                out.append("Inconsistência de fluxo: revise custos variáveis.")
            if receita_media > 100_000:
                out.append("Receita média elevada: avalie diversificação de mercados/produtos.")

        elif stage == "Maturidade":
            out.append("Foco em eficiência operacional e inovação.")
            if consist_fluxo > 0.9:
                out.append("Fluxo muito estável: considere P&D e expansão.")
            out.append("Mitigue estagnação com um programa de inovação contínua.")

        elif stage == "Declinio":
            out.append("Ação imediata: plano de recuperação.")
            if fluxo_medio < -10_000:
                out.append("Fluxo muito negativo: renegociação de dívidas e cortes de custos.")
            if consist_fluxo < 0.3:
                out.append("Fluxo inconsistente: adote controles de caixa diário.")

        elif stage == "Reestruturacao":
            out.append("Monitore indicadores de transformação e recuperação.")
            if fluxo_medio > 0:
                out.append("Fluxo positivo: sinais iniciais de recuperação.")
            if consist_fluxo > 0.5:
                out.append("Consistência de fluxo em melhora.")

        if confianca < 0.6:
            out.append(f"Aviso do modelo: confiança baixa ({confianca:.0%}); considere revisão manual.")
        elif confianca > 0.9:
            out.append(f"Aviso do modelo: confiança alta ({confianca:.0%}).")

        return out

    def comparative_benchmarks(self, company_data: Dict[str, Any]) -> List[str]:
        setor = str(company_data.get("setor", "OUTROS"))
        porte = str(company_data.get("porte", "MICRO"))
        receita_media = float(company_data.get("receita_media", 0))
        out = [f"Benchmark setorial: {setor}, porte {porte}."]
        if receita_media > 50_000:
            out.append("Receita acima da média para pares do mesmo porte.")
        elif receita_media < 10_000:
            out.append("Receita abaixo da média: oportunidade de crescimento.")
        return out

    def actionable_recommendations(self, stage: str, company_data: Dict[str, Any]) -> List[str]:
        consist_fluxo = float(company_data.get("consistencia_fluxo", 0))
        fluxo_medio = float(company_data.get("fluxo_medio", 0))
        recs: List[str] = []

        if stage == "Inicio":
            recs += [
                "Implemente controles semanais de fluxo de caixa.",
                "Defina 3–5 métricas de tração e acompanhe-as.",
                "Considere linhas de microcrédito e programas públicos.",
            ]
        elif stage == "Crescimento":
            recs += [
                "Implemente dashboard executivo com KPIs em tempo real.",
                "Profissionalize processos de RH e financeiro.",
                "Estruture crédito para capital de giro.",
            ]
        elif stage == "Maturidade":
            recs += [
                "Implemente programa de inovação contínua.",
                "Avalie expansão internacional ou mercados adjacentes.",
                "Invista em automação e transformação digital.",
            ]
        elif stage == "Declinio":
            recs += [
                "Acompanhe caixa diariamente.",
                "Corte custos não essenciais imediatamente.",
                "Renegocie prazos com fornecedores e credores.",
            ]
        elif stage == "Reestruturacao":
            recs += [
                "Defina plano de 90 dias com metas claras.",
                "Comunique mudanças de forma transparente.",
                "Foque no core business mais rentável.",
            ]

        if consist_fluxo < 0.4 and stage in ("Crescimento", "Maturidade"):
            recs.append("Endureça controle de despesas e previsões de receita para elevar consistência.")
        if fluxo_medio < 0 and stage != "Declinio":
            recs.append("Priorize geração de caixa antes de expandir investimentos.")

        return recs
