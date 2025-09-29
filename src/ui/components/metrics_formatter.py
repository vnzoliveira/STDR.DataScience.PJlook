# src/ui/components/metrics_formatter.py
"""
Padronização de rótulos e formatação de métricas para stakeholders.
Centraliza toda a formatação e nomenclatura para consistência na UI.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple

# === DICIONÁRIO DE RÓTULOS PADRONIZADOS ===
METRIC_LABELS = {
    # Métricas financeiras básicas
    "receita_mensal": "Receita Mensal",
    "despesa_mensal": "Despesa Mensal", 
    "fluxo_liquido": "Fluxo de Caixa",
    "VL_FATU": "Faturamento Anual",
    "VL_SLDO": "Saldo Atual",
    
    # Métricas de crescimento
    "g_receita_mom": "Crescimento MoM",
    "crescimento_mom_medio": "Crescimento Médio MoM",
    "tendencia_receita": "Tendência de Receita",
    "tendencia_fluxo": "Tendência de Fluxo",
    
    # Métricas de volatilidade/risco
    "vol_receita_3m": "Volatilidade 3M",
    "volatilidade_receita": "Volatilidade da Receita",
    "volatilidade_fluxo": "Volatilidade do Fluxo",
    "coef_variacao_receita": "Coeficiente de Variação",
    
    # Métricas de estabilidade
    "consistencia_fluxo_positivo": "Consistência de Fluxo+",
    "pct_meses_fluxo_pos": "% Meses Fluxo Positivo",
    "margem_media": "Margem Operacional",
    
    # Métricas temporais
    "tempo_operacao_anos": "Anos de Operação",
    "num_meses": "Meses Analisados",
    "taxa_recuperacao": "Taxa de Recuperação",
    
    # Outros
    "DS_CNAE": "Setor/CNAE",
    "DT_ABRT": "Data de Abertura",
    "estagio": "Estágio Empresarial",
    "confianca": "Confiança do Modelo"
}

# === DESCRIÇÕES DETALHADAS PARA TOOLTIPS ===
METRIC_DESCRIPTIONS = {
    "receita_mensal": "Receita bruta mensal da empresa (entradas de pagamentos)",
    "despesa_mensal": "Despesas mensais da empresa (saídas de pagamentos)",
    "fluxo_liquido": "Diferença entre receitas e despesas mensais",
    "g_receita_mom": "Taxa de crescimento da receita vs mês anterior (%)",
    "vol_receita_3m": "Desvio padrão da receita nos últimos 3 meses",
    "consistencia_fluxo_positivo": "Percentual de meses com fluxo de caixa positivo",
    "margem_media": "Fluxo líquido médio / Receita média (rentabilidade)",
    "tendencia_receita": "Inclinação da tendência linear de receita no período",
    "taxa_recuperacao": "Capacidade de recuperar após quedas de receita",
    "confianca": "Confiança estatística do modelo na classificação (0-1)"
}

# === UNIDADES E FORMATAÇÃO ===
METRIC_UNITS = {
    "receita_mensal": "R$",
    "despesa_mensal": "R$", 
    "fluxo_liquido": "R$",
    "VL_FATU": "R$",
    "VL_SLDO": "R$",
    "g_receita_mom": "%",
    "crescimento_mom_medio": "%",
    "vol_receita_3m": "R$",
    "volatilidade_receita": "R$",
    "volatilidade_fluxo": "R$",
    "consistencia_fluxo_positivo": "%",
    "margem_media": "%",
    "taxa_recuperacao": "%",
    "confianca": "%",
    "tempo_operacao_anos": "anos",
    "num_meses": "meses"
}

# === CORES PARA MÉTRICAS (tema dark) ===
METRIC_COLORS = {
    "positive": "#22c55e",  # Verde para valores positivos
    "negative": "#ef4444",  # Vermelho para valores negativos
    "neutral": "#64748b",   # Cinza para valores neutros
    "warning": "#f59e0b",   # Amarelo para alertas
    "info": "#3b82f6"       # Azul para informações
}

def format_currency_br(value: float, precision: int = 0) -> str:
    """Formatar valores monetários no padrão brasileiro."""
    try:
        if pd.isna(value) or value is None:
            return "R$ 0"
        
        # Converter para milhares/milhões se necessário
        abs_value = abs(float(value))
        if abs_value >= 1_000_000:
            formatted = f"R$ {value/1_000_000:.1f}M"
        elif abs_value >= 1_000:
            formatted = f"R$ {value/1_000:.1f}K"
        else:
            formatted = f"R$ {value:,.{precision}f}"
        
        return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "R$ 0"

def format_percentage(value: float, precision: int = 1) -> str:
    """Formatar percentuais."""
    try:
        if pd.isna(value) or value is None:
            return "0,0%"
        return f"{float(value)*100:,.{precision}f}%".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "0,0%"

def format_number_br(value: float, precision: int = 2) -> str:
    """Formatar números no padrão brasileiro."""
    try:
        if pd.isna(value) or value is None:
            return "-"
        return f"{float(value):,.{precision}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "-"

def get_metric_color(metric_name: str, value: float) -> str:
    """Determinar cor baseada no tipo de métrica e valor."""
    if pd.isna(value):
        return METRIC_COLORS["neutral"]
    
    # Métricas onde maior é melhor
    positive_metrics = [
        "receita_mensal", "g_receita_mom", "crescimento_mom_medio",
        "consistencia_fluxo_positivo", "margem_media", "taxa_recuperacao",
        "confianca", "fluxo_liquido"
    ]
    
    # Métricas onde menor é melhor  
    negative_metrics = [
        "vol_receita_3m", "volatilidade_receita", "volatilidade_fluxo",
        "coef_variacao_receita"
    ]
    
    if metric_name in positive_metrics:
        return METRIC_COLORS["positive"] if value > 0 else METRIC_COLORS["negative"]
    elif metric_name in negative_metrics:
        return METRIC_COLORS["negative"] if value > 0.3 else METRIC_COLORS["positive"]
    else:
        return METRIC_COLORS["neutral"]

def format_metric_value(metric_name: str, value: Any) -> str:
    """Formatar valor de métrica baseado no tipo."""
    if metric_name in METRIC_UNITS:
        unit = METRIC_UNITS[metric_name]
        
        if unit == "R$":
            return format_currency_br(value)
        elif unit == "%":
            # Se o valor já está em percentual (0-100) ou decimal (0-1)
            if isinstance(value, (int, float)) and abs(value) <= 1:
                return format_percentage(value)
            else:
                return format_percentage(value / 100)
        elif unit in ["anos", "meses"]:
            return f"{format_number_br(value, 0)} {unit}"
        else:
            return format_number_br(value)
    else:
        return str(value) if value is not None else "-"

def get_standardized_label(metric_name: str) -> str:
    """Obter rótulo padronizado para métrica."""
    return METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())

def get_metric_description(metric_name: str) -> str:
    """Obter descrição detalhada da métrica para tooltips."""
    return METRIC_DESCRIPTIONS.get(metric_name, "Métrica financeira da empresa")

def create_metric_card_data(metric_name: str, value: Any, 
                          include_description: bool = False) -> Dict[str, Any]:
    """Criar dados completos para card de métrica."""
    return {
        "name": metric_name,
        "label": get_standardized_label(metric_name),
        "value": value,
        "formatted_value": format_metric_value(metric_name, value),
        "color": get_metric_color(metric_name, value if isinstance(value, (int, float)) else 0),
        "unit": METRIC_UNITS.get(metric_name, ""),
        "description": get_metric_description(metric_name) if include_description else None
    }

def standardize_dataframe_columns(df: pd.DataFrame, 
                                 column_mapping: Dict[str, str] = None) -> pd.DataFrame:
    """Padronizar nomes de colunas de DataFrame para exibição."""
    df_display = df.copy()
    
    # Aplicar mapeamento customizado se fornecido
    if column_mapping:
        df_display = df_display.rename(columns=column_mapping)
    
    # Aplicar rótulos padronizados
    new_columns = {}
    for col in df_display.columns:
        if col in METRIC_LABELS:
            new_columns[col] = METRIC_LABELS[col]
        else:
            # Fallback: capitalizar e substituir underscores
            new_columns[col] = col.replace("_", " ").title()
    
    df_display = df_display.rename(columns=new_columns)
    return df_display

def get_kpi_summary(stage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Criar resumo de KPIs principais para exibição."""
    kpis = []
    
    # KPIs principais baseados no estágio
    main_metrics = [
        ("estagio", "Estágio"),
        ("confianca", "Confiança"),
        ("fluxo_liquido_medio", "Fluxo Médio"),
        ("crescimento_mom_medio", "Crescimento MoM")
    ]
    
    for metric_key, display_name in main_metrics:
        if metric_key in stage_data:
            kpis.append(create_metric_card_data(metric_key, stage_data[metric_key]))
    
    return kpis

# === CONFIGURAÇÕES DE TABELAS ===
TABLE_COLUMN_CONFIG = {
    "ID": {"width": "120px", "align": "left"},
    "Mês": {"width": "100px", "align": "center"},
    "Receita Mensal": {"width": "130px", "align": "right", "format": "currency"},
    "Despesa Mensal": {"width": "130px", "align": "right", "format": "currency"},
    "Fluxo de Caixa": {"width": "130px", "align": "right", "format": "currency"},
    "Crescimento MoM": {"width": "120px", "align": "right", "format": "percentage"},
    "Volatilidade 3M": {"width": "120px", "align": "right", "format": "currency"},
    "Estágio Empresarial": {"width": "150px", "align": "center"},
    "Confiança do Modelo": {"width": "130px", "align": "right", "format": "percentage"}
}

def get_table_column_config(column_name: str) -> Dict[str, str]:
    """Obter configuração de coluna para tabelas."""
    return TABLE_COLUMN_CONFIG.get(column_name, {"width": "auto", "align": "left", "format": "text"})
