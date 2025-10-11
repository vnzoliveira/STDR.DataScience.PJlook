# src/features/sector_analysis.py
"""
Sistema avançado de análise setorial e benchmarking.
Compara empresas com peers do mesmo CNAE e porte.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SectorMetrics:
    """Métricas agregadas do setor."""
    setor: str
    porte: str
    receita_p25: float
    receita_p50: float
    receita_p75: float
    margem_p25: float
    margem_p50: float
    margem_p75: float
    crescimento_p50: float
    fluxo_consistencia_p50: float
    empresas_total: int
    empresas_saudaveis_pct: float

class SectorBenchmarkEngine:
    """
    Motor de benchmarking setorial.
    Calcula percentis, rankings e análises comparativas.
    """
    
    def __init__(self):
        self.sector_cache: Dict[str, SectorMetrics] = {}
        self.peer_groups: Dict[str, pd.DataFrame] = {}
        
    def compute_sector_metrics(self, 
                              empresas_df: pd.DataFrame,
                              setor: str,
                              porte: str = None) -> SectorMetrics:
        """
        Calcula métricas agregadas para um setor/porte específico.
        """
        # Filtrar por setor e porte
        mask = empresas_df['setor'] == setor
        if porte:
            mask &= empresas_df['porte'] == porte
        
        sector_data = empresas_df[mask]
        
        if len(sector_data) < 3:
            # Poucos dados - usar todo o dataset como fallback
            sector_data = empresas_df
            
        # Calcular percentis
        metrics = SectorMetrics(
            setor=setor,
            porte=porte or "TODOS",
            receita_p25=sector_data['receita_media'].quantile(0.25),
            receita_p50=sector_data['receita_media'].quantile(0.50),
            receita_p75=sector_data['receita_media'].quantile(0.75),
            margem_p25=sector_data['margem_media'].quantile(0.25) if 'margem_media' in sector_data else 0,
            margem_p50=sector_data['margem_media'].quantile(0.50) if 'margem_media' in sector_data else 0,
            margem_p75=sector_data['margem_media'].quantile(0.75) if 'margem_media' in sector_data else 0,
            crescimento_p50=sector_data['crescimento_medio'].quantile(0.50) if 'crescimento_medio' in sector_data else 0,
            fluxo_consistencia_p50=sector_data['consistencia_fluxo'].quantile(0.50) if 'consistencia_fluxo' in sector_data else 0,
            empresas_total=len(sector_data),
            empresas_saudaveis_pct=(sector_data['estagio'].isin(['Crescimento', 'Maturidade']).mean() 
                                   if 'estagio' in sector_data else 0.5)
        )
        
        # Cache para performance
        cache_key = f"{setor}_{porte or 'TODOS'}"
        self.sector_cache[cache_key] = metrics
        
        return metrics
    
    def get_peer_ranking(self, 
                        empresas_df: pd.DataFrame,
                        empresa_id: str,
                        metrics_to_rank: List[str] = None) -> Dict[str, Any]:
        """
        Calcula o ranking da empresa vs peers do mesmo setor/porte.
        """
        if metrics_to_rank is None:
            metrics_to_rank = ['receita_media', 'margem_media', 'crescimento_medio', 'consistencia_fluxo']
            
        # Identificar empresa
        empresa = empresas_df[empresas_df['ID'] == empresa_id].iloc[0]
        setor = empresa['setor']
        porte = empresa['porte']
        
        # Filtrar peers
        peers = empresas_df[(empresas_df['setor'] == setor) & (empresas_df['porte'] == porte)]
        
        if len(peers) < 5:
            # Expandir para incluir portes adjacentes
            peers = empresas_df[empresas_df['setor'] == setor]
            
        rankings = {}
        for metric in metrics_to_rank:
            if metric not in peers.columns:
                continue
                
            # Calcular ranking (1 = melhor)
            values = peers[metric].dropna()
            if len(values) == 0:
                rankings[metric] = {'rank': 1, 'total': 1, 'percentile': 50}
                continue
                
            empresa_value = empresa[metric]
            rank = int((values > empresa_value).sum() + 1)
            percentile = int(((values <= empresa_value).mean()) * 100)
            
            rankings[metric] = {
                'value': float(empresa_value),
                'rank': rank,
                'total': len(values),
                'percentile': percentile,
                'quartile': self._get_quartile(percentile)
            }
            
        return {
            'empresa_id': empresa_id,
            'setor': setor,
            'porte': porte,
            'rankings': rankings,
            'peer_count': len(peers)
        }
    
    def _get_quartile(self, percentile: int) -> str:
        """Determina o quartil baseado no percentil."""
        if percentile >= 75:
            return "Q1 (Top 25%)"
        elif percentile >= 50:
            return "Q2 (25-50%)"
        elif percentile >= 25:
            return "Q3 (50-75%)"
        else:
            return "Q4 (Bottom 25%)"
    
    def generate_competitive_analysis(self,
                                     empresas_df: pd.DataFrame,
                                     empresa_id: str) -> Dict[str, Any]:
        """
        Gera análise competitiva completa da empresa.
        """
        empresa = empresas_df[empresas_df['ID'] == empresa_id].iloc[0]
        
        # Métricas do setor
        sector_metrics = self.compute_sector_metrics(
            empresas_df, 
            empresa['setor'], 
            empresa['porte']
        )
        
        # Rankings
        rankings = self.get_peer_ranking(empresas_df, empresa_id)
        
        # Análise SWOT simplificada baseada em dados
        swot = self._generate_swot(empresa, sector_metrics, rankings)
        
        # Score competitivo (0-100)
        competitive_score = self._calculate_competitive_score(rankings)
        
        return {
            'empresa_id': empresa_id,
            'setor': empresa['setor'],
            'porte': empresa['porte'],
            'sector_metrics': sector_metrics,
            'rankings': rankings,
            'swot': swot,
            'competitive_score': competitive_score,
            'market_position': self._get_market_position(competitive_score)
        }
    
    def _generate_swot(self, 
                      empresa: pd.Series,
                      sector_metrics: SectorMetrics,
                      rankings: Dict) -> Dict[str, List[str]]:
        """Gera análise SWOT baseada em dados."""
        swot = {
            'strengths': [],
            'weaknesses': [],
            'opportunities': [],
            'threats': []
        }
        
        # Forças
        if rankings['rankings'].get('receita_media', {}).get('percentile', 0) > 70:
            swot['strengths'].append("Receita acima do P70 do setor")
        if rankings['rankings'].get('margem_media', {}).get('percentile', 0) > 70:
            swot['strengths'].append("Margem superior à maioria dos competidores")
        if empresa.get('consistencia_fluxo', 0) > 0.8:
            swot['strengths'].append("Alta previsibilidade de fluxo de caixa")
            
        # Fraquezas
        if rankings['rankings'].get('crescimento_medio', {}).get('percentile', 0) < 30:
            swot['weaknesses'].append("Crescimento abaixo da média setorial")
        if empresa.get('fluxo_medio', 0) < 0:
            swot['weaknesses'].append("Fluxo de caixa negativo")
            
        # Oportunidades
        if sector_metrics.empresas_saudaveis_pct > 0.6:
            swot['opportunities'].append("Setor em expansão com alto % de empresas saudáveis")
        if empresa['porte'] in ['MICRO', 'PEQUENA']:
            swot['opportunities'].append("Potencial de crescimento para portes maiores")
            
        # Ameaças
        if sector_metrics.empresas_saudaveis_pct < 0.3:
            swot['threats'].append("Setor em dificuldade com muitas empresas em declínio")
        if rankings['peer_count'] > 100:
            swot['threats'].append("Alta competição no segmento")
            
        return swot
    
    def _calculate_competitive_score(self, rankings: Dict) -> float:
        """Calcula score competitivo de 0-100."""
        scores = []
        for metric, data in rankings['rankings'].items():
            percentile = data.get('percentile', 50)
            scores.append(percentile)
            
        return float(np.mean(scores)) if scores else 50.0
    
    def _get_market_position(self, score: float) -> str:
        """Classifica posição de mercado."""
        if score >= 80:
            return "Líder de Mercado"
        elif score >= 60:
            return "Competidor Forte"
        elif score >= 40:
            return "Posição Mediana"
        elif score >= 20:
            return "Desafiante"
        else:
            return "Posição Frágil"
    
    def get_sector_evolution(self,
                           historical_data: pd.DataFrame,
                           setor: str,
                           window_months: int = 12) -> pd.DataFrame:
        """
        Analisa evolução temporal do setor.
        """
        sector_data = historical_data[historical_data['setor'] == setor]
        
        if 'year_month' not in sector_data.columns:
            return pd.DataFrame()
            
        evolution = (sector_data.groupby('year_month')
                    .agg({
                        'receita_media': 'mean',
                        'margem_media': 'mean',
                        'consistencia_fluxo': 'mean',
                        'ID': 'count'
                    })
                    .rename(columns={'ID': 'empresas_ativas'})
                    .reset_index())
        
        # Adicionar tendências
        if len(evolution) > 1:
            evolution['receita_trend'] = evolution['receita_media'].pct_change()
            evolution['margem_trend'] = evolution['margem_media'].diff()
            
        return evolution

