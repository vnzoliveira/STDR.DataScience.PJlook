# src/cli/main.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..etl.ingest import ingest_bases
from ..etl.validate import run_checks
from ..features.metrics import build_f_empresa_mes
from ..features.advanced_stage import classify_advanced_stage
from ..features.supervised_stage import train_and_classify_stages
from ..features.ensemble_model import EnsembleStagePredictor
from ..features.sector_analysis import SectorBenchmarkEngine
from ..features.company_summary import build_companies_summary
from ..sna.relations import relations_monthly, relation_importance

EXPORTS_DIR = Path("reports/exports")
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def build_all() -> None:
    # 1) ETL
    log.info("Iniciando ingestão das bases...")
    ingest_bases()

    log.info("Validando artefatos processados...")
    checks = run_checks()
    if not checks["ok"]:
        raise SystemExit(f"Falhas de validação: {checks['issues']}")

    # 2) Feature mensal por empresa
    log.info("Gerando f_empresa_mes ...")
    f = build_f_empresa_mes()
    (EXPORTS_DIR / "f_empresa_mes.parquet").parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(EXPORTS_DIR / "f_empresa_mes.parquet", index=False)

    # 3) Não supervisionado avançado
    log.info("Classificando estágios (não-supervisionado avançado)...")
    est_advanced = classify_advanced_stage(f, months_window=6)
    est_advanced.to_parquet(EXPORTS_DIR / "estagio_unsupervised.parquet", index=False)

    # 4) Supervisionado/híbrido Base1+Base2
    try:
        log.info("Executando classificação supervisionada (híbrido)...")
        b1 = pd.read_parquet("data/processed/base1/base1.parquet")
        b2 = pd.read_parquet("data/processed/base2/base2.parquet")
        est_sup = train_and_classify_stages(b1, b2)
        est_sup.to_parquet(EXPORTS_DIR / "estagio.parquet", index=False)
        
        # 5) Modelo Ensemble avançado
        log.info("Treinando modelo ensemble...")
        ensemble = EnsembleStagePredictor(use_neural_network=False)  # Sem NN para simplicidade
        
        # Preparar dados combinados
        combined_df = est_sup.copy()
        if 'receita_media' not in combined_df.columns:
            combined_df = combined_df.merge(
                f.groupby('ID')['receita_mensal'].agg(['mean', 'sum', 'std']).reset_index(),
                on='ID', how='left'
            )
            combined_df.rename(columns={'mean': 'receita_media', 'sum': 'receita_total', 'std': 'volatilidade_receita'}, inplace=True)
        
        # Treinar e salvar resultados
        ensemble_results = ensemble.fit(combined_df)
        log.info(f"Ensemble treinado - RF: {ensemble_results['rf_score']:.3f}, GB: {ensemble_results['gb_score']:.3f}")
        
        # Predições do ensemble
        ensemble_preds = ensemble.predict(combined_df)
        ensemble_preds.to_parquet(EXPORTS_DIR / "estagio_ensemble.parquet", index=False)
        
        # 6) Análise setorial
        log.info("Executando análise setorial...")
        sector_engine = SectorBenchmarkEngine()
        
        # Calcular métricas setoriais para cada empresa
        sector_metrics = []
        for company_id in combined_df['ID'].unique()[:100]:  # Limitar para performance
            competitive_analysis = sector_engine.generate_competitive_analysis(combined_df, company_id)
            sector_metrics.append({
                'ID': company_id,
                'competitive_score': competitive_analysis['competitive_score'],
                'market_position': competitive_analysis['market_position']
            })
        
        sector_df = pd.DataFrame(sector_metrics)
        sector_df.to_parquet(EXPORTS_DIR / "sector_analysis.parquet", index=False)
        
    except Exception:
        log.exception("Falha no treinamento supervisionado; seguindo com artefatos não supervisionados.")
        est_advanced.to_parquet(EXPORTS_DIR / "estagio.parquet", index=False)

    try:
        b1 = pd.read_parquet("data/processed/base1/base1.parquet")
    except Exception:
        b1 = pd.DataFrame()

    est_final = pd.read_parquet(EXPORTS_DIR / "estagio.parquet")
    companies = build_companies_summary(f, est_final, b1)
    companies.to_parquet(EXPORTS_DIR / "companies.parquet", index=False)
    log.info("Gerado reports/exports/companies.parquet (%d empresas).", len(companies))

    # 7) Relações (SNA) agregadas
    log.info("Gerando relações mensais...")
    df_rel = relations_monthly()

    rows = []
    for cid in f["ID"].unique():
        top_in = relation_importance(df_rel, cid, "in", 12).head(10)
        if not top_in.empty:
            top_in["company_id"] = cid
            top_in["direction"] = "in"
            rows.append(top_in)
        top_out = relation_importance(df_rel, cid, "out", 12).head(10)
        if not top_out.empty:
            top_out["company_id"] = cid
            top_out["direction"] = "out"
            rows.append(top_out)

    rel_out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    rel_out.to_parquet(EXPORTS_DIR / "relations_latest.parquet", index=False)

    log.info("Pipeline concluído com sucesso.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-all"])
    args = ap.parse_args()

    if args.cmd == "build-all":
        build_all()


if __name__ == "__main__":
    main()
