import argparse
from ..etl.ingest import ingest_bases
from ..etl.validate import run_checks
from ..features.metrics import build_f_empresa_mes
from ..features.stage import classify_stage
from ..sna.relations import relations_monthly, relation_importance, concentration
from pathlib import Path
import pandas as pd

def build_all():
    ingest_bases()
    checks = run_checks()
    if not checks["ok"]:
        raise SystemExit(f"Falhas de validação: {checks['issues']}")

    f = build_f_empresa_mes()
    f.to_parquet("reports/exports/f_empresa_mes.parquet", index=False)

    est = classify_stage(f)
    est.to_parquet("reports/exports/estagio.parquet", index=False)

    df_rel = relations_monthly()
    # gerar top contrapartes e HHI por ID (exemplo simples)
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
    rel_out.to_parquet("reports/exports/relations_latest.parquet", index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-all"])
    args = ap.parse_args()
    if args.cmd == "build-all":
        build_all()

if __name__ == "__main__":
    main()