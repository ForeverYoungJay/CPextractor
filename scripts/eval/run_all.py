import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-click evaluation runner for CPextractor.")
    ap.add_argument("--gold", required=True, help="Gold labels (.json/.jsonl)")
    ap.add_argument("--pred-root", required=True, help="Root folder containing DOI subfolders")
    ap.add_argument("--qrels", required=True, help="Retrieval qrels jsonl")
    ap.add_argument("--runs", required=True, help="Retrieval runs jsonl")
    ap.add_argument("--pipeline-csv", required=True, help="pipeline_runs export CSV")
    ap.add_argument("--outdir", default="results/eval")
    ap.add_argument("--method-name", default="CPextractor")
    ap.add_argument("--ks", default="5,10")
    ap.add_argument("--usd-per-1k-input", type=float, default=0.0)
    ap.add_argument("--usd-per-1k-output", type=float, default=0.0)
    args = ap.parse_args()

    eval_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir)
    metrics_dir = outdir / "metrics"
    tables_dir = outdir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    gold_norm = metrics_dir / "gold_norm.json"
    pred_norm = metrics_dir / "pred_norm.json"
    field_json = metrics_dir / "metrics_field.json"
    numeric_json = metrics_dir / "metrics_numeric.json"
    unit_json = metrics_dir / "metrics_unit.json"
    citation_json = metrics_dir / "metrics_citation.json"
    retrieval_json = metrics_dir / "metrics_retrieval.json"
    cost_json = metrics_dir / "metrics_cost.json"
    error_json = metrics_dir / "metrics_errors.json"
    gate_json = metrics_dir / "quality_gate.json"

    # 1) normalize gold
    _run([
        py,
        str(eval_dir / "normalize_gold.py"),
        "--input",
        args.gold,
        "--output",
        str(gold_norm),
    ])

    # 2) normalize pred
    _run([
        py,
        str(eval_dir / "normalize_pred.py"),
        "--input-root",
        args.pred_root,
        "--output",
        str(pred_norm),
    ])

    # 3) field metrics
    _run([
        py,
        str(eval_dir / "field_metrics.py"),
        "--gold",
        str(gold_norm),
        "--pred",
        str(pred_norm),
        "--output",
        str(field_json),
    ])

    # 4) numeric metrics
    _run([
        py,
        str(eval_dir / "numeric_metrics.py"),
        "--gold",
        str(gold_norm),
        "--pred",
        str(pred_norm),
        "--output",
        str(numeric_json),
    ])

    # 5) unit metrics
    _run([
        py,
        str(eval_dir / "unit_metrics.py"),
        "--gold",
        str(gold_norm),
        "--pred",
        str(pred_norm),
        "--output",
        str(unit_json),
    ])

    # 6) citation metrics
    _run([
        py,
        str(eval_dir / "citation_metrics.py"),
        "--gold",
        str(gold_norm),
        "--pred",
        str(pred_norm),
        "--output",
        str(citation_json),
    ])

    # 7) retrieval metrics
    _run([
        py,
        str(eval_dir / "retrieval_metrics.py"),
        "--qrels",
        args.qrels,
        "--runs",
        args.runs,
        "--ks",
        args.ks,
        "--output",
        str(retrieval_json),
    ])

    # 8) cost/latency report
    _run([
        py,
        str(eval_dir / "cost_latency_report.py"),
        "--input-csv",
        args.pipeline_csv,
        "--usd-per-1k-input",
        str(args.usd_per_1k_input),
        "--usd-per-1k-output",
        str(args.usd_per_1k_output),
        "--output",
        str(cost_json),
    ])

    # 9) error buckets
    _run([
        py,
        str(eval_dir / "error_bucket.py"),
        "--gold",
        str(gold_norm),
        "--pred",
        str(pred_norm),
        "--output",
        str(error_json),
    ])

    # 10) quality gate (optional hard gate for paper-grade reporting)
    postprocess_report = Path(args.pred_root) / "postprocess_report.json"
    cmd = [
        py,
        str(eval_dir / "quality_gate.py"),
        "--metrics-dir",
        str(metrics_dir),
        "--output",
        str(gate_json),
    ]
    if postprocess_report.exists():
        cmd.extend(["--postprocess-report", str(postprocess_report)])
    _run(cmd)

    field = _load_json(field_json)
    numeric = _load_json(numeric_json)
    unit = _load_json(unit_json)
    citation = _load_json(citation_json)
    retrieval = _load_json(retrieval_json)
    cost = _load_json(cost_json)
    errors = _load_json(error_json)
    gate = _load_json(gate_json)

    # Table 1: main extraction result table
    table_main = [
        {
            "method": args.method_name,
            "field_precision": field.get("micro", {}).get("precision"),
            "field_recall": field.get("micro", {}).get("recall"),
            "field_f1": field.get("micro", {}).get("f1"),
            "numeric_mae": numeric.get("mae"),
            "numeric_mape": numeric.get("mape"),
            "unit_accuracy": unit.get("unit_accuracy"),
            "citation_precision": citation.get("precision"),
            "citation_recall": citation.get("recall"),
            "citation_f1": citation.get("f1"),
            "cost_per_paper_usd": cost.get("cost_per_paper_usd"),
            "time_seconds_per_paper": cost.get("time_seconds_per_paper"),
        }
    ]
    _write_csv(tables_dir / "table_main_results.csv", table_main)

    # Table 2: retrieval table
    table_retrieval = [
        {
            "method": args.method_name,
            "queries": retrieval.get("queries"),
            "mrr": retrieval.get("mrr"),
            **{k: v for k, v in retrieval.items() if str(k).startswith("recall@")},
        }
    ]
    _write_csv(tables_dir / "table_retrieval_results.csv", table_retrieval)

    # Table 3: error bucket table
    table_errors = []
    for row in errors.get("buckets", []):
        table_errors.append(
            {
                "method": args.method_name,
                "error_type": row.get("error_type"),
                "count": row.get("count"),
                "ratio": row.get("ratio"),
            }
        )
    _write_csv(tables_dir / "table_error_buckets.csv", table_errors)

    # Optional table: per-prefix field metrics
    by_prefix_rows = []
    for prefix, m in field.get("by_prefix", {}).items():
        by_prefix_rows.append(
            {
                "method": args.method_name,
                "prefix": prefix,
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "tp": m.get("tp"),
                "fp": m.get("fp"),
                "fn": m.get("fn"),
            }
        )
    _write_csv(tables_dir / "table_field_by_prefix.csv", by_prefix_rows)

    # Optional table: gate summary
    _write_csv(
        tables_dir / "table_quality_gate.csv",
        [{"method": args.method_name, "pass": gate.get("pass"), "failed_count": len(gate.get("failed", []))}],
    )

    print("\nDone. Outputs:")
    print(f"- metrics json: {metrics_dir}")
    print(f"- paper tables: {tables_dir}")


if __name__ == "__main__":
    main()
