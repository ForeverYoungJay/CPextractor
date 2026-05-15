import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def build_annotation_benchmark_summary(
    benchmark_claims: Dict[str, Any],
    benchmark_gate: Dict[str, Any],
    benchmark_slices: Dict[str, Any],
) -> Dict[str, Any]:
    field = benchmark_claims.get("field_accuracy") or {}
    claim_detection = benchmark_claims.get("claim_detection") or {}
    slices = benchmark_slices.get("slices") or []
    hardest_slice = max(slices, key=lambda row: float(row.get("gold_rows") or 0), default={})

    return {
        "gold_rows": benchmark_claims.get("gold_rows"),
        "gold_positive_rows": benchmark_claims.get("gold_positive_rows"),
        "pred_rows": benchmark_claims.get("pred_rows"),
        "claim_detection": {
            "precision": claim_detection.get("precision"),
            "recall": claim_detection.get("recall"),
            "f1": claim_detection.get("f1"),
            "tp": claim_detection.get("tp"),
            "fp": claim_detection.get("fp"),
            "fn": claim_detection.get("fn"),
        },
        "field_accuracy": {
            "canonical_name_accuracy": field.get("canonical_name_accuracy"),
            "value_accuracy": field.get("value_accuracy"),
            "unit_accuracy": field.get("unit_accuracy"),
            "grounding_accuracy": field.get("grounding_accuracy"),
            "grounding_annotation_coverage": field.get("grounding_annotation_coverage"),
        },
        "bundle": benchmark_claims.get("bundle") or {},
        "gate": {
            "paper_count": benchmark_gate.get("paper_count"),
            "blocked_rate": benchmark_gate.get("blocked_rate"),
            "gate_vs_gold_review_recommended_f1": ((benchmark_gate.get("gate_vs_gold_review_recommended") or {}).get("f1")),
            "gate_vs_gold_block_recommended_f1": ((benchmark_gate.get("gate_vs_gold_block_recommended") or {}).get("f1")),
        },
        "largest_slice": {
            "slice": hardest_slice.get("slice"),
            "gold_rows": hardest_slice.get("gold_rows"),
            "f1": hardest_slice.get("f1"),
            "value_accuracy": hardest_slice.get("value_accuracy"),
            "unit_accuracy": hardest_slice.get("unit_accuracy"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run claim-level annotation benchmarks and produce a compact summary.")
    ap.add_argument("--gold", required=True, help="gold_claims.jsonl")
    ap.add_argument("--pred-root", required=True, help="Root with DOI paper folders")
    ap.add_argument("--outdir", default="results/eval_annotation")
    ap.add_argument("--pred-source", default="materials_extracted.json")
    args = ap.parse_args()

    eval_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir)
    metrics_dir = outdir / "metrics"
    tables_dir = outdir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    benchmark_claims_json = metrics_dir / "benchmark_claims.json"
    benchmark_gate_json = metrics_dir / "benchmark_gate.json"
    benchmark_slices_json = metrics_dir / "benchmark_slices.json"

    _run([
        py,
        str(eval_dir / "benchmark_claims.py"),
        "--gold",
        args.gold,
        "--pred-root",
        args.pred_root,
        "--pred-source",
        args.pred_source,
        "--output",
        str(benchmark_claims_json),
        "--by-paper-csv",
        str(tables_dir / "table_bundle_completeness.csv"),
    ])

    _run([
        py,
        str(eval_dir / "benchmark_gate.py"),
        "--gold",
        args.gold,
        "--pred-root",
        args.pred_root,
        "--output",
        str(benchmark_gate_json),
        "--by-paper-csv",
        str(tables_dir / "table_gate_by_paper.csv"),
    ])

    _run([
        py,
        str(eval_dir / "benchmark_slices.py"),
        "--gold",
        args.gold,
        "--pred-root",
        args.pred_root,
        "--pred-source",
        args.pred_source,
        "--output",
        str(benchmark_slices_json),
        "--output-csv",
        str(tables_dir / "table_slice_results.csv"),
    ])

    summary = build_annotation_benchmark_summary(
        _load_json(benchmark_claims_json),
        _load_json(benchmark_gate_json),
        _load_json(benchmark_slices_json),
    )
    summary_path = metrics_dir / "annotation_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved annotation benchmark summary -> {summary_path}")


if __name__ == "__main__":
    main()
