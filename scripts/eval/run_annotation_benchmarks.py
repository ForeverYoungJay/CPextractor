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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.eval.evaluate_release import main as run_release
    run_release()


if __name__ == "__main__":
    main()
