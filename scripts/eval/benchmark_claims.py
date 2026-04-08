import argparse
from collections import Counter
from pathlib import Path

from common import save_csv, save_json
from claim_eval_common import (
    bundle_metrics,
    field_accuracy,
    gold_positive,
    load_gold_claim_rows,
    load_pred_claim_rows,
    match_gold_pred,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark claim-level extraction against annotated gold claims.")
    ap.add_argument("--gold", required=True, help="gold_claims.jsonl")
    ap.add_argument("--pred-root", required=True, help="Root with DOI paper folders")
    ap.add_argument("--pred-source", default="materials_extracted.json")
    ap.add_argument("--output", required=True, help="benchmark_claims.json")
    ap.add_argument("--by-paper-csv", default="", help="Optional per-paper bundle completeness CSV")
    args = ap.parse_args()

    gold_rows = load_gold_claim_rows(args.gold)
    pred_rows = load_pred_claim_rows(args.pred_root, args.pred_source)
    matched = match_gold_pred(gold_rows, pred_rows)
    field = field_accuracy(matched["matched_positive"])
    bundle = bundle_metrics(gold_rows, pred_rows)

    status_counts = Counter()
    for row in gold_rows:
        status = (row.get("annotation") or {}).get("status") or ""
        status_counts[str(status)] += 1

    out = {
        "gold_rows": len(gold_rows),
        "gold_positive_rows": sum(1 for r in gold_rows if gold_positive(r)),
        "pred_rows": len(pred_rows),
        "claim_detection": {
            **matched["prf"],
            "tp": matched["tp"],
            "fp": matched["fp"],
            "fn": matched["fn"],
        },
        "field_accuracy": field,
        "bundle": {
            "macro_bundle_completeness": bundle["macro_bundle_completeness"],
            "paper_count": bundle["paper_count"],
        },
        "gold_status_distribution": dict(status_counts),
        "diagnostics": {
            "matched_positive_count": len(matched["matched_positive"]),
            "missing_gold_positive_count": len(matched["missing_gold_positive"]),
            "matched_spurious_count": len(matched["matched_spurious"]),
            "unmatched_pred_count": len(matched["unmatched_pred"]),
        },
    }
    save_json(args.output, out)

    if args.by_paper_csv:
        save_csv(args.by_paper_csv, bundle["by_paper"])

    print(f"Saved claim benchmark -> {args.output}")


if __name__ == "__main__":
    main()
