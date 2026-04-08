import argparse

from common import save_csv, save_json
from claim_eval_common import gold_paper_labels, load_gate_rows, load_gold_claim_rows


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _evaluate(rows, gold_flag_key: str) -> dict:
    tp = fp = fn = 0
    for row in rows:
        pred = bool(row.get("blocked"))
        gold = bool(row.get(gold_flag_key))
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif (not pred) and gold:
            fn += 1
    return _prf(tp, fp, fn)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark ingest gate against annotated paper-level error signals.")
    ap.add_argument("--gold", required=True, help="gold_claims.jsonl")
    ap.add_argument("--pred-root", required=True, help="Root with DOI paper folders")
    ap.add_argument("--output", required=True, help="benchmark_gate.json")
    ap.add_argument("--by-paper-csv", default="", help="Optional paper-level gate CSV")
    args = ap.parse_args()

    gold_rows = load_gold_claim_rows(args.gold)
    gold_labels = {row["doi"]: row for row in gold_paper_labels(gold_rows)}
    gate_rows = load_gate_rows(args.pred_root)

    merged = []
    for row in gate_rows:
        gold = gold_labels.get(row["doi"], {})
        merged.append({**row, **gold})

    blocked_count = sum(1 for r in merged if r.get("blocked"))
    out = {
        "paper_count": len(merged),
        "blocked_count": blocked_count,
        "blocked_rate": blocked_count / len(merged) if merged else 0.0,
        "ingested_count": len(merged) - blocked_count,
        "gate_vs_gold_review_recommended": _evaluate(merged, "gold_review_recommended"),
        "gate_vs_gold_block_recommended": _evaluate(merged, "gold_block_recommended"),
    }

    reason_counts = {}
    for row in merged:
        for reason in row.get("gate_reasons") or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    out["gate_reason_distribution"] = reason_counts

    save_json(args.output, out)
    if args.by_paper_csv:
        save_csv(args.by_paper_csv, merged)

    print(f"Saved gate benchmark -> {args.output}")


if __name__ == "__main__":
    main()
