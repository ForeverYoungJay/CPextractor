import argparse
from collections import Counter

from common import save_csv, save_json
from claim_eval_common import (
    claim_match_key,
    field_accuracy,
    load_gold_claim_rows,
    load_pred_claim_rows,
    match_gold_pred,
    table_kind_for_row,
)


def _is_zero_value(row):
    try:
        return float(row.get("value")) == 0.0
    except Exception:
        return False


def _slice_name(row, paper_claim_counts, median_claim_count):
    names = []
    evidence = row.get("evidence") or {}
    prediction = row.get("prediction_context") or {}
    doi = str(row.get("doi") or "")

    if (evidence.get("kind") or "") == "table":
        names.append("table_backed")
        if table_kind_for_row(row) == "image_backed":
            names.append("image_backed_table")
    if (prediction.get("llm_verdict") or "") in {"warning", "fail"}:
        names.append("warning_or_fail")
    if prediction.get("review_required") is True:
        names.append("review_required")
    if not str(row.get("unit") or "").strip():
        names.append("missing_unit")
    if _is_zero_value(row):
        names.append("zero_value")
    row_name = str(evidence.get("row_name") or "")
    symbol = str(row.get("symbol") or "")
    if "," in row_name or "," in symbol:
        names.append("grouped_row_like")
    if paper_claim_counts.get(doi, 0) >= median_claim_count:
        names.append("high_claim_count_paper")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark extraction on important difficulty slices.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--pred-source", default="materials_extracted.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--output-csv", default="")
    args = ap.parse_args()

    gold_rows = load_gold_claim_rows(args.gold)
    pred_rows = load_pred_claim_rows(args.pred_root, args.pred_source)
    pred_by_key = {claim_match_key(r): r for r in pred_rows}

    paper_claim_counts = Counter(str(r.get("doi") or "") for r in gold_rows)
    count_values = sorted(v for v in paper_claim_counts.values() if v > 0)
    median_claim_count = count_values[len(count_values) // 2] if count_values else 0

    grouped = {}
    for row in gold_rows:
        for name in _slice_name(row, paper_claim_counts, median_claim_count):
            grouped.setdefault(name, {"gold": [], "pred": []})
            grouped[name]["gold"].append(row)
            pred = pred_by_key.get(claim_match_key(row))
            if pred is not None:
                grouped[name]["pred"].append(pred)

    out_rows = []
    for name, payload in sorted(grouped.items()):
        matched = match_gold_pred(payload["gold"], payload["pred"])
        field = field_accuracy(matched["matched_positive"])
        out_rows.append({
            "slice": name,
            "gold_rows": len(payload["gold"]),
            "tp": matched["tp"],
            "fp": matched["fp"],
            "fn": matched["fn"],
            "precision": matched["prf"]["precision"],
            "recall": matched["prf"]["recall"],
            "f1": matched["prf"]["f1"],
            "value_accuracy": field["value_accuracy"],
            "unit_accuracy": field["unit_accuracy"],
            "canonical_name_accuracy": field["canonical_name_accuracy"],
            "grounding_accuracy": field["grounding_accuracy"],
            "grounding_annotation_coverage": field["grounding_annotation_coverage"],
        })

    save_json(args.output, {"slices": out_rows, "median_claim_count_threshold": median_claim_count})
    if args.output_csv:
        save_csv(args.output_csv, out_rows)
    print(f"Saved slice benchmark -> {args.output}")


if __name__ == "__main__":
    main()
