import argparse
import csv
import json
from pathlib import Path
from postprocess.param_iter import iter_parameter_items


def low_confidence(item: dict) -> bool:
    c = (item.get("confidence") or "").strip().lower()
    if c == "low":
        return True
    src = item.get("source", {}) if isinstance(item.get("source", {}), dict) else {}
    rid = (src.get("adopted_from_reference_ids", []) or []) + (src.get("calibration_based_on_reference_ids", []) or [])
    if not rid and (src.get("origin_type") in {"adopted", "adopted_then_calibrated"}):
        return True
    return False


def review_reasons(item: dict) -> list[str]:
    reasons = []
    src = item.get("source", {}) if isinstance(item.get("source", {}), dict) else {}
    symbol = str(item.get("symbol") or "").strip().lower()
    origin = str(src.get("origin_type") or "").strip()

    if low_confidence(item):
        reasons.append("low_confidence_or_missing_refs")
    loc = src.get("evidence_location", {}) if isinstance(src.get("evidence_location", {}), dict) else {}
    has_loc = any(loc.get(k) not in (None, "") for k in ("kind", "id", "page"))
    if not (src.get("evidence_text") or src.get("evidence_section") or has_loc):
        reasons.append("missing_evidence")
    if origin == "adopted" and (src.get("calibration_based_on_reference_ids") or src.get("calibration_in_this_study")):
        reasons.append("origin_conflict_adopted_vs_calibrated")
    if origin == "calibrated" and src.get("adopted_from_reference_ids"):
        reasons.append("origin_conflict_calibrated_vs_adopted")
    if symbol in {"tau0", "τ0", "n", "h0", "g0"} and item.get("value") in (None, ""):
        reasons.append("key_parameter_missing_value")

    return reasons


def main() -> None:
    ap = argparse.ArgumentParser(description="Build human-review queue from extracted JSON")
    ap.add_argument("--root", default="data/fulltext")
    ap.add_argument("--output", default="output/review/review_queue.csv")
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.root).glob("*/materials_extracted.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        doi = doc.get("record_id") or (doc.get("source_document", {}) or {}).get("doi") or p.parent.name.replace("_", "/")

        counters = {"elastic": 0, "plastic": 0}
        for block, it in iter_parameter_items(doc):
            i = counters.get(block, 0)
            counters[block] = i + 1
            reasons = review_reasons(it)
            if reasons:
                rows.append({
                    "doi": doi,
                    "block": block,
                    "index": i,
                    "symbol": it.get("symbol"),
                    "canonical_name": it.get("canonical_name"),
                    "value": it.get("value"),
                    "unit": it.get("unit"),
                    "origin_type": (it.get("source", {}) or {}).get("origin_type"),
                    "confidence": it.get("confidence"),
                    "reasons": "|".join(reasons),
                    "status": "todo",
                    "reviewer": "",
                    "notes": "",
                })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "doi", "block", "index", "symbol", "canonical_name", "value", "unit",
        "origin_type", "confidence", "reasons", "status", "reviewer", "notes"
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved review queue -> {out} (rows={len(rows)})")


if __name__ == "__main__":
    main()
