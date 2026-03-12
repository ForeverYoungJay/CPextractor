import argparse
import csv
import json
from pathlib import Path


def low_confidence(item: dict) -> bool:
    c = (item.get("confidence") or "").strip().lower()
    if c == "low":
        return True
    src = item.get("source", {}) if isinstance(item.get("source", {}), dict) else {}
    rid = (src.get("adopted_from_reference_ids", []) or []) + (src.get("calibration_based_on_reference_ids", []) or []) + (src.get("reference_ids", []) or [])
    if not rid and (src.get("origin_type") in {"adopted", "mixed_adopted_and_calibrated"}):
        return True
    return False


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

        for block, key in (("elastic", "constants"), ("plastic", "parameters")):
            items = ((doc.get(f"{block}_parameters", {}) or {}).get(key, []) or [])
            for i, it in enumerate(items):
                if low_confidence(it):
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
                        "status": "todo",
                        "reviewer": "",
                        "notes": "",
                    })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "doi", "block", "index", "symbol", "canonical_name", "value", "unit",
        "origin_type", "confidence", "status", "reviewer", "notes"
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved review queue -> {out} (rows={len(rows)})")


if __name__ == "__main__":
    main()
