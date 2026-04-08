import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


LIST_FIELDS = {
    "scope.system_ids",
    "provenance.reference_ids",
    "provenance.adopted_from_reference_ids",
    "provenance.calibration_based_on_reference_ids",
    "prediction_context.policy_adjustments",
    "annotation.error_tags",
}

BOOL_FIELDS = {
    "provenance.calibration_in_this_study",
    "prediction_context.review_required",
}

INT_FIELDS = {
    "record_index",
}

FLOAT_FIELDS = {
    "value",
    "value_SI",
    "unit_SI",
    "evidence.page",
}


def _set_nested(record: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = record
    for part in parts[:-1]:
        node = cur.get(part)
        if not isinstance(node, dict):
            node = {}
            cur[part] = node
        cur = node
    cur[parts[-1]] = value


def _parse_value(field: str, raw: str) -> Any:
    raw = raw.strip()
    if raw == "":
        return None
    if field in LIST_FIELDS:
        try:
            value = json.loads(raw)
            return value if isinstance(value, list) else [raw]
        except Exception:
            return [part.strip() for part in raw.split("|") if part.strip()]
    if field in BOOL_FIELDS:
        return raw.lower() in {"true", "1", "yes"}
    if field in INT_FIELDS:
        try:
            return int(raw)
        except Exception:
            return raw
    if field in {"value", "value_SI", "evidence.page"}:
        try:
            return float(raw)
        except Exception:
            return raw
    return raw


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for flat in reader:
            record: Dict[str, Any] = {}
            for field, raw in flat.items():
                if field is None:
                    continue
                value = _parse_value(field, raw or "")
                _set_nested(record, field, value)
            rows.append(record)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Import edited annotation CSV back to jsonl.")
    ap.add_argument("--input", required=True, help="edited annotation csv")
    ap.add_argument("--output", required=True, help="gold_claims.jsonl")
    args = ap.parse_args()

    rows = _read_csv(Path(args.input))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Imported {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
