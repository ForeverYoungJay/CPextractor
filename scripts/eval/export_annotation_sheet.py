import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


FULL_FIELDS = [
    "doi",
    "paper_dir",
    "claim_id",
    "record_index",
    "material_name",
    "material_phase_mode",
    "canonical_name",
    "symbol",
    "domain",
    "value",
    "unit",
    "value_SI",
    "unit_SI",
    "scope.scope",
    "scope.family_id",
    "scope.system_ids",
    "provenance.origin_type",
    "provenance.provenance_id",
    "provenance.reference_ids",
    "provenance.adopted_from_reference_ids",
    "provenance.calibration_based_on_reference_ids",
    "provenance.calibration_in_this_study",
    "provenance.calibration_method",
    "evidence.kind",
    "evidence.file",
    "evidence.page",
    "evidence.row_name",
    "evidence.column_name",
    "evidence.value_text",
    "evidence.snippet",
    "prediction_context.source_file",
    "prediction_context.grounding_status",
    "prediction_context.llm_verdict",
    "prediction_context.review_required",
    "prediction_context.policy_adjustments",
    "annotation.status",
    "annotation.error_tags",
    "annotation.notes",
]

COMPACT_FIELDS = [
    "material_name",
    "canonical_name",
    "symbol",
    "value",
    "unit",
    "evidence.file",
    "evidence.row_name",
    "evidence.column_name",
    "evidence.value_text",
    "annotation.status",
    "annotation.notes",
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _get_nested(record: Dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _iter_rows(records: Iterable[Dict[str, Any]], fields: List[str]) -> Iterable[Dict[str, str]]:
    for record in records:
        yield {field: _stringify(_get_nested(record, field)) for field in fields}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export annotation draft jsonl to editable CSV sheet.")
    ap.add_argument("--input", required=True, help="annotation_draft.jsonl")
    ap.add_argument("--output", required=True, help="annotation_draft.csv")
    ap.add_argument(
        "--profile",
        default="compact",
        choices=["compact", "full"],
        help="Choose compact columns for lightweight editing or full for all fields.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = _load_jsonl(in_path)
    fields = COMPACT_FIELDS if args.profile == "compact" else FULL_FIELDS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in _iter_rows(rows, fields):
            writer.writerow(row)
    print(f"Exported {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
