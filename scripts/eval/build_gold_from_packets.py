import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


EDITABLE_FIELDS = [
    "material_name",
    "canonical_name",
    "symbol",
    "value",
    "unit",
    "scope.phase_id",
    "scope.family_name",
    "scope.mechanism",
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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except Exception:
        return text


def _apply_csv_edits(base_rows: List[Dict[str, Any]], csv_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if len(base_rows) != len(csv_rows):
        raise ValueError(f"Row count mismatch: claims.jsonl={len(base_rows)} claims.csv={len(csv_rows)}")

    merged: List[Dict[str, Any]] = []
    for base, edited in zip(base_rows, csv_rows):
        record = json.loads(json.dumps(base, ensure_ascii=False))
        claim_like = isinstance(record, dict) and any(
            k in record for k in ("reported_value", "reported_unit", "normalized_value", "confidence_score", "value", "unit", "value_SI")
        )
        if claim_like:
            if "material_name" in edited:
                record["gold_material_name"] = _parse_scalar(edited.get("material_name", ""))
            if "canonical_name" in edited:
                record["canonical_name"] = _parse_scalar(edited.get("canonical_name", ""))
            if "symbol" in edited:
                record["symbol"] = _parse_scalar(edited.get("symbol", ""))
            if "value" in edited:
                value = _parse_scalar(edited.get("value", ""))
                if "reported_value" in record:
                    record["reported_value"] = value
                else:
                    record["value"] = value
            if "unit" in edited:
                unit = _parse_scalar(edited.get("unit", ""))
                if "reported_unit" in record:
                    record["reported_unit"] = unit
                else:
                    record["unit"] = unit
            gold_scope = record.get("gold_scope")
            if not isinstance(gold_scope, dict):
                gold_scope = {}
                record["gold_scope"] = gold_scope
            for field in ("scope.phase_id", "scope.family_name", "scope.mechanism"):
                if field in edited:
                    _set_nested(record, field.replace("scope.", "gold_scope."), _parse_scalar(edited.get(field, "")))
            annotation = record.get("annotation")
            if not isinstance(annotation, dict):
                annotation = {}
                record["annotation"] = annotation
            if "annotation.status" in edited:
                annotation["status"] = _parse_scalar(edited.get("annotation.status", ""))
            if "annotation.notes" in edited:
                annotation["notes"] = _parse_scalar(edited.get("annotation.notes", ""))
        else:
            for field in EDITABLE_FIELDS:
                if field not in edited:
                    continue
                value = _parse_scalar(edited.get(field, ""))
                _set_nested(record, field, value)
        merged.append(record)
    return merged


def _iter_packet_dirs(root: Path) -> List[Path]:
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "claims.csv").exists() and (path / "claims.jsonl").exists()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build gold_claims.jsonl from edited per-paper annotation packets.")
    ap.add_argument("--packets-root", required=True, help="Root produced by export_annotation_packets.py")
    ap.add_argument("--output", required=True, help="Combined gold_claims.jsonl output path")
    ap.add_argument(
        "--write-per-packet-jsonl",
        action="store_true",
        help="Also write each packet's merged result to gold_claims.jsonl inside the packet folder.",
    )
    args = ap.parse_args()

    packets_root = Path(args.packets_root)
    packet_dirs = _iter_packet_dirs(packets_root)

    combined_rows: List[Dict[str, Any]] = []
    for packet_dir in packet_dirs:
        base_rows = _load_jsonl(packet_dir / "claims.jsonl")
        csv_rows = _read_csv(packet_dir / "claims.csv")
        merged_rows = _apply_csv_edits(base_rows, csv_rows)
        combined_rows.extend(merged_rows)

        if args.write_per_packet_jsonl:
            per_packet = packet_dir / "gold_claims.jsonl"
            with per_packet.open("w", encoding="utf-8") as f:
                for row in merged_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in combined_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Built {len(combined_rows)} gold claim rows from {len(packet_dirs)} packets -> {out_path}")


if __name__ == "__main__":
    main()
