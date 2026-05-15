import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.eval.export_annotation_draft import _claim_rows_for_paper, _iter_paper_dirs


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


def _doi_to_safe_id(doi: str) -> str:
    return doi.replace("/", "_")


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _iter_sheet_rows(records: Iterable[Dict[str, Any]], fields: List[str]) -> Iterable[Dict[str, str]]:
    for record in records:
        yield {field: _stringify(_get_nested(record, field)) for field in fields}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _claim_jsonl_rows_for_paper(paper_dir: Path, source_name: str) -> List[Dict[str, Any]]:
    extracted = _load_json(paper_dir / source_name)
    claims = extracted.get("parameter_claims") if isinstance(extracted, dict) else None
    if isinstance(claims, list) and claims:
        return [row for row in claims if isinstance(row, dict)]
    return _claim_rows_for_paper(paper_dir, source_name)


def _read_title(paper_dir: Path) -> str:
    extracted = _load_json(paper_dir / "materials_extracted.json")
    source_doc = _safe_dict(extracted.get("source_document"))
    title = str(source_doc.get("title") or "").strip()
    if title:
        return title
    paper_md = paper_dir / "paper.md"
    if not paper_md.exists():
        return ""
    first = paper_md.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
    if not first:
        return ""
    return first[0].lstrip("#").strip()


def _write_claim_csv(path: Path, records: List[Dict[str, Any]], *, fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in _iter_sheet_rows(records, fields):
            writer.writerow(row)


def _write_packet_readme(path: Path, *, doi: str, title: str, claim_count: int, paper_dir: str) -> None:
    lines = [
        f"# Annotation Packet: {doi}",
        "",
        f"- title: {title or '(unknown)'}",
        f"- claims: {claim_count}",
        f"- paper_dir: {paper_dir}",
        "",
        "Files:",
        "- `claims.csv`: edit this in Excel/Numbers",
        "- `claims.jsonl`: same content in JSONL form",
        "- `packet_meta.json`: packet metadata",
        "",
        "Suggested workflow:",
        "1. Edit `claims.csv`.",
        "2. Light annotation fields are: parameter name, symbol, value, unit, and table row/column context.",
        "3. Use `annotation.status` and `annotation.notes` for simple corrections or comments.",
        "4. `claims.jsonl` keeps the full context if you need to inspect provenance or grounding later.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export one annotation file per paper for manual annotation.")
    ap.add_argument("--input-root", default="data/fulltext")
    ap.add_argument("--output-root", required=True)
    ap.add_argument(
        "--source",
        default="materials_extracted.json",
        choices=["materials_extracted.json", "materials_extracted.extractor_raw.json", "materials_extracted.pre_evaluator.json"],
    )
    ap.add_argument(
        "--csv-profile",
        default="compact",
        choices=["compact", "full"],
        help="Choose compact claim CSV columns for simpler manual annotation, or full for all fields.",
    )
    args = ap.parse_args()

    root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_fields = COMPACT_FIELDS if args.csv_profile == "compact" else FULL_FIELDS

    manifest_rows: List[Dict[str, Any]] = []
    for paper_dir in _iter_paper_dirs(root):
        records = _claim_rows_for_paper(paper_dir, args.source)
        claim_jsonl_rows = _claim_jsonl_rows_for_paper(paper_dir, args.source)
        if not records or not claim_jsonl_rows:
            continue
        doi = str(records[0].get("doi") or paper_dir.name.replace("_", "/"))
        safe_id = _doi_to_safe_id(doi)
        packet_dir = output_root / safe_id
        packet_dir.mkdir(parents=True, exist_ok=True)

        claims_jsonl = packet_dir / "claims.jsonl"
        with claims_jsonl.open("w", encoding="utf-8") as f:
            for row in claim_jsonl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        _write_claim_csv(packet_dir / "claims.csv", records, fields=csv_fields)

        meta = {
            "doi": doi,
            "paper_dir": str(paper_dir),
            "title": _read_title(paper_dir),
            "claim_count": len(records),
            "source_file": args.source,
            "csv_profile": args.csv_profile,
        }
        (packet_dir / "packet_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_packet_readme(
            packet_dir / "README.md",
            doi=doi,
            title=meta["title"],
            claim_count=len(records),
            paper_dir=str(paper_dir),
        )

        manifest_rows.append({
            "doi": doi,
            "title": meta["title"],
            "paper_dir": str(paper_dir),
            "packet_dir": str(packet_dir),
            "claim_count": len(records),
            "source_file": args.source,
            "csv_profile": args.csv_profile,
        })

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["doi", "title", "paper_dir", "packet_dir", "claim_count", "source_file", "csv_profile"],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"Exported {len(manifest_rows)} paper packets -> {output_root}")


if __name__ == "__main__":
    main()
