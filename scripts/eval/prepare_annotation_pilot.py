import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from export_usable_parameter_annotation import (
    CSV_FIELDS,
    JSONL_FIELDS,
    _get_nested,
    _iter_paper_dirs,
    _stringify,
    build_usable_parameter_rows,
)


FLAGGED_VERDICTS = {"flagged", "rejected", "warning", "fail"}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def journal_key_from_doi(doi: str) -> str:
    match = re.search(r"/j\.([^.]+)", doi)
    if match:
        return match.group(1)
    return "other"


def _doi_to_safe_id(doi: str) -> str:
    return doi.replace("/", "_")


def _read_title(paper_dir: Path) -> str:
    extracted = {}
    try:
        extracted = json.loads((paper_dir / "materials_extracted.json").read_text(encoding="utf-8"))
    except Exception:
        extracted = {}
    document = _safe_dict(extracted.get("document"))
    title = str(document.get("title") or "").strip()
    if title:
        return title
    paper_md = paper_dir / "paper.md"
    if paper_md.exists():
        first = paper_md.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        if first:
            return first[0].lstrip("#").strip()
    return ""


def _write_csv(path: Path, records: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: _stringify(_get_nested(record, field)) for field in fields})


def _write_packet_readme(path: Path, *, doi: str, title: str, record_count: int, paper_dir: str) -> None:
    lines = [
        f"# Usable Parameter Annotation Pilot Packet: {doi}",
        "",
        f"- title: {title or '(unknown)'}",
        f"- usable_parameter_records: {record_count}",
        f"- paper_dir: {paper_dir}",
        "",
        "Files:",
        "- `usable_parameters.csv`: edit this in Excel/Numbers",
        "- `usable_parameters.jsonl`: full draft records with all schema blocks",
        "- `packet_meta.json`: packet metadata and selection rationale",
        "",
        "Suggested workflow:",
        "1. Review `usable_parameters.csv`.",
        "2. Focus on whether each record is scientifically reusable by another person.",
        "3. Check the six minimum-use blocks: material object, CP model, parameter body, scope, evidence, provenance.",
        "4. Keep `annotation.status=correct` unless you want to manually change it to `incorrect`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _is_zero_value(row: Dict[str, Any]) -> bool:
    try:
        return float(_safe_dict(row.get("parameter_body")).get("value")) == 0.0
    except Exception:
        return False


def _is_grouped_row_like(row: Dict[str, Any]) -> bool:
    evidence = _safe_dict(row.get("evidence"))
    snippet = str(evidence.get("snippet") or "")
    raw_name = str(_safe_dict(row.get("parameter_body")).get("raw_name") or "")
    return "," in snippet or "," in raw_name


def _is_image_backed_table(row: Dict[str, Any]) -> bool:
    evidence = _safe_dict(row.get("evidence"))
    snippet = str(evidence.get("snippet") or "")
    evidence_id = str(evidence.get("evidence_id") or "")
    if "image" in snippet.lower():
        return True
    paper_dir = Path(str(row.get("paper_dir") or ""))
    if not evidence_id or not paper_dir:
        return False
    try:
        extracted = json.loads((paper_dir / "materials_extracted.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    file_name = ""
    for candidate in extracted.get("evidence_objects", []) or []:
        if isinstance(candidate, dict) and str(candidate.get("evidence_id") or "") == evidence_id:
            file_name = str(candidate.get("source_file") or "")
            break
    if not file_name:
        return False
    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        return True
    table_path = paper_dir / "tables" / file_name
    if not table_path.exists():
        return False
    try:
        payload = json.loads(table_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(_safe_dict(payload).get("table_kind") or "").strip() == "image_backed"


def paper_candidate_from_dir(paper_dir: Path) -> Dict[str, Any] | None:
    records = build_usable_parameter_rows(paper_dir)
    if not records:
        return None

    doi = str(records[0].get("doi") or paper_dir.name.replace("_", "/"))
    flagged = review_required = missing_unit = grouped = image_backed = table_backed = zero_value = 0

    for row in records:
        prediction = _safe_dict(row.get("prediction_context"))
        evidence = _safe_dict(row.get("evidence"))
        parameter_body = _safe_dict(row.get("parameter_body"))
        if str(prediction.get("llm_verdict") or "").strip().lower() in FLAGGED_VERDICTS:
            flagged += 1
        if prediction.get("review_required") is True:
            review_required += 1
        if not str(parameter_body.get("unit") or "").strip():
            missing_unit += 1
        if _is_grouped_row_like(row):
            grouped += 1
        if _is_zero_value(row):
            zero_value += 1
        if str(evidence.get("evidence_type") or "").strip().startswith("table"):
            table_backed += 1
            if _is_image_backed_table(row):
                image_backed += 1

    record_count = len(records)
    difficulty_score = (
        image_backed * 5.0
        + flagged * 4.0
        + review_required * 3.0
        + grouped * 2.0
        + missing_unit * 1.5
        + zero_value * 1.0
        + min(record_count, 12) * 0.25
    )

    reasons = []
    if image_backed:
        reasons.append(f"image_backed_table={image_backed}")
    if flagged:
        reasons.append(f"flagged={flagged}")
    if review_required:
        reasons.append(f"review_required={review_required}")
    if grouped:
        reasons.append(f"grouped_row_like={grouped}")
    if missing_unit:
        reasons.append(f"missing_unit={missing_unit}")
    if zero_value:
        reasons.append(f"zero_value={zero_value}")
    if not reasons:
        reasons.append("baseline_coverage")

    return {
        "doi": doi,
        "title": _read_title(paper_dir),
        "paper_dir": str(paper_dir),
        "journal_key": journal_key_from_doi(doi),
        "record_count": record_count,
        "claim_count": record_count,
        "table_backed_claim_count": table_backed,
        "image_backed_claim_count": image_backed,
        "flagged_claim_count": flagged,
        "review_required_claim_count": review_required,
        "missing_unit_claim_count": missing_unit,
        "grouped_row_like_claim_count": grouped,
        "zero_value_claim_count": zero_value,
        "difficulty_score": difficulty_score,
        "selection_reason": ", ".join(reasons[:3]),
        "schema_type": "usable_parameter_annotation",
        "records": records,
    }


def collect_pilot_candidates(input_root: Path, source_name: str = "materials_extracted.json") -> List[Dict[str, Any]]:
    del source_name
    out = []
    for paper_dir in _iter_paper_dirs(input_root):
        candidate = paper_candidate_from_dir(paper_dir)
        if candidate is not None:
            out.append(candidate)
    return out


def select_pilot_candidates(candidates: List[Dict[str, Any]], n: int, strategy: str) -> List[Dict[str, Any]]:
    if n <= 0 or not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda row: (-float(row["difficulty_score"]), -int(row["record_count"]), str(row["doi"])),
    )
    if strategy == "difficulty":
        return ranked[: min(n, len(ranked))]

    by_journal: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        by_journal[str(row["journal_key"])].append(row)

    journal_order = sorted(
        by_journal.keys(),
        key=lambda key: (-len(by_journal[key]), -float(by_journal[key][0]["difficulty_score"]), key),
    )

    picked: List[Dict[str, Any]] = []
    while len(picked) < n:
        added = False
        for journal_key in journal_order:
            queue = by_journal[journal_key]
            if not queue:
                continue
            picked.append(queue.pop(0))
            added = True
            if len(picked) >= n:
                break
        if not added:
            break
    return picked


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "doi",
        "journal_key",
        "title",
        "paper_dir",
        "packet_dir",
        "record_count",
        "difficulty_score",
        "image_backed_claim_count",
        "flagged_claim_count",
        "review_required_claim_count",
        "missing_unit_claim_count",
        "grouped_row_like_claim_count",
        "selection_reason",
        "schema_type",
        "annotator",
        "annotation_status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_selection_summary(path: Path, rows: List[Dict[str, Any]], total_candidates: int, strategy: str) -> None:
    summary = {
        "total_candidates": total_candidates,
        "selected_papers": len(rows),
        "selection_strategy": strategy,
        "schema_type": "usable_parameter_annotation",
        "journal_distribution": {},
        "record_totals": {
            "records": sum(int(row["record_count"]) for row in rows),
            "image_backed_records": sum(int(row["image_backed_claim_count"]) for row in rows),
            "flagged_records": sum(int(row["flagged_claim_count"]) for row in rows),
            "review_required_records": sum(int(row["review_required_claim_count"]) for row in rows),
            "missing_unit_records": sum(int(row["missing_unit_claim_count"]) for row in rows),
            "grouped_row_like_records": sum(int(row["grouped_row_like_claim_count"]) for row in rows),
        },
        "selected": [
            {
                "doi": row["doi"],
                "journal_key": row["journal_key"],
                "record_count": row["record_count"],
                "difficulty_score": row["difficulty_score"],
                "selection_reason": row["selection_reason"],
            }
            for row in rows
        ],
    }
    for row in rows:
        key = str(row["journal_key"])
        summary["journal_distribution"][key] = summary["journal_distribution"].get(key, 0) + 1
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_readme(path: Path, *, output_root: Path, selected_count: int) -> None:
    lines = [
        "# Usable Parameter Annotation Pilot Packets",
        "",
        f"- selected_papers: {selected_count}",
        "- schema_doc: docs/usable_parameter_annotation_schema.md",
        "",
        "Files:",
        "- `manifest.csv`: pilot paper list and progress tracking",
        "- `selection_summary.json`: why these papers were selected",
        "- one folder per DOI containing `usable_parameters.csv`, `usable_parameters.jsonl`, `packet_meta.json`, and `README.md`",
        "",
        "Recommended workflow:",
        "1. Annotate `usable_parameters.csv` in each packet folder.",
        "2. Review each record as a reusable parameter record, not just a raw extracted claim.",
        "3. Check the six blocks: material object, CP model, parameter body, scope, evidence, provenance.",
        "4. Leave `annotation.status=correct` by default and only change it to `incorrect` when needed.",
        "",
        "Suggested next step:",
        "```bash",
        f"python3 scripts/eval/export_usable_parameter_annotation.py --input-root {output_root} --output-root data/annotations/tmp_unused_example",
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_pilot_packets(
    rows: List[Dict[str, Any]],
    *,
    output_root: Path,
    csv_profile: str,
    source_name: str,
    total_candidates: int,
    strategy: str,
) -> None:
    del source_name
    output_root.mkdir(parents=True, exist_ok=True)
    csv_fields = CSV_FIELDS if csv_profile == "compact" else JSONL_FIELDS

    manifest_rows: List[Dict[str, Any]] = []
    for row in rows:
        packet_dir = output_root / _doi_to_safe_id(str(row["doi"]))
        packet_dir.mkdir(parents=True, exist_ok=True)

        records_path = packet_dir / "usable_parameters.jsonl"
        with records_path.open("w", encoding="utf-8") as f:
            for record in row["records"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        _write_csv(packet_dir / "usable_parameters.csv", row["records"], fields=csv_fields)

        meta = {
            "doi": row["doi"],
            "paper_dir": row["paper_dir"],
            "title": row["title"],
            "record_count": row["record_count"],
            "schema_type": "usable_parameter_annotation",
            "schema_doc": "docs/usable_parameter_annotation_schema.md",
            "csv_profile": csv_profile,
            "selection_strategy": strategy,
            "difficulty_score": row["difficulty_score"],
            "selection_reason": row["selection_reason"],
            "difficulty_signals": {
                "image_backed_claim_count": row["image_backed_claim_count"],
                "flagged_claim_count": row["flagged_claim_count"],
                "review_required_claim_count": row["review_required_claim_count"],
                "missing_unit_claim_count": row["missing_unit_claim_count"],
                "grouped_row_like_claim_count": row["grouped_row_like_claim_count"],
                "zero_value_claim_count": row["zero_value_claim_count"],
            },
        }
        (packet_dir / "packet_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_packet_readme(
            packet_dir / "README.md",
            doi=str(row["doi"]),
            title=str(row["title"]),
            record_count=int(row["record_count"]),
            paper_dir=str(row["paper_dir"]),
        )

        manifest_rows.append({
            **{k: v for k, v in row.items() if k not in {"records"}},
            "packet_dir": str(packet_dir),
            "annotator": "",
            "annotation_status": "todo",
            "notes": "",
        })

    _write_manifest(output_root / "manifest.csv", manifest_rows)
    _write_selection_summary(output_root / "selection_summary.json", manifest_rows, total_candidates, strategy)
    _write_readme(output_root / "README.md", output_root=output_root, selected_count=len(manifest_rows))


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a usable-parameter annotation pilot packet set.")
    ap.add_argument("--input-root", default="data/fulltext")
    ap.add_argument("--output-root", default="data/annotations/pilot_packets")
    ap.add_argument("--n", type=int, default=50, help="How many papers to include in the pilot set.")
    ap.add_argument(
        "--strategy",
        default="journal_balanced",
        choices=["journal_balanced", "difficulty"],
        help="Use journal-balanced selection for diversity or pure difficulty ranking.",
    )
    ap.add_argument("--source", default="materials_extracted.json")
    ap.add_argument("--csv-profile", default="compact", choices=["compact", "full"])
    args = ap.parse_args()

    candidates = collect_pilot_candidates(Path(args.input_root), args.source)
    if not candidates:
        raise RuntimeError("No annotation pilot candidates found.")

    selected = select_pilot_candidates(candidates, n=min(args.n, len(candidates)), strategy=args.strategy)
    export_pilot_packets(
        selected,
        output_root=Path(args.output_root),
        csv_profile=args.csv_profile,
        source_name=args.source,
        total_candidates=len(candidates),
        strategy=args.strategy,
    )
    print(f"Prepared {len(selected)} usable-parameter pilot packets -> {args.output_root}")


if __name__ == "__main__":
    main()
