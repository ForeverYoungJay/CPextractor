import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from postprocess.claim_builder import build_parameter_claims
from postprocess.location_ids import claim_id_from_location
from postprocess.record_links import inflate_registry_from_claims, resolve_claim_record


def _maybe_number(v: str) -> Any:
    s = (v or "").strip()
    if not s:
        return None
    try:
        return int(s) if s.isdigit() else float(s)
    except Exception:
        return s


def _copy_item(item: dict) -> dict:
    return json.loads(json.dumps(item, ensure_ascii=False))


def _provenance_index(doc: dict) -> dict[str, dict]:
    out = {}
    for record in doc.get("provenance_records") or []:
        if isinstance(record, dict):
            pid = str(record.get("provenance_id") or "").strip()
            if pid:
                out[pid] = record
    return out


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "approved"}


def _is_falsey(value: Any) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "n", "rejected"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply reviewed queue corrections back to extracted JSON files.")
    ap.add_argument("--review-csv", required=True)
    ap.add_argument("--root", default="data/fulltext")
    ap.add_argument("--export-gold-jsonl", default="")
    ap.add_argument("--export-calibration-jsonl", default="output/review/judge_calibration.jsonl")
    ap.add_argument("--export-prompt-feedback-json", default="output/review/prompt_feedback.json")
    args = ap.parse_args()

    root = Path(args.root)
    rows = list(csv.DictReader(Path(args.review_csv).open("r", encoding="utf-8")))
    gold_rows = []
    calibration_rows = []
    error_type_counts = Counter()
    prompt_examples = []

    grouped = {}
    for row in rows:
        doi = row.get("doi") or ""
        grouped.setdefault(doi, []).append(row)

    for doi, doi_rows in grouped.items():
        doc_path = root / doi.replace("/", "_") / "materials_extracted.json"
        if not doc_path.exists():
            continue
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        doc = inflate_registry_from_claims(doc)
        registry = ((doc.get("parameters") or {}).get("registry") or [])
        if not isinstance(registry, list):
            continue
        provenance_by_id = _provenance_index(doc)
        report_path = doc_path.parent / "postprocess_report.json"
        review_decision_path = doc_path.parent / "review_decision.json"
        reports = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

        changed = False
        explicit_decision_rows = []
        for row in doi_rows:
            adjudication = (row.get("adjudication") or "").strip().lower()
            approved_raw = row.get("approved_for_ingest", "")
            if _is_truthy(approved_raw) or _is_falsey(approved_raw):
                explicit_decision_rows.append(row)
            loc = row.get("location") or ""
            idx = None
            if loc.startswith("parameters.registry["):
                try:
                    idx = int(loc.split("[", 1)[1].split("]", 1)[0])
                except Exception:
                    idx = None
            else:
                claim_id = claim_id_from_location(loc)
                if claim_id:
                    for j, entry in enumerate(registry):
                        if isinstance(entry, dict) and str(entry.get("claim_id") or "").strip() == claim_id:
                            idx = j
                            break
            if idx is None:
                continue
            if idx < 0 or idx >= len(registry) or not isinstance(registry[idx], dict):
                continue
            item = registry[idx]
            claim = resolve_claim_record(doc, item, idx)
            before = _copy_item(item)

            if adjudication in {"correct", "accepted"}:
                error_type_counts["accepted"] += 1
                calibration_rows.append({
                    "doi": doi,
                    "location": loc,
                    "adjudication": adjudication,
                    "human_error_type": row.get("human_error_type") or "accepted",
                    "before": before,
                    "after": before,
                })
                gold_rows.append(doc)
                continue

            if adjudication in {"fixed", "corrected", "incorrect", "rejected"}:
                if row.get("corrected_value", "").strip():
                    item["value"] = _maybe_number(row["corrected_value"])
                    if isinstance(claim, dict):
                        if "reported_value" in claim:
                            claim["reported_value"] = item["value"]
                        else:
                            claim["value"] = item["value"]
                if row.get("corrected_unit", "").strip():
                    item["unit"] = row["corrected_unit"].strip()
                    if isinstance(claim, dict):
                        if "reported_unit" in claim:
                            claim["reported_unit"] = item["unit"]
                        else:
                            claim["unit"] = item["unit"]
                if row.get("corrected_canonical_name", "").strip():
                    item["canonical_name"] = row["corrected_canonical_name"].strip()
                    if isinstance(claim, dict):
                        claim["canonical_name"] = item["canonical_name"]
                if row.get("corrected_origin_type", "").strip():
                    src = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
                    provenance_id = str(src.get("provenance_id") or (claim or {}).get("provenance_id") or "").strip()
                    if provenance_id and provenance_id in provenance_by_id:
                        provenance_by_id[provenance_id]["origin_type"] = row["corrected_origin_type"].strip()
                    src["origin_type"] = row["corrected_origin_type"].strip()
                    item["source"] = src
                if row.get("corrected_confidence", "").strip():
                    item["confidence"] = row["corrected_confidence"].strip()
                    if isinstance(claim, dict):
                        claim["confidence"] = item["confidence"]
                changed = True
                after = _copy_item(item)
                human_error_type = (row.get("human_error_type") or "corrected").strip().lower()
                error_type_counts[human_error_type] += 1
                calibration_rows.append({
                    "doi": doi,
                    "location": loc,
                    "adjudication": adjudication,
                    "human_error_type": human_error_type,
                    "before": before,
                    "after": after,
                })
                prompt_examples.append({
                    "doi": doi,
                    "location": loc,
                    "human_error_type": human_error_type,
                    "before": {
                        "canonical_name": before.get("canonical_name"),
                        "value": before.get("value"),
                        "unit": before.get("unit"),
                        "origin_type": ((before.get("source") or {}).get("origin_type") if isinstance(before.get("source"), dict) else None),
                    },
                    "after": {
                        "canonical_name": after.get("canonical_name"),
                        "value": after.get("value"),
                        "unit": after.get("unit"),
                        "origin_type": ((after.get("source") or {}).get("origin_type") if isinstance(after.get("source"), dict) else None),
                    },
                })

        if explicit_decision_rows:
            approved_rows = [row for row in explicit_decision_rows if _is_truthy(row.get("approved_for_ingest", ""))]
            rejected_rows = [row for row in explicit_decision_rows if _is_falsey(row.get("approved_for_ingest", ""))]
            unique_reviewers = sorted({str(row.get("reviewer") or "").strip() for row in explicit_decision_rows if str(row.get("reviewer") or "").strip()})
            note_parts = [str(row.get("notes") or "").strip() for row in explicit_decision_rows if str(row.get("notes") or "").strip()]
            decision_payload = {
                "approved_for_ingest": bool(approved_rows),
                "decision_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source_review_csv": args.review_csv,
                "reviewer": ", ".join(unique_reviewers) if unique_reviewers else None,
                "notes": "\n".join(note_parts) if note_parts else None,
                "rows_considered": len(explicit_decision_rows),
                "approval_count": len(approved_rows),
                "rejection_count": len(rejected_rows),
                "locations": [str(row.get("location") or "").strip() for row in explicit_decision_rows if str(row.get("location") or "").strip()],
            }
            review_decision_path.write_text(
                json.dumps(decision_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Updated review decision -> {review_decision_path}")

        if changed:
            doc, reports["parameter_claims"] = build_parameter_claims(
                doc,
                evaluation_report=reports.get("llm_evaluation"),
                confidence_report=reports.get("confidence_fusion"),
            )
            doc_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            if report_path:
                report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
            gold_rows.append(doc)
            print(f"Updated -> {doc_path}")

    if args.export_gold_jsonl:
        out = Path(args.export_gold_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            seen = set()
            for doc in gold_rows:
                rid = doc.get("record_id") or (doc.get("source_document", {}) or {}).get("doi")
                if rid and rid in seen:
                    continue
                if rid:
                    seen.add(rid)
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"Exported reviewed gold -> {out}")

    calibration_out = Path(args.export_calibration_jsonl)
    calibration_out.parent.mkdir(parents=True, exist_ok=True)
    with calibration_out.open("w", encoding="utf-8") as f:
        for row in calibration_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported judge calibration set -> {calibration_out}")

    prompt_feedback = {
        "error_type_counts": dict(error_type_counts),
        "examples": prompt_examples[:20],
        "generated_from": args.review_csv,
        "total_calibration_rows": len(calibration_rows),
    }
    prompt_out = Path(args.export_prompt_feedback_json)
    prompt_out.parent.mkdir(parents=True, exist_ok=True)
    prompt_out.write_text(json.dumps(prompt_feedback, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported prompt feedback -> {prompt_out}")


if __name__ == "__main__":
    main()
