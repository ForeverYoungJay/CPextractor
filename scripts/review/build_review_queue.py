import argparse
import csv
import json
import sys
from html import escape
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from postprocess.location_ids import claim_location, location_candidates, legacy_registry_location
from postprocess.param_iter import iter_parameter_items_with_index
from postprocess.record_links import resolve_provenance_record


def _normalize_verdict(value: str) -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "accepted": "accepted",
        "pass": "accepted",
        "flagged": "flagged",
        "warning": "flagged",
        "rejected": "rejected",
        "fail": "rejected",
    }
    return mapping.get(raw, raw)


def _load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _idx_from_location_text(loc: str) -> int | None:
    text = str(loc or "").strip()
    if not text.startswith("parameters.registry["):
        return None
    try:
        return int(text.split("[", 1)[1].split("]", 1)[0])
    except Exception:
        return None


def _priority_key(row: dict) -> tuple:
    reasons = set(str(row.get("reasons") or "").split("|"))
    llm_verdict = _normalize_verdict(row.get("llm_verdict") or "")
    confidence = str(row.get("confidence") or "").strip().lower()
    severity = 3
    if llm_verdict == "rejected":
        severity = 0
    elif "high_severity_rule_issue" in reasons or "semantic_extraction_risk" in reasons:
        severity = 1
    elif llm_verdict == "flagged" or confidence == "low":
        severity = 2
    return (
        severity,
        str(row.get("doi") or ""),
        str(row.get("canonical_name") or ""),
        str(row.get("location") or ""),
    )


def _compress_text(text: str, max_len: int = 220) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def _human_review_summary(audit: dict, reasons: list[str]) -> str:
    verdict = str(audit.get("verdict") or "").strip().lower()
    reason = _compress_text(str(audit.get("reason") or ""))
    recommendation = _compress_text(str(audit.get("recommendation") or ""))
    reason_labels = ", ".join(reasons[:3]) if reasons else ""

    parts = []
    if verdict:
        parts.append(f"LLM verdict: {verdict}")
    if reason:
        parts.append(reason)
    if recommendation and recommendation.lower() != "no action needed.":
        parts.append(f"Recommended action: {recommendation}")
    if reason_labels:
        parts.append(f"Flags: {reason_labels}")
    return " | ".join(parts)


REVIEW_COLUMNS = [
    "doi",
    "symbol",
    "canonical_name",
    "value",
    "unit",
    "origin_type",
    "confidence",
    "confidence_score",
    "llm_verdict",
    "llm_score",
    "human_review_summary",
    "llm_reason",
    "llm_recommendation",
    "reviewer",
    "adjudication",
    "approved_for_ingest",
    "corrected_value",
    "corrected_unit",
    "corrected_canonical_name",
    "corrected_origin_type",
    "corrected_confidence",
    "notes",
    "reasons",
    "uncertainty_types",
    "location",
    "claim_id",
    "evidence_ids",
    "legacy_location",
]


def _review_row(row: dict) -> dict:
    return {col: row.get(col, "") for col in REVIEW_COLUMNS}


def _write_review_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(_review_row(row))


def _write_review_html(path: Path, rows: list[dict]) -> None:
    headers = "".join(f"<th>{escape(col)}</th>" for col in REVIEW_COLUMNS)
    body_parts = []
    for row in rows:
        verdict = _normalize_verdict(row.get("llm_verdict") or "")
        confidence = str(row.get("confidence") or "").strip().lower()
        css = []
        if verdict == "rejected":
            css.append("fail")
        elif verdict == "flagged":
            css.append("warning")
        if confidence == "low":
            css.append("low")
        tr_class = f' class="{" ".join(css)}"' if css else ""
        cells = "".join(f"<td>{escape(str(row.get(col, '') or ''))}</td>" for col in REVIEW_COLUMNS)
        body_parts.append(f"<tr{tr_class}>{cells}</tr>")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Review Queue</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; color: #1f2937; }}
    h1 {{ margin: 0 0 8px; }}
    p {{ margin: 0 0 16px; color: #4b5563; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; vertical-align: top; text-align: left; }}
    th {{ position: sticky; top: 0; background: #111827; color: #fff; z-index: 1; }}
    tr:nth-child(even) td {{ background: #f9fafb; }}
    tr.fail td {{ background: #fef2f2; }}
    tr.warning td {{ background: #fffbeb; }}
    tr.low td {{ color: #7f1d1d; }}
    .table-wrap {{ overflow: auto; max-height: 85vh; border: 1px solid #d1d5db; }}
  </style>
</head>
<body>
  <h1>Review Queue</h1>
  <p>Use the CSV for editing and this HTML for quick scanning/filtering in a browser.</p>
  <div class="table-wrap">
    <table>
      <thead><tr>{headers}</tr></thead>
      <tbody>
        {''.join(body_parts)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _audit_map(report: dict) -> dict:
    audits = (((report.get("llm_evaluation") or {}).get("parameter_audits")) or [])
    out = {}
    for audit in audits:
        if isinstance(audit, dict) and audit.get("location"):
            out[audit["location"]] = audit
    return out


def _confidence_map(report: dict) -> dict:
    rows = ((report.get("confidence_fusion") or {}).get("parameter_confidence")) or []
    out = {}
    for row in rows:
        if isinstance(row, dict) and row.get("location"):
            out[row["location"]] = row
    return out


def _quality_issue_map(report: dict) -> dict:
    issues = ((report.get("quality_checks") or {}).get("issues")) or []
    out = {}
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        path = str(issue.get("path") or "").split(".source")[0].split(".value")[0].split(".unit")[0]
        if path:
            out.setdefault(path, []).append(issue)
    return out


def _evidence_map(report: dict) -> dict:
    rows = ((report.get("evidence_grounding") or {}).get("rows")) or []
    out = {}
    for row in rows:
        if isinstance(row, dict) and row.get("location"):
            out[row["location"]] = row
    return out


def review_reasons(item: dict, path: str, report: dict) -> list[str]:
    reasons = []
    src = item.get("source", {}) if isinstance(item.get("source", {}), dict) else {}
    symbol = str(item.get("symbol") or "").strip().lower()
    doc = report.get("_doc") or {}
    provenance = resolve_provenance_record(doc, src) if isinstance(doc, dict) else {"origin_type": src.get("origin_type")}
    origin = str(provenance.get("origin_type") or "").strip()

    conf_row = {}
    audit = {}
    issues = []
    evidence_row = {}
    idx = _idx_from_location_text(path)
    if idx is None:
        idx = _idx_from_location_text(str(item.get("legacy_location") or ""))
    for candidate in location_candidates(item, idx):
        conf_row = _confidence_map(report).get(candidate, conf_row)
        audit = _audit_map(report).get(candidate, audit)
        issues = _quality_issue_map(report).get(candidate, issues)
        evidence_row = _evidence_map(report).get(candidate, evidence_row)
        if conf_row or audit or issues or evidence_row:
            break

    if (conf_row.get("confidence") or item.get("confidence") or "").strip().lower() == "low":
        reasons.append("low_final_confidence")
    if audit.get("verdict") == "fail":
        reasons.append("llm_audit_fail")
    if audit.get("verdict") == "warning":
        reasons.append("llm_audit_warning")
    if audit.get("review_required"):
        reasons.append("manual_review_required")
    committee = audit.get("committee", {}) if isinstance(audit.get("committee"), dict) else {}
    if committee.get("disagreement"):
        reasons.append("committee_disagreement")
    if any((issue.get("severity") == "high") for issue in issues):
        reasons.append("high_severity_rule_issue")
    if any((issue.get("type") == "missing_evidence") for issue in issues):
        reasons.append("missing_evidence")
    if evidence_row.get("status") in {"not_found", "missing_evidence_text"}:
        reasons.append("evidence_grounding_failed")
    if any((issue.get("type", "").startswith("source_conflict")) for issue in issues):
        reasons.append("provenance_conflict")
    if any(err in {"wrong_value", "wrong_unit_conversion", "wrong_parameter_mapping", "condition_binding_error"} for err in (audit.get("error_types") or [])):
        reasons.append("semantic_extraction_risk")
    if audit.get("uncertainty_types"):
        reasons.append("typed_uncertainty_present")

    rid = (provenance.get("adopted_from_reference_ids", []) or []) + (provenance.get("calibration_based_on_reference_ids", []) or [])
    if not rid and origin in {"adopted", "adopted_then_calibrated"}:
        reasons.append("missing_reference_links")
    if symbol in {"tau0", "τ0", "n", "h0", "g0"} and item.get("value") in (None, ""):
        reasons.append("key_parameter_missing_value")

    return sorted(set(reasons))


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

        report = _load_json(p.parent / "postprocess_report.json")
        report["_doc"] = doc
        doi = doc.get("record_id") or (doc.get("source_document", {}) or {}).get("doi") or p.parent.name.replace("_", "/")

        for i, block, it in iter_parameter_items_with_index(doc):
            path = claim_location(it, i)
            reasons = review_reasons(it, path, report)
            if reasons:
                conf_row = {}
                audit = {}
                for candidate in location_candidates(it, i):
                    conf_row = _confidence_map(report).get(candidate, conf_row)
                    audit = _audit_map(report).get(candidate, audit)
                    if conf_row or audit:
                        break
                rows.append({
                    "doi": doi,
                    "block": block,
                    "index": i,
                    "location": path,
                    "legacy_location": legacy_registry_location(i),
                    "symbol": it.get("symbol"),
                    "canonical_name": it.get("canonical_name"),
                    "value": it.get("value"),
                    "unit": it.get("unit"),
                    "claim_id": it.get("claim_id") or "",
                    "origin_type": resolve_provenance_record(doc, (it.get("source", {}) or {})).get("origin_type"),
                    "confidence": conf_row.get("confidence") or it.get("confidence"),
                    "confidence_score": conf_row.get("score"),
                    "llm_verdict": audit.get("verdict"),
                    "llm_score": audit.get("score"),
                    "human_review_summary": _human_review_summary(audit, reasons),
                    "llm_reason": _compress_text(str(audit.get("reason") or ""), max_len=800),
                    "llm_recommendation": _compress_text(str(audit.get("recommendation") or ""), max_len=500),
                    "uncertainty_types": "|".join(audit.get("uncertainty_types", []) or []),
                    "evidence_ids": "|".join(((it.get("source", {}) or {}).get("evidence_ids") or [])),
                    "reasons": "|".join(reasons),
                    "status": "todo",
                    "reviewer": "",
                    "adjudication": "",
                    "approved_for_ingest": "",
                    "human_error_type": "",
                    "corrected_value": "",
                    "corrected_unit": "",
                    "corrected_canonical_name": "",
                    "corrected_origin_type": "",
                    "corrected_confidence": "",
                    "notes": "",
                })

    rows.sort(key=_priority_key)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "doi", "block", "index", "location", "legacy_location", "symbol", "canonical_name", "value", "unit",
        "claim_id", "origin_type", "confidence", "confidence_score", "llm_verdict", "llm_score",
        "human_review_summary", "llm_reason", "llm_recommendation", "uncertainty_types", "evidence_ids",
        "reasons", "status", "reviewer", "adjudication", "approved_for_ingest", "human_error_type",
        "corrected_value", "corrected_unit", "corrected_canonical_name",
        "corrected_origin_type", "corrected_confidence", "notes"
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    review_csv = out.with_name(f"{out.stem}.review.csv")
    review_html = out.with_name(f"{out.stem}.html")
    _write_review_csv(review_csv, rows)
    _write_review_html(review_html, rows)

    print(f"Saved review queue -> {out} (rows={len(rows)})")
    print(f"Saved review-friendly CSV -> {review_csv}")
    print(f"Saved review HTML -> {review_html}")


if __name__ == "__main__":
    main()
