from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common import load_json, load_jsonl, norm_text, prf
from scripts.eval.export_annotation_draft import _claim_rows_for_paper


POSITIVE_STATUSES = {
    "correct",
    "accepted",
    "wrong_value",
    "wrong_unit",
    "wrong_mapping",
    "wrong_binding",
    "wrong_provenance",
    "insufficient_evidence",
    "missing_from_prediction",
}

ERROR_STATUSES = POSITIVE_STATUSES - {"correct", "accepted"}

SEVERE_ERROR_STATUSES = {
    "wrong_value",
    "wrong_mapping",
    "wrong_binding",
    "wrong_provenance",
    "insufficient_evidence",
    "spurious_claim",
    "missing_from_prediction",
}


def load_any_json(path: str | Path) -> Any:
    text = str(path)
    return load_jsonl(path) if text.endswith(".jsonl") else load_json(path)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _norm_unit(value: Any) -> str:
    text = norm_text(value)
    aliases = {
        "pa": "pa",
        "kpa": "kpa",
        "mpa": "mpa",
        "gpa": "gpa",
        "s -1": "s^-1",
        "1/s": "s^-1",
        "s^-1": "s^-1",
        "strain": "strain",
        "percent": "percent",
    }
    return aliases.get(text, text)


def _to_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _scope_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    scope = _safe_dict(row.get("scope"))
    return (
        norm_text(scope.get("scope")),
        norm_text(scope.get("phase_id")),
        norm_text(scope.get("mechanism")),
        norm_text(scope.get("family_id")),
        norm_text(scope.get("family_name")),
    )


def claim_match_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    doi = str(row.get("doi") or row.get("record_id") or "").strip()
    claim_id = str(row.get("claim_id") or "").strip()
    if claim_id:
        return ("claim_id", doi, claim_id)
    return (
        "semantic",
        doi,
        norm_text(row.get("canonical_name")),
        norm_text(row.get("symbol")),
        *_scope_key(row),
    )


def gold_positive(row: Dict[str, Any]) -> bool:
    status = norm_text(_safe_dict(row.get("annotation")).get("status"))
    return status in POSITIVE_STATUSES


def gold_error(row: Dict[str, Any]) -> bool:
    status = norm_text(_safe_dict(row.get("annotation")).get("status"))
    return status in ERROR_STATUSES


def gold_severe_error(row: Dict[str, Any]) -> bool:
    status = norm_text(_safe_dict(row.get("annotation")).get("status"))
    return status in SEVERE_ERROR_STATUSES


def _field_equal_text(a: Any, b: Any) -> bool:
    return norm_text(a) == norm_text(b)


def _field_equal_value(a: Any, b: Any, atol: float = 1e-9, rtol: float = 1e-4) -> bool:
    af = _to_number(a)
    bf = _to_number(b)
    if af is None or bf is None:
        return _field_equal_text(a, b)
    return abs(af - bf) <= (atol + rtol * abs(bf))


def load_pred_claim_rows(pred_root: str | Path, source_name: str = "materials_extracted.json") -> List[Dict[str, Any]]:
    root = Path(pred_root)
    rows: List[Dict[str, Any]] = []
    for paper_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        rows.extend(_claim_rows_for_paper(paper_dir, source_name))
    return rows


def load_gold_claim_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows = load_any_json(path)
    if isinstance(rows, dict):
        return [rows]
    return [r for r in rows if isinstance(r, dict)]


def match_gold_pred(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pred_by_key = {claim_match_key(row): row for row in pred_rows}
    gold_by_key = {claim_match_key(row): row for row in gold_rows}

    tp = fp = fn = 0
    matched_positive: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    unmatched_pred: List[Dict[str, Any]] = []
    missing_gold_positive: List[Dict[str, Any]] = []
    matched_spurious: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for key, gold in gold_by_key.items():
        pred = pred_by_key.get(key)
        if gold_positive(gold):
            if pred is not None:
                tp += 1
                matched_positive.append((gold, pred))
            else:
                fn += 1
                missing_gold_positive.append(gold)
        else:
            if pred is not None:
                fp += 1
                matched_spurious.append((gold, pred))

    for key, pred in pred_by_key.items():
        gold = gold_by_key.get(key)
        if gold is None:
            fp += 1
            unmatched_pred.append(pred)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "prf": prf(tp, fp, fn),
        "matched_positive": matched_positive,
        "missing_gold_positive": missing_gold_positive,
        "matched_spurious": matched_spurious,
        "unmatched_pred": unmatched_pred,
    }


def field_accuracy(matched_positive: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    counters = Counter()
    total = 0
    grounding_annotated = 0
    grounding_correct = 0

    for gold, pred in matched_positive:
        total += 1
        if _field_equal_text(gold.get("canonical_name"), pred.get("canonical_name")):
            counters["canonical_name"] += 1
        if _field_equal_text(gold.get("symbol"), pred.get("symbol")):
            counters["symbol"] += 1
        if _field_equal_value(gold.get("value"), pred.get("value")):
            counters["value"] += 1
        if _norm_unit(gold.get("unit")) == _norm_unit(pred.get("unit")):
            counters["unit"] += 1

        ge = _safe_dict(gold.get("evidence"))
        pe = _safe_dict(pred.get("evidence"))
        if any(ge.get(k) not in (None, "", []) for k in ("kind", "file", "row_name", "column_name")):
            grounding_annotated += 1
            if (
                _field_equal_text(ge.get("kind"), pe.get("kind"))
                and _field_equal_text(ge.get("file"), pe.get("file"))
                and (_field_equal_text(ge.get("row_name"), pe.get("row_name")) or not ge.get("row_name"))
                and (_field_equal_text(ge.get("column_name"), pe.get("column_name")) or not ge.get("column_name"))
            ):
                grounding_correct += 1

    out = {
        "matched_positive_count": total,
        "canonical_name_accuracy": counters["canonical_name"] / total if total else 0.0,
        "symbol_accuracy": counters["symbol"] / total if total else 0.0,
        "value_accuracy": counters["value"] / total if total else 0.0,
        "unit_accuracy": counters["unit"] / total if total else 0.0,
        "grounding_annotation_coverage": grounding_annotated / total if total else 0.0,
        "grounding_accuracy": grounding_correct / grounding_annotated if grounding_annotated else None,
        "grounding_annotated_count": grounding_annotated,
    }
    return out


def bundle_metrics(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pred_keys = {claim_match_key(r) for r in pred_rows}
    per_paper_total = Counter()
    per_paper_matched = Counter()

    for row in gold_rows:
        if not gold_positive(row):
            continue
        doi = str(row.get("doi") or row.get("record_id") or "").strip()
        per_paper_total[doi] += 1
        if claim_match_key(row) in pred_keys:
            per_paper_matched[doi] += 1

    rows = []
    scores = []
    for doi in sorted(per_paper_total):
        total = per_paper_total[doi]
        matched = per_paper_matched[doi]
        completeness = matched / total if total else 0.0
        scores.append(completeness)
        rows.append({
            "doi": doi,
            "gold_claims": total,
            "matched_claims": matched,
            "bundle_completeness": completeness,
        })

    return {
        "macro_bundle_completeness": statistics.mean(scores) if scores else 0.0,
        "paper_count": len(rows),
        "by_paper": rows,
    }


def load_gate_rows(pred_root: str | Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    root = Path(pred_root)
    for paper_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        report_path = paper_dir / "postprocess_report.json"
        if not report_path.exists():
            continue
        report = load_json(report_path)
        gate = _safe_dict(report.get("ingest_gate"))
        llm_eval = load_json(paper_dir / "llm_evaluation.json") if (paper_dir / "llm_evaluation.json").exists() else {}
        doi = (
            gate.get("doi")
            or report.get("record_id")
            or _safe_dict(report.get("source_document")).get("doi")
            or paper_dir.name.replace("_", "/")
        )
        out.append({
            "doi": doi,
            "paper_dir": str(paper_dir),
            "blocked": bool(gate.get("blocked")),
            "decision": "gated" if gate.get("blocked") else "ingest",
            "document_confidence_score": gate.get("document_confidence_score"),
            "review_required_parameter_count": gate.get("review_required_parameter_count"),
            "rejected_parameter_count": gate.get("rejected_parameter_count"),
            "flagged_parameter_count": gate.get("flagged_parameter_count"),
            "fail_parameter_count": gate.get("fail_parameter_count"),
            "review_escalation_required": gate.get("review_escalation_required"),
            "review_escalation_blocks": gate.get("review_escalation_blocks"),
            "gate_reasons": gate.get("gate_reasons") or [],
            "document_verdict": gate.get("document_verdict") or _safe_dict(llm_eval).get("verdict"),
        })
    return out


def gold_paper_labels(gold_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        doi = str(row.get("doi") or row.get("record_id") or "").strip()
        if doi:
            grouped[doi].append(row)

    labels = []
    for doi, rows in sorted(grouped.items()):
        error_count = sum(1 for r in rows if gold_error(r))
        severe_count = sum(1 for r in rows if gold_severe_error(r))
        labels.append({
            "doi": doi,
            "gold_claim_count": len(rows),
            "gold_error_count": error_count,
            "gold_severe_error_count": severe_count,
            "gold_review_recommended": error_count > 0,
            "gold_block_recommended": severe_count > 0,
        })
    return labels


def paper_claim_count_map(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        doi = str(row.get("doi") or row.get("record_id") or "").strip()
        if doi:
            counts[doi] += 1
    return dict(counts)


def table_kind_for_row(row: Dict[str, Any]) -> str:
    evidence = _safe_dict(row.get("evidence"))
    file_name = str(evidence.get("file") or "").strip()
    paper_dir = str(row.get("paper_dir") or "").strip()
    if not file_name or not paper_dir:
        return ""
    table_path = Path(paper_dir) / "tables" / file_name
    if not table_path.exists():
        return ""
    try:
        payload = load_json(table_path)
    except Exception:
        return ""
    return str(_safe_dict(payload).get("table_kind") or "").strip()
