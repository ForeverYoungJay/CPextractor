from __future__ import annotations

from typing import Any, Dict, List, Tuple

from postprocess.location_ids import claim_location, legacy_registry_location, location_candidates
from postprocess.param_iter import iter_parameter_items_with_index
from postprocess.record_links import resolve_evidence_objects, resolve_provenance_record

_STRESS_FACTORS = {
    "pa": 1.0,
    "kpa": 1e3,
    "mpa": 1e6,
    "gpa": 1e9,
}


def _normalize_verdict(value: Any, default: str = "") -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "accepted": "accepted",
        "pass": "accepted",
        "passed": "accepted",
        "flagged": "flagged",
        "warning": "flagged",
        "warn": "flagged",
        "needs_review": "flagged",
        "rejected": "rejected",
        "fail": "rejected",
        "failed": "rejected",
    }
    return mapping.get(raw, default)


def _score_to_bucket(score: float) -> str:
    if score >= 85:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


def _quality_tier(*, doc_bucket: str, verdict: str | None, doc_score: float) -> str:
    verdict = _normalize_verdict(verdict)
    if doc_bucket == "high" and verdict == "accepted" and doc_score >= 85:
        return "gold"
    if doc_bucket in {"high", "medium"} and verdict in {"accepted", "flagged"} and doc_score >= 60:
        return "silver"
    return "candidate"


def _build_issue_map(quality_report: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for issue in quality_report.get("issues", []) or []:
        if not isinstance(issue, dict):
            continue
        candidates: List[str] = []
        loc = str(issue.get("location") or "").strip()
        if loc:
            candidates.append(loc)
        path = str(issue.get("path") or "").split(".source")[0].split(".value")[0].split(".unit")[0]
        if path:
            candidates.append(path)
        for candidate in candidates:
            out.setdefault(candidate, []).append(issue)
    return out


def _build_audit_map(parameter_audits: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for audit in parameter_audits:
        if not isinstance(audit, dict):
            continue
        path = str(audit.get("location") or "")
        if path:
            out[path] = audit
    return out


def _grounding_status(extracted_json: Dict[str, Any], item: Dict[str, Any]) -> str:
    src = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
    evidence_ids = src.get("evidence_ids") if isinstance(src.get("evidence_ids"), list) else []
    if not evidence_ids and isinstance(item.get("evidence_ids"), list):
        evidence_ids = item.get("evidence_ids")
    evidence_objects = resolve_evidence_objects(extracted_json, evidence_ids or [])
    if evidence_objects:
        return str(evidence_objects[0].get("status") or "").strip().lower()
    return str(item.get("grounding_status") or "").strip().lower()


def _is_table_based_claim(extracted_json: Dict[str, Any], item: Dict[str, Any]) -> bool:
    src = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
    loc = src.get("evidence_location", {}) if isinstance(src.get("evidence_location"), dict) else {}
    if str(loc.get("kind") or "").strip().lower() == "table":
        return True
    evidence_ids = src.get("evidence_ids") if isinstance(src.get("evidence_ids"), list) else []
    if not evidence_ids and isinstance(item.get("evidence_ids"), list):
        evidence_ids = item.get("evidence_ids")
    for obj in resolve_evidence_objects(extracted_json, evidence_ids or []):
        if not isinstance(obj, dict):
            continue
        file_name = str(obj.get("file") or obj.get("matched_file") or "").strip().lower()
        if file_name.startswith("table_") or (file_name.endswith(".json") and "/table_" in file_name):
            return True
    table_id = str(loc.get("id") or "").strip().lower()
    return table_id.startswith("table_")


def _is_high_risk_table_claim(item: Dict[str, Any], audit: Dict[str, Any]) -> bool:
    error_types = {str(v or "").strip().lower() for v in (audit.get("error_types") or []) if str(v or "").strip()}
    uncertainty_types = {str(v or "").strip().lower() for v in (audit.get("uncertainty_types") or []) if str(v or "").strip()}
    if error_types & {"wrong_value", "provenance_conflict", "missing_key_field", "cross_material_mixup", "model_variant_confusion"}:
        return True
    if _normalize_verdict(audit.get("verdict")) == "rejected" and not uncertainty_types.issubset({"missing_evidence", "weak_grounding", "table_parse_uncertain"}):
        return True
    if item.get("value") in (None, "") and item.get("value_SI") in (None, ""):
        return True
    applies_to = item.get("applies_to", {}) if isinstance(item.get("applies_to"), dict) else {}
    scope = str(applies_to.get("scope") or "").strip().lower()
    if scope == "system" and not (applies_to.get("system_ids") or []):
        return True
    return False


def _grounding_penalty(extracted_json: Dict[str, Any], item: Dict[str, Any]) -> float:
    status = _grounding_status(extracted_json, item)
    if status == "source_backed":
        return 1.0
    if _is_table_based_claim(extracted_json, item):
        if status in {"not_found", "missing_evidence_text"}:
            return 3.0
        if status == "read_error":
            return 5.0
    if status == "not_found":
        return 18.0
    if status == "missing_evidence_text":
        return 14.0
    if status == "read_error":
        return 10.0
    return 0.0


def _param_base_score(extracted_json: Dict[str, Any], item: Dict[str, Any]) -> float:
    src = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
    score = 70.0
    if src.get("evidence_text"):
        score += 10.0
    elif isinstance(item.get("evidence_ids"), list) and item.get("evidence_ids"):
        score += 5.0
    loc = src.get("evidence_location", {}) if isinstance(src.get("evidence_location"), dict) else {}
    if any(loc.get(k) not in (None, "") for k in ("kind", "id", "page")):
        score += 5.0
    origin = str(resolve_provenance_record(extracted_json, src).get("origin_type") or src.get("origin_type") or "").strip().lower()
    if origin in {"adopted", "calibrated", "adopted_then_calibrated", "original"}:
        score += 5.0
    return min(score, 100.0)


def _si_conversion_consistent(item: Dict[str, Any]) -> bool:
    unit = str(item.get("unit") or "").strip().lower()
    unit_si = str(item.get("unit_SI") or "").strip()
    if unit not in _STRESS_FACTORS or unit_si != "Pa":
        return False
    try:
        value = float(item.get("value"))
        value_si = float(item.get("value_SI"))
    except Exception:
        return False
    expected = value * _STRESS_FACTORS[unit]
    return abs(value_si - expected) <= max(1e-6, abs(expected) * 1e-6)


def fuse_confidence(
    extracted_json: Dict[str, Any],
    quality_report: Dict[str, Any],
    evaluation_report: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    evaluation_report = evaluation_report or {}
    parameter_audits = evaluation_report.get("parameter_audits", []) or []
    issue_map = _build_issue_map(quality_report)
    audit_map = _build_audit_map(parameter_audits)

    registry = ((extracted_json.get("parameters") or {}).get("registry") or [])
    param_scores: List[Dict[str, Any]] = []

    for idx, _, item in iter_parameter_items_with_index(extracted_json):
        path = claim_location(item, idx)
        base = _param_base_score(extracted_json, item)
        issues: List[Dict[str, Any]] = []
        audit: Dict[str, Any] = {}
        for candidate in location_candidates(item, idx):
            issues = issue_map.get(candidate, issues)
            audit = audit_map.get(candidate, audit)
            if issues or audit:
                break

        penalty = 0.0
        for issue in issues:
            sev = str(issue.get("severity") or "low")
            if sev == "high":
                penalty += 12.0
            elif sev == "medium":
                penalty += 4.0
            else:
                penalty += 1.0
        grounding_status = _grounding_status(extracted_json, item)
        penalty += _grounding_penalty(extracted_json, item)

        adjusted = audit.get("policy_adjusted_consensus") if isinstance(audit.get("policy_adjusted_consensus"), dict) else {}
        fused = (0.45 * max(0.0, 100.0 - penalty)) + (0.55 * base)

        llm_verdict = _normalize_verdict(adjusted.get("verdict") or audit.get("verdict"))
        error_types = {str(v or "").strip().lower() for v in (audit.get("error_types") or []) if str(v or "").strip()}
        uncertainty_types = {str(v or "").strip().lower() for v in (audit.get("uncertainty_types") or []) if str(v or "").strip()}
        if llm_verdict == "rejected" and "wrong_unit_conversion" in error_types and _si_conversion_consistent(item):
            llm_verdict = "flagged"
        table_based = _is_table_based_claim(extracted_json, item)
        if table_based and uncertainty_types and uncertainty_types.issubset({"missing_evidence", "weak_grounding", "table_parse_uncertain"}) and not _is_high_risk_table_claim(item, audit):
            if llm_verdict == "rejected":
                llm_verdict = "flagged"
            fused = max(fused, 72.0)
        if llm_verdict == "rejected":
            fused = min(fused, 54.0)
        elif llm_verdict == "flagged":
            fused = min(fused, 84.0)
        elif llm_verdict == "accepted":
            fused = max(fused, 70.0)

        review_required = bool(adjusted.get("review_required")) if "review_required" in adjusted else bool(audit.get("review_required"))
        if "wrong_unit_conversion" in error_types and _si_conversion_consistent(item):
            review_required = False
        if table_based and uncertainty_types and uncertainty_types.issubset({"missing_evidence", "weak_grounding", "table_parse_uncertain"}) and not _is_high_risk_table_claim(item, audit):
            review_required = False

        if review_required:
            fused = min(fused, 79.0)

        fused = max(0.0, fused)

        bucket = _score_to_bucket(fused)
        item["confidence"] = bucket
        item.setdefault("quality_assessment", {})
        item["quality_assessment"]["final_confidence_score"] = round(fused, 2)
        item["quality_assessment"]["rule_issue_count"] = len(issues)
        item["quality_assessment"]["llm_audited"] = bool(audit)
        item["quality_assessment"]["llm_verdict"] = llm_verdict or _normalize_verdict(audit.get("verdict"))
        item["quality_assessment"]["grounding_status"] = grounding_status

        param_scores.append({
            "location": path,
            "legacy_location": legacy_registry_location(idx),
            "canonical_name": item.get("canonical_name"),
            "symbol": item.get("symbol"),
            "score": round(fused, 2),
            "confidence": bucket,
            "rule_issue_count": len(issues),
            "llm_verdict": llm_verdict or _normalize_verdict(audit.get("verdict")),
            "review_required": review_required,
            "grounding_status": grounding_status,
            "table_based": table_based,
        })

    quality_skipped = bool(quality_report.get("skipped"))
    doc_rule_score = None if quality_skipped else float(quality_report.get("rule_score") or 0.0)
    if quality_skipped:
        final_doc_score = 100.0
    elif doc_rule_score is None:
        final_doc_score = 100.0
    else:
        final_doc_score = float(doc_rule_score or 0.0)

    rejected_count = sum(1 for r in param_scores if _normalize_verdict(r.get("llm_verdict")) == "rejected")
    flagged_count = sum(1 for r in param_scores if _normalize_verdict(r.get("llm_verdict")) == "flagged")
    review_required_count = sum(1 for r in param_scores if r.get("review_required"))
    not_grounded_count = sum(
        1 for r in param_scores
        if str(r.get("grounding_status") or "").strip().lower() in {"not_found", "missing_evidence_text"} and not r.get("table_based")
    )
    review_escalation = ((evaluation_report.get("review_escalation") or {}) if isinstance(evaluation_report, dict) else {})
    disagreement_count = int(review_escalation.get("disagreement_count") or 0)

    doc_verdict = _normalize_verdict(evaluation_report.get("verdict")) if isinstance(evaluation_report, dict) else ""
    if doc_verdict == "rejected":
        final_doc_score = min(final_doc_score, 54.0)
    elif doc_verdict == "flagged":
        final_doc_score = min(final_doc_score, 84.0)

    final_doc_score -= rejected_count * 8.0
    final_doc_score -= flagged_count * 3.0
    final_doc_score -= min(10.0, not_grounded_count * 2.5)
    final_doc_score -= min(8.0, disagreement_count * 1.5)
    final_doc_score = max(0.0, final_doc_score)

    doc_bucket = _score_to_bucket(final_doc_score)
    quality_tier = _quality_tier(doc_bucket=doc_bucket, verdict=doc_verdict, doc_score=float(final_doc_score))
    soft_review_only = (
        rejected_count == 0
        and doc_verdict == "accepted"
        and final_doc_score >= 85.0
        and review_required_count <= 2
    )
    if rejected_count or (review_required_count and not soft_review_only):
        quality_tier = "candidate"
    extracted_json["quality_tier"] = quality_tier
    report = {
        "document_confidence_score": round(final_doc_score, 2),
        "document_confidence": doc_bucket,
        "quality_tier": quality_tier,
        "rule_score": round(doc_rule_score, 2) if doc_rule_score is not None else None,
        "rule_checks_skipped": quality_skipped,
        "llm_score": None,
        "parameter_confidence": param_scores,
        "rejected_parameter_count": rejected_count,
        "flagged_parameter_count": flagged_count,
        "fail_parameter_count": rejected_count,
        "warning_parameter_count": flagged_count,
        "review_required_parameter_count": review_required_count,
        "disagreement_count": disagreement_count,
        "review_recommended": (
            doc_bucket == "low"
            or any(r["confidence"] == "low" for r in param_scores)
            or rejected_count > 0
            or review_required_count > 0
            or bool(review_escalation.get("required"))
        ),
    }
    return extracted_json, report
