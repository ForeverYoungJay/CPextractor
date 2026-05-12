from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def build_compact_summary(
    extracted_json: Dict[str, Any],
    postprocess_report: Dict[str, Any] | None = None,
    llm_evaluation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    postprocess_report = postprocess_report or {}
    llm_evaluation = llm_evaluation or {}

    document = _safe_dict(extracted_json.get("document"))
    materials = [m for m in _safe_list(extracted_json.get("materials")) if isinstance(m, dict)]
    constituents = [c for c in _safe_list(extracted_json.get("constituents")) if isinstance(c, dict)]
    claims: List[Dict[str, Any]] = []

    for claim in _safe_list(extracted_json.get("parameter_claims")):
        if not isinstance(claim, dict):
            continue
        parameter = _safe_dict(claim.get("parameter"))
        assertion = _safe_dict(claim.get("assertion"))
        applies_to = _safe_dict(claim.get("applies_to"))
        provenance = _safe_dict(claim.get("provenance")) or _safe_dict(claim.get("source"))
        claims.append({
            "claim_id": claim.get("claim_id"),
            "canonical_name": claim.get("canonical_name") or parameter.get("canonical_name"),
            "symbol": claim.get("symbol") or parameter.get("symbol_reported"),
            "material_id": applies_to.get("material_id"),
            "constituent_id": applies_to.get("constituent_id"),
            "process_state_id": applies_to.get("process_state_id"),
            "model_id": applies_to.get("model_id"),
            "condition_id": applies_to.get("condition_id"),
            "branch_ids": _safe_list(applies_to.get("branch_ids")),
            "scope": applies_to.get("scope"),
            "value": claim.get("value", assertion.get("reported_value")),
            "unit": claim.get("unit", assertion.get("reported_unit")),
            "origin_type": provenance.get("origin_type"),
            "governing_equation_ids": _safe_list(claim.get("governing_equation_ids")),
        })

    issues: List[Dict[str, Any]] = []
    for issue in _safe_list(_safe_dict(postprocess_report.get("quality_checks")).get("issues")):
        if isinstance(issue, dict):
            issues.append({
                "type": issue.get("type"),
                "severity": issue.get("severity"),
                "path": issue.get("path"),
                "message": issue.get("message"),
            })
    for issue in _safe_list(llm_evaluation.get("critical_issues")):
        if isinstance(issue, dict):
            issues.append({
                "type": issue.get("category"),
                "severity": issue.get("severity"),
                "path": issue.get("location"),
                "message": issue.get("issue"),
            })

    return {
        "schema_version": extracted_json.get("schema_version"),
        "document": {
            "doi": document.get("doi"),
            "title": document.get("title"),
            "year": document.get("year"),
            "journal": document.get("journal"),
        },
        "materials": {
            "count": len(materials),
            "primary_material": _safe_dict(materials[0]) if materials else {},
            "constituent_count": len(constituents),
        },
        "summary": {
            "quality_tier": extracted_json.get("quality_tier"),
            "document_confidence_score": _safe_dict(postprocess_report.get("confidence_fusion")).get("document_confidence_score"),
            "verdict": llm_evaluation.get("verdict"),
            "overall_score": llm_evaluation.get("overall_score"),
        },
        "claims": claims,
        "issues": issues,
    }


def write_compact_summary(
    paper_dir: str | Path,
    extracted_json: Dict[str, Any],
    postprocess_report: Dict[str, Any] | None = None,
    llm_evaluation: Dict[str, Any] | None = None,
    *,
    filename: str = "compact_summary.json",
) -> Path:
    paper_dir = Path(paper_dir)
    out_path = paper_dir / filename
    summary = build_compact_summary(extracted_json, postprocess_report, llm_evaluation)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
