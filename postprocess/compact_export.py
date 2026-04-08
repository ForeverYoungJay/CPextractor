from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def build_compact_summary(
    extracted_json: Dict[str, Any],
    postprocess_report: Dict[str, Any] | None = None,
    llm_evaluation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    postprocess_report = postprocess_report or {}
    llm_evaluation = llm_evaluation or {}
    material = _safe_dict(extracted_json.get("material"))
    source_document = _safe_dict(extracted_json.get("source_document"))
    document = _safe_dict(extracted_json.get("document"))
    materials = [m for m in (extracted_json.get("materials") or []) if isinstance(m, dict)]
    primary_material = _safe_dict(materials[0]) if materials else {}
    compact_claims: List[Dict[str, Any]] = []
    for claim in extracted_json.get("parameter_claims") or []:
        if not isinstance(claim, dict):
            continue
        binding = _safe_dict(claim.get("applies_to"))
        provenance = _safe_dict(claim.get("provenance")) or _safe_dict(claim.get("source"))
        evidence = _safe_dict(claim.get("evidence"))
        table_evidence = _safe_dict(evidence.get("table_evidence"))
        compact_claims.append({
            "claim_id": claim.get("claim_id"),
            "parameter": claim.get("canonical_name"),
            "symbol": claim.get("symbol"),
            "material_id": binding.get("material_id"),
            "sample_id": binding.get("sample_id"),
            "condition_id": binding.get("condition_id"),
            "value": claim.get("value", claim.get("reported_value")),
            "unit": claim.get("unit", claim.get("reported_unit")),
            "phase": _safe_dict(binding).get("phase_id"),
            "scope": _safe_dict(binding).get("scope"),
            "family": _safe_dict(binding).get("family_name") or _safe_dict(binding).get("family_id"),
            "origin_type": _safe_dict(provenance).get("origin_type"),
            "evidence": {
                "evidence_text": evidence.get("evidence_text"),
                "row_name": table_evidence.get("row_name"),
                "column_name": table_evidence.get("column_name"),
                "value": table_evidence.get("value"),
                "file": evidence.get("file"),
            },
        })

    issues = []
    for issue in ((_safe_dict(postprocess_report.get("quality_checks")).get("issues")) or []):
        if isinstance(issue, dict):
            issues.append({
                "type": issue.get("type"),
                "severity": issue.get("severity"),
                "path": issue.get("path"),
                "message": issue.get("message"),
            })
    for issue in (llm_evaluation.get("critical_issues") or []):
        if isinstance(issue, dict):
            issues.append({
                "type": issue.get("category"),
                "severity": issue.get("severity"),
                "path": issue.get("location"),
                "message": issue.get("issue"),
            })

    return {
        "doi": document.get("doi") or source_document.get("doi"),
        "title": document.get("title") or source_document.get("title"),
        "material": {
            "name": primary_material.get("name") or material.get("name"),
            "formula": primary_material.get("formula") or material.get("chemical_formula"),
            "phase_mode": extracted_json.get("study", {}).get("study_type") if isinstance(extracted_json.get("study"), dict) else material.get("phase"),
            "phases": primary_material.get("phases") or material.get("phases"),
        },
        "summary": {
            "quality_tier": extracted_json.get("quality_tier"),
            "document_confidence_score": _safe_dict(postprocess_report.get("confidence_fusion")).get("document_confidence_score"),
            "verdict": llm_evaluation.get("verdict"),
            "overall_score": llm_evaluation.get("overall_score"),
        },
        "claims": compact_claims,
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
