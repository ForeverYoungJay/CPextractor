from __future__ import annotations

from typing import Any, Dict, List, Tuple

from postprocess.location_ids import location_candidates
from postprocess.param_iter import iter_parameter_items_with_index


def _safe_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _compact_table_evidence(evidence: Dict[str, Any], grounded: Dict[str, Any]) -> Dict[str, Any] | None:
    table_evidence = _safe_dict(evidence.get("table_evidence"))
    row_name = _first_non_null(table_evidence.get("row_name"), grounded.get("row_name"))
    column_name = _first_non_null(table_evidence.get("column_name"), grounded.get("column_name"))
    value = _first_non_null(table_evidence.get("value"), grounded.get("value"))
    excerpt = _first_non_null(table_evidence.get("excerpt"), grounded.get("context_window"), grounded.get("snippet"))
    compact = {
        "row_name": row_name,
        "column_name": column_name,
        "value": value,
        "excerpt": excerpt,
    }
    return compact if any(v is not None and v != "" for v in compact.values()) else None


def _normalize_source_for_claim(source: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "origin_type": source.get("origin_type"),
        "evidence_ids": source.get("evidence_ids"),
        "evidence_location": source.get("evidence_location"),
        "reference_ids": source.get("reference_ids"),
        "adopted_from_reference_ids": source.get("adopted_from_reference_ids"),
        "calibration_based_on_reference_ids": source.get("calibration_based_on_reference_ids"),
        "calibration_in_this_study": source.get("calibration_in_this_study"),
        "calibration_method": source.get("calibration_method"),
        "references": source.get("references"),
        "adopted_from_references": source.get("adopted_from_references"),
        "calibration_based_on_references": source.get("calibration_based_on_references"),
        "notes": source.get("notes"),
    }
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def _normalize_evidence_for_claim(evidence: Dict[str, Any], grounded: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "evidence_text": _first_non_null(evidence.get("evidence_text"), grounded.get("snippet")),
        "file": _first_non_null(grounded.get("file"), grounded.get("matched_file")),
        "table_evidence": _compact_table_evidence(evidence, grounded),
        "notes": evidence.get("notes"),
    }
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def _is_v4_claim(item: Dict[str, Any]) -> bool:
    return isinstance(item, dict) and (
        isinstance(item.get("parameter"), dict)
        or isinstance(item.get("assertion"), dict)
        or "governing_equation_ids" in item
    )


def _canonical_parameter_payload(item: Dict[str, Any], original_claim: Dict[str, Any]) -> Dict[str, Any]:
    parameter = _safe_dict(original_claim.get("parameter")) or _safe_dict(item.get("parameter"))
    payload = {
        "canonical_name": _first_non_null(parameter.get("canonical_name"), original_claim.get("canonical_name"), item.get("canonical_name")),
        "parameter_family": parameter.get("parameter_family"),
        "raw_name": _first_non_null(parameter.get("raw_name"), original_claim.get("raw_name")),
        "symbol_reported": _first_non_null(parameter.get("symbol_reported"), original_claim.get("symbol"), item.get("symbol")),
        "domain": _first_non_null(parameter.get("domain"), original_claim.get("domain"), item.get("domain")),
        "description": _first_non_null(parameter.get("description"), original_claim.get("description"), item.get("description")),
    }
    return {k: v for k, v in payload.items() if v not in (None, "", [])}


def _canonical_assertion_payload(item: Dict[str, Any], original_claim: Dict[str, Any]) -> Dict[str, Any]:
    assertion = _safe_dict(original_claim.get("assertion")) or _safe_dict(item.get("assertion"))
    payload = {
        "value_type": _first_non_null(assertion.get("value_type"), "scalar" if _first_non_null(item.get("value"), original_claim.get("value")) not in (None, "") else None),
        "reported_value": _first_non_null(assertion.get("reported_value"), original_claim.get("value"), item.get("value")),
        "reported_unit": _first_non_null(assertion.get("reported_unit"), original_claim.get("unit"), item.get("unit")),
        "qualifier": assertion.get("qualifier"),
        "valid_range": _first_non_null(assertion.get("valid_range"), original_claim.get("valid_range"), item.get("valid_range")),
    }
    return {k: v for k, v in payload.items() if v not in (None, "", [])}


def _canonical_provenance_payload(source_payload: Dict[str, Any], original_claim: Dict[str, Any]) -> Dict[str, Any]:
    original_prov = _safe_dict(original_claim.get("provenance"))
    calibration = _safe_dict(original_prov.get("calibration"))
    payload = {
        "origin_type": _first_non_null(original_prov.get("origin_type"), source_payload.get("origin_type")),
        "reference_ids": _first_non_null(original_prov.get("reference_ids"), source_payload.get("reference_ids"), []),
        "adopted_from_reference_ids": _first_non_null(original_prov.get("adopted_from_reference_ids"), source_payload.get("adopted_from_reference_ids"), []),
        "calibration_based_on_reference_ids": _first_non_null(original_prov.get("calibration_based_on_reference_ids"), source_payload.get("calibration_based_on_reference_ids"), []),
        "calibration": {
            "method": _first_non_null(calibration.get("method"), source_payload.get("calibration_method")),
            "target_type": calibration.get("target_type"),
            "target_description": calibration.get("target_description"),
            "observation_scope": calibration.get("observation_scope"),
            "notes": _first_non_null(calibration.get("notes"), source_payload.get("notes")),
        },
    }
    if not any(payload["calibration"].values()):
        payload.pop("calibration", None)
    return {k: v for k, v in payload.items() if v not in (None, "", [])}


def build_parameter_claims(
    extracted_json: Dict[str, Any],
    evaluation_report: Dict[str, Any] | None = None,
    confidence_report: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    evaluation_report = evaluation_report or {}
    confidence_report = confidence_report or {}

    audits = {
        str(a.get("location")): a
        for a in (evaluation_report.get("parameter_audits") or [])
        if isinstance(a, dict) and a.get("location")
    }
    confidence_rows = {
        str(r.get("location")): r
        for r in (confidence_report.get("parameter_confidence") or [])
        if isinstance(r, dict) and r.get("location")
    }

    claims: List[Dict[str, Any]] = []
    evidence_objects = extracted_json.get("evidence_objects") or []
    evidence_by_id = {
        str(obj.get("evidence_id") or "").strip(): obj
        for obj in evidence_objects
        if isinstance(obj, dict) and str(obj.get("evidence_id") or "").strip()
    } if isinstance(evidence_objects, list) else {}
    binding_records = {
        str(obj.get("binding_id") or "").strip(): obj
        for obj in (extracted_json.get("binding_contexts") or [])
        if isinstance(obj, dict) and str(obj.get("binding_id") or "").strip()
    }
    raw_claims = extracted_json.get("parameter_claims") if isinstance(extracted_json.get("parameter_claims"), list) else []
    for idx, _, item in iter_parameter_items_with_index(extracted_json):
        source = _safe_dict(item.get("source"))
        evidence = _safe_dict(item.get("evidence"))
        claim_id = str(item.get("claim_id") or f"claim_{idx + 1:04d}")
        original_claim = raw_claims[idx] if idx < len(raw_claims) and isinstance(raw_claims[idx], dict) else {}
        if isinstance(item, dict):
            item["claim_id"] = claim_id
        audit = {}
        conf = {}
        for candidate in location_candidates({"claim_id": claim_id}, idx):
            audit = audits.get(candidate, audit)
            conf = confidence_rows.get(candidate, conf)
            if audit or conf:
                break
        evidence_ids = source.get("evidence_ids") if isinstance(source.get("evidence_ids"), list) else []
        if not evidence_ids and isinstance(item.get("evidence_ids"), list):
            evidence_ids = item.get("evidence_ids")
        if not evidence_ids and isinstance(original_claim.get("evidence_ids"), list):
            evidence_ids = original_claim.get("evidence_ids")
        first_evidence = {}
        for evidence_id in evidence_ids:
            grounded_evidence = evidence_by_id.get(str(evidence_id or "").strip())
            if isinstance(grounded_evidence, dict):
                claim_ids = grounded_evidence.get("claim_ids")
                if not isinstance(claim_ids, list):
                    claim_ids = []
                if claim_id not in claim_ids:
                    claim_ids.append(claim_id)
                grounded_evidence["claim_ids"] = claim_ids
                grounded_evidence.setdefault("claim_id", claim_id)
                if not first_evidence:
                    first_evidence = grounded_evidence
        binding_id = item.get("binding_id")
        applies_to = dict(binding_records.get(str(binding_id or "").strip()) or _safe_dict(item.get("applies_to")))
        applies_to.pop("binding_id", None)
        if applies_to.get("phase_id") and not applies_to.get("constituent_id"):
            applies_to["constituent_id"] = applies_to.pop("phase_id")
        source_payload = _normalize_source_for_claim(source)
        evidence_payload = _normalize_evidence_for_claim(evidence, first_evidence)
        confidence_payload = {
            "score": conf.get("score"),
            "label": conf.get("confidence"),
        }
        confidence_payload = {k: v for k, v in confidence_payload.items() if v not in (None, "", [])}

        claim_payload = {
            "claim_id": claim_id,
            "parameter": _canonical_parameter_payload(item, original_claim),
            "assertion": _canonical_assertion_payload(item, original_claim),
            "applies_to": applies_to,
            "provenance": _canonical_provenance_payload(source_payload, original_claim),
            "governing_equation_ids": (
                original_claim.get("governing_equation_ids")
                if isinstance(original_claim.get("governing_equation_ids"), list)
                else item.get("governing_equation_ids", [])
            ),
            "evidence_ids": evidence_ids,
            "notes": _first_non_null(original_claim.get("notes"), item.get("notes")),
        }
        claim_payload = {k: v for k, v in claim_payload.items() if v not in (None, "", [])}
        claims.append(claim_payload)

    extracted_json["parameter_claims"] = claims
    extracted_json.pop("provenance_records", None)
    extracted_json.pop("binding_contexts", None)
    extracted_json.pop("references", None)
    extracted_json.pop("parameters", None)
    return extracted_json, {
        "claims_built": len(claims),
        "claim_unit": "one v5.0.2 parameter claim per normalized item",
        "registry_mode": "removed_legacy_registry",
    }
