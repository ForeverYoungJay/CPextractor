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
    compact_registry: List[Dict[str, Any]] = []
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
    for idx, _, item in iter_parameter_items_with_index(extracted_json):
        source = _safe_dict(item.get("source"))
        evidence = _safe_dict(item.get("evidence"))
        claim_id = str(item.get("claim_id") or f"claim_{idx + 1:04d}")
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
        source_payload = _normalize_source_for_claim(source)
        evidence_payload = _normalize_evidence_for_claim(evidence, first_evidence)
        confidence_payload = {
            "score": conf.get("score"),
            "label": conf.get("confidence"),
        }
        confidence_payload = {k: v for k, v in confidence_payload.items() if v not in (None, "", [])}

        claims.append({
            "claim_id": claim_id,
            "domain": item.get("domain"),
            "canonical_name": item.get("canonical_name"),
            "canonical_name_raw": item.get("canonical_name_raw"),
            "canonical_name_normalized": item.get("canonical_name_normalized"),
            "symbol": item.get("symbol"),
            "description": item.get("description"),
            "value": item.get("value"),
            "unit": item.get("unit"),
            "value_SI": _first_non_null(item.get("value_SI"), item.get("value")),
            "unit_SI": _first_non_null(item.get("unit_SI"), item.get("unit")),
            "applies_to": applies_to,
            "provenance": source_payload,
            "source": source_payload,
            "evidence": evidence_payload,
            "confidence": confidence_payload,
            "temperature_dependent": item.get("temperature_dependent"),
            "strain_rate_dependent": item.get("strain_rate_dependent"),
            "valid_range": item.get("valid_range"),
            "notes": item.get("notes"),
            "parameter_location": f"parameters.registry[{idx}]",
        })
        compact_registry.append({
            "claim_id": claim_id,
        })

    extracted_json["parameter_claims"] = claims
    extracted_json.pop("provenance_records", None)
    extracted_json.pop("binding_contexts", None)
    extracted_json.pop("references", None)
    extracted_json.setdefault("parameters", {})
    extracted_json["parameters"]["registry"] = compact_registry
    extracted_json["parameters"].pop("registry_full", None)
    extracted_json["parameters"].pop("registry_compact", None)
    return extracted_json, {
        "claims_built": len(claims),
        "claim_unit": "one raw-like parameter claim per registry item",
        "registry_mode": "compact_claim_index_only",
    }
