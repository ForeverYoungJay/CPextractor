from __future__ import annotations

from typing import Any, Dict, List


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def provenance_map(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for record in _safe_list(extracted_json.get("provenance_records")):
        if not isinstance(record, dict):
            continue
        pid = str(record.get("provenance_id") or "").strip()
        if pid:
            out[pid] = record
    return out


def binding_map(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for record in _safe_list(extracted_json.get("binding_contexts")):
        if not isinstance(record, dict):
            continue
        bid = str(record.get("binding_id") or "").strip()
        if bid:
            out[bid] = record
    return out


def evidence_map(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for record in _safe_list(extracted_json.get("evidence_objects")):
        if not isinstance(record, dict):
            continue
        eid = str(record.get("evidence_id") or "").strip()
        if eid:
            out[eid] = record
    return out


def claim_map(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for record in _safe_list(extracted_json.get("parameter_claims")):
        if not isinstance(record, dict):
            continue
        cid = str(record.get("claim_id") or "").strip()
        if cid:
            out[cid] = record
    return out


def resolve_primary_phase(extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    materials = _safe_list(extracted_json.get("materials"))
    for material in materials:
        if not isinstance(material, dict):
            continue
        phases = _safe_list(material.get("phases"))
        for phase in phases:
            if isinstance(phase, dict):
                return phase
    material = _safe_dict(extracted_json.get("material"))
    phases = _safe_list(material.get("phases"))
    for phase in phases:
        if isinstance(phase, dict):
            return phase
    return {}


def resolve_primary_crystal_structure(extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    phase = resolve_primary_phase(extracted_json)
    phase_cs = _safe_dict(phase.get("crystal_structure"))
    if phase_cs:
        return phase_cs
    materials = _safe_list(extracted_json.get("materials"))
    if materials:
        first_material = _safe_dict(materials[0])
        phases = _safe_list(first_material.get("phases"))
        if phases:
            return _safe_dict(_safe_dict(phases[0]).get("crystal_structure"))
    material = _safe_dict(extracted_json.get("material"))
    return _safe_dict(material.get("crystal_structure"))


def is_compact_registry(extracted_json: Dict[str, Any]) -> bool:
    registry = _safe_list(_safe_dict(extracted_json.get("parameters")).get("registry"))
    if not registry:
        return False
    if not _safe_list(extracted_json.get("parameter_claims")):
        return False
    compact_like = 0
    for item in registry:
        if not isinstance(item, dict):
            continue
        if item.get("claim_id") and all(k not in item for k in ("value", "source", "applies_to")):
            compact_like += 1
    return compact_like == len([it for it in registry if isinstance(it, dict)])


def inflate_registry_from_claims(extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    params = _safe_dict(extracted_json.get("parameters"))
    preserved_registry = _safe_list(params.get("registry_full"))
    if preserved_registry:
        params["registry"] = preserved_registry
        extracted_json["parameters"] = params
        return extracted_json
    if not is_compact_registry(extracted_json):
        return extracted_json
    registry = _safe_list(params.get("registry"))
    inflated: List[Dict[str, Any]] = []
    for idx, item in enumerate(registry):
        if isinstance(item, dict):
            inflated.append(project_parameter_item(extracted_json, idx, item))
    params["registry"] = inflated
    params["registry_full"] = inflated
    extracted_json["parameters"] = params
    return extracted_json


def resolve_provenance_record(extracted_json: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    source = _safe_dict(source)
    pid = str(source.get("provenance_id") or "").strip()
    if pid:
        record = provenance_map(extracted_json).get(pid)
        if isinstance(record, dict):
            return record
    return {
        "provenance_id": None,
        "origin_type": source.get("origin_type"),
        "references": source.get("references"),
        "adopted_from_references": source.get("adopted_from_references"),
        "calibration_based_on_references": source.get("calibration_based_on_references"),
        "calibration_in_this_study": source.get("calibration_in_this_study"),
        "calibration_method": source.get("calibration_method"),
        "notes": source.get("notes"),
    }


def resolve_binding_record(extracted_json: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    item = _safe_dict(item)
    bid = str(item.get("binding_id") or "").strip()
    if bid:
        record = binding_map(extracted_json).get(bid)
        if isinstance(record, dict):
            return record
    return {
        "binding_id": None,
        "scope": _safe_dict(item.get("applies_to")).get("scope"),
        "phase_id": _safe_dict(item.get("applies_to")).get("phase_id"),
        "mechanism": _safe_dict(item.get("applies_to")).get("mechanism"),
        "family_id": _safe_dict(item.get("applies_to")).get("family_id"),
        "family_name": _safe_dict(item.get("applies_to")).get("family_name"),
        "system_ids": _safe_dict(item.get("applies_to")).get("system_ids"),
        "system_count": _safe_dict(item.get("applies_to")).get("system_count"),
        "notes": _safe_dict(item.get("applies_to")).get("notes"),
    }


def resolve_evidence_objects(extracted_json: Dict[str, Any], evidence_ids: List[Any]) -> List[Dict[str, Any]]:
    e_map = evidence_map(extracted_json)
    out: List[Dict[str, Any]] = []
    for eid in evidence_ids or []:
        record = e_map.get(str(eid or "").strip())
        if isinstance(record, dict):
            out.append(record)
    return out


def resolve_claim_record(extracted_json: Dict[str, Any], item: Dict[str, Any], idx: int | None = None) -> Dict[str, Any]:
    item = _safe_dict(item)
    cid = str(item.get("claim_id") or "").strip()
    claims = claim_map(extracted_json)
    if cid and cid in claims:
        return claims[cid]
    target_location = f"parameters.registry[{idx}]" if idx is not None else None
    if target_location:
        for claim in _safe_list(extracted_json.get("parameter_claims")):
            if isinstance(claim, dict) and claim.get("parameter_location") == target_location:
                return claim
    return {}


def project_parameter_item(extracted_json: Dict[str, Any], idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
    item = _safe_dict(item)
    claim = resolve_claim_record(extracted_json, item, idx)
    if not claim:
        return item

    source = _safe_dict(claim.get("source")) or _safe_dict(claim.get("provenance"))
    applies_to = _safe_dict(claim.get("applies_to"))
    if not applies_to:
        binding = resolve_binding_record(extracted_json, {
            "binding_id": claim.get("binding_id") or item.get("binding_id"),
            "applies_to": item.get("applies_to"),
        })
        applies_to = {
            "scope": binding.get("scope"),
            "phase_id": binding.get("phase_id"),
            "mechanism": binding.get("mechanism"),
            "family_id": binding.get("family_id"),
            "family_name": binding.get("family_name"),
            "system_ids": binding.get("system_ids"),
            "system_count": binding.get("system_count"),
            "notes": binding.get("notes"),
        }

    projected = dict(item)
    projected.update({
        "claim_id": claim.get("claim_id"),
        "domain": claim.get("domain", item.get("domain")),
        "canonical_name": claim.get("canonical_name", item.get("canonical_name")),
        "canonical_name_raw": claim.get("canonical_name_raw", item.get("canonical_name_raw")),
        "canonical_name_normalized": claim.get("canonical_name_normalized", item.get("canonical_name_normalized")),
        "symbol": claim.get("symbol", item.get("symbol")),
        "description": claim.get("description", item.get("description")),
        "value": claim.get("value", claim.get("reported_value")),
        "unit": claim.get("unit", claim.get("reported_unit")),
        "value_SI": claim.get("value_SI", claim.get("normalized_value")),
        "unit_SI": claim.get("unit_SI", claim.get("normalized_unit")),
        "applies_to": applies_to,
        "source": source,
        "evidence": _safe_dict(claim.get("evidence", item.get("evidence"))),
    })
    return projected
