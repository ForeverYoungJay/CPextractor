from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from postprocess.record_links import project_parameter_item


_ELASTIC_CANONICALS = {"c11", "c12", "c13", "c33", "c44", "c55", "c66", "e", "nu", "g", "k"}


def _is_expanded_registry_item(item: Dict[str, Any]) -> bool:
    return any(k in item for k in ("value", "unit", "source", "applies_to"))


def _infer_block_from_registry_item(item: Dict[str, Any]) -> str:
    domain = str(item.get("domain") or "").strip().lower()
    if domain == "elastic":
        return "elastic"
    if domain in {"plastic", "twinning", "damage", "thermal", "numerical"}:
        return "plastic"
    cname = str(item.get("canonical_name") or "").strip().lower()
    if cname in _ELASTIC_CANONICALS:
        return "elastic"
    return "plastic"


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _is_compact_registry(registry: list[Any]) -> bool:
    if not registry:
        return False
    compact_like = 0
    dict_count = 0
    for item in registry:
        if not isinstance(item, dict):
            continue
        dict_count += 1
        if item.get("claim_id") and all(k not in item for k in ("value", "unit", "source", "applies_to")):
            compact_like += 1
    return dict_count > 0 and compact_like == dict_count


def _claim_to_parameter_item(claim: Dict[str, Any]) -> Dict[str, Any]:
    claim = _safe_dict(claim)
    source = _safe_dict(claim.get("source")) or _safe_dict(claim.get("provenance"))
    return {
        "claim_id": claim.get("claim_id"),
        "domain": claim.get("domain"),
        "canonical_name": claim.get("canonical_name"),
        "canonical_name_raw": claim.get("canonical_name_raw"),
        "canonical_name_normalized": claim.get("canonical_name_normalized"),
        "symbol": claim.get("symbol"),
        "description": claim.get("description"),
        "value": claim.get("value", claim.get("reported_value")),
        "unit": claim.get("unit", claim.get("reported_unit")),
        "value_SI": claim.get("value_SI", claim.get("normalized_value")),
        "unit_SI": claim.get("unit_SI", claim.get("normalized_unit")),
        "applies_to": _safe_dict(claim.get("applies_to")),
        "source": source,
        "evidence": _safe_dict(claim.get("evidence")),
        "notes": claim.get("notes"),
        "temperature_dependent": claim.get("temperature_dependent"),
        "strain_rate_dependent": claim.get("strain_rate_dependent"),
        "valid_range": claim.get("valid_range"),
    }


def iter_parameter_items(extracted_json: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yield (block, item) with block in {"elastic","plastic"}.
    Prefer schema-v2.1.1 parameters.registry; fallback to legacy blocks.
    """
    registry = (extracted_json.get("parameters", {}) or {}).get("registry", [])
    claims = _safe_list(extracted_json.get("parameter_claims"))
    if claims and (not isinstance(registry, list) or not registry or _is_compact_registry(registry)):
        for claim in claims:
            if isinstance(claim, dict):
                projected = _claim_to_parameter_item(claim)
                yield _infer_block_from_registry_item(projected), projected
        return
    if isinstance(registry, list):
        for idx, it in enumerate(registry):
            if isinstance(it, dict):
                projected = it if _is_expanded_registry_item(it) else project_parameter_item(extracted_json, idx, it)
                yield _infer_block_from_registry_item(projected), projected
        return

    for it in ((extracted_json.get("elastic_parameters", {}) or {}).get("constants", []) or []):
        if isinstance(it, dict):
            yield "elastic", it
    for it in ((extracted_json.get("plastic_parameters", {}) or {}).get("parameters", []) or []):
        if isinstance(it, dict):
            yield "plastic", it


def iter_parameter_items_with_index(extracted_json: Dict[str, Any]) -> Iterable[Tuple[int, str, Dict[str, Any]]]:
    """
    Yield (registry_index, block, item).
    For legacy schemas, the index is a synthetic running index.
    """
    registry = (extracted_json.get("parameters", {}) or {}).get("registry", [])
    claims = _safe_list(extracted_json.get("parameter_claims"))
    if claims and (not isinstance(registry, list) or not registry or _is_compact_registry(registry)):
        for idx, claim in enumerate(claims):
            if isinstance(claim, dict):
                projected = _claim_to_parameter_item(claim)
                yield idx, _infer_block_from_registry_item(projected), projected
        return
    if isinstance(registry, list):
        for idx, it in enumerate(registry):
            if isinstance(it, dict):
                projected = it if _is_expanded_registry_item(it) else project_parameter_item(extracted_json, idx, it)
                yield idx, _infer_block_from_registry_item(projected), projected
        return

    idx = 0
    for it in ((extracted_json.get("elastic_parameters", {}) or {}).get("constants", []) or []):
        if isinstance(it, dict):
            yield idx, "elastic", it
            idx += 1
    for it in ((extracted_json.get("plastic_parameters", {}) or {}).get("parameters", []) or []):
        if isinstance(it, dict):
            yield idx, "plastic", it
            idx += 1
