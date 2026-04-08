from __future__ import annotations

from typing import Any, Dict, Tuple


def _with_claim_id_first(item: Dict[str, Any], claim_id: str) -> Dict[str, Any]:
    reordered: Dict[str, Any] = {"claim_id": claim_id}
    for key, value in item.items():
        if key in {"claim_id", "parameter_id"}:
            continue
        reordered[key] = value
    return reordered


def assign_stable_claim_ids(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params = extracted_json.get("parameters") or {}
    registry = params.get("registry") or []
    assigned = 0
    preserved = 0

    if not isinstance(registry, list):
        return extracted_json, {
            "assigned_count": 0,
            "preserved_count": 0,
            "registry_count": 0,
        }

    for idx, item in enumerate(registry):
        if not isinstance(item, dict):
            continue
        existing = str(item.get("claim_id") or item.get("parameter_id") or "").strip()
        if existing:
            preserved += 1
            registry[idx] = _with_claim_id_first(item, existing)
            continue
        claim_id = f"claim_{idx + 1:04d}"
        registry[idx] = _with_claim_id_first(item, claim_id)
        assigned += 1

    extracted_json.setdefault("parameters", {})
    extracted_json["parameters"]["registry"] = registry
    return extracted_json, {
        "assigned_count": assigned,
        "preserved_count": preserved,
        "registry_count": len([it for it in registry if isinstance(it, dict)]),
    }
