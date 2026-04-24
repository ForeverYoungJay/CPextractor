from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def normalize_material_phases(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    materials = [m for m in _safe_list(extracted_json.get("materials")) if isinstance(m, dict)]
    constituents = [c for c in _safe_list(extracted_json.get("constituents")) if isinstance(c, dict)]

    material_ids_filled = 0

    for idx, material in enumerate(materials, start=1):
        if not str(material.get("material_id") or "").strip():
            material["material_id"] = f"mat_{idx:03d}"
            material_ids_filled += 1

    extracted_json["materials"] = materials
    extracted_json["constituents"] = constituents
    extracted_json.pop("material", None)
    return extracted_json, {
        "schema_version": "5.0.2",
        "materials_checked": len(materials),
        "constituents_checked": len(constituents),
        "material_ids_filled": material_ids_filled,
        "constituent_ids_filled": 0,
        "inferred_single_phase_constituents": 0,
    }
