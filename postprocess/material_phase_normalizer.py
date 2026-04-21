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
    constituent_ids_filled = 0
    inferred_single_phase = 0

    for idx, material in enumerate(materials, start=1):
        if not str(material.get("material_id") or "").strip():
            material["material_id"] = f"mat_{idx:03d}"
            material_ids_filled += 1

    if not constituents and len(materials) == 1:
        material = materials[0]
        crystal_structure = _safe_dict(material.get("crystal_structure"))
        if material.get("name") or crystal_structure:
            constituents.append({
                "constituent_id": "const_001",
                "material_id": material.get("material_id"),
                "process_state_id": None,
                "constituent_type": "phase",
                "name": material.get("name"),
                "aliases": [],
                "role": "matrix",
                "fraction": {
                    "value": 1.0,
                    "unit": "fraction",
                    "reported_value": 1,
                    "reported_unit": "fraction",
                    "basis": "volume",
                    "notes": "Auto-created single-phase constituent for v5.0.2 normalization",
                },
                "crystal_structure": crystal_structure or None,
                "evidence_ids": [],
                "notes": material.get("notes"),
            })
            inferred_single_phase = 1

    for idx, constituent in enumerate(constituents, start=1):
        if not str(constituent.get("constituent_id") or "").strip():
            constituent["constituent_id"] = f"const_{idx:03d}"
            constituent_ids_filled += 1

    material_ids = {
        str(material.get("material_id") or "").strip()
        for material in materials
        if str(material.get("material_id") or "").strip()
    }
    default_material_id = next(iter(material_ids), None) if len(material_ids) == 1 else None
    for constituent in constituents:
        if not constituent.get("material_id") and default_material_id:
            constituent["material_id"] = default_material_id

    for material in materials:
        if not material.get("phase_mode"):
            linked = [
                c for c in constituents
                if str(c.get("material_id") or "").strip() == str(material.get("material_id") or "").strip()
            ]
            if len(linked) > 1:
                material["phase_mode"] = "multi_phase"
            elif len(linked) == 1:
                material["phase_mode"] = "single_phase"

    extracted_json["materials"] = materials
    extracted_json["constituents"] = constituents
    extracted_json.pop("material", None)
    return extracted_json, {
        "schema_version": "5.0.2",
        "materials_checked": len(materials),
        "constituents_checked": len(constituents),
        "material_ids_filled": material_ids_filled,
        "constituent_ids_filled": constituent_ids_filled,
        "inferred_single_phase_constituents": inferred_single_phase,
    }
