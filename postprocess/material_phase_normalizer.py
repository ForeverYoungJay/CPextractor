from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _combined_material_notes(extracted_json: Dict[str, Any], material: Dict[str, Any]) -> str:
    parts: List[str] = []
    for value in (
        material.get("notes"),
        _safe_dict(extracted_json.get("microstructure")).get("notes"),
        extracted_json.get("global_notes"),
    ):
        text = str(value or "").strip()
        if text:
            parts.append(text)
    for studied in _safe_list(_safe_dict(extracted_json.get("paper_profile")).get("studied_materials")):
        if isinstance(studied, dict):
            text = str(studied.get("notes") or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _lattice_bundle_for_token(token: str) -> Dict[str, str]:
    token = str(token or "").strip().lower()
    if token == "bcc":
        return {"crystal_system": "cubic", "bravais_lattice": "I", "lattice_type": "bcc"}
    if token == "fcc":
        return {"crystal_system": "cubic", "bravais_lattice": "F", "lattice_type": "fcc"}
    if token == "hcp":
        return {"crystal_system": "hexagonal", "bravais_lattice": "P", "lattice_type": "hcp"}
    return {}


def _propagate_shared_lattice_from_notes(
    extracted_json: Dict[str, Any],
    material: Dict[str, Any],
    phases: List[Dict[str, Any]],
) -> int:
    notes = _combined_material_notes(extracted_json, material).lower()
    if not notes or len(phases) < 2:
        return 0

    phase_name_map = {}
    for phase in phases:
        name = str(phase.get("phase_name") or "").strip().lower()
        if name:
            phase_name_map[name] = phase

    propagated = 0
    present_names = list(phase_name_map.keys())
    lattice_patterns = {
        "bcc": [r"body[-\s]*centered cubic", r"\bbcc\b"],
        "fcc": [r"face[-\s]*centered cubic", r"\bfcc\b"],
        "hcp": [r"hexagonal close[-\s]*packed", r"\bhcp\b"],
    }

    for i, name_a in enumerate(present_names):
        for name_b in present_names[i + 1:]:
            if name_a not in notes or name_b not in notes:
                continue
            for token, patterns in lattice_patterns.items():
                if not any(re.search(pattern, notes) for pattern in patterns):
                    continue
                lattice_bundle = _lattice_bundle_for_token(token)
                for phase_name in (name_a, name_b):
                    phase = phase_name_map[phase_name]
                    cs = _safe_dict(phase.get("crystal_structure"))
                    if cs.get("lattice_type"):
                        continue
                    cs.update({k: v for k, v in lattice_bundle.items() if v and not cs.get(k)})
                    phase["crystal_structure"] = cs
                    propagated += 1
                return propagated
    return propagated


def normalize_material_phases(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    material = _safe_dict(extracted_json.get("material"))
    phases = [p for p in _safe_list(material.get("phases")) if isinstance(p, dict)]
    top_cs = _safe_dict(material.get("crystal_structure"))

    created = 0
    migrated = 0
    ids_filled = 0

    if not phases and (str(material.get("phase") or "").strip().lower() == "single" or top_cs or material.get("name")):
        phases = [{
            "phase_id": "phase_1",
            "phase_name": material.get("name"),
            "role": "matrix",
            "volume_fraction": {
                "value_SI": 1.0 if str(material.get("phase") or "").strip().lower() == "single" else None,
                "unit_SI": "fraction",
                "reported_value": 1 if str(material.get("phase") or "").strip().lower() == "single" else None,
                "reported_unit": "fraction" if str(material.get("phase") or "").strip().lower() == "single" else None,
                "notes": "Auto-created single phase record" if str(material.get("phase") or "").strip().lower() == "single" else None,
            },
            "crystal_structure": top_cs or None,
            "notes": None,
        }]
        created += 1

    for i, phase in enumerate(phases, start=1):
        if not phase.get("phase_id"):
            phase["phase_id"] = f"phase_{i}"
            ids_filled += 1

    if len(phases) == 1 and top_cs:
        phase_cs = _safe_dict(phases[0].get("crystal_structure"))
        if not phase_cs:
            phases[0]["crystal_structure"] = top_cs
            migrated += 1
        material.pop("crystal_structure", None)

    propagated_shared_lattice = _propagate_shared_lattice_from_notes(extracted_json, material, phases)

    material["phases"] = phases
    extracted_json["material"] = material
    return extracted_json, {
        "phase_records": len(phases),
        "phase_records_created": created,
        "phase_ids_filled": ids_filled,
        "crystal_structure_migrated_to_phase": migrated,
        "shared_lattice_propagated_from_notes": propagated_shared_lattice,
        "material_level_crystal_structure_present": bool(_safe_dict(material.get("crystal_structure"))),
    }
