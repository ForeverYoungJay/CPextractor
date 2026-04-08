from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        if value not in (None, "", [], {}):
            return value
    return None


def _compact_value_unit(value: Dict[str, Any]) -> Dict[str, Any] | None:
    value = _safe_dict(value)
    out = {
        "value": value.get("value"),
        "unit": value.get("unit"),
        "notes": value.get("notes"),
    }
    out = {k: v for k, v in out.items() if v not in (None, "", [], {})}
    return out or None


def _legacy_grain_size_payload(value: Dict[str, Any]) -> Dict[str, Any] | None:
    value = _safe_dict(value)
    out = {
        "value": value.get("value"),
        "unit": value.get("unit"),
        "distribution": value.get("distribution"),
        "notes": value.get("notes"),
    }
    out = {k: v for k, v in out.items() if v not in (None, "", [], {})}
    return out or None


def _legacy_microstructure_to_phase_microstructure(extracted_json: Dict[str, Any]) -> Dict[str, Any] | None:
    micro = _safe_dict(extracted_json.get("microstructure"))
    if not micro:
        return None

    orientation = _safe_dict(micro.get("orientation_texture"))
    defect_state = _safe_dict(micro.get("initial_defect_state"))
    dislocation_density = _safe_dict(defect_state.get("dislocation_density"))
    out = {
        "grain_structure": micro.get("grain_structure"),
        "grain_size": {
            "value": _safe_dict(micro.get("grain_size")).get("value"),
            "unit": _safe_dict(micro.get("grain_size")).get("unit"),
            "distribution": _safe_dict(micro.get("grain_size")).get("distribution"),
            "notes": _safe_dict(micro.get("grain_size")).get("notes"),
        },
        "texture": {
            "description": orientation.get("description"),
            "method": orientation.get("texture_type"),
            "notes": orientation.get("notes"),
        },
        "morphology": None,
        "defect_state": {
            "dislocation_density": {
                "value": dislocation_density.get("value"),
                "unit": dislocation_density.get("unit"),
                "notes": dislocation_density.get("notes"),
            },
            "precipitates": defect_state.get("precipitate_state"),
            "porosity": None,
            "notes": defect_state.get("notes"),
        },
        "notes": micro.get("notes"),
    }
    if not any(v not in (None, "", [], {}) for v in out.values()):
        return None
    return out


def _material_level_microstructure_summary(extracted_json: Dict[str, Any]) -> Dict[str, Any] | None:
    micro = _safe_dict(extracted_json.get("microstructure"))
    if not micro:
        return None

    parts: List[str] = []
    grain_structure = str(micro.get("grain_structure") or "").strip()
    if grain_structure:
        parts.append(grain_structure.replace("_", " "))
    grain_size = _safe_dict(micro.get("grain_size"))
    if grain_size.get("value") not in (None, ""):
        unit = str(grain_size.get("unit") or "").strip()
        value = grain_size.get("value")
        parts.append(f"grain size {value}{(' ' + unit) if unit else ''}".strip())
    texture = _safe_dict(micro.get("orientation_texture"))
    if str(texture.get("description") or "").strip():
        parts.append(str(texture.get("description")).strip())
    notes = _first_non_empty(micro.get("notes"), _safe_dict(micro.get("initial_defect_state")).get("notes"))
    out = {
        "summary": "; ".join(parts) if parts else None,
        "notes": notes,
    }
    return out if any(v not in (None, "", [], {}) for v in out.values()) else None


def _map_phase(phase: Dict[str, Any], *, include_legacy_microstructure: bool) -> Dict[str, Any]:
    phase = _safe_dict(phase)
    out = {
        "phase_id": phase.get("phase_id"),
        "name": phase.get("phase_name"),
        "role": phase.get("role"),
        "crystal_structure": _safe_dict(phase.get("crystal_structure")) or None,
        "volume_fraction": {
            "value": _safe_dict(phase.get("volume_fraction")).get("value_SI"),
            "unit": _safe_dict(phase.get("volume_fraction")).get("unit_SI"),
            "reported_value": _safe_dict(phase.get("volume_fraction")).get("reported_value"),
            "reported_unit": _safe_dict(phase.get("volume_fraction")).get("reported_unit"),
            "notes": _safe_dict(phase.get("volume_fraction")).get("notes"),
        } if _safe_dict(phase.get("volume_fraction")) else None,
        "notes": phase.get("notes"),
    }
    if include_legacy_microstructure:
        out["microstructure"] = include_legacy_microstructure
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def _material_from_legacy(legacy_material: Dict[str, Any], extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    legacy_material = _safe_dict(legacy_material)
    phases = [
        _map_phase(phase, include_legacy_microstructure=_legacy_microstructure_to_phase_microstructure(extracted_json))
        for phase in _safe_list(legacy_material.get("phases"))
        if isinstance(phase, dict)
    ]
    return {
        "material_id": "mat_001",
        "name": legacy_material.get("name"),
        "formula": legacy_material.get("chemical_formula"),
        "material_class": None,
        "composition": None,
        "processing_history": [],
        "phases": phases,
        "material_level_microstructure": _material_level_microstructure_summary(extracted_json),
        "notes": legacy_material.get("notes"),
    }


def _final_materials(extracted_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    paper_profile = _safe_dict(extracted_json.get("paper_profile"))
    studied_materials = [m for m in _safe_list(paper_profile.get("studied_materials")) if isinstance(m, dict)]
    legacy_material = _safe_dict(extracted_json.get("material"))
    legacy_phases = [p for p in _safe_list(legacy_material.get("phases")) if isinstance(p, dict)]
    legacy_phase_by_id = {
        str(phase.get("phase_id") or "").strip(): phase
        for phase in legacy_phases
        if str(phase.get("phase_id") or "").strip()
    }
    legacy_micro = _legacy_microstructure_to_phase_microstructure(extracted_json)
    material_level_micro = _material_level_microstructure_summary(extracted_json)

    if not studied_materials:
        material = _material_from_legacy(legacy_material, extracted_json)
        return [material] if any(material.values()) else []

    final_materials: List[Dict[str, Any]] = []
    for idx, studied in enumerate(studied_materials, start=1):
        material_id = str(studied.get("material_id") or f"mat_{idx:03d}")
        phase_ids = [
            str(pid).strip()
            for pid in _safe_list(studied.get("phase_ids"))
            if str(pid).strip()
        ]
        linked_phases = [
            _map_phase(legacy_phase_by_id[pid], include_legacy_microstructure=legacy_micro if len(legacy_phases) == 1 else None)
            for pid in phase_ids
            if pid in legacy_phase_by_id
        ]
        if not linked_phases and idx == 1 and legacy_phases:
            linked_phases = [
                _map_phase(phase, include_legacy_microstructure=legacy_micro if len(legacy_phases) == 1 else None)
                for phase in legacy_phases
            ]

        composition = _safe_dict(studied.get("composition"))
        final_materials.append({
            "material_id": material_id,
            "name": _first_non_empty(studied.get("name"), legacy_material.get("name") if idx == 1 else None),
            "formula": _first_non_empty(studied.get("chemical_formula"), legacy_material.get("chemical_formula") if idx == 1 else None),
            "material_class": studied.get("material_class"),
            "composition": {
                "basis": composition.get("basis"),
                "components": _safe_list(composition.get("rows")),
                "notes": composition.get("notes"),
            } if composition else None,
            "processing_history": [],
            "phases": linked_phases,
            "material_level_microstructure": material_level_micro if idx == 1 else None,
            "notes": _first_non_empty(studied.get("notes"), legacy_material.get("notes") if idx == 1 else None),
        })

    return final_materials


def _final_samples(extracted_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    paper_profile = _safe_dict(extracted_json.get("paper_profile"))
    sample_profiles = [s for s in _safe_list(paper_profile.get("sample_profiles")) if isinstance(s, dict)]
    out: List[Dict[str, Any]] = []
    for idx, sample in enumerate(sample_profiles, start=1):
        condition_id = sample.get("condition_id")
        condition_ids = [condition_id] if condition_id not in (None, "") else []
        microstructure_overrides = {
            "grain_size": _legacy_grain_size_payload(sample.get("grain_size")),
            "texture_or_orientation": sample.get("texture_or_orientation"),
            "phase_fractions": _safe_list(sample.get("phase_fractions")),
            "selected_grains": _safe_list(sample.get("selected_grains")),
            "notes": sample.get("notes"),
        }
        microstructure_overrides = {
            k: v for k, v in microstructure_overrides.items()
            if v not in (None, "", [], {})
        }
        out.append({
            "sample_id": str(sample.get("sample_id") or f"samp_{idx:03d}"),
            "material_id": sample.get("material_id"),
            "label": sample.get("label"),
            "processing_state": sample.get("processing_state"),
            "condition_ids": condition_ids,
            "microstructure_overrides": microstructure_overrides or None,
            "notes": sample.get("notes"),
        })
    return out


def _has_default_condition_payload(payload: Dict[str, Any]) -> bool:
    payload = _safe_dict(payload)
    return any(
        payload.get(key) not in (None, "", [], {})
        for key in ("loading_mode", "stress_state", "loading_path", "strain_rate", "temperature", "fatigue", "environment", "indentation", "notes")
    )


def _final_conditions(extracted_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    condition_profiles = [c for c in _safe_list(extracted_json.get("condition_profiles")) if isinstance(c, dict)]
    out: List[Dict[str, Any]] = []
    for idx, condition in enumerate(condition_profiles, start=1):
        out.append({
            "condition_id": str(condition.get("condition_id") or f"cond_{idx:03d}"),
            "label": condition.get("label"),
            "temperature": _compact_value_unit(condition.get("temperature")),
            "strain_rate": _compact_value_unit(condition.get("strain_rate")),
            "loading_mode": condition.get("loading_mode"),
            "stress_state": condition.get("stress_state"),
            "loading_path": _safe_dict(condition.get("loading_path")) or None,
            "fatigue": _safe_dict(condition.get("fatigue")) or None,
            "notes": condition.get("notes"),
        })

    default_condition = _safe_dict(extracted_json.get("deformation_conditions"))
    if _has_default_condition_payload(default_condition):
        existing_ids = {str(row.get("condition_id") or "").strip() for row in out}
        default_id = "cond_default"
        if default_id not in existing_ids:
            out.insert(0, {
                "condition_id": default_id,
                "label": "paper_default_condition",
                "temperature": _compact_value_unit(default_condition.get("temperature")),
                "strain_rate": _compact_value_unit(default_condition.get("strain_rate")),
                "loading_mode": default_condition.get("loading_mode"),
                "stress_state": default_condition.get("stress_state"),
                "loading_path": _safe_dict(default_condition.get("loading_path")) or None,
                "fatigue": _safe_dict(default_condition.get("fatigue")) or None,
                "notes": default_condition.get("notes"),
            })
    return out


def _final_models(extracted_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    model = _safe_dict(extracted_json.get("constitutive_model"))
    if not model:
        return []
    if not any(model.get(k) not in (None, "", [], {}) for k in ("class", "framework", "implementation", "kinematics", "rate_dependence", "notes")):
        return []
    return [{
        "model_id": "model_001",
        "class": model.get("class"),
        "framework": model.get("framework"),
        "implementation": _safe_dict(model.get("implementation")) or None,
        "kinematics": model.get("kinematics"),
        "rate_dependence": model.get("rate_dependence"),
        "notes": model.get("notes"),
    }]


def _final_mechanisms(extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    mechanisms = _safe_dict(extracted_json.get("deformation_mechanisms"))
    return {
        "slip_families": _safe_list(mechanisms.get("slip_families")),
        "twinning_families": _safe_list(mechanisms.get("twinning_families")),
        "other_mechanisms": (
            _safe_list(mechanisms.get("cleavage_families"))
            + _safe_list(mechanisms.get("damage_mechanisms"))
            + _safe_list(mechanisms.get("transformation_mechanisms"))
            + _safe_list(mechanisms.get("other_mechanisms"))
        ),
        "notes": mechanisms.get("notes"),
    }


def _study_type(materials: List[Dict[str, Any]], samples: List[Dict[str, Any]], conditions: List[Dict[str, Any]]) -> str | None:
    if len(materials) > 1 and len(conditions) > 1:
        return "comparative"
    if len(materials) > 1:
        return "multi_material"
    if len(conditions) > 1 or len(samples) > 1:
        return "multi_condition"
    if len(materials) == 1:
        return "single_material"
    return None


def _phase_lookup(materials: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for material in materials:
        for phase in _safe_list(material.get("phases")):
            if not isinstance(phase, dict):
                continue
            phase_id = str(phase.get("phase_id") or "").strip()
            if phase_id:
                out[phase_id] = phase
    return out


def _sample_lookup(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        sample_id = str(_safe_dict(sample).get("sample_id") or "").strip()
        if sample_id:
            out[sample_id] = sample
    return out


def _bundle_lookup(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for bundle in _safe_list(extracted_json.get("parameter_bundles")):
        if not isinstance(bundle, dict):
            continue
        bundle_id = str(bundle.get("bundle_id") or "").strip()
        if bundle_id:
            out[bundle_id] = bundle
    return out


def _enrich_claims(
    extracted_json: Dict[str, Any],
    *,
    materials: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    conditions: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    claims = [c for c in _safe_list(extracted_json.get("parameter_claims")) if isinstance(c, dict)]
    if not claims:
        return []

    primary_material_id = str(_safe_dict(materials[0]).get("material_id") or "").strip() if materials else None
    primary_condition_id = str(_safe_dict(conditions[0]).get("condition_id") or "").strip() if conditions else None
    primary_model_id = str(_safe_dict(models[0]).get("model_id") or "").strip() if models else None
    phase_by_id = _phase_lookup(materials)
    sample_by_id = _sample_lookup(samples)
    bundle_by_id = _bundle_lookup(extracted_json)

    out: List[Dict[str, Any]] = []
    for claim in claims:
        applies_to = dict(_safe_dict(claim.get("applies_to")))
        provenance = dict(_safe_dict(claim.get("provenance")) or _safe_dict(claim.get("source")))
        evidence = dict(_safe_dict(claim.get("evidence")))

        bundle = _safe_dict(bundle_by_id.get(str(applies_to.get("bundle_id") or "").strip()))
        if not applies_to.get("material_id") and bundle.get("material_id"):
            applies_to["material_id"] = bundle.get("material_id")
        if not applies_to.get("sample_id") and bundle.get("sample_id"):
            applies_to["sample_id"] = bundle.get("sample_id")
        if not applies_to.get("condition_id") and bundle.get("condition_id"):
            applies_to["condition_id"] = bundle.get("condition_id")

        sample = _safe_dict(sample_by_id.get(str(applies_to.get("sample_id") or "").strip()))
        if not applies_to.get("material_id") and sample.get("material_id"):
            applies_to["material_id"] = sample.get("material_id")
        if not applies_to.get("condition_id"):
            sample_conditions = [c for c in _safe_list(sample.get("condition_ids")) if c]
            if len(sample_conditions) == 1:
                applies_to["condition_id"] = sample_conditions[0]

        if not applies_to.get("material_id") and len(materials) == 1 and primary_material_id:
            applies_to["material_id"] = primary_material_id
        material = {}
        if applies_to.get("material_id"):
            material = next(
                (m for m in materials if str(_safe_dict(m).get("material_id") or "").strip() == str(applies_to.get("material_id") or "").strip()),
                {},
            )

        if not applies_to.get("phase_id"):
            phases = [p for p in _safe_list(_safe_dict(material).get("phases")) if isinstance(p, dict)]
            if len(phases) == 1:
                applies_to["phase_id"] = phases[0].get("phase_id")

        if not applies_to.get("condition_id") and len(conditions) == 1 and primary_condition_id:
            applies_to["condition_id"] = primary_condition_id

        if not applies_to.get("material_id") and applies_to.get("phase_id"):
            phase = _safe_dict(phase_by_id.get(str(applies_to.get("phase_id") or "").strip()))
            if phase:
                for mat in materials:
                    phases = _safe_list(_safe_dict(mat).get("phases"))
                    if phase in phases:
                        applies_to["material_id"] = _safe_dict(mat).get("material_id")
                        break

        confidence = _safe_dict(claim.get("confidence"))
        if confidence and not confidence.get("label") and claim.get("confidence"):
            confidence["label"] = claim.get("confidence")

        enriched = dict(claim)
        if primary_model_id and not enriched.get("model_id"):
            enriched["model_id"] = primary_model_id
        enriched["applies_to"] = applies_to
        enriched["provenance"] = provenance
        enriched["source"] = provenance
        if confidence:
            enriched["confidence"] = confidence
        enriched["evidence"] = evidence
        out.append(enriched)
    return out


def build_final_hierarchy(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    document = {
        "doi": _safe_dict(extracted_json.get("source_document")).get("doi"),
        "title": _safe_dict(extracted_json.get("source_document")).get("title"),
        "authors": _safe_list(_safe_dict(extracted_json.get("source_document")).get("authors")),
        "year": _safe_dict(extracted_json.get("source_document")).get("year"),
        "journal": _safe_dict(extracted_json.get("source_document")).get("journal_or_venue"),
    }
    materials = _final_materials(extracted_json)
    samples = _final_samples(extracted_json)
    conditions = _final_conditions(extracted_json)
    models = _final_models(extracted_json)
    mechanisms = _final_mechanisms(extracted_json)
    parameter_claims = _enrich_claims(
        extracted_json,
        materials=materials,
        samples=samples,
        conditions=conditions,
        models=models,
    )

    extracted_json["schema_version"] = "3.0.0"
    extracted_json["document"] = document
    extracted_json["study"] = {
        "study_type": _study_type(materials, samples, conditions),
        "notes": extracted_json.get("global_notes"),
    }
    extracted_json["materials"] = materials
    extracted_json["samples"] = samples
    extracted_json["conditions"] = conditions
    extracted_json["models"] = models
    extracted_json["mechanisms"] = mechanisms
    extracted_json["parameter_claims"] = parameter_claims
    return extracted_json, {
        "schema_version": "3.0.0",
        "materials": len(materials),
        "samples": len(samples),
        "conditions": len(conditions),
        "models": len(models),
        "parameter_claims": len(parameter_claims),
    }
