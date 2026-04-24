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


def _normalize_document(extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    document = _safe_dict(extracted_json.get("document"))
    source_document = _safe_dict(extracted_json.get("source_document"))
    return {
        "doi": _first_non_empty(document.get("doi"), source_document.get("doi")),
        "title": _first_non_empty(document.get("title"), source_document.get("title")),
        "authors": _safe_list(document.get("authors")) or _safe_list(source_document.get("authors")),
        "year": _first_non_empty(document.get("year"), source_document.get("year")),
        "journal": _first_non_empty(document.get("journal"), source_document.get("journal_or_venue")),
        "notes": document.get("notes"),
    }


def _normalize_study(extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    study = _safe_dict(extracted_json.get("study"))
    return {
        "study_type": study.get("study_type"),
        "primary_focus": study.get("primary_focus"),
        "notes": _first_non_empty(study.get("notes"), extracted_json.get("global_notes")),
    }


def _normalize_materials(extracted_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    materials = [m for m in _safe_list(extracted_json.get("materials")) if isinstance(m, dict)]
    if not materials:
        legacy_material = _safe_dict(extracted_json.get("material"))
        if legacy_material:
            materials = [{
                "material_id": "mat_001",
                "name": legacy_material.get("name"),
                "chemical_formula": legacy_material.get("chemical_formula"),
                "material_class": legacy_material.get("material_class"),
                "phase_mode": legacy_material.get("phase_mode"),
                "crystal_aggregate": legacy_material.get("crystal_aggregate"),
                "composition": _safe_dict(legacy_material.get("composition")) or None,
                "evidence_ids": [],
                "notes": legacy_material.get("notes"),
            }]
    ids_filled = 0
    out: List[Dict[str, Any]] = []
    for idx, material in enumerate(materials, start=1):
        row = dict(material)
        if not str(row.get("material_id") or "").strip():
            row["material_id"] = f"mat_{idx:03d}"
            ids_filled += 1
        row.setdefault("evidence_ids", [])
        out.append(row)
    return out, ids_filled


def _normalize_constituents(extracted_json: Dict[str, Any], materials: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    constituents = [c for c in _safe_list(extracted_json.get("constituents")) if isinstance(c, dict)]
    if not constituents:
        legacy_material = _safe_dict(extracted_json.get("material"))
        for idx, phase in enumerate(_safe_list(legacy_material.get("phases")), start=1):
            if not isinstance(phase, dict):
                continue
            volume_fraction = _safe_dict(phase.get("volume_fraction"))
            constituents.append({
                "constituent_id": phase.get("phase_id") or f"const_{idx:03d}",
                "material_id": _safe_dict(materials[0]).get("material_id") if materials else None,
                "process_state_id": None,
                "constituent_type": "phase",
                "name": _first_non_empty(phase.get("name"), phase.get("phase_name")),
                "aliases": [],
                "role": phase.get("role"),
                "fraction": {
                    "value": _first_non_empty(volume_fraction.get("value"), volume_fraction.get("value_SI")),
                    "unit": _first_non_empty(volume_fraction.get("unit"), volume_fraction.get("unit_SI")),
                    "reported_value": volume_fraction.get("reported_value"),
                    "reported_unit": volume_fraction.get("reported_unit"),
                    "basis": volume_fraction.get("basis"),
                    "notes": volume_fraction.get("notes"),
                } if volume_fraction else None,
                "crystal_structure": _safe_dict(phase.get("crystal_structure")) or None,
                "evidence_ids": [],
                "notes": phase.get("notes"),
            })
    ids_filled = 0
    out: List[Dict[str, Any]] = []
    for idx, constituent in enumerate(constituents, start=1):
        row = dict(constituent)
        if not str(row.get("constituent_id") or "").strip():
            row["constituent_id"] = f"const_{idx:03d}"
            ids_filled += 1
        row.setdefault("evidence_ids", [])
        out.append(row)
    return out, ids_filled


def _normalize_process_states(extracted_json: Dict[str, Any], materials: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    process_states = [s for s in _safe_list(extracted_json.get("process_states")) if isinstance(s, dict)]
    if not process_states:
        for idx, sample in enumerate(_safe_list(extracted_json.get("samples")), start=1):
            if not isinstance(sample, dict):
                continue
            process_states.append({
                "process_state_id": sample.get("sample_id") or f"ps_{idx:03d}",
                "material_id": sample.get("material_id") or (_safe_dict(materials[0]).get("material_id") if len(materials) == 1 else None),
                "label": sample.get("label"),
                "state_type": [],
                "processing_route": sample.get("processing_state"),
                "processing_steps": [],
                "state_descriptors": [],
                "evidence_ids": [],
                "notes": sample.get("notes"),
            })
    ids_filled = 0
    out: List[Dict[str, Any]] = []
    for idx, process_state in enumerate(process_states, start=1):
        row = dict(process_state)
        if not str(row.get("process_state_id") or "").strip():
            row["process_state_id"] = f"ps_{idx:03d}"
            ids_filled += 1
        row.setdefault("evidence_ids", [])
        out.append(row)
    return out, ids_filled


def _normalize_conditions(extracted_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    conditions = [c for c in _safe_list(extracted_json.get("conditions")) if isinstance(c, dict)]
    if not conditions:
        condition_profiles = [c for c in _safe_list(extracted_json.get("condition_profiles")) if isinstance(c, dict)]
        conditions = [dict(c) for c in condition_profiles]
        default_condition = _safe_dict(extracted_json.get("deformation_conditions"))
        if default_condition and any(default_condition.get(k) not in (None, "", [], {}) for k in default_condition.keys()):
            conditions.insert(0, default_condition)
    ids_filled = 0
    out: List[Dict[str, Any]] = []
    for idx, condition in enumerate(conditions, start=1):
        row = dict(condition)
        if not str(row.get("condition_id") or "").strip():
            row["condition_id"] = f"cond_{idx:03d}"
            ids_filled += 1
        row.setdefault("evidence_ids", [])
        out.append(row)
    return out, ids_filled


def _normalize_models(extracted_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, int]:
    models = [m for m in _safe_list(extracted_json.get("models")) if isinstance(m, dict)]
    if not models:
        legacy_model = _safe_dict(extracted_json.get("constitutive_model"))
        if legacy_model:
            models = [{
                "model_id": "model_001",
                "name": legacy_model.get("framework"),
                "model_type": legacy_model.get("class"),
                "model_role": "primary_simulation",
                "constituent_scope": [],
                "material_scope": [],
                "mechanism_scope": {},
                "solver_framework": {},
                "implementation": _safe_dict(legacy_model.get("implementation")) or None,
                "constitutive_description": {
                    "kinematics": legacy_model.get("kinematics"),
                    "flow_kinetics": {"rate_dependence": legacy_model.get("rate_dependence")},
                },
                "constitutive_branches": [],
                "equation_ids": [],
                "evidence_ids": [],
                "notes": legacy_model.get("notes"),
            }]
    model_ids_filled = 0
    branch_ids_filled = 0
    out: List[Dict[str, Any]] = []
    for model_idx, model in enumerate(models, start=1):
        row = dict(model)
        if not str(row.get("model_id") or "").strip():
            row["model_id"] = f"model_{model_idx:03d}"
            model_ids_filled += 1
        branches = [b for b in _safe_list(row.get("constitutive_branches")) if isinstance(b, dict)]
        norm_branches: List[Dict[str, Any]] = []
        for branch_idx, branch in enumerate(branches, start=1):
            branch_row = dict(branch)
            if not str(branch_row.get("branch_id") or "").strip():
                branch_row["branch_id"] = f"{row['model_id']}_branch_{branch_idx:02d}"
                branch_ids_filled += 1
            branch_row.setdefault("evidence_ids", [])
            branch_row["governing_equation_ids"] = _safe_list(branch_row.get("governing_equation_ids"))
            norm_branches.append(branch_row)
        row["constitutive_branches"] = norm_branches
        row["equation_ids"] = _safe_list(row.get("equation_ids"))
        row.setdefault("evidence_ids", [])
        out.append(row)
    return out, model_ids_filled, branch_ids_filled


def _normalize_microstructure_features(extracted_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    features = [f for f in _safe_list(extracted_json.get("microstructure_features")) if isinstance(f, dict)]
    ids_filled = 0
    out: List[Dict[str, Any]] = []
    for idx, feature in enumerate(features, start=1):
        row = dict(feature)
        if not str(row.get("feature_id") or "").strip():
            row["feature_id"] = f"feat_{idx:03d}"
            ids_filled += 1
        row.setdefault("evidence_ids", [])
        out.append(row)
    return out, ids_filled


def _normalize_parameter_claims(extracted_json: Dict[str, Any], materials: List[Dict[str, Any]], models: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    claims = [c for c in _safe_list(extracted_json.get("parameter_claims")) if isinstance(c, dict)]
    ids_filled = 0
    primary_material_id = _safe_dict(materials[0]).get("material_id") if len(materials) == 1 else None
    primary_model_id = _safe_dict(models[0]).get("model_id") if len(models) == 1 else None
    out: List[Dict[str, Any]] = []
    for idx, claim in enumerate(claims, start=1):
        parameter = _safe_dict(claim.get("parameter"))
        assertion = _safe_dict(claim.get("assertion"))
        provenance = _safe_dict(claim.get("provenance"))
        row = {
            "claim_id": claim.get("claim_id"),
            "parameter": {
                "canonical_name": parameter.get("canonical_name"),
                "parameter_family": parameter.get("parameter_family"),
                "raw_name": parameter.get("raw_name"),
                "symbol_reported": parameter.get("symbol_reported"),
                "domain": parameter.get("domain"),
                "description": parameter.get("description"),
            },
            "assertion": {
                "value_type": assertion.get("value_type"),
                "reported_value": assertion.get("reported_value"),
                "reported_unit": assertion.get("reported_unit"),
                "qualifier": assertion.get("qualifier"),
                "valid_range": assertion.get("valid_range"),
            },
            "applies_to": dict(_safe_dict(claim.get("applies_to"))),
            "provenance": {
                "origin_type": provenance.get("origin_type"),
                "reference_ids": _safe_list(provenance.get("reference_ids")),
                "adopted_from_reference_ids": _safe_list(provenance.get("adopted_from_reference_ids")),
                "calibration_based_on_reference_ids": _safe_list(provenance.get("calibration_based_on_reference_ids")),
                "calibration": _safe_dict(provenance.get("calibration")) or None,
            },
            "governing_equation_ids": _safe_list(claim.get("governing_equation_ids")),
            "evidence_ids": _safe_list(claim.get("evidence_ids")),
            "notes": claim.get("notes"),
        }
        if not str(row.get("claim_id") or "").strip():
            row["claim_id"] = f"claim_{idx:04d}"
            ids_filled += 1
        applies_to = dict(_safe_dict(row.get("applies_to")))
        if primary_material_id and not applies_to.get("material_id"):
            applies_to["material_id"] = primary_material_id
        if primary_model_id and not applies_to.get("model_id"):
            applies_to["model_id"] = primary_model_id
        row["applies_to"] = applies_to
        row["parameter"] = {k: v for k, v in _safe_dict(row.get("parameter")).items() if v not in (None, "", [])}
        row["assertion"] = {k: v for k, v in _safe_dict(row.get("assertion")).items() if v not in (None, "", [])}
        prov = {
            k: v for k, v in _safe_dict(row.get("provenance")).items()
            if v not in (None, "", []) and v != {}
        }
        row["provenance"] = prov
        row["governing_equation_ids"] = _safe_list(row.get("governing_equation_ids"))
        row["evidence_ids"] = _safe_list(row.get("evidence_ids"))
        out.append(row)
    return out, ids_filled


def _normalize_evidence_objects(extracted_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    evidence_objects = [e for e in _safe_list(extracted_json.get("evidence_objects")) if isinstance(e, dict)]
    ids_filled = 0
    out: List[Dict[str, Any]] = []
    for idx, evidence in enumerate(evidence_objects, start=1):
        row = dict(evidence)
        if not str(row.get("evidence_id") or "").strip():
            row["evidence_id"] = f"ev_{idx:04d}"
            ids_filled += 1
        row.pop("claim_id", None)
        row.pop("claim_ids", None)
        out.append(row)
    return out, ids_filled


def build_final_hierarchy(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    extracted = dict(extracted_json)
    document = _normalize_document(extracted)
    study = _normalize_study(extracted)
    materials, material_ids_filled = _normalize_materials(extracted)
    constituents, constituent_ids_filled = _normalize_constituents(extracted, materials)
    process_states, process_state_ids_filled = _normalize_process_states(extracted, materials)
    conditions, condition_ids_filled = _normalize_conditions(extracted)
    models, model_ids_filled, branch_ids_filled = _normalize_models(extracted)
    microstructure_features, feature_ids_filled = _normalize_microstructure_features(extracted)
    parameter_claims, claim_ids_filled = _normalize_parameter_claims(extracted, materials, models)
    evidence_objects, evidence_ids_filled = _normalize_evidence_objects(extracted)

    extracted["schema_version"] = "5.0.2"
    extracted["document"] = document
    extracted["study"] = study
    extracted["materials"] = materials
    extracted["process_states"] = process_states
    extracted["constituents"] = constituents
    extracted["models"] = models
    extracted["conditions"] = conditions
    extracted["microstructure_features"] = microstructure_features
    extracted["parameter_claims"] = parameter_claims
    extracted["evidence_objects"] = evidence_objects

    # Drop legacy views so downstream operates on the v5.0.2 hierarchy only.
    for key in (
        "source_document",
        "material",
        "paper_profile",
        "microstructure",
        "constitutive_model",
        "parameters",
        "parameter_bundles",
        "deformation_conditions",
        "condition_profiles",
        "deformation_mechanisms",
        "samples",
        "mechanisms",
    ):
        extracted.pop(key, None)

    return extracted, {
        "schema_version": "5.0.2",
        "materials": len(materials),
        "constituents": len(constituents),
        "process_states": len(process_states),
        "conditions": len(conditions),
        "models": len(models),
        "microstructure_features": len(microstructure_features),
        "parameter_claims": len(parameter_claims),
        "evidence_objects": len(evidence_objects),
        "material_ids_filled": material_ids_filled,
        "constituent_ids_filled": constituent_ids_filled,
        "process_state_ids_filled": process_state_ids_filled,
        "condition_ids_filled": condition_ids_filled,
        "model_ids_filled": model_ids_filled,
        "branch_ids_filled": branch_ids_filled,
        "feature_ids_filled": feature_ids_filled,
        "claim_ids_filled": claim_ids_filled,
        "evidence_ids_filled": evidence_ids_filled,
    }
