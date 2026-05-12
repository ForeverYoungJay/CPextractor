from __future__ import annotations

from typing import Any, Dict, List

from postprocess.record_links import resolve_evidence_objects


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _document_title(extracted_json: Dict[str, Any]) -> str:
    document = _safe_dict(extracted_json.get("document"))
    if document.get("title"):
        return _first_non_empty(document.get("title"))
    return _first_non_empty(_safe_dict(extracted_json.get("source_document")).get("title"))


def _materials(extracted_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    materials = [m for m in _safe_list(extracted_json.get("materials")) if isinstance(m, dict)]
    if materials:
        return materials
    legacy = _safe_dict(extracted_json.get("material"))
    if not legacy:
        return []
    return [{
        "material_id": None,
        "name": legacy.get("name"),
        "formula": legacy.get("chemical_formula"),
        "constituents": _safe_list(legacy.get("phases")),
    }]


def _material_lookup(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for material in _materials(extracted_json):
        material_id = str(material.get("material_id") or "").strip()
        if material_id:
            out[material_id] = material
    return out


def _process_state_lookup(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for process_state in _safe_list(extracted_json.get("process_states")):
        if not isinstance(process_state, dict):
            continue
        process_state_id = str(process_state.get("process_state_id") or "").strip()
        if process_state_id:
            out[process_state_id] = process_state
    if out:
        return out
    for sample in _safe_list(extracted_json.get("samples")):
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "").strip()
        if sample_id:
            out[sample_id] = sample
    return out


def _condition_lookup(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for condition in _safe_list(extracted_json.get("conditions")):
        if not isinstance(condition, dict):
            continue
        condition_id = str(condition.get("condition_id") or "").strip()
        if condition_id:
            out[condition_id] = condition
    return out


def _model_lookup(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for model in _safe_list(extracted_json.get("models")):
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("model_id") or "").strip()
        if model_id:
            out[model_id] = model
    return out


def _deformation_system_lookup(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for system in _safe_list(extracted_json.get("deformation_systems")):
        if not isinstance(system, dict):
            continue
        system_id = str(system.get("system_id") or "").strip()
        if system_id:
            out[system_id] = system
    return out


def _items_by_model(extracted_json: Dict[str, Any], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for item in _safe_list(extracted_json.get(key)):
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id") or "").strip()
        if model_id:
            out.setdefault(model_id, []).append(item)
    return out


def _constituent_lookup(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for constituent in _safe_list(extracted_json.get("constituents")):
        if not isinstance(constituent, dict):
            continue
        constituent_id = str(constituent.get("constituent_id") or "").strip()
        if constituent_id:
            out[constituent_id] = constituent
    if out:
        return out
    for material in _materials(extracted_json):
        if not isinstance(material, dict):
            continue
        for constituent in _safe_list(material.get("constituents") or material.get("phases")):
            if not isinstance(constituent, dict):
                continue
            constituent_id = str(
                constituent.get("constituent_id")
                or constituent.get("phase_id")
                or ""
            ).strip()
            if constituent_id:
                out[constituent_id] = constituent
    return out


def _material_summary_from_entity(material: Dict[str, Any], extracted_json: Dict[str, Any]) -> str:
    parts = [
        _first_non_empty(material.get("name")),
        _first_non_empty(material.get("formula")),
        _first_non_empty(_safe_dict(material.get("material_level_microstructure")).get("summary")),
        _document_title(extracted_json),
    ]
    return ". ".join(part for part in parts if part)


def _material_composition_summary(material: Dict[str, Any]) -> str:
    composition = _safe_dict(material.get("composition"))
    components = [c for c in _safe_list(composition.get("components")) if isinstance(c, dict)]
    bits = []
    for comp in components[:5]:
        bits.append(
            " ".join(
                part for part in (
                    str(comp.get("component") or "").strip(),
                    str(comp.get("value") or "").strip(),
                    str(comp.get("unit") or "").strip(),
                )
                if part
            )
        )
    return "; ".join(bit for bit in bits if bit)


def _constituent_summary(constituent: Dict[str, Any]) -> str:
    constituent = _safe_dict(constituent)
    parts = [
        _first_non_empty(constituent.get("name")),
        _first_non_empty(_safe_dict(constituent.get("crystal_structure")).get("lattice_type")),
        _first_non_empty(_safe_dict(constituent.get("microstructure")).get("grain_structure")),
    ]
    grain_size = _safe_dict(_safe_dict(constituent.get("microstructure")).get("grain_size"))
    if grain_size.get("value") not in (None, ""):
        parts.append(
            f"grain size {grain_size.get('value')} {_first_non_empty(grain_size.get('unit'))}".strip()
        )
    return "; ".join(part for part in parts if part)


def _process_state_summary(process_state: Dict[str, Any]) -> str:
    process_state = _safe_dict(process_state)
    parts = [
        _first_non_empty(process_state.get("name"), process_state.get("label")),
        _first_non_empty(process_state.get("processing_state")),
    ]
    overrides = _safe_dict(process_state.get("microstructure_overrides"))
    grain_size = _safe_dict(overrides.get("grain_size"))
    if grain_size.get("value") not in (None, ""):
        parts.append(f"grain size {grain_size.get('value')} {_first_non_empty(grain_size.get('unit'))}".strip())
    if overrides.get("texture_or_orientation"):
        parts.append(str(overrides.get("texture_or_orientation")).strip())
    return "; ".join(part for part in parts if part)


def _condition_summary(condition: Dict[str, Any]) -> str:
    condition = _safe_dict(condition)
    parts = [
        _first_non_empty(condition.get("label")),
        _first_non_empty(condition.get("loading_mode")),
        _first_non_empty(condition.get("stress_state")),
    ]
    temperature = _safe_dict(condition.get("temperature"))
    if temperature.get("value") not in (None, ""):
        parts.append(f"T={temperature.get('value')} {_first_non_empty(temperature.get('unit'))}".strip())
    strain_rate = _safe_dict(condition.get("strain_rate"))
    if strain_rate.get("value") not in (None, ""):
        parts.append(f"strain_rate={strain_rate.get('value')} {_first_non_empty(strain_rate.get('unit'))}".strip())
    return "; ".join(part for part in parts if part)


def _geometry_summary(geometry: Dict[str, Any]) -> str:
    geometry = _safe_dict(geometry)
    parts = [
        _first_non_empty(geometry.get("geometry_type")),
        _first_non_empty(geometry.get("mesh_type")),
        _first_non_empty(geometry.get("grid_size")),
    ]
    if geometry.get("number_of_grains") not in (None, ""):
        parts.append(f"grains {geometry.get('number_of_grains')}")
    if geometry.get("periodic_geometry"):
        parts.append(f"periodic {geometry.get('periodic_geometry')}")
    return "; ".join(part for part in parts if part)


def _orientation_summary(orientation: Dict[str, Any]) -> str:
    orientation = _safe_dict(orientation)
    return "; ".join(
        part for part in (
            _first_non_empty(orientation.get("source")),
            _first_non_empty(orientation.get("representation")),
            _first_non_empty(orientation.get("texture_type")),
        )
        if part
    )


def _numerical_summary(method: Dict[str, Any]) -> str:
    method = _safe_dict(method)
    return "; ".join(
        part for part in (
            _first_non_empty(method.get("time_integration")),
            _first_non_empty(method.get("nonlinear_solver")),
            _first_non_empty(method.get("regularization")),
        )
        if part
    )


def _evidence_snippet_for_claim(claim: Dict[str, Any], extracted_json: Dict[str, Any]) -> str:
    evidence = _safe_dict(claim.get("evidence"))
    table_evidence = _safe_dict(evidence.get("table_evidence"))
    source = _safe_dict(claim.get("source")) or _safe_dict(claim.get("provenance"))
    if evidence.get("evidence_text") or table_evidence:
        return _first_non_empty(evidence.get("evidence_text"), table_evidence.get("excerpt"), table_evidence.get("value"), source.get("notes"))
    evidence_ids = [str(v).strip() for v in _safe_list(claim.get("evidence_ids")) if str(v).strip()]
    for obj in resolve_evidence_objects(extracted_json, evidence_ids):
        if not isinstance(obj, dict):
            continue
        locator = _safe_dict(obj.get("locator"))
        return _first_non_empty(
            obj.get("snippet"),
            locator.get("excerpt"),
            locator.get("value"),
            source.get("notes"),
        )
    return _first_non_empty(source.get("notes"))


def _provenance_for_claim(claim: Dict[str, Any], extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    source = _safe_dict(claim.get("provenance")) or _safe_dict(claim.get("source"))
    if source:
        return source
    return {}


def build_parameter_vector_rows(doi: str, extracted_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims = _safe_list(extracted_json.get("parameter_claims"))
    material_by_id = _material_lookup(extracted_json)
    process_state_by_id = _process_state_lookup(extracted_json)
    condition_by_id = _condition_lookup(extracted_json)
    constituent_by_id = _constituent_lookup(extracted_json)
    model_by_id = _model_lookup(extracted_json)
    deformation_system_by_id = _deformation_system_lookup(extracted_json)
    geometries_by_model = _items_by_model(extracted_json, "simulation_geometries")
    orientations_by_model = _items_by_model(extracted_json, "orientation_inputs")
    numerics_by_model = _items_by_model(extracted_json, "numerical_methods")
    all_materials = _materials(extracted_json)
    rows: List[Dict[str, Any]] = []

    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or "").strip()
        if not claim_id:
            continue
        claim_class = _first_non_empty(claim.get("claim_class"))

        parameter = _safe_dict(claim.get("parameter"))
        assertion = _safe_dict(claim.get("assertion"))
        canonical_name = _first_non_empty(claim.get("canonical_name"), parameter.get("canonical_name"))
        symbol = _first_non_empty(claim.get("symbol"), parameter.get("symbol_reported"))
        domain = _first_non_empty(claim.get("domain"), parameter.get("domain"))
        reported_value = _first_non_empty(claim.get("value"), claim.get("reported_value"), assertion.get("reported_value"))
        reported_unit = _first_non_empty(claim.get("unit"), claim.get("reported_unit"), assertion.get("reported_unit"))
        provenance = _provenance_for_claim(claim, extracted_json)
        origin_type = _first_non_empty(_safe_dict(provenance).get("origin_type"))

        binding = _safe_dict(claim.get("applies_to"))
        material_id = _first_non_empty(binding.get("material_id"))
        process_state_id = _first_non_empty(binding.get("process_state_id"), binding.get("sample_id"))
        condition_id = _first_non_empty(binding.get("condition_id"))
        constituent_id = _first_non_empty(binding.get("constituent_id"), binding.get("phase_id"))
        mechanism = _first_non_empty(binding.get("mechanism"))
        family_id = _first_non_empty(binding.get("family_id"))
        family_name = _first_non_empty(binding.get("family_name"))
        model_id = _first_non_empty(binding.get("model_id"), claim.get("model_id"))
        branch_id = _first_non_empty(binding.get("branch_id"))
        system_ids = [str(s).strip() for s in _safe_list(binding.get("system_ids")) if str(s).strip()]
        branch_ids = [str(s).strip() for s in _safe_list(binding.get("branch_ids")) if str(s).strip()]
        if not branch_id and branch_ids:
            branch_id = branch_ids[0]

        material = material_by_id.get(material_id) if material_id else {}
        if not material and len(all_materials) == 1:
            material = all_materials[0]
            material_id = _first_non_empty(material.get("material_id"), material_id)
        process_state = process_state_by_id.get(process_state_id) if process_state_id else {}
        condition = condition_by_id.get(condition_id) if condition_id else {}
        constituent = constituent_by_id.get(constituent_id) if constituent_id else {}
        model = model_by_id.get(model_id) if model_id else {}
        system_summaries = [
            "; ".join(
                part for part in (
                    _first_non_empty(_safe_dict(deformation_system_by_id.get(system_id)).get("system_type")),
                    _first_non_empty(_safe_dict(deformation_system_by_id.get(system_id)).get("family_name")),
                    _first_non_empty(_safe_dict(deformation_system_by_id.get(system_id)).get("plane")),
                    _first_non_empty(_safe_dict(deformation_system_by_id.get(system_id)).get("direction")),
                )
                if part
            )
            for system_id in system_ids
            if deformation_system_by_id.get(system_id)
        ]
        geometry_summary = "; ".join(
            summary for summary in (_geometry_summary(row) for row in geometries_by_model.get(model_id, []))
            if summary
        )
        orientation_summary = "; ".join(
            summary for summary in (_orientation_summary(row) for row in orientations_by_model.get(model_id, []))
            if summary
        )
        numerical_summary = "; ".join(
            summary for summary in (_numerical_summary(row) for row in numerics_by_model.get(model_id, []))
            if summary
        )

        material_name = _first_non_empty(material.get("name"))
        process_state_name = _first_non_empty(process_state.get("name"), process_state.get("label"))
        condition_label = _first_non_empty(condition.get("label"))
        constituent_name = _first_non_empty(constituent.get("name"))

        source = _safe_dict(claim.get("source")) or _safe_dict(claim.get("provenance"))
        evidence = _safe_dict(claim.get("evidence"))
        table_evidence = _safe_dict(evidence.get("table_evidence"))
        evidence_ids = [str(v).strip() for v in _safe_list(claim.get("evidence_ids")) if str(v).strip()]
        evidence_object = resolve_evidence_objects(extracted_json, evidence_ids)
        first_evidence = evidence_object[0] if evidence_object else {}
        evidence_kind = "table_cell" if table_evidence else _first_non_empty(first_evidence.get("evidence_type"), "inline")
        evidence_file = _first_non_empty(evidence.get("file"), first_evidence.get("source_file"))
        evidence_snippet = _evidence_snippet_for_claim(claim, extracted_json)
        provenance_bits = []
        for key in ("references", "adopted_from_references", "calibration_based_on_references"):
            refs = _safe_list(_safe_dict(provenance).get(key))
            for ref in refs[:3]:
                if isinstance(ref, dict):
                    provenance_bits.append(
                        _first_non_empty(ref.get("citation"), ref.get("doi"), ref.get("reference_id"))
                    )
        provenance_text = "; ".join(bit for bit in provenance_bits if bit)

        retrieval_text = "\n".join(
            [
                f"DOI: {doi}",
                f"Title: {_first_non_empty(_document_title(extracted_json), 'unknown')}",
                f"Claim class: {_first_non_empty(claim_class, 'unknown')}",
                f"Material ID: {_first_non_empty(material_id, 'none')}",
                f"Material: {_first_non_empty(material_name, _material_summary_from_entity(material, extracted_json) if material else 'unknown')}",
                f"Composition: {_first_non_empty(_material_composition_summary(material), 'none')}",
                f"Process state ID: {_first_non_empty(process_state_id, 'none')}",
                f"Process state: {_first_non_empty(process_state_name, _process_state_summary(process_state), 'none')}",
                f"Condition ID: {_first_non_empty(condition_id, 'none')}",
                f"Condition: {_first_non_empty(condition_label, _condition_summary(condition), 'none')}",
                f"Constituent ID: {_first_non_empty(constituent_id, 'none')}",
                f"Constituent: {_first_non_empty(constituent_name, _constituent_summary(constituent), 'none')}",
                f"Mechanism: {_first_non_empty(mechanism, 'none')}",
                f"Family: {_first_non_empty(family_name, family_id, 'none')}",
                f"Model ID: {_first_non_empty(model_id, 'none')}",
                f"Branch ID: {_first_non_empty(branch_id, 'none')}",
                f"System IDs: {_first_non_empty(', '.join(system_ids), 'none')}",
                f"Systems: {_first_non_empty(' | '.join(system_summaries), 'none')}",
                f"Model: {_first_non_empty(_safe_dict(model).get('framework'), 'unknown')}",
                f"Geometry: {_first_non_empty(geometry_summary, 'none')}",
                f"Orientation input: {_first_non_empty(orientation_summary, 'none')}",
                f"Numerical method: {_first_non_empty(numerical_summary, 'none')}",
                f"Domain: {_first_non_empty(domain, 'unknown')}",
                f"Parameter: {_first_non_empty(canonical_name, 'unknown')}",
                f"Symbol: {_first_non_empty(symbol, 'none')}",
                f"Value: {_first_non_empty(reported_value, 'null')} {_first_non_empty(reported_unit)}".strip(),
                f"Origin: {_first_non_empty(origin_type, 'unknown')}",
                f"Provenance: {_first_non_empty(provenance_text, 'none')}",
                f"Evidence kind: {_first_non_empty(evidence_kind, 'unknown')}",
                f"Evidence file: {_first_non_empty(evidence_file, 'unknown')}",
                f"Evidence: {_first_non_empty(evidence_snippet, 'none')}",
            ]
        )

        rows.append(
            {
                "doi": doi,
                "claim_id": claim_id,
                "claim_class": claim_class,
                "material_id": material_id,
                "material_name": material_name,
                "process_state_id": process_state_id,
                "process_state_name": process_state_name,
                "sample_id": process_state_id,
                "sample_label": process_state_name,
                "condition_id": condition_id,
                "condition_label": condition_label,
                "canonical_name": canonical_name,
                "symbol": symbol,
                "domain": domain,
                "constituent_id": constituent_id,
                "constituent_name": constituent_name,
                "phase_id": constituent_id,
                "phase_name": constituent_name,
                "mechanism": mechanism,
                "family_id": family_id,
                "family_name": family_name,
                "model_id": model_id,
                "branch_id": branch_id,
                "system_ids": system_ids,
                "value_text": _first_non_empty(reported_value),
                "unit": reported_unit,
                "origin_type": origin_type,
                "evidence_file": evidence_file,
                "evidence_kind": evidence_kind,
                "evidence_snippet": evidence_snippet,
                "retrieval_text": retrieval_text,
                "metadata": {
                    "origin_type": origin_type,
                    "claim_class": claim_class,
                    "material_id": material_id,
                    "process_state_id": process_state_id,
                    "sample_id": process_state_id,
                    "condition_id": condition_id,
                    "constituent_id": constituent_id,
                    "phase_id": constituent_id,
                    "model_id": model_id,
                    "branch_id": branch_id,
                    "branch_ids": branch_ids,
                    "system_ids": system_ids,
                    "system_summaries": system_summaries,
                    "geometry_summary": geometry_summary,
                    "orientation_summary": orientation_summary,
                    "numerical_summary": numerical_summary,
                },
            }
        )

    return rows
