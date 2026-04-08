from __future__ import annotations

from typing import Any, Dict, List


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
        "phases": _safe_list(legacy.get("phases")),
    }]


def _material_lookup(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for material in _materials(extracted_json):
        material_id = str(material.get("material_id") or "").strip()
        if material_id:
            out[material_id] = material
    return out


def _sample_lookup(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
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


def _phase_lookup(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for material in _materials(extracted_json):
        if not isinstance(material, dict):
            continue
        for phase in _safe_list(material.get("phases")):
            if not isinstance(phase, dict):
                continue
            phase_id = str(phase.get("phase_id") or "").strip()
            if phase_id:
                out[phase_id] = phase
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


def _phase_summary(phase: Dict[str, Any]) -> str:
    phase = _safe_dict(phase)
    parts = [
        _first_non_empty(phase.get("name")),
        _first_non_empty(_safe_dict(phase.get("crystal_structure")).get("lattice_type")),
        _first_non_empty(_safe_dict(phase.get("microstructure")).get("grain_structure")),
    ]
    grain_size = _safe_dict(_safe_dict(phase.get("microstructure")).get("grain_size"))
    if grain_size.get("value") not in (None, ""):
        parts.append(
            f"grain size {grain_size.get('value')} {_first_non_empty(grain_size.get('unit'))}".strip()
        )
    return "; ".join(part for part in parts if part)


def _sample_summary(sample: Dict[str, Any]) -> str:
    sample = _safe_dict(sample)
    parts = [
        _first_non_empty(sample.get("label")),
        _first_non_empty(sample.get("processing_state")),
    ]
    overrides = _safe_dict(sample.get("microstructure_overrides"))
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


def _evidence_snippet_for_claim(claim: Dict[str, Any], extracted_json: Dict[str, Any]) -> str:
    evidence = _safe_dict(claim.get("evidence"))
    table_evidence = _safe_dict(evidence.get("table_evidence"))
    source = _safe_dict(claim.get("source")) or _safe_dict(claim.get("provenance"))
    return _first_non_empty(evidence.get("evidence_text"), table_evidence.get("excerpt"), table_evidence.get("value"), source.get("notes"))


def _provenance_for_claim(claim: Dict[str, Any], extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    source = _safe_dict(claim.get("provenance")) or _safe_dict(claim.get("source"))
    if source:
        return source
    return {}


def build_parameter_vector_rows(doi: str, extracted_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims = _safe_list(extracted_json.get("parameter_claims"))
    material_by_id = _material_lookup(extracted_json)
    sample_by_id = _sample_lookup(extracted_json)
    condition_by_id = _condition_lookup(extracted_json)
    phase_by_id = _phase_lookup(extracted_json)
    model_by_id = _model_lookup(extracted_json)
    all_materials = _materials(extracted_json)
    rows: List[Dict[str, Any]] = []

    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or "").strip()
        if not claim_id:
            continue

        canonical_name = _first_non_empty(claim.get("canonical_name"))
        symbol = _first_non_empty(claim.get("symbol"))
        domain = _first_non_empty(claim.get("domain"))
        reported_value = claim.get("value", claim.get("reported_value"))
        reported_unit = _first_non_empty(claim.get("unit", claim.get("reported_unit")))
        provenance = _provenance_for_claim(claim, extracted_json)
        origin_type = _first_non_empty(_safe_dict(provenance).get("origin_type"))

        binding = _safe_dict(claim.get("applies_to"))
        material_id = _first_non_empty(binding.get("material_id"))
        sample_id = _first_non_empty(binding.get("sample_id"))
        condition_id = _first_non_empty(binding.get("condition_id"))
        phase_id = _first_non_empty(binding.get("phase_id"))
        mechanism = _first_non_empty(binding.get("mechanism"))
        family_id = _first_non_empty(binding.get("family_id"))
        family_name = _first_non_empty(binding.get("family_name"))
        model_id = _first_non_empty(claim.get("model_id"))

        material = material_by_id.get(material_id) if material_id else {}
        if not material and len(all_materials) == 1:
            material = all_materials[0]
            material_id = _first_non_empty(material.get("material_id"), material_id)
        sample = sample_by_id.get(sample_id) if sample_id else {}
        condition = condition_by_id.get(condition_id) if condition_id else {}
        phase = phase_by_id.get(phase_id) if phase_id else {}
        model = model_by_id.get(model_id) if model_id else {}

        material_name = _first_non_empty(material.get("name"))
        sample_label = _first_non_empty(sample.get("label"))
        condition_label = _first_non_empty(condition.get("label"))
        phase_name = _first_non_empty(phase.get("name"))

        source = _safe_dict(claim.get("source")) or _safe_dict(claim.get("provenance"))
        evidence = _safe_dict(claim.get("evidence"))
        table_evidence = _safe_dict(evidence.get("table_evidence"))
        evidence_kind = "table_cell" if table_evidence else "inline"
        evidence_file = _first_non_empty(evidence.get("file"))
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
                f"Material ID: {_first_non_empty(material_id, 'none')}",
                f"Material: {_first_non_empty(material_name, _material_summary_from_entity(material, extracted_json) if material else 'unknown')}",
                f"Composition: {_first_non_empty(_material_composition_summary(material), 'none')}",
                f"Sample ID: {_first_non_empty(sample_id, 'none')}",
                f"Sample: {_first_non_empty(sample_label, _sample_summary(sample), 'none')}",
                f"Condition ID: {_first_non_empty(condition_id, 'none')}",
                f"Condition: {_first_non_empty(condition_label, _condition_summary(condition), 'none')}",
                f"Phase: {_first_non_empty(phase_id, 'none')}",
                f"Phase name: {_first_non_empty(phase_name, _phase_summary(phase), 'none')}",
                f"Mechanism: {_first_non_empty(mechanism, 'none')}",
                f"Family: {_first_non_empty(family_name, family_id, 'none')}",
                f"Model ID: {_first_non_empty(model_id, 'none')}",
                f"Model: {_first_non_empty(_safe_dict(model).get('framework'), 'unknown')}",
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
                "material_id": material_id,
                "material_name": material_name,
                "sample_id": sample_id,
                "sample_label": sample_label,
                "condition_id": condition_id,
                "condition_label": condition_label,
                "canonical_name": canonical_name,
                "symbol": symbol,
                "domain": domain,
                "phase_id": phase_id,
                "phase_name": phase_name,
                "mechanism": mechanism,
                "family_id": family_id,
                "family_name": family_name,
                "model_id": model_id,
                "value_text": _first_non_empty(reported_value),
                "unit": reported_unit,
                "origin_type": origin_type,
                "evidence_file": evidence_file,
                "evidence_kind": evidence_kind,
                "evidence_snippet": evidence_snippet,
                "retrieval_text": retrieval_text,
                "metadata": {
                    "origin_type": origin_type,
                    "material_id": material_id,
                    "sample_id": sample_id,
                    "condition_id": condition_id,
                    "phase_id": phase_id,
                    "model_id": model_id,
                },
            }
        )

    return rows
