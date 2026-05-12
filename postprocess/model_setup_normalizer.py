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


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    out = []
    prev_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        elif not prev_sep:
            out.append("_")
            prev_sep = True
    return "".join(out).strip("_")


def _map_feature_source(method: Any) -> str | None:
    raw = str(method or "").strip().lower()
    mapping = {
        "ebsd": "ebsd",
        "xrd": "xrd_odf",
        "3dxrd": "xrd_odf",
    }
    return mapping.get(raw)


def _infer_orientation_representation(feature: Dict[str, Any]) -> str | None:
    haystack = " ".join(
        str(feature.get(k) or "")
        for k in ("feature_name", "description", "notes", "value")
    ).lower()
    if "euler" in haystack:
        return "euler_angles"
    if "quaternion" in haystack:
        return "quaternion"
    if "matrix" in haystack:
        return "orientation_matrix"
    if "pole figure" in haystack:
        return "pole_figure"
    if "odf" in haystack:
        return "odf"
    if "ipf" in haystack:
        return "ipf_map"
    return None


def _infer_texture_type(feature: Dict[str, Any]) -> str | None:
    haystack = " ".join(
        str(feature.get(k) or "")
        for k in ("feature_name", "description", "notes", "value")
    ).lower()
    for token, label in (
        ("random", "random"),
        ("measured", "measured"),
        ("ideal", "ideal"),
        ("fiber", "fiber"),
        ("rolling", "rolling"),
        ("extrusion", "extrusion"),
    ):
        if token in haystack:
            return label
    return None


def _model_implies_fcc_slip(model: Dict[str, Any], constituents: List[Dict[str, Any]], materials: List[Dict[str, Any]]) -> bool:
    slip_description = _safe_dict(_safe_dict(model.get("constitutive_description")).get("slip_description"))
    haystack = " ".join(
        str(value or "")
        for value in (
            slip_description.get("notes"),
            model.get("notes"),
            model.get("name"),
        )
    ).lower()
    if "fcc" in haystack:
        return True
    model_constituent_scope = {
        str(cid or "").strip()
        for cid in _safe_list(model.get("constituent_scope"))
        if str(cid or "").strip()
    }
    for constituent in constituents:
        if not isinstance(constituent, dict):
            continue
        constituent_id = str(constituent.get("constituent_id") or "").strip()
        if model_constituent_scope and constituent_id and constituent_id not in model_constituent_scope:
            continue
        crystal = _safe_dict(constituent.get("crystal_structure"))
        if str(crystal.get("bravais_lattice") or "").strip().lower() == "fcc":
            return True
        if str(crystal.get("lattice_type") or "").strip().lower() == "fcc":
            return True
    for material in materials:
        if not isinstance(material, dict):
            continue
        hay = " ".join(
            str(material.get(k) or "")
            for k in ("name", "notes", "material_class")
        ).lower()
        if "austenitic" in hay or "fcc" in hay:
            return True
    return False


def _legacy_mechanism_root(extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    raw = extracted_json.get("deformation_mechanisms")
    if isinstance(raw, dict):
        return raw
    raw = extracted_json.get("mechanisms")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        rows = [m for m in raw if isinstance(m, dict)]
        return {
            "slip_families": [
                m for m in rows
                if str(m.get("mechanism_type") or "").strip().lower() == "slip"
                and str(m.get("level") or "").strip().lower() == "family"
            ],
            "twinning_families": [
                m for m in rows
                if str(m.get("mechanism_type") or "").strip().lower() == "twinning"
                and str(m.get("level") or "").strip().lower() == "family"
            ],
            "transformation_mechanisms": [
                m for m in rows
                if str(m.get("mechanism_type") or "").strip().lower() == "transformation"
            ],
        }
    return {}


def _build_deformation_system_from_family(
    family: Dict[str, Any],
    *,
    system_type: str,
    model_id: str | None,
    claim_ids: List[str],
) -> Dict[str, Any]:
    family_name = _first_non_empty(
        family.get("family_name"),
        family.get("name"),
        family.get("label"),
    )
    systems = [s for s in _safe_list(family.get("systems")) if isinstance(s, dict)]
    first_system = systems[0] if systems else {}
    plane = _first_non_empty(
        _safe_dict(first_system.get("plane")).get("as_written"),
        first_system.get("plane"),
    )
    direction = _first_non_empty(
        _safe_dict(first_system.get("direction")).get("as_written"),
        first_system.get("direction"),
    )
    constituent_id = _first_non_empty(
        family.get("constituent_id"),
        family.get("phase_id"),
    )
    return {
        "system_id": _first_non_empty(
            family.get("system_id"),
            family.get("family_id"),
            f"sys_{system_type}_{_slug(family_name or constituent_id or '1')}",
        ),
        "model_id": model_id,
        "constituent_id": constituent_id,
        "system_type": system_type,
        "family_name": family_name,
        "plane": plane,
        "direction": direction,
        "number_of_systems": _first_non_empty(family.get("num_systems"), len(systems) or None),
        "schmid_tensor_defined": None,
        "non_schmid_effects": family.get("non_schmid_effects"),
        "associated_parameter_claim_ids": claim_ids,
        "evidence_ids": _safe_list(family.get("evidence_ids")),
        "notes": family.get("notes"),
    }


def normalize_model_setup(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report = {
        "deformation_systems_added": 0,
        "deformation_system_links_added": 0,
        "simulation_geometries_added": 0,
        "orientation_inputs_added": 0,
        "numerical_methods_added": 0,
        "simulation_outputs_added": 0,
        "model_evaluations_added": 0,
    }
    models = [m for m in _safe_list(extracted_json.get("models")) if isinstance(m, dict)]
    constituents = [c for c in _safe_list(extracted_json.get("constituents")) if isinstance(c, dict)]
    materials = [m for m in _safe_list(extracted_json.get("materials")) if isinstance(m, dict)]
    single_model_id = str(models[0].get("model_id") or "").strip() if len(models) == 1 else ""
    claims = [c for c in _safe_list(extracted_json.get("parameter_claims")) if isinstance(c, dict)]

    deformation_systems = [d for d in _safe_list(extracted_json.get("deformation_systems")) if isinstance(d, dict)]
    if not deformation_systems:
        mechanism_root = _legacy_mechanism_root(extracted_json)
        generated: List[Dict[str, Any]] = []
        for system_type, key in (
            ("slip", "slip_families"),
            ("twin", "twinning_families"),
            ("transformation", "transformation_mechanisms"),
        ):
            for family in _safe_list(mechanism_root.get(key)):
                if not isinstance(family, dict):
                    continue
                family_id = str(family.get("family_id") or "").strip()
                family_name = str(
                    _first_non_empty(family.get("family_name"), family.get("name"), family.get("label")) or ""
                ).strip()
                related_claim_ids = []
                for claim in claims:
                    claim_id = str(claim.get("claim_id") or "").strip()
                    applies_to = _safe_dict(claim.get("applies_to"))
                    if not claim_id:
                        continue
                    if family_id and str(applies_to.get("family_id") or "").strip() == family_id:
                        related_claim_ids.append(claim_id)
                        continue
                    if family_name and str(applies_to.get("family_name") or "").strip().lower() == family_name.lower():
                        related_claim_ids.append(claim_id)
                generated.append(
                    _build_deformation_system_from_family(
                        family,
                        system_type=system_type,
                        model_id=single_model_id or None,
                        claim_ids=related_claim_ids,
                    )
                )
        if not generated:
            for idx, model in enumerate(models, start=1):
                constitutive_description = _safe_dict(model.get("constitutive_description"))
                slip_description = _safe_dict(constitutive_description.get("slip_description"))
                if str(slip_description.get("slip_families_defined") or "").strip().lower() != "yes":
                    continue
                if not _model_implies_fcc_slip(model, constituents, materials):
                    continue
                generated.append({
                    "system_id": f"{str(model.get('model_id') or f'model_{idx:03d}').strip()}_fcc_12",
                    "model_id": model.get("model_id"),
                    "constituent_id": _safe_list(model.get("constituent_scope"))[0] if _safe_list(model.get("constituent_scope")) else None,
                    "system_type": "slip",
                    "family_name": "fcc_12",
                    "plane": "{111}",
                    "direction": "<110>",
                    "number_of_systems": 12,
                    "schmid_tensor_defined": None,
                    "non_schmid_effects": None,
                    "associated_parameter_claim_ids": [],
                    "evidence_ids": [],
                    "notes": "Inferred placeholder from explicit statement that FCC slip systems were used; family/plane/direction were not explicitly enumerated in the paper.",
                })
        deformation_systems = generated
        extracted_json["deformation_systems"] = deformation_systems
        report["deformation_systems_added"] = len(generated)

    slip_ids_by_model: Dict[str, List[str]] = {}
    twin_ids_by_model: Dict[str, List[str]] = {}
    for system in deformation_systems:
        system_id = str(system.get("system_id") or "").strip()
        if not system_id:
            continue
        model_id = str(system.get("model_id") or single_model_id or "").strip()
        system_type = str(system.get("system_type") or "").strip().lower()
        if system_type == "slip":
            slip_ids_by_model.setdefault(model_id, []).append(system_id)
        elif system_type == "twin":
            twin_ids_by_model.setdefault(model_id, []).append(system_id)

    for model in models:
        model_id = str(model.get("model_id") or "").strip()
        constitutive_description = _safe_dict(model.get("constitutive_description"))
        slip_description = _safe_dict(constitutive_description.get("slip_description"))
        twinning = _safe_dict(constitutive_description.get("twinning"))
        if slip_ids_by_model.get(model_id) and not _safe_list(slip_description.get("deformation_system_ids")):
            slip_description["deformation_system_ids"] = slip_ids_by_model[model_id]
            report["deformation_system_links_added"] += len(slip_ids_by_model[model_id])
        if twin_ids_by_model.get(model_id) and not _safe_list(twinning.get("deformation_system_ids")):
            twinning["deformation_system_ids"] = twin_ids_by_model[model_id]
            report["deformation_system_links_added"] += len(twin_ids_by_model[model_id])
        if slip_description:
            constitutive_description["slip_description"] = slip_description
        if twinning:
            constitutive_description["twinning"] = twinning
        if constitutive_description:
            model["constitutive_description"] = constitutive_description

    simulation_geometries = [g for g in _safe_list(extracted_json.get("simulation_geometries")) if isinstance(g, dict)]
    if not simulation_geometries:
        generated_geometries: List[Dict[str, Any]] = []
        for idx, model in enumerate(models, start=1):
            solver = _safe_dict(model.get("solver_framework"))
            discretization = str(solver.get("discretization") or "").strip().lower()
            grain_resolution = str(solver.get("grain_resolution") or "").strip().lower()
            boundary_style = str(solver.get("boundary_condition_style") or "").strip().lower()
            legacy_geometry = str(solver.get("geometry_representation") or "").strip().lower()
            mesh_type = None
            if legacy_geometry == "voxelized":
                mesh_type = "voxel"
            elif legacy_geometry == "tessellated":
                mesh_type = "other"
            elif discretization in {"fft", "spectral"}:
                mesh_type = "spectral_grid"
            geometry_type = None
            if grain_resolution == "grain_resolved" or str(solver.get("scale") or "").strip().lower() in {"polycrystal", "aggregate"}:
                geometry_type = "grain_aggregate"
            if not any((geometry_type, mesh_type, discretization, grain_resolution, boundary_style)):
                continue
            generated_geometries.append({
                "geometry_id": f"geom_{idx:03d}",
                "model_id": model.get("model_id"),
                "geometry_type": geometry_type,
                "dimensions": None,
                "number_of_grains": None,
                "number_of_elements": None,
                "mesh_type": mesh_type,
                "element_type": None,
                "grid_size": None,
                "periodic_geometry": "yes" if boundary_style == "periodic" else None,
                "grain_shape_assumption": None,
                "evidence_ids": _safe_list(model.get("evidence_ids")),
                "notes": _first_non_empty(
                    model.get("notes"),
                    solver.get("notes"),
                    f"Backfilled from solver framework discretization={discretization or 'null'} and grain_resolution={grain_resolution or 'null'}",
                ),
            })
        simulation_geometries = generated_geometries
        extracted_json["simulation_geometries"] = simulation_geometries
        report["simulation_geometries_added"] = len(generated_geometries)

    geometry_by_model: Dict[str, str] = {}
    for geometry in simulation_geometries:
        model_id = str(geometry.get("model_id") or "").strip()
        geometry_id = str(geometry.get("geometry_id") or "").strip()
        if model_id and geometry_id and model_id not in geometry_by_model:
            geometry_by_model[model_id] = geometry_id

    orientation_inputs = [o for o in _safe_list(extracted_json.get("orientation_inputs")) if isinstance(o, dict)]
    if not orientation_inputs:
        generated_orientations: List[Dict[str, Any]] = []
        features = [f for f in _safe_list(extracted_json.get("microstructure_features")) if isinstance(f, dict)]
        for idx, feature in enumerate(features, start=1):
            family = str(feature.get("feature_family") or "").strip().lower()
            if family not in {"texture", "orientation"}:
                continue
            source = _map_feature_source(feature.get("method"))
            representation = _infer_orientation_representation(feature)
            texture_type = _infer_texture_type(feature)
            if not any((source, representation, texture_type)):
                continue
            model_id = single_model_id or ""
            generated_orientations.append({
                "orientation_id": f"ori_{idx:03d}",
                "model_id": model_id or None,
                "geometry_id": geometry_by_model.get(model_id) if model_id else None,
                "source": source,
                "representation": representation,
                "texture_type": texture_type,
                "number_of_orientations": None,
                "evidence_ids": _safe_list(feature.get("evidence_ids")),
                "notes": _first_non_empty(feature.get("description"), feature.get("notes")),
            })
        orientation_inputs = generated_orientations
        extracted_json["orientation_inputs"] = orientation_inputs
        report["orientation_inputs_added"] = len(generated_orientations)

    # Keep these extractor-owned sections untouched. If they are absent, do not
    # synthesize them from calibration provenance during postprocessing.

    return extracted_json, report
