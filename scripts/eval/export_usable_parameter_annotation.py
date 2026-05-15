import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


JSONL_FIELDS = [
    "doi",
    "paper_dir",
    "claim_id",
    "record_index",
    "material_object.material_name",
    "material_object.material_class",
    "material_object.phase_mode",
    "material_object.constituent_name",
    "material_object.constituent_type",
    "material_object.crystal_structure",
    "material_object.process_state_label",
    "cp_model.model_label",
    "cp_model.model_name",
    "cp_model.model_type",
    "cp_model.model_role",
    "cp_model.kinematics",
    "cp_model.flow_rule_form",
    "cp_model.rate_dependence",
    "cp_model.hardening_law",
    "cp_model.solver_scale",
    "cp_model.discretization",
    "cp_model.software",
    "cp_model.code_name",
    "parameter_body.canonical_name",
    "parameter_body.symbol",
    "parameter_body.parameter_family",
    "parameter_body.raw_name",
    "parameter_body.domain",
    "parameter_body.value",
    "parameter_body.unit",
    "parameter_scope.scope_level",
    "parameter_scope.scope_target",
    "parameter_scope.mechanism",
    "parameter_scope.family_name",
    "parameter_scope.system_names",
    "parameter_scope.condition_label",
    "parameter_scope.temperature_text",
    "parameter_scope.strain_rate_text",
    "evidence.evidence_type",
    "evidence.table_id",
    "evidence.snippet",
    "provenance.origin_type",
    "provenance.reference_ids",
    "provenance.adopted_from_reference_ids",
    "provenance.calibration_based_on_reference_ids",
    "provenance.calibration_method",
    "provenance.target_type",
    "annotation.status",
]


CSV_FIELDS = [
    "material_object.material_name",
    "material_object.constituent_name",
    "material_object.process_state_label",
    "cp_model.model_type",
    "parameter_body.canonical_name",
    "parameter_body.symbol",
    "parameter_body.domain",
    "parameter_body.value",
    "parameter_body.unit",
    "parameter_scope.scope_level",
    "parameter_scope.scope_target",
    "parameter_scope.condition_label",
    "parameter_scope.temperature_text",
    "evidence.evidence_type",
    "evidence.snippet",
    "provenance.origin_type",
    "annotation.status",
]


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _iter_paper_dirs(root: Path) -> Iterable[Path]:
    for paper_dir in sorted(root.iterdir()):
        if paper_dir.is_dir() and (paper_dir / "materials_extracted.json").exists():
            yield paper_dir


def _build_lookup(rows: List[Any], key: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get(key) or "").strip()
        if value:
            out[value] = row
    return out


def _audit_map(llm_evaluation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _safe_list(llm_evaluation.get("parameter_audits")):
        if not isinstance(row, dict):
            continue
        location = str(row.get("location") or "").strip()
        if location:
            out[location] = row
    return out


def _norm_value_text(value: Any, unit: Any = None) -> str | None:
    if value in (None, "", []):
        return None
    if unit not in (None, "", []):
        return f"{value} {unit}"
    return str(value)


def _scalar_measure_text(obj: Dict[str, Any]) -> str | None:
    if not obj:
        return None
    reported_value = obj.get("reported_value")
    reported_unit = obj.get("reported_unit")
    value = obj.get("value")
    unit = obj.get("unit")
    return _norm_value_text(
        reported_value if reported_value not in (None, "") else value,
        reported_unit if reported_unit not in (None, "") else unit,
    )


def _source_scope(origin_type: str) -> str | None:
    normalized = origin_type.strip().lower()
    if normalized in {"calibrated", "original", "numerical"}:
        return "this_study"
    if normalized == "adopted":
        return "reference"
    if normalized == "adopted_then_calibrated":
        return "reference_plus_this_study"
    return None


def _crystal_structure_text(constituent: Dict[str, Any]) -> str | None:
    structure = _safe_dict(constituent.get("crystal_structure"))
    return (
        structure.get("lattice_type")
        or structure.get("bravais_lattice")
        or structure.get("crystal_system")
        or None
    )


def _branch_lookup(model: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in _safe_list(model.get("constitutive_branches")):
        if not isinstance(row, dict):
            continue
        branch_id = str(row.get("branch_id") or "").strip()
        if not branch_id:
            continue
        label = str(row.get("name") or row.get("branch_type") or "").strip()
        out[branch_id] = label
    return out


def _system_lookup(rows: List[Any]) -> Dict[str, Dict[str, Any]]:
    return _build_lookup(rows, "system_id")


def _family_rows(
    deformation_system_rows: List[Any],
    *,
    family_id: str,
    model_id: str,
    constituent_id: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in deformation_system_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("family_id") or "") != family_id:
            continue
        if model_id and str(row.get("model_id") or "") != model_id:
            continue
        if constituent_id and str(row.get("constituent_id") or "") != constituent_id:
            continue
        rows.append(row)
    return rows


def _family_target_text(
    applies_to: Dict[str, Any],
    deformation_system_rows: List[Any],
) -> str | None:
    family_id = str(applies_to.get("family_id") or "").strip()
    if not family_id:
        return None
    rows = _family_rows(
        deformation_system_rows,
        family_id=family_id,
        model_id=str(applies_to.get("model_id") or ""),
        constituent_id=str(applies_to.get("constituent_id") or ""),
    )
    if not rows:
        family_name = str(applies_to.get("family_name") or "").strip()
        return f"family:{family_name}" if family_name else f"family_id:{family_id}"

    first = rows[0]
    family_name = str(first.get("family_name") or applies_to.get("family_name") or family_id).strip()
    plane = str(first.get("plane") or "").strip()
    direction = str(first.get("direction") or "").strip()
    number_of_systems = first.get("number_of_systems")
    detail_bits = []
    if plane or direction:
        detail_bits.append(f"{plane}{direction}".strip())
    if number_of_systems not in (None, "", []):
        detail_bits.append(f"n={number_of_systems}")
    if detail_bits:
        return f"family:{family_name} ({', '.join(detail_bits)})"
    return f"family:{family_name}"


def _system_target_text(system_names: List[str], system_ids: List[Any], deformation_system_map: Dict[str, Dict[str, Any]]) -> str:
    labels = []
    for system_id in system_ids:
        row = _safe_dict(deformation_system_map.get(str(system_id)))
        family_name = str(row.get("family_name") or "").strip()
        plane = str(row.get("plane") or "").strip()
        direction = str(row.get("direction") or "").strip()
        label = family_name or str(row.get("system_type") or system_id)
        if plane or direction:
            label = f"{label} [{plane}{direction}]".strip()
        labels.append(label)
    if labels:
        return "systems:" + ", ".join(labels)
    if system_names:
        return "systems:" + ", ".join(system_names)
    return "systems"


def _model_label(model: Dict[str, Any]) -> str | None:
    model_name = str(model.get("name") or "").strip()
    constitutive = _safe_dict(model.get("constitutive_description"))
    flow = _safe_dict(constitutive.get("flow_kinetics"))
    hardening = _safe_dict(constitutive.get("hardening"))
    model_type = str(model.get("model_type") or "").strip()
    bits = [bit for bit in [model_name or model_type, flow.get("flow_rule_form"), hardening.get("slip_hardening_law")] if bit]
    if bits:
        return " | ".join(str(bit) for bit in bits)
    return model_name or model_type or None


def _scope_target(
    applies_to: Dict[str, Any],
    material: Dict[str, Any],
    constituent: Dict[str, Any],
    systems: List[str],
    deformation_system_map: Dict[str, Dict[str, Any]],
    deformation_system_rows: List[Any],
) -> str:
    scope = str(applies_to.get("scope") or "").strip()
    family_name = str(applies_to.get("family_name") or "").strip()
    constituent_name = str(constituent.get("name") or "").strip()
    material_name = str(material.get("name") or "").strip()
    system_ids = _safe_list(applies_to.get("system_ids"))
    if scope == "system":
        return _system_target_text(systems, system_ids, deformation_system_map)
    if scope == "family":
        family_target = _family_target_text(applies_to, deformation_system_rows)
        if family_target:
            return family_target
        if family_name:
            return f"family:{family_name}"
    if scope == "constituent" and constituent_name:
        return f"constituent:{constituent_name}"
    if scope == "material" and material_name:
        return f"material:{material_name}"
    if scope == "global":
        return "global"
    family_target = _family_target_text(applies_to, deformation_system_rows)
    if family_target:
        return family_target
    if family_name:
        return f"family:{family_name}"
    if constituent_name:
        return f"constituent:{constituent_name}"
    if material_name:
        return f"material:{material_name}"
    return scope or "unknown"


def _primary_evidence(claim: Dict[str, Any], evidence_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for evidence_id in _safe_list(claim.get("evidence_ids")):
        evidence = _safe_dict(evidence_map.get(str(evidence_id)))
        if evidence:
            locator = _safe_dict(evidence.get("locator"))
            return {
                "evidence_id": evidence.get("evidence_id"),
                "evidence_type": evidence.get("evidence_type"),
                "table_id": locator.get("table_id"),
                "snippet": locator.get("excerpt") or evidence.get("snippet"),
                "section_heading": evidence.get("section_heading"),
                "page": evidence.get("page"),
            }
    return {
        "evidence_id": None,
        "evidence_type": None,
        "table_id": None,
        "snippet": None,
        "section_heading": None,
        "page": None,
    }


def build_usable_parameter_rows(paper_dir: Path) -> List[Dict[str, Any]]:
    extracted = _load_json(paper_dir / "materials_extracted.json")
    llm_eval = _load_json(paper_dir / "llm_evaluation.json")
    if not extracted:
        return []

    document = _safe_dict(extracted.get("document"))
    material_map = _build_lookup(_safe_list(extracted.get("materials")), "material_id")
    constituent_map = _build_lookup(_safe_list(extracted.get("constituents")), "constituent_id")
    process_state_map = _build_lookup(_safe_list(extracted.get("process_states")), "process_state_id")
    condition_map = _build_lookup(_safe_list(extracted.get("conditions")), "condition_id")
    model_map = _build_lookup(_safe_list(extracted.get("models")), "model_id")
    evidence_map = _build_lookup(_safe_list(extracted.get("evidence_objects")), "evidence_id")
    deformation_system_rows = _safe_list(extracted.get("deformation_systems"))
    deformation_system_map = _system_lookup(deformation_system_rows)
    audit_map = _audit_map(llm_eval)

    rows: List[Dict[str, Any]] = []
    for idx, claim in enumerate(_safe_list(extracted.get("parameter_claims"))):
        if not isinstance(claim, dict):
            continue
        parameter = _safe_dict(claim.get("parameter"))
        assertion = _safe_dict(claim.get("assertion"))
        applies_to = _safe_dict(claim.get("applies_to"))
        provenance = _safe_dict(claim.get("provenance"))
        material = _safe_dict(material_map.get(str(applies_to.get("material_id") or "")))
        constituent = _safe_dict(constituent_map.get(str(applies_to.get("constituent_id") or "")))
        process_state = _safe_dict(process_state_map.get(str(applies_to.get("process_state_id") or "")))
        condition = _safe_dict(condition_map.get(str(applies_to.get("condition_id") or "")))
        model = _safe_dict(model_map.get(str(applies_to.get("model_id") or "")))
        branch_map = _branch_lookup(model)
        system_names = []
        for system_id in _safe_list(applies_to.get("system_ids")):
            row = _safe_dict(deformation_system_map.get(str(system_id)))
            label = str(row.get("family_name") or row.get("system_type") or system_id)
            system_names.append(label)
        evidence = _primary_evidence(claim, evidence_map)
        claim_id = str(claim.get("claim_id") or f"claim_{idx + 1:04d}")
        audit = _safe_dict(audit_map.get(f"claim:{claim_id}") or audit_map.get(f"parameters.registry[{idx}]"))

        rows.append({
            "doi": document.get("doi") or paper_dir.name.replace("_", "/"),
            "paper_dir": str(paper_dir),
            "claim_id": claim_id,
            "record_index": idx,
            "material_object": {
                "material_id": applies_to.get("material_id"),
                "material_name": material.get("name"),
                "material_class": material.get("material_class"),
                "phase_mode": material.get("phase_mode"),
                "constituent_id": applies_to.get("constituent_id"),
                "constituent_name": constituent.get("name"),
                "constituent_type": constituent.get("constituent_type"),
                "crystal_structure": _crystal_structure_text(constituent),
                "process_state_id": applies_to.get("process_state_id"),
                "process_state_label": process_state.get("label"),
            },
            "cp_model": {
                "model_id": applies_to.get("model_id"),
                "model_label": _model_label(model),
                "model_name": model.get("name"),
                "model_type": model.get("model_type"),
                "model_role": model.get("model_role"),
                "kinematics": _safe_dict(model.get("constitutive_description")).get("kinematics"),
                "flow_rule_form": _safe_dict(_safe_dict(model.get("constitutive_description")).get("flow_kinetics")).get("flow_rule_form"),
                "rate_dependence": _safe_dict(_safe_dict(model.get("constitutive_description")).get("flow_kinetics")).get("rate_dependence"),
                "hardening_law": _safe_dict(_safe_dict(model.get("constitutive_description")).get("hardening")).get("slip_hardening_law"),
                "solver_scale": _safe_dict(model.get("solver_framework")).get("scale"),
                "discretization": _safe_dict(model.get("solver_framework")).get("discretization"),
                "software": _safe_dict(model.get("implementation")).get("software"),
                "code_name": _safe_dict(model.get("implementation")).get("code_name"),
                "branch_ids": _safe_list(applies_to.get("branch_ids")),
                "branch_labels": [branch_map.get(str(branch_id), str(branch_id)) for branch_id in _safe_list(applies_to.get("branch_ids"))],
            },
            "parameter_body": {
                "canonical_name": parameter.get("canonical_name"),
                "symbol": parameter.get("symbol_reported"),
                "parameter_family": parameter.get("parameter_family"),
                "raw_name": parameter.get("raw_name"),
                "domain": parameter.get("domain"),
                "value": assertion.get("reported_value"),
                "unit": assertion.get("reported_unit"),
            },
            "parameter_scope": {
                "scope_level": applies_to.get("scope"),
                "scope_target": _scope_target(
                    applies_to,
                    material,
                    constituent,
                    system_names,
                    deformation_system_map,
                    deformation_system_rows,
                ),
                "mechanism": applies_to.get("mechanism"),
                "family_id": applies_to.get("family_id"),
                "family_name": applies_to.get("family_name"),
                "system_ids": _safe_list(applies_to.get("system_ids")),
                "system_names": system_names,
                "condition_id": applies_to.get("condition_id"),
                "condition_label": condition.get("label"),
                "temperature_text": _scalar_measure_text(_safe_dict(condition.get("temperature"))),
                "strain_rate_text": _scalar_measure_text(_safe_dict(condition.get("strain_rate"))),
                "notes": applies_to.get("notes"),
            },
            "evidence": evidence,
            "provenance": {
                "origin_type": provenance.get("origin_type"),
                "reference_ids": _safe_list(provenance.get("reference_ids")),
                "adopted_from_reference_ids": _safe_list(provenance.get("adopted_from_reference_ids")),
                "calibration_based_on_reference_ids": _safe_list(provenance.get("calibration_based_on_reference_ids")),
                "calibration_method": _safe_dict(provenance.get("calibration")).get("method"),
                "target_type": _safe_dict(provenance.get("calibration")).get("target_type"),
                "target_description": _safe_dict(provenance.get("calibration")).get("target_description"),
                "observation_scope": _safe_dict(provenance.get("calibration")).get("observation_scope"),
                "notes": _safe_dict(provenance.get("calibration")).get("notes"),
            },
            "prediction_context": {
                "llm_verdict": audit.get("verdict"),
                "review_required": audit.get("review_required"),
            },
            "annotation": {
                "status": "correct",
            },
        })
    return rows


def _get_nested(record: Dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _iter_sheet_rows(records: List[Dict[str, Any]], fields: List[str]) -> Iterable[Dict[str, str]]:
    for record in records:
        yield {field: _stringify(_get_nested(record, field)) for field in fields}


def _write_csv(path: Path, records: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in _iter_sheet_rows(records, fields):
            writer.writerow(row)


def _safe_id(doi: str) -> str:
    return doi.replace("/", "_")


def _write_packet_readme(path: Path) -> None:
    lines = [
        "# Usable Parameter Annotation Packet",
        "",
        "Edit `usable_parameters.csv` for manual review.",
        "",
        "Recommended editable fields:",
        "- material / constituent context",
        "- model type / hardening law",
        "- parameter identity, value, unit",
        "- scope target",
        "- evidence summary",
        "- provenance origin",
        "- `annotation.status`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_usable_parameter_packets(input_root: Path, output_root: Path) -> Dict[str, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for paper_dir in _iter_paper_dirs(input_root):
        rows = build_usable_parameter_rows(paper_dir)
        if not rows:
            continue
        all_rows.extend(rows)
        doi = str(rows[0].get("doi") or paper_dir.name.replace("_", "/"))
        packet_dir = output_root / "packets" / _safe_id(doi)
        packet_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = packet_dir / "usable_parameters.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        _write_csv(packet_dir / "usable_parameters.csv", rows, CSV_FIELDS)
        (packet_dir / "packet_meta.json").write_text(
            json.dumps(
                {
                    "doi": doi,
                    "paper_dir": str(paper_dir),
                    "record_count": len(rows),
                    "schema_doc": "docs/usable_parameter_annotation_schema.md",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_packet_readme(packet_dir / "README.md")
        manifest_rows.append({
            "doi": doi,
            "paper_dir": str(paper_dir),
            "packet_dir": str(packet_dir),
            "record_count": len(rows),
        })

    combined_jsonl = output_root / "usable_parameter_annotation_draft.jsonl"
    with combined_jsonl.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_csv(output_root / "usable_parameter_annotation_sheet.csv", all_rows, CSV_FIELDS)
    _write_csv(output_root / "manifest.csv", manifest_rows, ["doi", "paper_dir", "packet_dir", "record_count"])

    readme = output_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Usable Parameter Annotation Export",
                "",
                f"- input_root: {input_root}",
                f"- records: {len(all_rows)}",
                f"- papers: {len(manifest_rows)}",
                "",
                "Files:",
                "- `usable_parameter_annotation_draft.jsonl`: full draft records",
                "- `usable_parameter_annotation_sheet.csv`: editable cross-paper sheet",
                "- `manifest.csv`: packet index",
                "- `packets/<doi>/usable_parameters.jsonl`: per-paper full records",
                "- `packets/<doi>/usable_parameters.csv`: per-paper editable sheet",
            ]
        ),
        encoding="utf-8",
    )
    return {"papers": len(manifest_rows), "records": len(all_rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export parameter-centric usable-parameter annotation packets.")
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    stats = export_usable_parameter_packets(Path(args.input_root), Path(args.output_root))
    print(f"Exported {stats['records']} usable parameter records across {stats['papers']} papers -> {args.output_root}")


if __name__ == "__main__":
    main()
