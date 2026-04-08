from __future__ import annotations

from typing import Any, Dict, Tuple

from postprocess.reference_resolver import resolve_references
from postprocess.claim_id_assigner import assign_stable_claim_ids
from postprocess.slip_system_resolver import resolve_slip_systems
from postprocess.parameter_normalizer import normalize_parameters
from postprocess.parameter_table_resolver import resolve_parameter_tables
from postprocess.unit_normalizer import normalize_extracted_units
from postprocess.provenance_normalizer import normalize_provenance
from postprocess.document_backfill import backfill_document_metadata
from postprocess.material_phase_normalizer import normalize_material_phases
from postprocess.condition_binding import resolve_condition_bindings
from postprocess.quality_checks import run_quality_checks
from postprocess.evidence_grounding import verify_evidence_grounding
from postprocess.confidence_fusion import fuse_confidence
from postprocess.claim_builder import build_parameter_claims
from postprocess.final_hierarchy import build_final_hierarchy


def run_structure_normalization(
    extracted_json: Dict[str, Any],
    *,
    paper_dir: str,
    doi_hint: str | None = None,
    reference_map: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    reports: Dict[str, Any] = {}
    extracted = extracted_json
    extracted, reports["claim_id_assignment"] = assign_stable_claim_ids(extracted)
    if reference_map:
        extracted, reports["reference_resolution"] = resolve_references(extracted, reference_map)
    extracted, reports["slip_system_resolution"] = resolve_slip_systems(extracted, paper_dir)
    extracted, reports["parameter_normalization"] = normalize_parameters(extracted)
    extracted, reports["unit_normalization"] = normalize_extracted_units(extracted)
    extracted, reports["document_backfill"] = backfill_document_metadata(
        extracted,
        paper_dir=paper_dir,
        doi_hint=doi_hint,
    )
    extracted, reports["material_phase_normalization"] = normalize_material_phases(extracted)
    return extracted, reports


def run_evidence_linking(
    extracted_json: Dict[str, Any],
    *,
    paper_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    reports: Dict[str, Any] = {}
    extracted = extracted_json
    extracted, reports["provenance_normalization"] = normalize_provenance(extracted)
    extracted, reports["parameter_table_resolution"] = resolve_parameter_tables(extracted, paper_dir)
    extracted, reports["condition_binding"] = resolve_condition_bindings(extracted)
    extracted, reports["evidence_grounding"] = verify_evidence_grounding(extracted, paper_dir)
    return extracted, reports


def run_linking(
    extracted_json: Dict[str, Any],
    *,
    paper_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return run_evidence_linking(extracted_json, paper_dir=paper_dir)


def run_deterministic_validation(
    extracted_json: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    extracted, quality_report = run_quality_checks(extracted_json)
    return extracted, {"quality_checks": quality_report}


def run_finalization(
    extracted_json: Dict[str, Any],
    *,
    evaluation_report: Dict[str, Any] | None,
    quality_report: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    reports: Dict[str, Any] = {}
    extracted = extracted_json
    extracted, reports["confidence_fusion"] = fuse_confidence(
        extracted,
        quality_report or {},
        evaluation_report,
    )
    extracted, reports["parameter_claims"] = build_parameter_claims(
        extracted,
        evaluation_report=evaluation_report,
        confidence_report=reports.get("confidence_fusion"),
    )
    extracted, reports["final_hierarchy"] = build_final_hierarchy(extracted)
    return extracted, reports
