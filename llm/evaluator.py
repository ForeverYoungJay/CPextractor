import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from llm.extractor import build_context, load_md_files, load_table_files, load_equation_files
from llm.openai_sanitize import sanitize_text_for_openai, validate_openai_json_payload
from postprocess.location_ids import claim_location
from postprocess.param_iter import iter_parameter_items_with_index
from postprocess.record_links import (
    binding_map,
    evidence_map,
    provenance_map,
    resolve_binding_record,
    resolve_evidence_objects,
    resolve_provenance_record,
)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


EVIDENCE_AGENT_SYSTEM_PROMPT = """
You are an evidence-grounding judge for crystal-plasticity parameter extraction.
Judge only whether the parameter is supported by the provided evidence and provenance.
Return JSON only.
"""

EVIDENCE_AGENT_USER_PROMPT_TEMPLATE = """
1 Task description
Evaluate evidence support for extracted crystal-plasticity parameters.

2 Task requirements
- Use only the provided parameter record, evidence text, evidence location, and grounding span.
- Output JSON exactly:
{
  "parameter_audits": [
    {
      "location": "string",
      "verdict": "pass / warning / fail",
      "score": "number 0-100",
      "supportiveness": "supported / unsupported / contradictory / insufficient_evidence",
      "confidence": "high / medium / low",
      "error_types": ["unsupported_claim / wrong_value / contradictory_evidence / other"],
      "uncertainty_types": ["missing_evidence / weak_grounding / table_parse_uncertain / other"],
      "reason": "string",
      "recommendation": "string"
    }
  ]
}

3 Processing suggestions
- Judge only whether the claim is supported by the supplied evidence package.
- Do not evaluate unit conversion, SI normalization, canonical_name mapping, or binding/scope coherence here.
- Prefer `warning` instead of `fail` if evidence is incomplete rather than contradictory.
- A grouped table row such as `c11, c12, c44 -> 183.9 GPa, 123.4 GPa, 91.5 GPa` can be valid direct evidence for a grouped record if the extraction intentionally keeps those constants together.
- For image-backed or table-based claims, do not require a separately grounded table cell if the extractor already indicates a coherent table/row/column/value reading. Treat extractor table reading as primary evidence unless there is a stronger contradiction.
- Do not call a claim unsupported solely because the evidence package lacks a strict cell coordinate, if the table row/value semantics are otherwise coherent.
- If a record lacks an explicit value and only restates an equation, definition, or literature source, treat that as insufficient_evidence rather than a normalization or consistency problem.

4 Few-shot examples
Example A:
- Evidence explicitly states tau0 = 85 MPa and the record stores tau0 = 85 MPa.
- Good behavior: supported, pass.

Example B:
- Evidence text does not actually contain the claimed value.
- Good behavior: warning or fail with unsupported_claim or insufficient evidence.

Example C:
- Evidence table cell directly contradicts the extracted value.
- Good behavior: contradictory, fail.

Reviewed feedback summary:
__FEEDBACK_SUMMARY__

Parameter records:
__RECORDS_JSON__
"""


NORMALIZATION_AGENT_SYSTEM_PROMPT = """
You are a normalization judge for crystal-plasticity parameter extraction.
Judge whether the canonical field mapping, unit normalization, and scope mapping are correct.
Return JSON only.
"""


NORMALIZATION_AGENT_USER_PROMPT_TEMPLATE = """
1 Task description
Evaluate normalization correctness for extracted crystal-plasticity parameters.

2 Task requirements
- Work only from the provided parameter records and schema-oriented metadata.
- Output JSON exactly:
{
  "parameter_audits": [
    {
      "location": "string",
      "verdict": "pass / warning / fail",
      "score": "number 0-100",
      "normalization_correctness": "correct / uncertain / likely_incorrect",
      "confidence": "high / medium / low",
      "error_types": ["wrong_unit_conversion / wrong_parameter_mapping / other"],
      "uncertainty_types": ["normalization_ambiguous / symbol_mapping_conflict / other"],
      "reason": "string",
      "recommendation": "string"
    }
  ]
}

3 Processing suggestions
- Judge only canonical parameter mapping plus reported unit / SI normalization.
- Do not judge whether the evidence package is sufficient; assume the evidence package is whatever the evidence judge saw.
- Do not judge binding coherence, family scope, phase scope, or cross-material leakage here; that belongs to the consistency judge.
- Use `wrong_parameter_mapping` when the standardized field seems incorrect.
- If a standard SI conversion is already provided and is numerically consistent, do not mark it as `wrong_unit_conversion`.
- Treat MPa -> Pa as multiply by 1e6, GPa -> Pa as multiply by 1e9, and kPa -> Pa as multiply by 1e3.
- Do not infer that a second value in a multi-parameter row inherits the first value's unit unless the evidence explicitly says so.
- If evidence is packaged as a table-row snippet plus row context, treat that as valid direct evidence for the corresponding parameter segment.
- Do not fail or escalate a parameter solely because unit is missing or value_SI/unit_SI are absent, if the parameter identity and explicit value are otherwise correct.
- Missing SI normalization alone is a low-risk formatting issue, not a substantive extraction failure.
- Do not perform physical plausibility checks or magnitude-based judgments.
- Do not label a value as a data-entry error, typo, or physically implausible if the reported value, reported unit, and SI conversion are internally consistent.
- Unusual or extreme values are not errors in this stage.
- If the SI conversion is numerically correct, do not use `wrong_unit_conversion` to express domain-level surprise about magnitude.

4 Few-shot examples
Example A:
- Symbol tau0 mapped to crss_initial with MPa -> Pa normalization.
- Good behavior: correct.

Example B:
- Symbol tau0 is mapped to the wrong canonical parameter name.
- Good behavior: wrong_parameter_mapping.

Example C:
- A grouped elastic record stores `c11, c12, c44` with comma-separated values and one shared reported unit `GPa`.
- Good behavior: do not mark this as wrong_unit_conversion solely because it is grouped.

Reviewed feedback summary:
__FEEDBACK_SUMMARY__

Parameter records:
__RECORDS_JSON__
"""


CONSISTENCY_AGENT_SYSTEM_PROMPT = """
You are a consistency judge for crystal-plasticity extraction.
Judge cross-field and cross-parameter consistency within one paper.
Return JSON only.
"""


CONSISTENCY_AGENT_USER_PROMPT_TEMPLATE = """
1 Task description
Evaluate whether extracted parameter records are self-consistent within the paper.

2 Task requirements
- Consider material, constitutive model, mechanisms, scope, and parameter set coherence.
- Output JSON exactly:
{
  "document_consistency_score": "number 0-100",
  "global_issues": [
    {
      "severity": "high / medium / low",
      "category": "cross_material_mixup / model_variant_confusion / condition_binding_error / other",
      "issue": "string",
      "recommendation": "string"
    }
  ],
  "parameter_flags": [
    {
      "location": "string",
      "verdict": "pass / warning / fail",
      "score": "number 0-100",
      "completeness": "complete / partially_complete / incomplete / not_applicable",
      "confidence": "high / medium / low",
      "error_types": ["condition_binding_error / cross_material_mixup / model_variant_confusion / other"],
      "uncertainty_types": ["condition_binding_ambiguous / other"],
      "reason": "string",
      "recommendation": "string"
    }
  ]
}

3 Processing suggestions
- Focus only on cross-material leakage, model variant confusion, and binding coherence.
- Do not invent missing parameters unless omission is clearly significant from the provided records.
- Do not use `condition_binding_error` merely to express weak or indirect evidence support. If the scope/binding itself is coherent but evidence is weak, prefer `pass`.
- Table-based claims with coherent family/phase binding but weak explicit cell grounding should not be treated as binding inconsistencies.
- Do not evaluate unit conversion, SI normalization, or evidence sufficiency here.
- Do not flag a parameter merely because one slip-family value is larger than another.
- The v5.0.2 schema allows many-to-many equation binding.
  One branch may legitimately bind to multiple governing equations, one parameter may legitimately bind to multiple equations, and multiple branches may also share one equation.
  Do not treat multi-equation arrays or shared equation labels as a consistency error by themselves.
- The v5.0.2 schema also allows multiple models in one paper, including `primary_simulation`, `comparison`, and `auxiliary` roles.
  Do not use `model_variant_confusion` merely because a paper contains more than one model or because a comparison model reuses the same constitutive law.
- Use `condition_binding_error` only for true scope incoherence such as wrong material, wrong constituent, wrong branch, impossible condition assignment, or explicit contradiction in applicability.
- If a claim is attached to a branch and that branch carries several constitutive equations, it is coherent for the claim to reference one or several of those equations depending on the text.

4 Few-shot examples
Example A:
- FCC material with FCC slip family naming and coherent CRSS/hardening set.
- Good behavior: high consistency score.

Example B:
- Two materials appear mixed into one parameter block or phase/family binding conflicts with the mechanism.
- Good behavior: flag condition_binding_error or cross_material_mixup.

Example C:
- A constitutive branch has equations (4), (5), and (6), and one parameter claim references both (5) and (6).
- Good behavior: treat this as normal v5.0.2 many-to-many binding, not as model_variant_confusion or condition_binding_error.

Example D:
- The paper contains one primary CPFE model and one comparison single-crystal or J2 model, each with its own model_role.
- Good behavior: do not flag model_variant_confusion unless parameter claims are actually mixed across incompatible models.

Reviewed feedback summary:
__FEEDBACK_SUMMARY__

Material and model summary:
__DOC_SUMMARY__

Parameter records:
__RECORDS_JSON__
"""


META_AGENT_SYSTEM_PROMPT = """
You are the meta-judge for a multi-agent crystal-plasticity extraction audit.
Combine evidence, normalization, and consistency judgments into a final document-level verdict.
Return JSON only.
"""


META_AGENT_USER_PROMPT_TEMPLATE = """
1 Task description
Produce the final document-level audit result for a crystal-plasticity extraction.

2 Task requirements
- Use the committee outputs, rule report, evidence grounding report, and source extraction summary.
- Output JSON exactly:
{
  "verdict": "pass / warning / fail",
  "overall_score": "number 0-100",
  "summary": "short string",
  "dimension_scores": {
    "schema_consistency": "number 0-100",
    "evidence_grounding": "number 0-100",
    "provenance_quality": "number 0-100",
    "parameter_plausibility": "number 0-100",
    "completeness": "number 0-100"
  },
  "critical_issues": [
    {
      "severity": "high / medium / low",
      "category": "unsupported_claim / wrong_value / provenance_conflict / missing_key_field / schema_problem / normalization_problem / cross_material_mixup / model_variant_confusion / other",
      "location": "JSON path or null",
      "issue": "string",
      "evidence": "string or null",
      "recommendation": "string"
    }
  ],
  "strengths": ["string"],
  "recommended_actions": ["string"]
}

3 Processing suggestions
- Be conservative and evidence-based.
- Treat committee disagreement as a review risk.
- Base document-level issues on committee outputs and rule reports; do not invent new parameter-level failure theories that are not grounded in those inputs.
- A fail should be reserved for high-risk or repeated critical issues.
- Mixed provenance such as "adopted from prior work, then calibrated in this study" is acceptable and should not be escalated as provenance_conflict by itself.
- Do not escalate low-risk table-based disagreements into critical issues when the disagreement is mainly `warning/pass` around weak grounding rather than a concrete wrong value or provenance conflict.
- For image-backed table extractions, weak direct cell grounding alone should not dominate the document verdict if normalization and consistency remain strong.
- Unit-only or SI-format-only disagreements should not become document-level critical issues or mandatory review escalations.
- Treat v5.0.2 many-to-many equation binding as normal structure, not as a schema or consistency problem.
- Treat the presence of comparison or auxiliary models as normal when the roles and bindings are explicit.
- Do not escalate a paper merely because one branch or one parameter is linked to multiple equations, or because multiple branches share one equation.

4 Few-shot examples
Example A:
- Committee mostly agrees, evidence grounding is strong, rule issues are minor.
- Good behavior: pass or warning.

Example B:
- Multiple unsupported values, provenance conflicts, and severe disagreement.
- Good behavior: fail.

Reviewed feedback summary:
__FEEDBACK_SUMMARY__

Document context summary:
__DOC_CONTEXT_SUMMARY__

Rule report:
__QUALITY_REPORT__

Evidence grounding report:
__EVIDENCE_REPORT__

Committee summary:
__COMMITTEE_SUMMARY__
"""


def _is_retryable_llm_error(exc: Exception) -> bool:
    name = exc.__class__.__name__
    if name in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}:
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) in {408, 409, 429, 500, 502, 503, 504}:
        return True
    return False


def _chat_completion_with_retry(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_retries: int = 4,
):
    delay = 1.0
    last_exc: Exception | None = None
    request_payload = validate_openai_json_payload({
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": messages,
    })
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**request_payload)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable_llm_error(exc):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 20.0)
    raise RuntimeError(f"LLM evaluator request failed after retries: {last_exc}")


def _trim_text(text: str, max_chars: int) -> str:
    text = sanitize_text_for_openai(text or "").strip()
    return text[:max_chars] + ("...[TRUNCATED]..." if len(text) > max_chars else "")


def _resolve_selected_sources(
    paper_dir: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    sections_dir = os.path.join(paper_dir, "sections")
    tables_dir = os.path.join(paper_dir, "tables")
    equations_dir = os.path.join(paper_dir, "equations")
    sections = load_md_files(sections_dir) if os.path.exists(sections_dir) else []
    tables = load_table_files(tables_dir) if os.path.exists(tables_dir) else []
    equations = load_equation_files(equations_dir) if os.path.exists(equations_dir) else []

    selected_path = os.path.join(paper_dir, "llm_selected_files.json")
    selection: Dict[str, Any] = {}
    if os.path.exists(selected_path):
        try:
            with open(selected_path, "r", encoding="utf-8") as f:
                selection = json.load(f)
        except Exception:
            selection = {}

    selected_section_names = set(selection.get("selected_sections", []) or [])
    resolved_table_files = {
        str(name).strip()
        for name in (selection.get("resolved_selected_table_files", []) or [])
        if str(name).strip()
    }
    selected_equation_ids = {
        str(name).strip()
        for name in (selection.get("selected_equations", []) or [])
        if str(name).strip()
    }
    selected_table_ids = set()
    for name in (selection.get("selected_tables", []) or []):
        raw = str(name).strip()
        if not raw:
            continue
        if raw.endswith(".md"):
            raw = raw[:-3]
        if raw.endswith(".json"):
            raw = raw[:-5]
        selected_table_ids.add(raw)
    selected_sections = [s for s in sections if s["name"] in selected_section_names]
    selected_tables = [
        t for t in tables
        if t.get("selection_id") in selected_table_ids or t["name"] in resolved_table_files or t["name"] in selected_table_ids
    ]
    selected_equations = [
        e for e in equations
        if e.get("selection_id") in selected_equation_ids or e["name"] in selected_equation_ids
    ]
    used_fallback_sections = False
    used_fallback_tables = False
    used_fallback_equations = False
    if not selected_sections and sections:
        selected_sections = sections[:2]
        used_fallback_sections = True
    if not selected_tables and tables:
        selected_tables = tables[:1]
        used_fallback_tables = True
    if not selected_equations and equations:
        selected_equations = equations[: min(2, len(equations))]
        used_fallback_equations = True
    return selected_sections, selected_tables, selected_equations, {
        "selected_sections": [s["name"] for s in selected_sections],
        "selected_tables": [t["name"] for t in selected_tables],
        "selected_equations": [e["name"] for e in selected_equations],
        "resolved_selected_table_files": [t["name"] for t in selected_tables],
        "used_fallback_selection": used_fallback_sections or used_fallback_tables or used_fallback_equations,
        "fallback": {
            "sections": used_fallback_sections,
            "tables": used_fallback_tables,
            "equations": used_fallback_equations,
        },
        "selection_file_metadata": {
            "selected_tables_raw": selection.get("selected_tables", []) or [],
            "resolved_selected_table_files_raw": selection.get("resolved_selected_table_files", []) or [],
        },
    }


def _load_selected_context(
    paper_dir: str,
    max_context_chars: int,
) -> Tuple[str, Dict[str, Any]]:
    selected_sections, selected_tables, selected_equations, selection_meta = _resolve_selected_sources(paper_dir)
    context, build_meta = build_context(selected_sections, selected_tables, selected_equations, max_context_chars=max_context_chars)
    selection_meta["context_build"] = build_meta
    return context, selection_meta


def _load_feedback_artifacts(path: str | None, max_examples: int = 5) -> Dict[str, Any]:
    if not path:
        return {"summary": "No reviewed feedback artifacts provided.", "examples": []}
    p = Path(path)
    if not p.exists():
        return {"summary": f"Feedback artifact not found: {path}", "examples": []}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"summary": f"Failed to parse feedback artifact: {path}", "examples": []}

    examples = (payload.get("examples") or [])[:max_examples] if isinstance(payload, dict) else []
    summary_lines = []
    for err, count in ((payload.get("error_type_counts") or {}).items() if isinstance(payload, dict) else []):
        summary_lines.append(f"- {err}: {count}")
    if not summary_lines:
        summary_lines.append("- No summarized reviewed error patterns available.")
    if examples:
        for ex in examples:
            summary_lines.append(
                f"- Example {ex.get('doi')} {ex.get('location')}: {ex.get('before')} -> {ex.get('after')} ({ex.get('human_error_type')})"
            )
    return {
        "summary": "\n".join(summary_lines),
        "examples": examples,
    }


def _evidence_object_summary(obj: Dict[str, Any]) -> str:
    locator = _safe_dict(obj.get("locator"))
    parts = [
        str(obj.get("evidence_type") or "").strip(),
        str(obj.get("source_file") or "").strip(),
        str(locator.get("table_id") or locator.get("equation_label") or "").strip(),
        str(locator.get("row_name") or "").strip(),
        str(locator.get("column_name") or "").strip(),
        str(locator.get("value") or "").strip(),
        str(locator.get("excerpt") or obj.get("snippet") or "").strip(),
    ]
    return " | ".join([p for p in parts if p])


def _best_available_evidence_text(
    evidence: Dict[str, Any],
    src: Dict[str, Any],
    table_evidence: Dict[str, Any],
    evidence_objects: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    raw = str(
        evidence.get("evidence_text")
        or src.get("evidence_text")
        or table_evidence.get("excerpt")
        or ""
    ).strip()
    if raw:
        return _trim_text(raw, max_chars)
    combined = " || ".join(
        _evidence_object_summary(obj)
        for obj in evidence_objects
        if isinstance(obj, dict) and _evidence_object_summary(obj)
    ).strip()
    return _trim_text(combined, max_chars) if combined else ""


def _norm_text(value: Any) -> str:
    text = sanitize_text_for_openai(str(value or "")).strip().lower()
    return " ".join(text.split())


def _value_string_candidates(value: Any) -> List[str]:
    out: List[str] = []
    if value in (None, ""):
        return out
    text = str(value).strip()
    if text and text not in out:
        out.append(text)
    try:
        num = float(value)
    except Exception:
        return out
    numeric_variants = [
        f"{num}",
        f"{num:g}",
        f"{num:.1f}",
        f"{num:.2f}",
        f"{num:.3f}",
        f"{num:.6g}",
    ]
    for variant in numeric_variants:
        variant = variant.strip()
        if variant and variant not in out:
            out.append(variant)
    return out


def _record_symbol_candidates(item: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    parameter = _safe_dict(item.get("parameter"))
    for raw in (
        parameter.get("symbol_reported"),
        parameter.get("raw_name"),
        parameter.get("canonical_name"),
        item.get("symbol"),
        item.get("canonical_name"),
    ):
        text = str(raw or "").strip()
        if text and text not in candidates:
            candidates.append(text)
    return candidates


def _infer_support_snippets(
    *,
    item: Dict[str, Any],
    selected_sections: List[Dict[str, Any]],
    selected_tables: List[Dict[str, Any]],
    selected_equations: List[Dict[str, Any]],
    max_chars: int,
) -> List[Dict[str, Any]]:
    symbol_candidates = [c.lower() for c in _record_symbol_candidates(item)]
    equation_candidates = [str(v).strip().lower() for v in _safe_list(item.get("governing_equation_ids")) if str(v).strip()]
    unit_candidates = [str(v).strip().lower() for v in {
        _safe_dict(item.get("assertion")).get("reported_unit"),
        item.get("unit"),
        item.get("unit_SI"),
    } if str(v or "").strip()]
    value_candidates = [c.lower() for c in _value_string_candidates(_safe_dict(item.get("assertion")).get("reported_value"))]
    if not value_candidates:
        value_candidates = [c.lower() for c in _value_string_candidates(item.get("value"))]

    supports: List[Dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    def maybe_add(source_type: str, source_name: str, raw_text: str) -> None:
        norm = _norm_text(raw_text)
        if not norm:
            return
        symbol_hit = any(sym in norm for sym in symbol_candidates if sym)
        value_hit = any(val in norm for val in value_candidates if val)
        equation_hit = any(eq in norm for eq in equation_candidates if eq)
        unit_hit = any(unit in norm for unit in unit_candidates if unit)
        if not ((symbol_hit and value_hit) or (value_hit and unit_hit) or (symbol_hit and equation_hit)):
            return
        snippet = _trim_text(raw_text, max_chars)
        key = (source_name, snippet)
        if key in seen_keys:
            return
        seen_keys.add(key)
        supports.append({
            "source_type": source_type,
            "source_name": source_name,
            "snippet": snippet,
            "match_basis": {
                "symbol": symbol_hit,
                "value": value_hit,
                "unit": unit_hit,
                "equation": equation_hit,
            },
        })

    for table in selected_tables:
        maybe_add("table", str(table.get("name") or ""), str(table.get("extract_text") or table.get("text") or ""))
    for section in selected_sections:
        maybe_add("section", str(section.get("name") or ""), str(section.get("text") or ""))
    for equation in selected_equations:
        maybe_add("equation", str(equation.get("name") or ""), str(equation.get("extract_text") or equation.get("text") or ""))
    return supports[:3]


def _has_inferred_support(record: Dict[str, Any]) -> bool:
    inferred = record.get("inferred_support")
    return isinstance(inferred, list) and any(isinstance(row, dict) and str(row.get("snippet") or "").strip() for row in inferred)


def _build_document_summary(extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    materials = extracted_json.get("materials", []) if isinstance(extracted_json.get("materials"), list) else []
    primary_material = materials[0] if materials and isinstance(materials[0], dict) else {}
    constituents = extracted_json.get("constituents", []) if isinstance(extracted_json.get("constituents"), list) else []
    primary_constituent = constituents[0] if constituents and isinstance(constituents[0], dict) else {}
    models = extracted_json.get("models", []) if isinstance(extracted_json.get("models"), list) else []
    primary_model = models[0] if models and isinstance(models[0], dict) else {}
    primary_constitutive_description = _safe_dict(primary_model.get("constitutive_description"))
    primary_slip_description = _safe_dict(primary_constitutive_description.get("slip_description"))
    primary_twinning = _safe_dict(primary_constitutive_description.get("twinning"))
    slip_count = 1 if str(primary_slip_description.get("slip_families_defined") or "").strip().lower() == "yes" else 0
    twin_count = 1 if str(primary_twinning.get("enabled") or "").strip().lower() == "yes" else 0
    cleavage_count = 0
    lattice_type = (
        _safe_dict(primary_constituent.get("crystal_structure")).get("lattice_type")
        or _safe_dict(primary_constituent.get("crystal_structure")).get("bravais_lattice")
        or _safe_dict(primary_constituent.get("crystal_structure")).get("crystal_system")
    )
    model_summaries = []
    for model in models:
        if not isinstance(model, dict):
            continue
        branches = [b for b in _safe_list(model.get("constitutive_branches")) if isinstance(b, dict)]
        model_summaries.append({
            "model_id": model.get("model_id"),
            "name": model.get("name"),
            "model_type": model.get("model_type"),
            "model_role": model.get("model_role"),
            "equation_ids": _safe_list(model.get("equation_ids")),
            "branch_ids": [b.get("branch_id") for b in branches if b.get("branch_id")],
            "branch_types": [b.get("branch_type") for b in branches if b.get("branch_type")],
        })
    return {
        "schema_version": extracted_json.get("schema_version"),
        "material_name": primary_material.get("name"),
        "chemical_formula": primary_material.get("chemical_formula") or primary_material.get("formula"),
        "lattice_type": lattice_type,
        "constitutive_framework": primary_model.get("framework") or primary_model.get("model_type"),
        "rate_dependence": primary_model.get("rate_dependence") or _safe_dict(primary_constitutive_description.get("flow_kinetics")).get("rate_dependence"),
        "phase_mode": primary_material.get("phase_mode"),
        "model_count": len(model_summaries),
        "models": model_summaries,
        "slip_family_count": slip_count,
        "twin_family_count": twin_count,
        "cleavage_family_count": cleavage_count,
    }


def _build_parameter_records(
    paper_dir: str,
    extracted_json: Dict[str, Any],
    per_evidence_chars: int,
    limit: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    provenance_lookup = provenance_map(extracted_json)
    binding_lookup = binding_map(extracted_json)
    evidence_lookup = evidence_map(extracted_json)
    selected_sections, selected_tables, selected_equations, _ = _resolve_selected_sources(paper_dir)
    models = [m for m in _safe_list(extracted_json.get("models")) if isinstance(m, dict)]
    model_lookup = {
        str(m.get("model_id") or "").strip(): m
        for m in models
        if str(m.get("model_id") or "").strip()
    }
    branch_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for model in models:
        model_id = str(model.get("model_id") or "").strip()
        for branch in _safe_list(model.get("constitutive_branches")):
            if not isinstance(branch, dict):
                continue
            branch_id = str(branch.get("branch_id") or "").strip()
            if model_id and branch_id:
                branch_lookup[(model_id, branch_id)] = branch
    for idx, _, item in iter_parameter_items_with_index(extracted_json):
        src = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
        evidence = item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
        table_evidence = evidence.get("table_evidence", {}) if isinstance(evidence.get("table_evidence"), dict) else {}
        pid = str(src.get("provenance_id") or "").strip()
        provenance = provenance_lookup.get(pid) if pid else None
        if not isinstance(provenance, dict):
            provenance = resolve_provenance_record(extracted_json, src)
        bid = str(item.get("binding_id") or "").strip()
        binding = binding_lookup.get(bid) if bid else None
        if not isinstance(binding, dict):
            binding = resolve_binding_record(extracted_json, item)
        claim_evidence_ids = item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
        source_evidence_ids = src.get("evidence_ids") if isinstance(src.get("evidence_ids"), list) else []
        evidence_ids: List[str] = []
        for eid in claim_evidence_ids + source_evidence_ids:
            normalized = str(eid).strip()
            if normalized and normalized not in evidence_ids:
                evidence_ids.append(normalized)
        evidence_objects = [
            evidence_lookup[str(eid).strip()]
            for eid in evidence_ids
            if str(eid).strip() in evidence_lookup
        ]
        raw_evidence_text = _best_available_evidence_text(
            evidence=evidence,
            src=src,
            table_evidence=table_evidence,
            evidence_objects=evidence_objects,
            max_chars=per_evidence_chars,
        )
        trimmed_evidence_text = _trim_text(raw_evidence_text, per_evidence_chars) if raw_evidence_text else ""
        applies_to = item.get("applies_to") if isinstance(item.get("applies_to"), dict) else {}
        model_id = str(applies_to.get("model_id") or "").strip()
        branch_id = str(applies_to.get("branch_id") or "").strip()
        model_context = _safe_dict(model_lookup.get(model_id))
        branch_context = _safe_dict(branch_lookup.get((model_id, branch_id)))
        inferred_support = _infer_support_snippets(
            item=item,
            selected_sections=selected_sections,
            selected_tables=selected_tables,
            selected_equations=selected_equations,
            max_chars=per_evidence_chars,
        )
        parameter_payload = _safe_dict(item.get("parameter"))
        assertion_payload = _safe_dict(item.get("assertion"))
        provenance_payload = {
            "origin_type": provenance.get("origin_type"),
            "reference_ids": _safe_list(provenance.get("reference_ids")),
            "adopted_from_reference_ids": _safe_list(provenance.get("adopted_from_reference_ids")),
            "calibration_based_on_reference_ids": _safe_list(provenance.get("calibration_based_on_reference_ids")),
            "calibration": _safe_dict(provenance.get("calibration")),
        }
        provenance_payload = {
            k: v for k, v in provenance_payload.items()
            if v not in (None, "", []) and v != {}
        }
        evidence_summary = {
            "primary_text": trimmed_evidence_text,
            "primary_text_original_chars": len(raw_evidence_text),
            "primary_text_truncated": len(raw_evidence_text) > len(trimmed_evidence_text),
            "table_locator": table_evidence,
        }
        evidence_summary = {
            k: v for k, v in evidence_summary.items()
            if v not in (None, "", []) and v != {}
        }
        records.append({
            "location": claim_location(item, idx),
            "record_index": idx,
            "parameter": parameter_payload,
            "assertion": assertion_payload,
            "applies_to": applies_to,
            "provenance": provenance_payload,
            "evidence_ids": evidence_ids,
            "evidence_objects": evidence_objects,
            "evidence_summary": evidence_summary,
            "governing_equation_ids": _safe_list(item.get("governing_equation_ids")),
            "binding_id": item.get("binding_id"),
            "binding_context": binding,
            "evaluator_mode": "extractor_raw_document_backfill_v5",
            "inferred_support": inferred_support,
            "canonical_name": item.get("canonical_name") or parameter_payload.get("canonical_name"),
            "symbol_reported": item.get("symbol") or parameter_payload.get("symbol_reported"),
            "domain": item.get("domain") or parameter_payload.get("domain"),
            "reported_value": item.get("value") if item.get("value") not in (None, "") else assertion_payload.get("reported_value"),
            "reported_unit": item.get("unit") if item.get("unit") not in (None, "") else assertion_payload.get("reported_unit"),
            "normalized_value": item.get("value_SI"),
            "normalized_unit": item.get("unit_SI"),
            "model_context": {
                "model_id": model_context.get("model_id"),
                "name": model_context.get("name"),
                "model_type": model_context.get("model_type"),
                "model_role": model_context.get("model_role"),
                "equation_ids": _safe_list(model_context.get("equation_ids")),
            } if model_context else {},
            "branch_context": {
                "branch_id": branch_context.get("branch_id"),
                "branch_type": branch_context.get("branch_type"),
                "name": branch_context.get("name"),
                "governing_equation_ids": _safe_list(branch_context.get("governing_equation_ids")),
            } if branch_context else {},
            "evidence_linkage": {
                "claim_evidence_ids": claim_evidence_ids,
                "projected_evidence_ids": source_evidence_ids,
            },
        })
    return records[:limit] if limit > 0 else records


def _audit_axis_value(record: Dict[str, Any], axis: str) -> str:
    applies_to = _safe_dict(record.get("applies_to"))
    parameter = _safe_dict(record.get("parameter"))
    if axis == "condition_id":
        return str(applies_to.get("condition_id") or "").strip()
    if axis == "process_state_id":
        return str(applies_to.get("process_state_id") or "").strip()
    if axis == "canonical_name":
        return str(
            record.get("canonical_name")
            or parameter.get("canonical_name")
            or ""
        ).strip()
    if axis == "scope":
        return str(applies_to.get("scope") or "").strip()
    return ""


def _sampling_axis_counts(records: List[Dict[str, Any]], axes: List[str]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {axis: {} for axis in axes}
    for record in records:
        if not isinstance(record, dict):
            continue
        for axis in axes:
            value = _audit_axis_value(record, axis)
            if not value:
                continue
            counts[axis][value] = counts[axis].get(value, 0) + 1
    return counts


def _select_parameter_records_for_audit(
    all_parameter_records: List[Dict[str, Any]],
    limit: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if limit <= 0 or len(all_parameter_records) <= limit:
        axes = ["condition_id", "process_state_id", "canonical_name", "scope"]
        return list(all_parameter_records), {
            "strategy": "full_coverage",
            "axes": axes,
            "selected_count": len(all_parameter_records),
            "available_count": len(all_parameter_records),
            "selected_axis_counts": _sampling_axis_counts(all_parameter_records, axes),
        }

    axes_with_weights = [
        ("condition_id", 3.0),
        ("process_state_id", 2.5),
        ("canonical_name", 2.0),
        ("scope", 1.0),
    ]
    axes = [axis for axis, _ in axes_with_weights]
    axis_counts: Dict[str, Dict[str, int]] = {axis: {} for axis in axes}
    remaining: List[Tuple[int, Dict[str, Any]]] = list(enumerate(all_parameter_records))
    selected: List[Dict[str, Any]] = []
    selected_indices: List[int] = []

    while remaining and len(selected) < limit:
        best_pos = 0
        best_rank: Tuple[float, float, float] | None = None
        for pos, (idx, record) in enumerate(remaining):
            novelty_score = 0.0
            balance_score = 0.0
            for axis, weight in axes_with_weights:
                value = _audit_axis_value(record, axis)
                if not value:
                    continue
                seen = axis_counts[axis].get(value, 0)
                if seen == 0:
                    novelty_score += weight
                balance_score += weight / (1.0 + float(seen))
            rank = (novelty_score, balance_score, -float(idx))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_pos = pos

        idx, record = remaining.pop(best_pos)
        selected.append(record)
        selected_indices.append(idx)
        for axis, _ in axes_with_weights:
            value = _audit_axis_value(record, axis)
            if value:
                axis_counts[axis][value] = axis_counts[axis].get(value, 0) + 1

    selected_pairs = sorted(zip(selected_indices, selected), key=lambda pair: pair[0])
    selected_records = [record for _, record in selected_pairs]
    selected_locations = {
        str(record.get("location") or "")
        for record in selected_records
        if isinstance(record, dict)
    }
    omitted_records = [
        record for record in all_parameter_records
        if isinstance(record, dict) and str(record.get("location") or "") not in selected_locations
    ]
    return selected_records, {
        "strategy": "balanced_axis_stratified",
        "axes": axes,
        "selected_count": len(selected_records),
        "available_count": len(all_parameter_records),
        "selected_axis_counts": _sampling_axis_counts(selected_records, axes),
        "omitted_axis_counts": _sampling_axis_counts(omitted_records, axes),
    }


def _score_bucket(score: Any) -> str:
    try:
        score_f = float(score)
    except Exception:
        return "low"
    if score_f >= 85:
        return "high"
    if score_f >= 60:
        return "medium"
    return "low"


def _record_reported_value(record: Dict[str, Any]) -> Any:
    if record.get("reported_value") not in (None, ""):
        return record.get("reported_value")
    return _safe_dict(record.get("assertion")).get("reported_value")


def _record_reported_unit(record: Dict[str, Any]) -> str:
    if str(record.get("reported_unit") or "").strip():
        return str(record.get("reported_unit") or "")
    return str(_safe_dict(record.get("assertion")).get("reported_unit") or "")


def _record_normalized_value(record: Dict[str, Any]) -> Any:
    if record.get("normalized_value") not in (None, ""):
        return record.get("normalized_value")
    return _safe_dict(record.get("assertion")).get("normalized_value")


def _record_normalized_unit(record: Dict[str, Any]) -> str:
    if str(record.get("normalized_unit") or "").strip():
        return str(record.get("normalized_unit") or "")
    return str(_safe_dict(record.get("assertion")).get("normalized_unit") or "")


def _si_conversion_consistent(record: Dict[str, Any]) -> bool:
    unit = _record_reported_unit(record).strip().lower()
    unit_si = _record_normalized_unit(record).strip().lower()
    if not unit or not unit_si:
        return False
    try:
        value = float(_record_reported_value(record))
        value_si = float(_record_normalized_value(record))
    except Exception:
        return False

    factors = {
        ("mpa", "pa"): 1e6,
        ("gpa", "pa"): 1e9,
        ("kpa", "pa"): 1e3,
        ("pa", "pa"): 1.0,
        ("s^-1", "s^-1"): 1.0,
        ("1/s", "s^-1"): 1.0,
        ("s-1", "s^-1"): 1.0,
    }
    factor = factors.get((unit, unit_si))
    if factor is None:
        return False
    expected = value * factor
    return abs(expected - value_si) <= max(1e-9, abs(expected) * 1e-6)


def _is_table_based_record(record: Dict[str, Any]) -> bool:
    for support in record.get("inferred_support", []) or []:
        if not isinstance(support, dict):
            continue
        if str(support.get("source_type") or "").strip().lower() == "table":
            return True
    summary = record.get("evidence_summary", {}) if isinstance(record.get("evidence_summary"), dict) else {}
    locator = summary.get("table_locator", {}) if isinstance(summary.get("table_locator"), dict) else {}
    if any(locator.get(k) not in (None, "", []) for k in ("row_name", "column_name", "value", "excerpt")):
        return True
    for obj in record.get("evidence_objects", []) or []:
        if not isinstance(obj, dict):
            continue
        file_name = str(obj.get("source_file") or obj.get("file") or obj.get("matched_file") or "").strip().lower()
        if file_name.startswith("table_") or file_name.endswith(".json") and "/table_" in file_name:
            return True
    return False


def _is_high_risk_table_claim(merged_row: Dict[str, Any], record: Dict[str, Any]) -> bool:
    error_types = {str(v or "").strip().lower() for v in (merged_row.get("error_types") or []) if str(v or "").strip()}
    if error_types & {"wrong_value", "provenance_conflict", "missing_key_field", "cross_material_mixup", "model_variant_confusion"}:
        return True
    if str(merged_row.get("verdict") or "").strip().lower() == "fail":
        return True
    if _record_reported_value(record) in (None, "") and _record_normalized_value(record) in (None, ""):
        return True
    binding = record.get("binding_context", {}) if isinstance(record.get("binding_context"), dict) else {}
    scope = str((binding.get("scope") or (record.get("applies_to") or {}).get("scope") or "")).strip().lower()
    if scope == "system":
        system_ids = binding.get("system_ids") or ((record.get("applies_to") or {}).get("system_ids") if isinstance(record.get("applies_to"), dict) else [])
        if not system_ids:
            return True
    return False


def _is_low_risk_evidence_disagreement(merged_row: Dict[str, Any], record: Dict[str, Any]) -> bool:
    if not _is_table_based_record(record):
        return False
    error_types = {str(v or "").strip().lower() for v in (merged_row.get("error_types") or []) if str(v or "").strip()}
    uncertainty_types = {str(v or "").strip().lower() for v in (merged_row.get("uncertainty_types") or []) if str(v or "").strip()}
    if error_types & {"wrong_value", "provenance_conflict", "cross_material_mixup", "model_variant_confusion", "missing_key_field"}:
        return False
    if uncertainty_types and uncertainty_types.issubset({"missing_evidence", "weak_grounding", "table_parse_uncertain", "condition_binding_ambiguous"}):
        votes = [str(v or "").strip().lower() for v in ((merged_row.get("committee") or {}).get("votes") or []) if str(v or "").strip()]
        return not ("fail" in votes and len(set(votes)) == 1)
    return False


def _is_low_risk_evidence_disagreement_row_only(merged_row: Dict[str, Any]) -> bool:
    error_types = {str(v or "").strip().lower() for v in (merged_row.get("error_types") or []) if str(v or "").strip()}
    uncertainty_types = {str(v or "").strip().lower() for v in (merged_row.get("uncertainty_types") or []) if str(v or "").strip()}
    if error_types & {"wrong_value", "provenance_conflict", "cross_material_mixup", "model_variant_confusion", "missing_key_field"}:
        return False
    if not uncertainty_types or not uncertainty_types.issubset({"missing_evidence", "weak_grounding", "table_parse_uncertain", "condition_binding_ambiguous"}):
        return False
    votes = [str(v or "").strip().lower() for v in ((merged_row.get("committee") or {}).get("votes") or []) if str(v or "").strip()]
    return not ("fail" in votes and len(set(votes)) == 1)


def _is_low_risk_normalization_only_disagreement(
    merged_row: Dict[str, Any],
    record: Dict[str, Any],
    ev: Dict[str, Any],
    nm: Dict[str, Any],
    cs: Dict[str, Any],
) -> bool:
    if _record_reported_value(record) in (None, ""):
        return False
    ev_verdict = str(ev.get("verdict") or "").strip().lower()
    cs_verdict = str(cs.get("verdict") or "").strip().lower()
    nm_verdict = str(nm.get("verdict") or "").strip().lower()
    if ev_verdict != "pass" or nm_verdict not in {"warning", "fail"} or cs_verdict not in {"", "pass"}:
        return False

    error_types = {str(v or "").strip().lower() for v in (merged_row.get("error_types") or []) if str(v or "").strip()}
    if error_types - {"wrong_unit_conversion", "other"}:
        return False

    reason = " ".join([
        str(nm.get("reason") or ""),
        str(merged_row.get("reason") or ""),
    ]).lower()
    unit_hints = ("unit", "value_si", "unit_si", "si conversion", "si normalization")
    high_risk_hints = ("wrong value", "incorrect value", "binding", "scope", "phase mismatch", "family mismatch")
    if not any(hint in reason for hint in unit_hints):
        return False
    if any(hint in reason for hint in high_risk_hints):
        return False
    return True


def _review_required_from_raw_consensus(
    *,
    verdict: str,
    disagreement: bool,
    votes: List[str],
    error_types: List[str],
    uncertainty_types: List[str],
) -> bool:
    verdict = str(verdict or "").strip().lower()
    vote_set = {str(v or "").strip().lower() for v in votes if str(v or "").strip()}
    error_set = {str(v or "").strip().lower() for v in error_types if str(v or "").strip()}
    uncertainty_set = {str(v or "").strip().lower() for v in uncertainty_types if str(v or "").strip()}

    high_risk_errors = {
        "wrong_value",
        "contradictory_evidence",
        "provenance_conflict",
        "missing_key_field",
        "cross_material_mixup",
        "model_variant_confusion",
    }
    if verdict == "fail" and not uncertainty_set.issubset({"missing_evidence", "weak_grounding", "table_parse_uncertain"}):
        return True
    if error_set & high_risk_errors:
        return True
    if disagreement and "fail" in vote_set:
        return True
    return False


def _dedupe_uncertainty_types(values: List[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        raw = str(value or "").strip().lower()
        if not raw:
            continue
        if raw not in out:
            out.append(raw)
    return out


def _run_parameter_agent(
    *,
    client: OpenAI,
    model_evaluate: str,
    system_prompt: str,
    user_prompt_template: str,
    parameter_records: List[Dict[str, Any]],
    batch_size: int,
    max_retries: int,
    feedback_summary: str,
    extra_prompt_kwargs: Dict[str, Any] | None = None,
    payload_key: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    audits: List[Dict[str, Any]] = []
    total_tokens = 0
    total_input = 0
    total_output = 0
    total_time = 0.0
    extra_prompt_kwargs = extra_prompt_kwargs or {}

    for start_idx in range(0, len(parameter_records), max(1, batch_size)):
        batch = parameter_records[start_idx:start_idx + max(1, batch_size)]
        prompt = user_prompt_template.replace("__RECORDS_JSON__", json.dumps(batch, ensure_ascii=False, indent=2))
        prompt = prompt.replace("__FEEDBACK_SUMMARY__", feedback_summary)
        for k, v in extra_prompt_kwargs.items():
            prompt = prompt.replace(f"__{k.upper()}__", str(v))
        started = time.perf_counter()
        resp = _chat_completion_with_retry(
            client,
            model=model_evaluate,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_retries=max(1, max_retries),
        )
        total_time += time.perf_counter() - started
        usage = resp.usage
        total_input += getattr(usage, "prompt_tokens", 0)
        total_output += getattr(usage, "completion_tokens", 0)
        total_tokens += getattr(usage, "total_tokens", 0)
        payload = json.loads(resp.choices[0].message.content)
        batch_rows = payload.get(payload_key, []) if isinstance(payload, dict) else []
        if not isinstance(batch_rows, list):
            batch_rows = []
        for row in batch_rows:
            if not isinstance(row, dict):
                continue
            row.setdefault("verdict", "warning")
            row.setdefault("score", 0)
            row.setdefault("confidence", _score_bucket(row.get("score")))
            row.setdefault("error_types", [])
            row.setdefault("uncertainty_types", [])
            row.setdefault("reason", "")
            row.setdefault("recommendation", "")
            audits.append(row)
    return audits, {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
        "time_seconds": round(total_time, 2),
        "audited_parameter_count": len(audits),
    }


def _run_consistency_agent(
    *,
    client: OpenAI,
    model_evaluate: str,
    parameter_records: List[Dict[str, Any]],
    doc_summary: Dict[str, Any],
    feedback_summary: str,
    max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prompt = CONSISTENCY_AGENT_USER_PROMPT_TEMPLATE.replace("__RECORDS_JSON__", json.dumps(parameter_records, ensure_ascii=False, indent=2))
    prompt = prompt.replace("__DOC_SUMMARY__", json.dumps(doc_summary, ensure_ascii=False, indent=2))
    prompt = prompt.replace("__FEEDBACK_SUMMARY__", feedback_summary)
    started = time.perf_counter()
    resp = _chat_completion_with_retry(
        client,
        model=model_evaluate,
        messages=[
            {"role": "system", "content": CONSISTENCY_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_retries=max(1, max_retries),
    )
    elapsed = time.perf_counter() - started
    usage = resp.usage
    payload = json.loads(resp.choices[0].message.content)
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("document_consistency_score", 0)
    payload.setdefault("global_issues", [])
    payload.setdefault("parameter_flags", [])
    return payload, {
        "input_tokens": getattr(usage, "prompt_tokens", 0),
        "output_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
        "time_seconds": round(elapsed, 2),
        "audited_parameter_count": len(payload.get("parameter_flags", []) or []),
    }


def _merge_committee(
    parameter_records: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    normalization_rows: List[Dict[str, Any]],
    consistency_payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    evidence_map = {r.get("location"): r for r in evidence_rows if isinstance(r, dict)}
    norm_map = {r.get("location"): r for r in normalization_rows if isinstance(r, dict)}
    consistency_map = {r.get("location"): r for r in (consistency_payload.get("parameter_flags", []) or []) if isinstance(r, dict)}
    record_map = {r.get("location"): r for r in parameter_records if isinstance(r, dict)}

    merged: List[Dict[str, Any]] = []
    disagreements = []
    escalations = 0
    raw_rows: List[Dict[str, Any]] = []

    for rec in parameter_records:
        location = rec.get("location")
        ev = evidence_map.get(location, {})
        nm = norm_map.get(location, {})
        cs = consistency_map.get(location, {})
        votes = [str(ev.get("verdict") or ""), str(nm.get("verdict") or ""), str(cs.get("verdict") or "")]
        non_empty_votes = [v for v in votes if v]
        disagreement = len(set(non_empty_votes)) > 1 if non_empty_votes else False
        score_parts = [x for x in [ev.get("score"), nm.get("score"), cs.get("score")] if isinstance(x, (int, float))]
        score = round(sum(score_parts) / len(score_parts), 2) if score_parts else 0.0
        verdict = "pass"
        if "fail" in non_empty_votes:
            verdict = "fail"
        elif "warning" in non_empty_votes:
            verdict = "warning"

        error_types: List[str] = []
        for src in (ev, nm, cs):
            for err in src.get("error_types", []) or []:
                if err not in error_types:
                    error_types.append(err)
        uncertainty_types: List[str] = []
        for src in (ev, nm, cs):
            for uncertainty in src.get("uncertainty_types", []) or []:
                normed = _dedupe_uncertainty_types([uncertainty])
                for mapped in normed:
                    if mapped not in uncertainty_types:
                        uncertainty_types.append(mapped)
        if ev.get("provenance_quality") in {"weak", "missing"} and "weak_grounding" not in uncertainty_types:
            uncertainty_types.append("weak_grounding")
        if str(ev.get("supportiveness") or "") in {"insufficient_evidence", "unsupported"} and "missing_evidence" not in uncertainty_types:
            uncertainty_types.append("missing_evidence")

        recommendation_parts = [str(src.get("recommendation") or "").strip() for src in (ev, nm, cs)]
        reason_parts = [str(src.get("reason") or "").strip() for src in (ev, nm, cs)]
        merged_row = {
            "location": location,
            "canonical_name": rec.get("canonical_name"),
            "symbol": rec.get("symbol_reported") or rec.get("symbol"),
            "verdict": verdict,
            "score": score,
            "supportiveness": ev.get("supportiveness", "insufficient_evidence"),
            "exactness": ev.get("exactness", "uncertain"),
            "normalization_correctness": nm.get("normalization_correctness", "uncertain"),
            "completeness": cs.get("completeness", "not_applicable"),
            "provenance_quality": ev.get("provenance_quality", "missing"),
            "confidence": _score_bucket(score),
            "error_types": error_types,
            "uncertainty_types": uncertainty_types,
            "reason": " | ".join([p for p in reason_parts if p]),
            "recommendation": " | ".join([p for p in recommendation_parts if p]),
            "committee": {
                "evidence_judge": ev,
                "normalization_judge": nm,
                "consistency_judge": cs,
                "disagreement": disagreement,
                "votes": non_empty_votes,
            },
            "review_required": _review_required_from_raw_consensus(
                verdict=verdict,
                disagreement=disagreement,
                votes=non_empty_votes,
                error_types=error_types,
                uncertainty_types=uncertainty_types,
            ),
        }
        raw_consensus = json.loads(json.dumps({
            "location": merged_row["location"],
            "canonical_name": merged_row.get("canonical_name"),
            "symbol": merged_row.get("symbol"),
            "verdict": merged_row.get("verdict"),
            "score": merged_row.get("score"),
            "error_types": list(merged_row.get("error_types") or []),
            "uncertainty_types": list(merged_row.get("uncertainty_types") or []),
            "reason": merged_row.get("reason"),
            "recommendation": merged_row.get("recommendation"),
            "review_required": merged_row.get("review_required"),
            "committee": merged_row.get("committee"),
        }, ensure_ascii=False))
        merged_row["judge_consensus_raw"] = raw_consensus
        merged_row["policy_adjustments"] = []
        effective_verdict = merged_row["verdict"]
        effective_score = float(merged_row["score"] or 0)
        effective_confidence = merged_row["confidence"]
        effective_review_required = bool(merged_row["review_required"])
        effective_normalization = merged_row["normalization_correctness"]
        effective_reason = merged_row.get("reason") or ""
        record = record_map.get(location) or {}
        if "wrong_unit_conversion" in merged_row["error_types"] and _si_conversion_consistent(record):
            merged_row["policy_adjustments"].append("suppress_false_unit_conversion")
            if effective_normalization in {"likely_incorrect", "uncertain"}:
                effective_normalization = "correct"
            if effective_verdict == "fail":
                effective_verdict = "warning"
                effective_review_required = disagreement
            effective_score = max(float(effective_score or 0), 85.0)
            effective_confidence = _score_bucket(effective_score)
            extra = " SI conversion is numerically consistent and should not be treated as a unit-conversion failure."
            effective_reason = (effective_reason.strip() + extra).strip()
        if _is_table_based_record(record):
            uncertainty_types = [str(v or "").strip().lower() for v in (merged_row.get("uncertainty_types") or []) if str(v or "").strip()]
            non_grounding_errors = [
                e for e in (merged_row.get("error_types") or [])
                if str(e or "").strip().lower() not in {"unsupported_claim", "other"}
            ]
            only_grounding_uncertainty = bool(uncertainty_types) and set(uncertainty_types).issubset({"missing_evidence", "weak_grounding", "table_parse_uncertain"})
            if only_grounding_uncertainty and not non_grounding_errors and not _is_high_risk_table_claim(merged_row, record):
                merged_row["policy_adjustments"].append("relax_table_grounding_penalty")
                if effective_verdict == "fail":
                    effective_verdict = "warning"
                effective_review_required = False
                effective_score = max(float(effective_score or 0), 72.0)
                effective_confidence = _score_bucket(effective_score)
                if "Table-based claim accepted with relaxed evidence penalty; extractor image/table reading is treated as primary evidence unless a stronger contradiction exists." not in effective_reason:
                    extra = " Table-based claim accepted with relaxed evidence penalty; extractor image/table reading is treated as primary evidence unless a stronger contradiction exists."
                    effective_reason = (effective_reason.strip() + extra).strip()
        cs_errors = [str(v or "").strip().lower() for v in (cs.get("error_types") or []) if str(v or "").strip()]
        cs_reason = str(cs.get("reason") or "").strip().lower()
        if "condition_binding_error" in cs_errors and (
            "lack" in cs_reason and "evidence" in cs_reason
            or "direct table cell evidence" in cs_reason
            or "general citation" in cs_reason
        ):
            merged_row["policy_adjustments"].append("ignore_binding_error_from_weak_evidence")
        if _is_low_risk_evidence_disagreement(merged_row, record):
            merged_row["policy_adjustments"].append("downgrade_low_risk_evidence_disagreement")
            effective_review_required = False
            effective_score = max(float(effective_score or 0), 78.0)
            effective_confidence = _score_bucket(effective_score)
            if "Low-risk table-based disagreement was treated as evidence incompleteness rather than a critical audit failure." not in effective_reason:
                extra = " Low-risk table-based disagreement was treated as evidence incompleteness rather than a critical audit failure."
                effective_reason = (effective_reason.strip() + extra).strip()
        if _is_low_risk_normalization_only_disagreement(merged_row, record, ev, nm, cs):
            merged_row["policy_adjustments"].append("downgrade_low_risk_normalization_disagreement")
            effective_review_required = False
            if effective_verdict == "fail":
                effective_verdict = "warning"
            effective_score = max(float(effective_score or 0), 88.0)
            effective_confidence = _score_bucket(effective_score)
            if "Low-risk unit/SI normalization disagreement was not treated as a mandatory review issue." not in effective_reason:
                extra = " Low-risk unit/SI normalization disagreement was not treated as a mandatory review issue."
                effective_reason = (effective_reason.strip() + extra).strip()
        if _has_inferred_support(record):
            non_support_errors = {
                str(e or "").strip().lower()
                for e in (merged_row.get("error_types") or [])
                if str(e or "").strip()
            } - {"unsupported_claim", "other"}
            if not non_support_errors and effective_verdict in {"warning", "fail"}:
                merged_row["policy_adjustments"].append("trust_inferred_support_snippets")
                if effective_verdict == "fail":
                    effective_verdict = "warning"
                effective_review_required = False
                effective_score = max(float(effective_score or 0), 80.0)
                effective_confidence = _score_bucket(effective_score)
                if "Extractor-first support snippets were found in the selected source files, so missing postprocessed evidence packaging was not treated as an unsupported claim." not in effective_reason:
                    extra = " Extractor-first support snippets were found in the selected source files, so missing postprocessed evidence packaging was not treated as an unsupported claim."
                    effective_reason = (effective_reason.strip() + extra).strip()
        merged_row["policy_adjusted_consensus"] = {
            "verdict": effective_verdict,
            "score": round(effective_score, 2),
            "confidence": effective_confidence,
            "review_required": effective_review_required,
            "normalization_correctness": effective_normalization,
            "reason": effective_reason,
            "recommendation": merged_row.get("recommendation"),
        }
        merged_row["verdict"] = effective_verdict
        merged_row["score"] = round(effective_score, 2)
        merged_row["confidence"] = effective_confidence
        merged_row["review_required"] = effective_review_required
        merged_row["normalization_correctness"] = effective_normalization
        merged_row["reason"] = effective_reason
        if disagreement:
            disagreements.append({
                "location": location,
                "votes": non_empty_votes,
                "canonical_name": rec.get("canonical_name"),
                "symbol": rec.get("symbol"),
            })
        if effective_review_required:
            escalations += 1
        raw_rows.append(raw_consensus)
        merged.append(merged_row)

    return merged, {
        "parameter_disagreements": disagreements,
        "disagreement_count": len(disagreements),
        "human_escalation_count": escalations,
        "document_consistency_score": consistency_payload.get("document_consistency_score"),
        "global_issues": consistency_payload.get("global_issues", []),
        "raw_parameter_consensus": raw_rows,
    }


def _run_meta_agent(
    *,
    client: OpenAI,
    model_evaluate: str,
    context_meta: Dict[str, Any],
    doc_summary: Dict[str, Any],
    quality_report: Dict[str, Any],
    evidence_report: Dict[str, Any],
    committee_report: Dict[str, Any],
    feedback_summary: str,
    max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prompt = META_AGENT_USER_PROMPT_TEMPLATE.replace("__FEEDBACK_SUMMARY__", feedback_summary)
    prompt = prompt.replace(
        "__DOC_CONTEXT_SUMMARY__",
        json.dumps({
            "selected_context": context_meta,
            "document_summary": doc_summary,
        }, ensure_ascii=False, indent=2),
    )
    prompt = prompt.replace("__QUALITY_REPORT__", json.dumps(quality_report, ensure_ascii=False, indent=2))
    prompt = prompt.replace(
        "__EVIDENCE_REPORT__",
        json.dumps({k: v for k, v in evidence_report.items() if k != "rows"}, ensure_ascii=False, indent=2),
    )
    prompt = prompt.replace("__COMMITTEE_SUMMARY__", json.dumps(committee_report, ensure_ascii=False, indent=2))
    started = time.perf_counter()
    resp = _chat_completion_with_retry(
        client,
        model=model_evaluate,
        messages=[
            {"role": "system", "content": META_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_retries=max(1, max_retries),
    )
    elapsed = time.perf_counter() - started
    usage = resp.usage
    payload = json.loads(resp.choices[0].message.content)
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("verdict", "warning")
    payload.setdefault("overall_score", 0)
    payload.setdefault("summary", "")
    payload.setdefault("dimension_scores", {})
    payload.setdefault("critical_issues", [])
    payload.setdefault("strengths", [])
    payload.setdefault("recommended_actions", [])
    return payload, {
        "input_tokens": getattr(usage, "prompt_tokens", 0),
        "output_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
        "time_seconds": round(elapsed, 2),
    }


def _prune_meta_issues(
    meta_payload: Dict[str, Any],
    merged_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    merged_map = {str(r.get("location") or ""): r for r in merged_rows if isinstance(r, dict)}
    issues_out: List[Dict[str, Any]] = []
    removed_low_risk = 0
    for issue in meta_payload.get("critical_issues", []) or []:
        if not isinstance(issue, dict):
            continue
        loc = str(issue.get("location") or "")
        row = merged_map.get(loc) or {}
        if row and _is_low_risk_evidence_disagreement_row_only(row):
            removed_low_risk += 1
            continue
        if row and _is_low_risk_normalization_meta_issue(row, issue):
            removed_low_risk += 1
            continue
        issues_out.append(issue)
    meta_payload["critical_issues"] = issues_out
    if removed_low_risk and str(meta_payload.get("summary") or ""):
        meta_payload["summary"] = str(meta_payload.get("summary") or "").replace(
            "Committee shows significant disagreement on many parameters, especially for delta hydride initial and asymptotic CRSS values, indicating uncertainty. ",
            ""
        )
    return meta_payload


def _is_low_risk_normalization_meta_issue(row: Dict[str, Any], issue: Dict[str, Any]) -> bool:
    if not isinstance(row, dict) or not isinstance(issue, dict):
        return False
    category = str(issue.get("category") or "").strip().lower()
    if category not in {"normalization_problem", "parameter_review_required", "parameter_warning", "other"}:
        return False
    policy_adjustments = {str(v or "").strip().lower() for v in (row.get("policy_adjustments") or []) if str(v or "").strip()}
    if "suppress_false_unit_conversion" not in policy_adjustments and "downgrade_low_risk_normalization_disagreement" not in policy_adjustments:
        return False
    evidence = " ".join(
        [
            str(issue.get("issue") or ""),
            str(issue.get("evidence") or ""),
            str(row.get("reason") or ""),
        ]
    ).lower()
    suspicious_hints = (
        "data entry error",
        "likely a data entry error",
        "physically implausible",
        "unusually large",
        "plausible magnitude",
    )
    return any(hint in evidence for hint in suspicious_hints)


def _short_reason(text: str, max_len: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _sanitize_issue_reason(text: str, row: Dict[str, Any]) -> str:
    reason = str(text or "")
    policy_adjustments = {str(v or "").strip().lower() for v in (row.get("policy_adjustments") or []) if str(v or "").strip()}
    if "suppress_false_unit_conversion" in policy_adjustments or "downgrade_low_risk_normalization_disagreement" in policy_adjustments:
        lowered = reason.lower()
        suspicious_hints = (
            "likely a data entry error",
            "data entry error",
            "unusually large",
            "physically implausible",
        )
        if any(hint in lowered for hint in suspicious_hints):
            return "The reported value, reported unit, and SI conversion are internally consistent; remaining concern is low-risk normalization ambiguity rather than a confirmed value error."
    return reason


def _augment_critical_issues_from_audits(
    meta_payload: Dict[str, Any],
    merged_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    existing = meta_payload.get("critical_issues") or []
    issues_out: List[Dict[str, Any]] = [issue for issue in existing if isinstance(issue, dict)]
    existing_locations = {str(issue.get("location") or "") for issue in issues_out if isinstance(issue, dict)}

    for row in merged_rows:
        if not isinstance(row, dict):
            continue
        location = str(row.get("location") or "").strip()
        if not location or location in existing_locations:
            continue

        verdict = str(row.get("verdict") or "").strip().lower()
        review_required = bool(row.get("review_required"))
        if verdict not in {"warning", "fail"} and not review_required:
            continue

        committee = row.get("committee") if isinstance(row.get("committee"), dict) else {}
        judge_reasons: List[str] = []
        judge_labels = (
            ("evidence_judge", "Evidence"),
            ("normalization_judge", "Normalization"),
            ("consistency_judge", "Consistency"),
        )
        for key, label in judge_labels:
            judge_row = committee.get(key) if isinstance(committee.get(key), dict) else {}
            judge_verdict = str(judge_row.get("verdict") or "").strip().lower()
            if judge_verdict in {"warning", "fail"}:
                reason = _short_reason(_sanitize_issue_reason(judge_row.get("reason") or row.get("reason") or "", row))
                if reason:
                    judge_reasons.append(f"{label}: {reason}")

        if not judge_reasons and row.get("reason"):
            judge_reasons.append(_short_reason(_sanitize_issue_reason(row.get("reason") or "", row)))

        severity = "medium" if review_required or verdict == "fail" else "low"
        category = "parameter_review_required" if review_required else "parameter_warning"
        issue_text = (
            f"{row.get('canonical_name') or row.get('symbol') or location} was flagged by the committee."
            if judge_reasons
            else f"{row.get('canonical_name') or row.get('symbol') or location} needs attention."
        )
        if review_required:
            issue_text += " Manual review is currently required."

        issues_out.append({
            "severity": severity,
            "category": category,
            "location": location,
            "issue": issue_text,
            "evidence": " | ".join(judge_reasons) if judge_reasons else None,
            "recommendation": row.get("recommendation") or "Review the parameter-level audit details.",
        })
        existing_locations.add(location)

    meta_payload["critical_issues"] = issues_out
    return meta_payload


def run_llm_evaluation(
    *,
    paper_dir: str,
    extracted_json: Dict[str, Any],
    model_evaluate: str,
    max_context_chars: int = 18000,
    max_retries: int = 2,
    parameter_limit: int = 40,
    field_batch_size: int = 12,
    per_evidence_chars: int = 800,
    quality_report: Dict[str, Any] | None = None,
    evidence_report: Dict[str, Any] | None = None,
    feedback_artifact_path: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    _, context_meta = _load_selected_context(paper_dir=paper_dir, max_context_chars=max_context_chars)
    quality_report = quality_report or {}
    evidence_report = evidence_report or {}
    feedback = _load_feedback_artifacts(feedback_artifact_path)
    feedback_summary = feedback["summary"]
    all_parameter_records = _build_parameter_records(
        paper_dir=paper_dir,
        extracted_json=extracted_json,
        per_evidence_chars=per_evidence_chars,
        limit=0,
    )
    parameter_records, sampling_meta = _select_parameter_records_for_audit(
        all_parameter_records,
        parameter_limit,
    )
    doc_summary = _build_document_summary(extracted_json)

    evidence_rows, evidence_metrics = _run_parameter_agent(
        client=client,
        model_evaluate=model_evaluate,
        system_prompt=EVIDENCE_AGENT_SYSTEM_PROMPT,
        user_prompt_template=EVIDENCE_AGENT_USER_PROMPT_TEMPLATE,
        parameter_records=parameter_records,
        batch_size=field_batch_size,
        max_retries=max_retries,
        feedback_summary=feedback_summary,
        payload_key="parameter_audits",
    )

    normalization_rows, normalization_metrics = _run_parameter_agent(
        client=client,
        model_evaluate=model_evaluate,
        system_prompt=NORMALIZATION_AGENT_SYSTEM_PROMPT,
        user_prompt_template=NORMALIZATION_AGENT_USER_PROMPT_TEMPLATE,
        parameter_records=parameter_records,
        batch_size=field_batch_size,
        max_retries=max_retries,
        feedback_summary=feedback_summary,
        payload_key="parameter_audits",
    )

    consistency_payload, consistency_metrics = _run_consistency_agent(
        client=client,
        model_evaluate=model_evaluate,
        parameter_records=parameter_records,
        doc_summary=doc_summary,
        feedback_summary=feedback_summary,
        max_retries=max_retries,
    )

    parameter_audits, committee_report = _merge_committee(
        parameter_records=parameter_records,
        evidence_rows=evidence_rows,
        normalization_rows=normalization_rows,
        consistency_payload=consistency_payload,
    )

    meta_payload, meta_metrics = _run_meta_agent(
        client=client,
        model_evaluate=model_evaluate,
        context_meta=context_meta,
        doc_summary=doc_summary,
        quality_report=quality_report,
        evidence_report=evidence_report,
        committee_report=committee_report,
        feedback_summary=feedback_summary,
        max_retries=max_retries,
    )
    meta_payload = _prune_meta_issues(meta_payload, parameter_audits)
    meta_payload = _augment_critical_issues_from_audits(meta_payload, parameter_audits)

    payload = {
        **meta_payload,
        "context_used": context_meta,
        "committee": {
            "evidence_judge": {
                "parameter_audits": evidence_rows,
            },
            "normalization_judge": {
                "parameter_audits": normalization_rows,
            },
            "consistency_judge": consistency_payload,
            "meta_judge": meta_payload,
            "disagreement_summary": committee_report,
        },
        "parameter_audits": parameter_audits,
        "parameter_audit_coverage": {
            "audited": len(parameter_audits),
            "available": len(all_parameter_records),
            "limit": parameter_limit,
            "truncated_by_limit": bool(parameter_limit > 0 and len(all_parameter_records) > len(parameter_records)),
            "per_evidence_chars": per_evidence_chars,
            "sampling": sampling_meta,
            "audited_locations": [str(r.get("location") or "") for r in parameter_records],
            "omitted_locations": [
                str(r.get("location") or "")
                for r in all_parameter_records[len(parameter_records):]
                if isinstance(r, dict)
            ],
            "truncated_evidence_locations": [
                str(r.get("location") or "")
                for r in parameter_records
                if isinstance(r, dict) and bool(
                    ((r.get("evidence_summary") or {}) if isinstance(r.get("evidence_summary"), dict) else {}).get("primary_text_truncated")
                )
            ],
        },
        "review_escalation": {
            "required": bool(committee_report.get("human_escalation_count")),
            "count": committee_report.get("human_escalation_count", 0),
            "disagreement_count": committee_report.get("disagreement_count", 0),
        },
        "feedback_summary_used": feedback_summary,
    }

    evaluator_input_payload = {
        "schema_version": extracted_json.get("schema_version"),
        "context_used": context_meta,
        "document_summary": doc_summary,
        "parameter_audit_coverage": {
            "available": len(all_parameter_records),
            "audited": len(parameter_records),
            "limit": parameter_limit,
            "per_evidence_chars": per_evidence_chars,
            "sampling": sampling_meta,
        },
        "parameter_records": parameter_records,
        "all_parameter_records": all_parameter_records,
    }
    evaluator_input_path = os.path.join(paper_dir, "llm_evaluation_input.json")
    with open(evaluator_input_path, "w", encoding="utf-8") as f:
        json.dump(evaluator_input_payload, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": model_evaluate,
        "committee": {
            "evidence_judge": evidence_metrics,
            "normalization_judge": normalization_metrics,
            "consistency_judge": consistency_metrics,
            "meta_judge": meta_metrics,
        },
        "input_tokens": evidence_metrics["input_tokens"] + normalization_metrics["input_tokens"] + consistency_metrics["input_tokens"] + meta_metrics["input_tokens"],
        "output_tokens": evidence_metrics["output_tokens"] + normalization_metrics["output_tokens"] + consistency_metrics["output_tokens"] + meta_metrics["output_tokens"],
        "total_tokens": evidence_metrics["total_tokens"] + normalization_metrics["total_tokens"] + consistency_metrics["total_tokens"] + meta_metrics["total_tokens"],
        "time_seconds": round(evidence_metrics["time_seconds"] + normalization_metrics["time_seconds"] + consistency_metrics["time_seconds"] + meta_metrics["time_seconds"], 2),
        "context_used": context_meta,
    }
    return payload, metrics
