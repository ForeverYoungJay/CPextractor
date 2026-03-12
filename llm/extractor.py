import os, re, json, glob
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import time

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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


def _chat_completion_with_retry(*, model: str, messages: List[Dict[str, str]], max_retries: int = 4):
    delay = 1.0
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable_llm_error(exc):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 20.0)

    raise RuntimeError(f"LLM request failed after retries: {last_exc}")

def trim_text(text: str, max_chars: int) -> str:
    text = text.strip()
    return text[:max_chars] + ("...[TRUNCATED]..." if len(text) > max_chars else "")

def load_md_files(folder: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(folder, "*.md")))
    out = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        out.append({
            "name": os.path.basename(path),
            "path": path,
            "text": txt,
            "length": len(txt)
        })
    return out

# ----------------------------
# Stage 1: LLM selection
# ----------------------------
SELECTION_SYSTEM_PROMPT = """
You are a crystal-plasticity literature triage assistant.
Follow the 4-section protocol in the user prompt.
Return JSON only.
"""


SELECTION_USER_PROMPT_TEMPLATE = """
1 Task description
Select the MINIMUM section and table files needed to reliably extract crystal-plasticity parameters.

2 Task requirements
- Prioritize files containing: constitutive equations, parameter tables, calibration/validation details, slip/twin systems.
- Avoid over-selection; include only files needed to extract:
  material identity, model type/framework, elastic constants, plastic parameters, and parameter provenance.
- Output JSON schema exactly:
{{
  "selected_sections": ["filename.md"],
  "selected_tables": ["table_001.md"],
  "why_selected": "short reason"
}}

3 Processing suggestions
- Tables with parameters are highest priority.
- Methods/simulation sections are next priority.
- Results/discussion sections are included only when they contain calibration/validation targets.
- Abstract alone is never sufficient.

4 Few-shot examples
Example A:
- Input cues: method section + parameter table exist.
- Output behavior: select those two first, add one results section only if calibration details appear there.

Example B:
- Input cues: no explicit table, parameters embedded in text.
- Output behavior: select relevant method/result sections, leave selected_tables empty.

Current paper file catalog:
Sections:
{sections_catalog}

Tables:
{tables_catalog}
"""

def build_catalog(files: List[Dict[str, Any]], max_snippet_chars: int) -> str:
    parts = []
    for f in files:
        snippet = trim_text(f["text"], max_snippet_chars).replace("\n", " ")
        parts.append(f"- {f['name']} | len={f['length']} | snippet=\"{snippet}\"")
    return "\n".join(parts)

def llm_select_files(sections, tables, model: str, max_snippet_chars: int) -> Dict[str, Any]:
    prompt = SELECTION_USER_PROMPT_TEMPLATE.format(
        sections_catalog=build_catalog(sections, max_snippet_chars),
        tables_catalog=build_catalog(tables, max_snippet_chars),
    )
    start = time.perf_counter()
    resp = _chat_completion_with_retry(
        model=model,
        messages=[
            {"role": "system", "content": SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )
    elapsed = time.perf_counter() - start
    usage = resp.usage

    return json.loads(resp.choices[0].message.content), usage, elapsed


# ----------------------------
# Stage 2: LLM Extraction
# ----------------------------
EXTRACT_SYSTEM_PROMPT = """
You are a crystal-plasticity parameter extraction engine.
Follow the 4-section protocol in the user prompt.
Extract only stated facts; do not guess.
Return JSON only.
"""

EXTRACT_SCHEMA_JSON_TEMPLATE = r"""
{
  "schema_version": "2.1.1",
  "record_id": "string or null",
  "source_document": {
    "title": "string or null",
    "authors": ["string"],
    "year": "number or null",
    "journal_or_venue": "string or null",
    "doi": "string or null"
  },
  "material": {
    "name": "string or null",
    "chemical_formula": "string or null",
    "crystal_structure": {
      "crystal_system": "cubic / hexagonal / tetragonal / orthorhombic / trigonal_rhombohedral / monoclinic / triclinic / null",
      "bravais_lattice": "P / I / F / R / A / B / C / null",
      "lattice_type": "fcc / bcc / hcp / diamond_cubic / simple_cubic / other / null",
      "structure_prototype": "fluorite / rocksalt / perovskite / spinel / wurtzite / zincblende / corundum / other / null",
      "space_group": "string or null",
      "notes": "string or null"
    },
    "phase": "single / multi / null",
    "phases": [
      {
        "phase_id": "string",
        "phase_name": "string or null",
        "role": "matrix / precipitate / inclusion / transformed_product / pore / other / null",
        "volume_fraction": {
          "value_SI": "number or null",
          "unit_SI": "fraction",
          "reported_value": "number or null",
          "reported_unit": "fraction / % / null",
          "notes": "string or null"
        },
        "notes": "string or null"
      }
    ],
    "notes": "string or null"
  },
  "microstructure": {
    "grain_structure": "single_crystal / polycrystal / bicrystal / null",
    "grain_size": {
      "value": "number or null",
      "unit": "um / mm / null",
      "distribution": "string or null",
      "notes": "string or null"
    },
    "orientation_texture": {
      "description": "string or null",
      "texture_type": "odf / pole_figure / ebsd / none / null",
      "texture_data_available": "yes / no / null",
      "data_location": "string or null",
      "notes": "string or null"
    },
    "initial_defect_state": {
      "dislocation_density": {
        "value": "number or null",
        "unit": "m^-2 / null",
        "notes": "string or null"
      },
      "precipitate_state": "string or null",
      "solute_state": "string or null",
      "prestrain": {
        "value": "number or null",
        "unit": "strain / percent / null",
        "notes": "string or null"
      },
      "notes": "string or null"
    },
    "notes": "string or null"
  },
  "constitutive_model": {
    "class": "crystal_plasticity / phase_field / continuum_damage / null",
    "framework": "cpfe / fft / evpfft / vpsc / damask / umat / custom / null",
    "implementation": {
      "code_name": "string or null",
      "platform": "abaqus / damask / vpsc / custom / other / null",
      "subroutine_or_solver": "umat / vumat / spectral / fem / null",
      "version": "string or null",
      "notes": "string or null"
    },
    "kinematics": "finite_strain / small_strain / null",
    "rate_dependence": "rate_dependent / rate_independent / null",
    "single_or_poly": "single_crystal / polycrystal / null",
    "homogenization_assumption": "taylor / self_consistent / full_field / mean_field / null",
    "notes": "string or null"
  },
  "parameters": {
    "registry": [
      {
        "parameter_id": "string or null",
        "domain": "elastic / plastic / twinning / damage / thermal / numerical / other / null",
        "canonical_name": "string or null",
        "symbol": "string or null",
        "description": "string or null",
        "value": "number or string or null",
        "unit": "string or null",
        "value_SI": "number or string or null",
        "unit_SI": "string or null",
        "applies_to": {
          "scope": "global / phase / family / system / all_slip / all_twin / all_mechanisms / null",
          "phase_id": "string or null",
          "mechanism": "slip / twinning / cleavage / damage / basal_slip / prismatic_slip / pyramidal_slip / all_slip / all_twin / all_mechanisms / null",
          "family_id": "string or null",
          "family_name": "string or null",
          "system_ids": ["string"],
          "system_count": "number or null",
          "notes": "string or null"
        },
        "temperature_dependent": "yes / no / null",
        "strain_rate_dependent": "yes / no / null",
        "valid_range": "string or null",
        "source": {
          "origin_type": "original / adopted / calibrated / adopted_then_calibrated / null",
          "reference_ids": ["string"],
          "adopted_from_reference_ids": ["string"],
          "calibration_based_on_reference_ids": ["string"],
          "calibration_in_this_study": "yes / no / null",
          "calibration_method": "manual_fitting / inverse_modeling / optimization / bayesian / null",
          "evidence_text": "string or null",
          "evidence_location": {
            "kind": "figure / table / section / supplement / code / null",
            "id": "string or null",
            "page": "number or null"
          },
          "notes": "string or null"
        },
        "notes": "string or null",
        "confidence": "high / medium / low / null"
      }
    ],
    "notes": "string or null"
  },
  "deformation_conditions": {
    "loading_mode": "uniaxial_tension / compression / shear / indentation / cyclic / creep / torsion / bending / null",
    "stress_state": "uniaxial / plane_strain / biaxial / triaxial / multiaxial / null",
    "loading_path": {
      "control": "strain_controlled / stress_controlled / displacement_controlled / mixed / null",
      "loading_direction": "rd / td / nd / crystal_axis / null",
      "crystal_axis": "string or null",
      "description": "string or null"
    },
    "strain_rate": {"value": "number or null", "unit": "s^-1 / null", "range": "string or null"},
    "temperature": {"value": "number or null", "unit": "K / C / null", "history": "isothermal / non_isothermal / null"},
    "environment": {
      "pressure": {"value": "number or null", "unit": "Pa / MPa / bar / null", "notes": "string or null"},
      "medium": "air / vacuum / liquid / inert_gas / hydrogen / null",
      "notes": "string or null"
    },
    "indentation": {
      "indenter_type": "berkovich / spherical / cono_spherical / vickers / knoop / custom / null",
      "tip_radius": {"value": "number or null", "unit": "nm / um / mm / null", "notes": "string or null"},
      "target_depths": ["number"],
      "target_depth_unit": "nm / um / mm / null",
      "max_load": {"value": "number or null", "unit": "mN / N / null", "notes": "string or null"},
      "loading_rate": {"type": "displacement_rate / load_rate / indentation_strain_rate / unknown / null", "value": "number or null", "unit": "nm/s / um/s / mN/s / N/s / s^-1 / null", "notes": "string or null"},
      "hold_time_at_peak": {"value": "number or null", "unit": "s / null", "notes": "string or null"},
      "contact_assumption": "frictionless / frictional / unknown / null",
      "notes": "string or null"
    },
    "calibration_strain_range": {"min": "number or null", "max": "number or null", "unit": "strain / percent / null", "notes": "string or null"},
    "notes": "string or null"
  },
  "deformation_mechanisms": {
    "slip_families": [
      {
        "family_id": "string or null",
        "family_name": "basal / prismatic / pyramidal_ca / cubic_100_110 / cubic_110_111 / cubic_111_110 / other / null",
        "plane_direction": "string or null",
        "num_systems": "number or null",
        "systems": [
          {
            "system_id": "string or null",
            "plane": {"as_written": "string or null", "indices": ["number"], "basis": "hkl / hkil / null"},
            "direction": {"as_written": "string or null", "indices": ["number"], "basis": "uvw / uvtw / null"}
          }
        ],
        "active": "yes / no / null",
        "notes": "string or null"
      }
    ],
    "twinning_families": [
      {
        "family_id": "string or null",
        "family_name": "extension_twin / contraction_twin / other / null",
        "plane_direction": "string or null",
        "num_systems": "number or null",
        "systems": [
          {
            "system_id": "string or null",
            "plane": {"as_written": "string or null", "indices": ["number"], "basis": "hkl / hkil / null"},
            "direction": {"as_written": "string or null", "indices": ["number"], "basis": "uvw / uvtw / null"}
          }
        ],
        "active": "yes / no / null",
        "reorientation_rule": "string or null",
        "notes": "string or null"
      }
    ],
    "cleavage_families": [
      {
        "family_id": "string or null",
        "family_name": "string or null",
        "plane_direction": "string or null",
        "num_systems": "number or null",
        "systems": [
          {
            "system_id": "string or null",
            "plane": {"as_written": "string or null", "indices": ["number"], "basis": "hkl / hkil / null"},
            "direction": {"as_written": "string or null", "indices": ["number"], "basis": "uvw / uvtw / null"}
          }
        ],
        "active": "yes / no / null",
        "notes": "string or null"
      }
    ],
    "damage_mechanisms": [
      {
        "mechanism_id": "string or null",
        "name": "string or null",
        "description": "string or null",
        "active": "yes / no / null",
        "notes": "string or null"
      }
    ],
    "transformation_mechanisms": [
      {
        "mechanism_id": "string or null",
        "name": "string or null",
        "description": "string or null",
        "active": "yes / no / null",
        "notes": "string or null"
      }
    ],
    "other_mechanisms": [
      {
        "mechanism_id": "string or null",
        "name": "string or null",
        "description": "string or null",
        "active": "yes / no / null",
        "notes": "string or null"
      }
    ],
    "notes": "string or null"
  },
  "numerical_settings": {
    "rve_type": "single_element / taylor / voronoi_grains / fft_voxels / fe_mesh / null",
    "boundary_conditions": {"type": "pbc / mixed / free_surface / null", "description": "string or null"},
    "mesh_or_voxels": {"element_type": "string or null", "element_count": "number or null", "voxel_resolution": "string or null", "notes": "string or null"},
    "num_grains": "number or null",
    "integration_scheme": "implicit / explicit / semi_implicit / null",
    "time_step_or_increment": "string or null",
    "convergence_settings": "string or null",
    "notes": "string or null"
  },
  "code_compatibility": {
    "targets": [
      {
        "code": "damask / vpsc / abaqus_umat / abaqus_vumat / custom / null",
        "version": "string or null",
        "parameter_map": [{"canonical_name": "string or null", "code_parameter_name": "string or null", "expected_unit": "string or null", "scope": "global / phase / family / system / null", "notes": "string or null"}],
        "family_map": [{"family_id": "string or null", "code_family_name": "string or null", "notes": "string or null"}],
        "system_map": [{"system_id": "string or null", "code_system_index": "number or string or null", "notes": "string or null"}],
        "notes": "string or null"
      }
    ],
    "notes": "string or null"
  },
  "fit_quality": {
    "reported_metrics": [{"name": "rmse / r2 / mae / max_error / null", "value": "number or string or null", "unit": "string or null", "extraction_location": "string or null", "notes": "string or null"}],
    "qualitative_assessment": "good / medium / poor / null",
    "validated_on_independent_case": "yes / no / null",
    "validation_cases": ["string"],
    "notes": "string or null"
  },
  "references": [
    {
      "reference_id": "string or null",
      "type": "paper / dataset / thesis / report / code / other / null",
      "citation": "string or null",
      "doi": "string or null",
      "url": "string or null",
      "notes": "string or null"
    }
  ],
  "global_notes": "string or null"
}
"""

EXTRACT_USER_PROMPT_TEMPLATE = """
1 Task description
Extract crystal-plasticity information from the provided paper excerpt into a CP schema.

2 Task requirements
- Use only explicit evidence in the excerpt.
- If unknown, return null or empty list.
- Keep parameter provenance carefully (adopted references vs calibration references).
- Extract the following schema from the paper excerpt:
__SCHEMA_JSON__

3 Processing suggestions
- Prefer table values over narrative values when both are present.
- Keep the original reported unit in value/unit; do not force SI conversion here.
- If both adopted and calibrated are stated, use origin_type as adopted_then_calibrated.
- Put bracketed citation labels like 12, 60, 61 into reference id arrays.
- For source, only output reference ID arrays; do not fabricate reference title/doi.
- Do not use placeholders like this_study/present_study as reference IDs; put such info in evidence_text.
- Use snake_case enum values exactly.

4 Few-shot examples
Example A: Text says parameters adopted from [12,13] and calibrated against stress-strain curves.
Expected behavior: source.origin_type is adopted_then_calibrated, adopted_from_reference_ids includes 12 and 13, and evidence_text mentions calibration target.

Example B: Text states a parameter value but no source citation.
Expected behavior: parameter extracted with null/empty source reference arrays and confidence as low or medium.

Paper excerpt:
----------------
{context}
----------------

Return JSON only.
"""


def _schema_skeleton_from_json_template(template: str) -> Dict[str, Any]:
    text = template.strip().replace("{{", "{").replace("}}", "}")
    l = text.find("{")
    r = text.rfind("}")
    if l < 0 or r < 0 or r <= l:
        raise RuntimeError("Failed to parse extraction schema template")
    return json.loads(text[l:r + 1])


def _coerce_to_schema_shape(schema_node: Any, payload_node: Any) -> Any:
    if isinstance(schema_node, dict):
        src = payload_node if isinstance(payload_node, dict) else {}
        out: Dict[str, Any] = {}
        for k, sv in schema_node.items():
            out[k] = _coerce_to_schema_shape(sv, src.get(k))
        # Keep extra keys from model output for audit/debug.
        if isinstance(src, dict):
            for k, v in src.items():
                if k not in out:
                    out[k] = v
        return out

    if isinstance(schema_node, list):
        if not isinstance(payload_node, list):
            return []
        if not schema_node:
            return payload_node
        item_schema = schema_node[0]
        return [_coerce_to_schema_shape(item_schema, item) for item in payload_node]

    # Leaf placeholder in template (e.g., "string or null"): prefer payload value, else null.
    if payload_node is None:
        return None
    return payload_node


EXTRACT_SCHEMA_SKELETON = _schema_skeleton_from_json_template(EXTRACT_SCHEMA_JSON_TEMPLATE)


def _validate_extracted_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["payload is not an object"]

    for key in EXTRACT_SCHEMA_SKELETON.keys():
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    has_registry = isinstance(payload.get("parameters", {}).get("registry", []), list)
    if not has_registry:
        if not isinstance(payload.get("elastic_parameters", {}).get("constants", []), list):
            errors.append("elastic_parameters.constants must be a list")
        if not isinstance(payload.get("plastic_parameters", {}).get("parameters", []), list):
            errors.append("plastic_parameters.parameters must be a list")

    def _walk(schema_node: Any, value_node: Any, path: str):
        if isinstance(schema_node, dict):
            if not isinstance(value_node, dict):
                errors.append(f"{path or 'root'} must be an object")
                return
            for k, sv in schema_node.items():
                _walk(sv, value_node.get(k), f"{path}.{k}" if path else k)
            return

        if isinstance(schema_node, list):
            if not isinstance(value_node, list):
                errors.append(f"{path} must be a list")
                return
            if schema_node:
                for i, item in enumerate(value_node):
                    _walk(schema_node[0], item, f"{path}[{i}]")
            return

    _walk(EXTRACT_SCHEMA_SKELETON, payload, "")
    return errors


def _build_extract_prompt(context: str) -> str:
    # Avoid str.format() here because the template contains literal braces such as crystallographic {111}.
    prompt = EXTRACT_USER_PROMPT_TEMPLATE.replace("__SCHEMA_JSON__", EXTRACT_SCHEMA_JSON_TEMPLATE)
    return prompt.replace("{context}", context)


SOURCE_ENRICH_SYSTEM_PROMPT = """
You refine provenance fields for previously extracted CP parameters.
Return JSON only.
"""


SOURCE_ENRICH_USER_PROMPT_TEMPLATE = """
1 Task description
Refine and complete parameter source/provenance fields using the excerpt.

2 Task requirements
- Only fill source fields for existing parameters by index.
- Do not change numeric values or symbols.
- Output JSON:
{{
  "elastic_sources": [
    {{
      "index": "number",
      "source": {{
        "origin_type": "original / adopted / calibrated / adopted_then_calibrated / null",
        "reference_ids": ["string"],
        "adopted_from_reference_ids": ["string"],
        "calibration_based_on_reference_ids": ["string"],
        "calibration_in_this_study": "yes / no / null",
        "calibration_method": "string or null",
        "evidence_text": "string or null",
        "evidence_location": {{
          "kind": "figure / table / section / supplement / code / null",
          "id": "string or null",
          "page": "number or null"
        }},
        "notes": "string or null"
      }},
      "confidence": "high / medium / low / null"
    }}
  ],
  "plastic_sources": [
    {{
      "index": "number",
      "source": {{
        "origin_type": "original / adopted / calibrated / adopted_then_calibrated / null",
        "reference_ids": ["string"],
        "adopted_from_reference_ids": ["string"],
        "calibration_based_on_reference_ids": ["string"],
        "calibration_in_this_study": "yes / no / null",
        "calibration_method": "string or null",
        "evidence_text": "string or null",
        "evidence_location": {{
          "kind": "figure / table / section / supplement / code / null",
          "id": "string or null",
          "page": "number or null"
        }},
        "notes": "string or null"
      }},
      "confidence": "high / medium / low / null"
    }}
  ]
}}

3 Processing suggestions
- If text says both adopted and calibrated, use adopted_then_calibrated.
- Copy citation labels (numbers) into the reference id arrays.
- Do not put this_study/present_study/current_study into reference id arrays.

4 Few-shot examples
If text says adopted from [60,61] then fitted to curves, then include both adopted and calibration refs.

Excerpt:
----------------
{context}
----------------

Current extracted JSON:
----------------
{extracted_json}
----------------

Return JSON only.
"""


def _build_source_enrich_prompt(context: str, extracted: Dict[str, Any]) -> str:
    prompt = SOURCE_ENRICH_USER_PROMPT_TEMPLATE.replace("{context}", context)
    return prompt.replace("{extracted_json}", json.dumps(extracted, ensure_ascii=False)[:20000])


def _merge_source_enrichment(extracted: Dict[str, Any], enrich: Dict[str, Any]) -> Dict[str, Any]:
    registry = extracted.get("parameters", {}).get("registry", [])
    if not isinstance(registry, list):
        return extracted

    elastic_idx = []
    plastic_idx = []
    for i, it in enumerate(registry):
        if not isinstance(it, dict):
            continue
        domain = str(it.get("domain") or "").strip().lower()
        cname = str(it.get("canonical_name") or "").strip().lower()
        if domain == "elastic" or cname in _ELASTIC_CANONICALS:
            elastic_idx.append(i)
        else:
            plastic_idx.append(i)

    for item in enrich.get("elastic_sources", []) or []:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(elastic_idx):
            ridx = elastic_idx[idx]
            if isinstance(item.get("source"), dict):
                registry[ridx]["source"] = item["source"]
            if item.get("confidence") is not None:
                registry[ridx]["confidence"] = item.get("confidence")

    for item in enrich.get("plastic_sources", []) or []:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(plastic_idx):
            ridx = plastic_idx[idx]
            if isinstance(item.get("source"), dict):
                registry[ridx]["source"] = item["source"]
            if item.get("confidence") is not None:
                registry[ridx]["confidence"] = item.get("confidence")

    return extracted


_ELASTIC_CANONICALS = {"c11", "c12", "c13", "c33", "c44", "c55", "c66", "e", "nu", "g", "k"}


def _strip_internal_keys(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            out[k] = _strip_internal_keys(v)
        return out
    if isinstance(node, list):
        return [_strip_internal_keys(x) for x in node]
    return node


def _drop_legacy_parameter_blocks(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict):
        payload.pop("elastic_parameters", None)
        payload.pop("plastic_parameters", None)
    return payload




def build_context(selected_sections, selected_tables, max_context_chars: int) -> str:
    parts = []
    total = 0

    for t in selected_tables:
        content = trim_text(t["text"], max_context_chars)
        chunk = f"\n\n=== TABLE: {t['name']} ===\n{content}"
        parts.append(chunk)
        total += len(chunk)
        if total > max_context_chars:
            return "\n".join(parts)

    for s in selected_sections:
        content = trim_text(s["text"], max_context_chars)
        chunk = f"\n\n=== SECTION: {s['name']} ===\n{content}"
        parts.append(chunk)
        total += len(chunk)
        if total > max_context_chars:
            break

    return "\n".join(parts).strip()

def llm_extract(context: str, model: str, max_retries: int = 2) -> Tuple[Dict[str, Any], Any, float]:
    prompt = _build_extract_prompt(context)
    attempts = max(1, max_retries + 1)
    start_all = time.perf_counter()
    last_errors: List[str] = []
    last_usage = None

    for attempt in range(1, attempts + 1):
        resp = _chat_completion_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_retries=4,
        )

        last_usage = resp.usage
        raw_payload = json.loads(resp.choices[0].message.content)
        payload = _coerce_to_schema_shape(EXTRACT_SCHEMA_SKELETON, raw_payload)
        errors = _validate_extracted_payload(payload)
        if not errors:
            elapsed = time.perf_counter() - start_all
            return payload, resp.usage, elapsed

        last_errors = errors
        if attempt < attempts:
            prompt = (
                _build_extract_prompt(context)
                + "\n\nValidation errors from your previous output:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\nPlease regenerate and return valid JSON only."
            )

    elapsed = time.perf_counter() - start_all
    raise RuntimeError(f"Extraction JSON validation failed after {attempts} attempts: {last_errors}")

def run_llm_on_paper_dir(
    paper_dir: str,
    model_select: str,
    model_extract: str,
    max_snippet_chars: int,
    max_context_chars: int,
    max_extract_retries: int = 2,
    enable_source_enrichment: bool = True,
):
    sections_dir = os.path.join(paper_dir, "sections")
    tables_dir = os.path.join(paper_dir, "tables")

    sections = load_md_files(sections_dir) if os.path.exists(sections_dir) else []
    tables = load_md_files(tables_dir) if os.path.exists(tables_dir) else []

    selection, sel_usage, sel_time = llm_select_files(
        sections, tables,
        model=model_select,
        max_snippet_chars=max_snippet_chars
    )

    selected_section_names = set(selection.get("selected_sections", []))
    selected_table_names = set(selection.get("selected_tables", []))

    selected_sections = [s for s in sections if s["name"] in selected_section_names]
    selected_tables = [t for t in tables if t["name"] in selected_table_names]

    # Fallback for robustness when selection stage returns empty.
    if not selected_sections and sections:
        selected_sections = sections[:2]
    if not selected_tables and tables:
        selected_tables = tables[:1]

    with open(os.path.join(paper_dir, "llm_selected_files.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    context = build_context(selected_sections, selected_tables, max_context_chars=max_context_chars)
    extracted, ext_usage, ext_time = llm_extract(
        context,
        model=model_extract,
        max_retries=max_extract_retries,
    )

    if enable_source_enrichment:
        try:
            enrich_resp = _chat_completion_with_retry(
                model=model_extract,
                messages=[
                    {"role": "system", "content": SOURCE_ENRICH_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_source_enrich_prompt(context, extracted)},
                ],
                max_retries=3,
            )
            enrich_payload = json.loads(enrich_resp.choices[0].message.content)
            extracted = _merge_source_enrichment(extracted, enrich_payload)
        except Exception:
            # Keep pipeline robust: provenance enrichment is optional.
            pass

    extracted = _strip_internal_keys(extracted)
    extracted = _drop_legacy_parameter_blocks(extracted)
    with open(os.path.join(paper_dir, "materials_extracted.json"), "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)
    print(
        "Extraction complete with "
        f"{ext_usage.total_tokens+sel_usage.total_tokens} total tokens, "
        f"in {sel_time+ext_time:.2f} seconds."
    )
    return {
        "selection": selection,
        "extracted": extracted,
        "metrics": {
            "select": {
                "input_tokens": sel_usage.prompt_tokens,
                "output_tokens": sel_usage.completion_tokens,
                "total_tokens": sel_usage.total_tokens,
                "time_seconds": sel_time,
            },
            "extract": {
                "input_tokens": ext_usage.prompt_tokens,
                "output_tokens": ext_usage.completion_tokens,
                "total_tokens": ext_usage.total_tokens,
                "time_seconds": ext_time,
            },
        }
    }
