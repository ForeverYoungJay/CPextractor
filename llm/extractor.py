import os, re, json, glob, base64
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import time
from elsevier.fulltext_parser import download_table_image
from llm.openai_sanitize import sanitize_text_for_openai, validate_openai_json_payload

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


def _chat_completion_with_retry(*, model: str, messages: List[Dict[str, Any]], max_retries: int = 4):
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

    raise RuntimeError(f"LLM request failed after retries: {last_exc}")

def trim_text(text: str, max_chars: int) -> str:
    text = sanitize_text_for_openai(text).strip()
    return text[:max_chars] + ("...[TRUNCATED]..." if len(text) > max_chars else "")


def image_to_data_url(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".") or "jpeg"
    mime = "image/jpeg"
    if ext == "png":
        mime = "image/png"
    elif ext == "gif":
        mime = "image/gif"
    elif ext == "webp":
        mime = "image/webp"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _section_selection_preview(text: str, max_lines: int = 3) -> str:
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if not line:
            continue
        lines.append(line)
    if len(lines) <= 1:
        return ""
    preview_lines = lines[1:max_lines + 1]
    return " || ".join(preview_lines)

def load_md_files(folder: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(folder, "*.md")))
    out = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            txt = sanitize_text_for_openai(f.read())
        title = None
        for line in txt.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip() or None
            else:
                title = stripped[:200]
            break
        out.append({
            "name": os.path.basename(path),
            "path": path,
            "text": txt,
            "length": len(txt),
            "title": title,
            "selection_preview": _section_selection_preview(txt),
        })
    return out


def _table_json_summary(table_json: Dict[str, Any], max_rows: int = 8, max_cells_per_row: int = 10) -> str:
    caption = str(table_json.get("caption") or "").strip()
    rows = table_json.get("rows")
    if table_json.get("table_kind") == "image_backed":
        label = str(table_json.get("table_label") or "").strip()
        image = table_json.get("image") or {}
        local_path = str(image.get("local_path") or "").strip()
        parts = []
        if label:
            parts.append(f"Label: {label}")
        if caption:
            parts.append(f"Caption: {caption}")
        if local_path:
            parts.append("This table has a local image and can be passed directly to the extraction model.")
        return "\n".join(parts).strip()
    if not isinstance(rows, list):
        return caption

    lines: List[str] = []
    if caption:
        lines.append(f"Caption: {caption}")
    for idx, row in enumerate(rows[:max_rows], start=1):
        if not isinstance(row, list):
            continue
        cells = [str(c).strip() for c in row[:max_cells_per_row] if str(c).strip()]
        if not cells:
            continue
        lines.append(f"Row {idx}: " + " | ".join(cells))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines).strip()


def _table_json_full_text(table_json: Dict[str, Any], max_cells_per_row: int = 20) -> str:
    caption = str(table_json.get("caption") or "").strip()
    rows = table_json.get("rows")
    if table_json.get("table_kind") == "image_backed":
        return _table_json_summary(table_json, max_rows=1000, max_cells_per_row=max_cells_per_row)
    if not isinstance(rows, list):
        return caption

    lines: List[str] = []
    if caption:
        lines.append(f"Caption: {caption}")
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            continue
        cells = [str(c).strip() for c in row[:max_cells_per_row] if str(c).strip()]
        if not cells:
            continue
        lines.append(f"Row {idx}: " + " | ".join(cells))
    return "\n".join(lines).strip()


def _fallback_table_score(table: Dict[str, Any]) -> int:
    haystack = " ".join(
        str(table.get(k) or "")
        for k in ("display_label", "json_summary", "extract_text", "text", "name", "table_kind")
    ).lower()
    keywords = (
        "parameter",
        "hardening",
        "elastic",
        "plastic",
        "constitutive",
        "slip",
        "twin",
        "calibration",
        "moduli",
        "strength",
        "chemical composition",
        "composition",
        "phase fraction",
        "phase constitution",
        "grain size",
        "texture",
        "microstructure",
        "material input",
    )
    score = sum(1 for kw in keywords if kw in haystack)
    if str(table.get("table_kind") or "").strip().lower() == "image_backed":
        score += 1
    return score


def _fallback_select_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        tables,
        key=lambda t: (_fallback_table_score(t), -int(t.get("length") or 0)),
        reverse=True,
    )
    if not ranked:
        return []
    top_score = _fallback_table_score(ranked[0])
    if top_score <= 0:
        return ranked[:1]
    return [t for t in ranked[:2] if _fallback_table_score(t) == top_score] or ranked[:1]


def _is_material_profile_table(table: Dict[str, Any]) -> bool:
    haystack = " ".join(
        str(table.get(k) or "")
        for k in ("display_label", "json_summary", "extract_text", "text", "name")
    ).lower()
    keywords = (
        "chemical composition",
        "composition",
        "phase fraction",
        "phase constitution",
        "grain size",
        "texture",
        "microstructure",
        "material input",
        "euler angles",
    )
    return any(kw in haystack for kw in keywords)


def _table_semantic_type(table: Dict[str, Any]) -> str:
    haystack = " ".join(
        str(table.get(k) or "")
        for k in ("display_label", "json_summary", "extract_text", "text", "name")
    ).lower()
    if any(kw in haystack for kw in ("chemical composition", "composition", "wt.%", "at.%")):
        return "composition_matrix"
    if any(kw in haystack for kw in ("phase fraction", "phase constitution", "volume fraction")):
        return "phase_fraction_matrix"
    if any(kw in haystack for kw in ("temperature", "grain size", "method")) and any(
        kw in haystack for kw in ("τ", "tau", "h0", "h 0", "parameter", "model parameters", "crss", "gamma", "γ")
    ):
        return "parameter_bundle_table"
    if any(kw in haystack for kw in ("grain size", "texture", "euler angle", "orientation", "microstructure")):
        return "sample_profile_matrix"
    if any(kw in haystack for kw in ("elastic constants", "slip strength", "zener ratio")):
        return "comparative_parameter_matrix"
    if any(kw in haystack for kw in ("parameter", "hardening", "elastic", "plastic", "constitutive")):
        return "parameter_table"
    return "generic_table"


def _table_semantic_hint(table: Dict[str, Any]) -> str:
    semantic_type = _table_semantic_type(table)
    hints = {
        "composition_matrix": "Interpret this as a material-composition matrix. Prefer populating materials[].composition rather than creating parameter claims.",
        "phase_fraction_matrix": "Interpret this as a process-state or condition profile table. Prefer populating microstructure_features[] and linking them to process_states[] or conditions[].",
        "sample_profile_matrix": "Interpret this as a process-state profile table. Prefer populating process_states[] and microstructure_features[] with grain size, texture, orientation, or processing-state facts.",
        "comparative_parameter_matrix": "Interpret this as a comparative parameter matrix spanning multiple materials or phases. Keep material/sample identity explicit and avoid collapsing all columns into one material.",
        "parameter_bundle_table": "Interpret this as a multi-condition parameter table. Expand rows or columns into separate parameter claims and link them through sample_id and condition_id when conditions are explicit.",
        "parameter_table": "Interpret this as a parameter table. Extract all explicit parameter values completely into parameter_claims[].",
        "generic_table": "Interpret this conservatively and only extract explicit, well-supported facts.",
    }
    return f"Table semantic type: {semantic_type}. {hints.get(semantic_type, '')}".strip()


_EXPLICIT_PARAMETER_NAME_RE = re.compile(
    r"\b(c11|c12|c13|c33|c44|c55|c66|tau0|tau_0|tau1|theta0|theta1|h0|h1|g0|g1|xi0|xi_inf|gamma0|gammadot0|qab|gamma|crss|elastic constant|hardening|slip resistance|reference shear rate)\b",
    re.IGNORECASE,
)
_EXPLICIT_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def _table_has_explicit_parameter_values(table: Dict[str, Any]) -> bool:
    semantic_type = _table_semantic_type(table)
    if semantic_type not in {"parameter_table", "comparative_parameter_matrix", "parameter_bundle_table"}:
        return False
    haystack = " ".join(
        str(table.get(k) or "")
        for k in ("display_label", "extract_text", "json_summary", "text", "name")
    )
    return bool(_EXPLICIT_PARAMETER_NAME_RE.search(haystack) and _EXPLICIT_NUMBER_RE.search(haystack))


def _section_has_explicit_parameter_values(section: Dict[str, Any]) -> bool:
    text = str(section.get("text") or "")
    return bool(_EXPLICIT_PARAMETER_NAME_RE.search(text) and _EXPLICIT_NUMBER_RE.search(text))


def _should_skip_extraction_no_explicit_parameters(
    selected_sections: List[Dict[str, Any]],
    selected_tables: List[Dict[str, Any]],
) -> bool:
    if any(_table_has_explicit_parameter_values(table) for table in selected_tables):
        return False
    if any(_section_has_explicit_parameter_values(section) for section in selected_sections):
        return False
    return True


def _empty_extraction_payload(reason: str) -> Dict[str, Any]:
    payload = _coerce_to_schema_shape(EXTRACT_SCHEMA_SKELETON, {})
    payload["global_notes"] = reason
    return payload


def load_table_files(folder: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(folder, "table_*.json")))
    base_dir = os.path.dirname(folder)
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                table_json = json.load(f)
        except Exception:
            continue
        if not isinstance(table_json, dict):
            continue
        json_summary = _table_json_summary(table_json)
        extract_text = _table_json_full_text(table_json)
        selection_id = Path(path).stem
        out.append({
            "name": os.path.basename(path),
            "selection_id": selection_id,
            "display_label": str(table_json.get("table_label") or selection_id),
            "path": path,
            "text": json_summary,
            "length": len(json_summary),
            "table_json": table_json,
            "json_summary": json_summary,
            "extract_text": extract_text,
            "table_kind": table_json.get("table_kind") or "text",
            "image_local_path": (
                os.path.join(base_dir, str((table_json.get("image") or {}).get("local_path")))
                if str((table_json.get("image") or {}).get("local_path") or "").strip()
                else None
            ),
        })
    return out


def ensure_image_backed_table_images(
    paper_dir: str,
    tables: List[Dict[str, Any]],
    *,
    api_key: str | None = None,
    inst_token: str | None = None,
) -> List[Dict[str, Any]]:
    tables_dir = os.path.join(paper_dir, "tables")
    image_dir = os.path.join(tables_dir, "images")

    for table in tables:
        if str(table.get("table_kind") or "") != "image_backed":
            continue

        image_path = str(table.get("image_local_path") or "").strip()
        if image_path and os.path.exists(image_path):
            continue

        table_json = table.get("table_json") or {}
        image_info = table_json.get("image") or {}
        if not isinstance(image_info, dict):
            continue

        downloaded = download_table_image(
            image_info,
            image_dir,
            Path(table["name"]).stem,
            api_key=api_key,
            inst_token=inst_token,
        )
        if not downloaded:
            continue

        rel = os.path.relpath(downloaded, paper_dir)
        table_json.setdefault("image", {})["local_path"] = rel
        table["image_local_path"] = downloaded
        table["table_json"] = table_json
        try:
            Path(table["path"]).write_text(
                json.dumps(table_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    return tables

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
Select the MINIMUM section and table files needed to reliably extract crystal-plasticity parameters and the core material-profile context needed to interpret them.

2 Task requirements
- Prioritize files containing: constitutive equations, parameter tables, calibration/validation details, slip/twin systems.
- Also include files needed to recover essential material-profile context when present:
  - chemical composition
  - phase fractions / phase constitution
  - grain size / texture / microstructure descriptors
  - deformation-condition sections and tables (fatigue, tension/compression, strain rate, temperature, sample geometry) when they define how parameters are used or calibrated
  - sample or material-input tables for multi-material or multi-sample studies
- Avoid over-selection; include only files needed to extract:
  material identity, composition or microstructure context, model type/framework, elastic constants, plastic parameters, and parameter provenance.
- Output JSON schema exactly:
{{
  "selected_sections": ["filename.md"],
  "selected_tables": ["table_001"],
  "why_selected": "short reason"
}}

3 Processing suggestions
- Tables with parameters are highest priority.
- Composition, phase-fraction, grain-size, texture, and material-input tables are second priority and should be included when they define the studied material system.
- Fatigue-test, loading-condition, temperature, strain-rate, and calibration-target sections are also high priority when they define deformation conditions.
- Methods/simulation sections are next priority.
- Results/discussion sections are included only when they contain calibration/validation targets.
- Abstract alone is never sufficient.
- For multi-material or comparative papers, do not select only the parameter table if a separate composition or material table is needed to identify which material each parameter bundle belongs to.
- Prefer specific subsections such as `Fatigue test`, `Material`, `Microstructural characterization`, or `Loading conditions` over relying only on a broad parent methods section.

4 Few-shot examples
Example A:
- Input cues: method section + parameter table exist.
- Output behavior: select those two first, add one results section only if calibration details appear there.

Example B:
- Input cues: no explicit table, parameters embedded in text.
- Output behavior: select relevant method/result sections, leave selected_tables empty.

Example C:
- Input cues: one table has CP parameters, another table has chemical composition or phase fractions.
- Output behavior: select both, because the second table is needed to recover material identity or sample context.

Current paper file catalog:
Sections:
{sections_catalog}

Tables:
{tables_catalog}
"""

def build_catalog(files: List[Dict[str, Any]], max_snippet_chars: int) -> str:
    parts = []
    for f in files:
        if "selection_id" in f:
            label = str(f.get("display_label") or f.get("selection_id") or "").strip()
            caption = str((f.get("table_json") or {}).get("caption") or "").strip()
            caption = trim_text(caption, max_snippet_chars).replace("\n", " ") if caption else ""
            kind = str(f.get("table_kind") or "text")
            preview = str(f.get("json_summary") or "").strip()
            preview_lines = [line.strip() for line in preview.splitlines() if line.strip()]
            preview_lines = [line for line in preview_lines if not line.lower().startswith("caption:")]
            row_preview = ""
            if preview_lines:
                row_preview = " || ".join(preview_lines[:3])
                row_preview = trim_text(row_preview, max_snippet_chars).replace("\n", " ")
            item = f"- {f['selection_id']} | label=\"{label}\" | kind={kind} | caption=\"{caption}\""
            if row_preview:
                item += f" | preview=\"{row_preview}\""
            parts.append(item)
        else:
            title = str(f.get("title") or "").strip()
            title = trim_text(title, max_snippet_chars).replace("\n", " ") if title else ""
            preview = str(f.get("selection_preview") or "").strip()
            preview = trim_text(preview, max_snippet_chars).replace("\n", " ") if preview else ""
            item = f"- {f['name']} | title=\"{title}\""
            if preview:
                item += f" | preview=\"{preview}\""
            parts.append(item)
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
Do not emit placeholder parameter records that lack an explicit value.
For selected parameter tables, optimize for completeness: extract every explicit parameter value that can be bound to a symbol/name/scope, including zero values and auxiliary constants.
Separate paper-level material/sample profile facts from parameter-level facts whenever possible.
Return JSON only.
"""

EXTRACT_SCHEMA_JSON_TEMPLATE = r"""
{
  "schema_version": "3.1.0",
  "document": {
    "doi": "string or null",
    "title": "string or null",
    "authors": ["string"],
    "year": "number or null",
    "journal": "string or null"
  },
  "materials": [
    {
      "material_id": "string or null",
      "name": "string or null",
      "formula": "string or null",
      "material_class": "steel / titanium_alloy / nickel_superalloy / magnesium_alloy / zirconium_alloy / aluminum_alloy / copper_alloy / ceramic / intermetallic / polymer / composite / other / null",
      "composition": {
        "basis": "wt_percent / at_percent / mol_percent / fraction / null",
        "components": [
          {
            "component": "string or null",
            "value": "number or string or null",
            "unit": "string or null",
            "notes": "string or null"
          }
        ],
        "notes": "string or null"
      },
      "processing_history": [
        {
          "step_id": "string or null",
          "type": "string or null",
          "temperature": {"value": "number or null", "unit": "K / C / null", "notes": "string or null"},
          "time": {"value": "number or null", "unit": "s / min / h / null", "notes": "string or null"},
          "notes": "string or null"
        }
      ],
      "phases": [
        {
          "phase_id": "string or null",
          "name": "string or null",
          "role": "matrix / precipitate / inclusion / transformed_product / pore / parent / initial_product / existing_phase / other / null",
          "crystal_structure": {
            "crystal_system": "cubic / hexagonal / tetragonal / orthorhombic / trigonal_rhombohedral / monoclinic / triclinic / null",
            "bravais_lattice": "P / I / F / R / A / B / C / null",
            "lattice_type": "fcc / bcc / hcp / diamond_cubic / simple_cubic / other / null",
            "structure_prototype": "fluorite / rocksalt / perovskite / spinel / wurtzite / zincblende / corundum / other / null",
            "space_group": "string or null",
            "notes": "string or null"
          },
          "volume_fraction": {
            "value": "number or null",
            "unit": "fraction / % / null",
            "reported_value": "number or string or null",
            "reported_unit": "fraction / % / null",
            "notes": "string or null"
          },
          "notes": "string or null"
        }
      ],
      "notes": "string or null"
    }
  ],
  "process_states": [
    {
      "process_state_id": "string or null",
      "material_id": "string or null",
      "label": "string or null",
      "processing_steps": [
        {
          "step_id": "string or null",
          "step_type": "casting / rolling / forging / annealing / solution_treatment / aging / quenching / extrusion / additive_manufacturing / machining / polishing / other / null",
          "temperature": {"value": "number or null", "unit": "K / C / null", "notes": "string or null"},
          "time": {"value": "number or null", "unit": "s / min / h / null", "notes": "string or null"},
          "strain": {"value": "number or null", "unit": "strain / % / null", "notes": "string or null"},
          "notes": "string or null"
        }
      ],
      "linked_condition_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "conditions": [
    {
      "condition_id": "string or null",
      "label": "string or null",
      "loading_mode": "uniaxial_tension / compression / shear / indentation / cyclic / fatigue / creep / torsion / bending / null",
      "stress_state": "uniaxial / plane_strain / biaxial / triaxial / multiaxial / null",
      "loading_path": {
        "control": "strain_controlled / stress_controlled / displacement_controlled / mixed / null",
        "loading_direction": "rd / td / nd / crystal_axis / null",
        "crystal_axis": "string or null",
        "description": "string or null"
      },
      "strain_rate": {"value": "number or null", "unit": "string or null", "range": "string or null"},
      "temperature": {"value": "number or null", "unit": "K / C / null", "history": "isothermal / non_isothermal / null"},
      "fatigue": {
        "mode": "lcf / hcf / vhcf / strain_controlled / stress_controlled / null",
        "load_ratio": "string or null",
        "frequency": {"value": "number or null", "unit": "Hz / null", "notes": "string or null"},
        "notes": "string or null"
      },
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
        "loading_rate": {"type": "displacement_rate / load_rate / indentation_strain_rate / unknown / null", "value": "number or null", "unit": "string or null", "notes": "string or null"},
        "hold_time_at_peak": {"value": "number or null", "unit": "s / null", "notes": "string or null"},
        "contact_assumption": "frictionless / frictional / unknown / null",
        "notes": "string or null"
      },
      "calibration_strain_range": {"min": "number or null", "max": "number or null", "unit": "strain / percent / null", "notes": "string or null"},
      "notes": "string or null"
    }
  ],
  "models": [
    {
      "model_id": "string or null",
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
      "constitutive_laws": {
        "slip_kinetics": "power_law / thermal_activation / other / null",
        "hardening": "voce / kalidindi / dislocation_based / other / null",
        "twinning": "detwinning_enabled / ptr / other / null",
        "elasticity": "anisotropic / isotropic / other / null",
        "damage": "none / phenomenological / continuum_damage / other / null",
        "notes": "string or null"
      },
      "notes": "string or null"
    }
  ],
  "mechanisms": {
    "slip_families": [
      {
        "family_id": "string or null",
        "phase_id": "string or null",
        "name": "string or null",
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
        "phase_id": "string or null",
        "name": "string or null",
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
        "phase_id": "string or null",
        "name": "string or null",
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
  "microstructure_features": [
    {
      "feature_id": "string or null",
      "type": "grain_structure / grain_size / texture / orientation / phase_fraction / precipitates / porosity / dislocation_density / selected_grain / morphology / other / null",
      "description": "string or null",
      "value": "number or string or null",
      "unit": "string or null",
      "method": "ebsd / xrd / sem / tem / om / narrative / table / other / null",
      "applies_to": {
        "material_id": "string or null",
        "phase_id": "string or null",
        "process_state_id": "string or null",
        "condition_id": "string or null",
        "notes": "string or null"
      },
      "notes": "string or null"
    }
  ],
  "parameter_claims": [
    {
      "claim_id": "string or null",
      "model_id": "string or null",
      "parameter": {
        "canonical_name": "string or null",
        "symbol": "string or null",
        "domain": "elastic / plastic / twinning / damage / thermal / numerical / other / null",
        "description": "string or null",
        "units": {
          "reported_unit": "string or null",
          "si_unit": "string or null"
        }
      },
      "assertion": {
        "value": "number or string or null",
        "reported_value": "number or string or null",
        "reported_unit": "string or null",
        "value_si": "number or string or null",
        "si_unit": "string or null",
        "qualifier": "string or null",
        "valid_range": "string or null"
      },
      "context": {
        "material_id": "string or null",
        "phase_id": "string or null",
        "process_state_id": "string or null",
        "condition_id": "string or null",
        "mechanism_scope": {
          "level": "global / phase / family / system / null",
          "mechanism_type": "slip / twinning / cleavage / damage / mixed / null",
          "family_id": "string or null",
          "family_name": "string or null",
          "system_ids": ["string"],
          "notes": "string or null"
        },
        "notes": "string or null"
      },
      "provenance": {
        "origin_type": "original / adopted / calibrated / adopted_then_calibrated / null",
        "reference_ids": ["string"],
        "adopted_from_reference_ids": ["string"],
        "calibration_based_on_reference_ids": ["string"],
        "calibration_in_this_study": "yes / no / null",
        "calibration_method": "manual_fitting / inverse_modeling / optimization / bayesian / null",
        "notes": "string or null"
      },
      "evidence": {
        "text": "string or null",
        "table_evidence": {
          "row_name": "string or null",
          "column_name": "string or null",
          "value": "string or null",
          "excerpt": "string or null"
        },
        "notes": "string or null"
      },
      "confidence": {
        "label": "high / medium / low / null",
        "score": "number or null"
      },
      "notes": "string or null"
    }
  ],
  "global_notes": "string or null"
}
"""

EXTRACT_USER_PROMPT_TEMPLATE = """
1 Task description
Extract crystal-plasticity information from the provided paper excerpt into the v3.1 hierarchical CP schema.

2 Task requirements
- Use only explicit evidence in the excerpt.
- If unknown, return null or empty list.
- Keep parameter provenance carefully (adopted references vs calibration references).
- Keep the hierarchy explicit: `phase` belongs inside its parent `materials[]` item as a sub-item.
- Do not output a top-level `study` block.
- Do not use `samples[]`; use `process_states[]` for material processing-state variants and `conditions[]` for loading/testing conditions.
- Express microstructure as `microstructure_features[]`, not as fixed nested microstructure trees.
- Keep `parameter_claims[]` claim-centric: separate `parameter`, `assertion`, `context`, `provenance`, and `evidence`.
- Extract the following schema from the paper excerpt:
__SCHEMA_JSON__

3 Processing suggestions
- Prefer table values over narrative values when both are present.
- Keep the original reported unit in value/unit; do not force SI conversion here.
- Use `evidence` only for the direct supporting text/table snippet.
- Use `provenance` only for provenance and origin tracing.
- If both adopted and calibrated are stated, use origin_type as adopted_then_calibrated.
- Put bracketed citation labels like 12, 60, 61 into reference id arrays.
- For provenance, only output reference ID arrays; do not fabricate reference title/doi.
- Do not use placeholders like this_study/present_study as reference IDs; put such info in `provenance.notes` or `evidence.text`.
- For comparative or multi-material papers, populate `materials[]` instead of collapsing everything into one material name.
- For papers with multiple heat treatments, processing routes, or initial states, populate `process_states[]`.
- Use `conditions[]` only for deformation/testing context such as temperature, strain rate, fatigue mode, indentation settings, or loading mode.
- Use `microstructure_features[]` for grain size, texture, phase fraction, precipitates, porosity, selected grains, and other structure descriptors.
- When a parameter is clearly tied to one material/process/condition, fill `context.material_id`, `context.process_state_id`, and `context.condition_id`.
- Split context scope into two dimensions:
  `context.mechanism_scope.level` is the hierarchy level.
  `context.mechanism_scope.mechanism_type` is the mechanism category.
- Keep the parameter identity in `parameter`, the numeric statement in `assertion`, and the applicability in `context`.
- Put range/bounds or qualifiers inside `assertion`, not at top level.
- If the claim comes from a table or image-backed table, fill `evidence.table_evidence` when possible.
- Tables may be row-oriented, column-oriented, transposed, matrix-like, multi-level-header, grouped-row, or image-backed. Interpret all of these as valid parameter sources.
- Resolve units from the nearest reliable source: the value cell, row label, column header, section header, or shared table note. Do not duplicate or invent units.
- Preserve table structure instead of flattening it. Keep row/column identity, grouped headers, section headers, and shared values explicit in `evidence.table_evidence` and `context`.
- `evidence.table_evidence.row_name/column_name/value` should be human-readable labels. Do not emit row_index, column_index, or evidence_location.
- `evidence.table_evidence.excerpt` should be the smallest self-contained supporting snippet: usually one table row or one grouped-row segment with the parameter label and value together. Do not dump the whole table, and do not reduce it to a naked number.
- If you know the literature source or table section but not the exact numeric cell span, still provide `evidence.text` and `evidence.table_evidence` rather than leaving evidence blank.
- `parameter.canonical_name` should already use the project-standard canonical name when it is clear from symbol/description/context. Avoid verbose phrase-like names if a stable canonical label is available.
- Use snake_case enum values exactly.
- Only emit a parameter claim when you can bind it to an explicit numeric/string value in the excerpt or to a concrete grouped table row with one-to-one value mapping.
- If the excerpt only gives a definition, equation form, literature provenance, or says a parameter was calibrated/adopted without stating its value, omit that parameter from `parameter_claims`.
- For grouped rows such as `c11, c12, c44 -> 183.9 GPa, 123.4 GPa, 91.5 GPa`, emit separate parameter claims for each value instead of one null-valued grouped shell.
- For comparison tables spanning many materials or slip modes, only extract entries that can be bound to the focal studied material/phase/slip family from context; otherwise omit them rather than producing a generic null-valued parameter.
- If a grouped row gives a numeric value but omits a unit for one segment, still emit that parameter with the explicit value and set unit to null; do not fabricate a unit.
- Missing unit alone is not a reason to omit a parameter when the value itself is explicit and the parameter identity is clear.
- For rows like `h, hD -> 3555 MPa, 245`, emit both parameters. Keep `h=3555 MPa`; keep `hD=245` with unit null if no unit is explicitly shown for `hD`.
- For selected parameter tables, scan the whole table from top to bottom. Do not stop after extracting only the most salient plastic parameters.
- Treat `0` as a valid explicit parameter value. Never omit a parameter solely because its value is zero.
- Extract elastic constants from parameter tables with the same priority as plastic parameters.
- Extract auxiliary calibrated constants and numerical/material constants from selected parameter tables when they have explicit values, including items like `R`, `T`, `A`, `d`, `h`, and `hD`.
- If one table row covers multiple scopes, expand it into multiple parameter claims when the mapping is explicit.
- Example: `Prism and Basal slip systems | γ̇0 | 3.5e-4` should become two claims if the value is shared by both prism and basal.
- Example: `Prism and Basal slip systems | n | 20` should become two claims, not one ambiguous shared record.
- For grouped rows with multiple labels and multiple values, preserve order and map each label to the corresponding value segment.
- For chemical-composition matrices, each column/material should become a separate entry in `materials[]`; do not flatten the entire table into one material string.
- For grain-size / phase-fraction / texture tables, populate `microstructure_features[]` even if the table contains no CP parameters.
- For selected-grain tables, populate `microstructure_features[]` with `type=selected_grain` and keep the grain identity in `description` or `notes`.
- Keep material-state hierarchy explicit: detailed per-material composition/phase data belongs in `materials[]`; per-process-state variation belongs in `process_states[]`; loading/test variation belongs in `conditions[]`.
- For multiple deformation conditions in one paper, populate `conditions[]` and link `parameter_claims[]` through `context.condition_id` when explicit.
- For parameter tables organized by temperature, grain size, process state, or method, expand each condition row into distinct parameter claims and link them to the right `process_state_id` / `condition_id`.
- Distinguish calibration bounds from final calibrated parameters. If a table gives bounds or search ranges, do not convert the bound itself into a standalone calibrated parameter. Preserve it in `assertion.valid_range` when the mapping is explicit.
- For image-backed parameter tables, apply the same completeness rule as text tables: recover all explicit numeric entries, including zeros, shared rows, and elastic blocks.
- Do not omit a tail row of a selected parameter table merely because earlier rows already provided more prominent parameters.

4 Few-shot examples
Example A: Text says parameters adopted from [12,13] and calibrated against stress-strain curves.
Expected behavior: provenance.origin_type is adopted_then_calibrated, adopted_from_reference_ids includes 12 and 13, and `evidence.text` mentions calibration target.

Example B: Text states a parameter value but no source citation.
Expected behavior: parameter extracted with null/empty provenance reference arrays.

Example C: Table row says `c11, c12, c44` and value cell says `183.9 GPa, 123.4 GPa, 91.5 GPa`.
Expected behavior: emit three separate elastic parameter claims with explicit values.

Example D: Section says `Slip shear rate equation from Ref. [47]` but gives no numeric parameter value.
Expected behavior: do not emit a parameter record for slip shear rate.

Example E: Table row says `h, hD` and value cell says `3555 MPa, 245`.
Expected behavior: emit two parameter claims. `h` keeps unit `MPa`; `hD` keeps value `245` with unit null if no explicit unit is given for the second value.

Example F: Table row says `Prism and Basal slip systems | n | 20`.
Expected behavior: emit one `n=20` record for prism and one `n=20` record for basal.

Example G: Table row says `Prism slip system | θ1 | 0`.
Expected behavior: emit the parameter with value `0`; do not treat zero as missing.

Example H: A selected parameter table contains elastic constants at the top and fitted constants such as `R`, `T`, `A`, `d` near the bottom.
Expected behavior: extract all of them if explicit values are shown; do not stop at the first few plastic parameters.

Example I: A composition table lists materials as columns and elements as rows.
Expected behavior: populate `materials[]` with one material entry per column and keep composition rows under each material.

Example J: A parameter table is indexed by temperature and grain size for one alloy after different heat treatments.
Expected behavior: create one `process_states[]` entry per heat-treatment state when explicit, store the grain-size fact in `microstructure_features[]`, and link the corresponding parameter claims with `process_state_id`.

Example K: A table lists process states as rows and phase fractions as columns for room temperature and cryogenic temperature.
Expected behavior: populate `conditions[]` for room temperature and cryogenic temperature, store phase-fraction observations in `microstructure_features[]`, and do not convert phase fractions into CP parameter claims.

Example L: A comparative table lists many materials as rows or columns, with elastic constants and slip strengths for each material.
Expected behavior: preserve one material entry per material and keep parameter claims tied to the correct material_id or phase_id instead of mixing all values into one generic phase.

Example M: A selected-grain table lists grain IDs, phase labels, and grain sizes for a nanoindentation study.
Expected behavior: populate `microstructure_features[]` with `type=selected_grain` entries instead of collapsing the information into a free-text note.

Example N: One table gives GA calibration bounds, and later tables give GA/T&E final calibrated parameters.
Expected behavior: keep the bounds as `assertion.valid_range` for the corresponding parameter when the mapping is explicit; do not create standalone `*_bound` parameters unless the paper explicitly treats the bound as a model quantity.

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


def _project_v3_claim_to_legacy_registry_item(claim: Dict[str, Any]) -> Dict[str, Any]:
    claim = _safe_dict(claim)
    parameter = _safe_dict(claim.get("parameter"))
    assertion = _safe_dict(claim.get("assertion"))
    context = _safe_dict(claim.get("context"))
    mechanism_scope = _safe_dict(context.get("mechanism_scope"))
    provenance = _safe_dict(claim.get("provenance"))
    evidence = _safe_dict(claim.get("evidence"))
    legacy_item = {
        "claim_id": claim.get("claim_id"),
        "domain": parameter.get("domain"),
        "canonical_name": parameter.get("canonical_name"),
        "symbol": parameter.get("symbol"),
        "description": parameter.get("description"),
        "value": assertion.get("reported_value", assertion.get("value")),
        "unit": assertion.get("reported_unit", _safe_dict(parameter.get("units")).get("reported_unit")),
        "value_SI": assertion.get("value_si"),
        "unit_SI": assertion.get("si_unit", _safe_dict(parameter.get("units")).get("si_unit")),
        "applies_to": {
            "scope": mechanism_scope.get("level"),
            "material_id": context.get("material_id"),
            "sample_id": context.get("process_state_id"),
            "bundle_id": None,
            "condition_id": context.get("condition_id"),
            "phase_id": context.get("phase_id"),
            "mechanism": mechanism_scope.get("mechanism_type"),
            "family_id": mechanism_scope.get("family_id"),
            "family_name": mechanism_scope.get("family_name"),
            "system_ids": _safe_list(mechanism_scope.get("system_ids")),
            "system_count": len(_safe_list(mechanism_scope.get("system_ids"))),
            "notes": _first_non_empty(context.get("notes"), mechanism_scope.get("notes")),
        },
        "temperature_dependent": claim.get("temperature_dependent"),
        "strain_rate_dependent": claim.get("strain_rate_dependent"),
        "valid_range": assertion.get("valid_range"),
        "source": {
            "origin_type": provenance.get("origin_type"),
            "reference_ids": _safe_list(provenance.get("reference_ids")),
            "adopted_from_reference_ids": _safe_list(provenance.get("adopted_from_reference_ids")),
            "calibration_based_on_reference_ids": _safe_list(provenance.get("calibration_based_on_reference_ids")),
            "calibration_in_this_study": provenance.get("calibration_in_this_study"),
            "calibration_method": provenance.get("calibration_method"),
            "notes": provenance.get("notes"),
        },
        "evidence": {
            "evidence_text": evidence.get("text"),
            "table_evidence": _safe_dict(evidence.get("table_evidence")) or None,
            "notes": evidence.get("notes"),
        },
        "notes": claim.get("notes"),
    }
    return legacy_item


def _feature_matches(feature: Dict[str, Any], *, material_id: Any = None, phase_id: Any = None, process_state_id: Any = None) -> bool:
    applies_to = _safe_dict(feature.get("applies_to"))
    if material_id and applies_to.get("material_id") not in (None, material_id):
        return False
    if phase_id and applies_to.get("phase_id") not in (None, phase_id):
        return False
    if process_state_id and applies_to.get("process_state_id") not in (None, process_state_id):
        return False
    return True


def _first_feature(
    features: List[Dict[str, Any]],
    feature_type: str,
    *,
    material_id: Any = None,
    phase_id: Any = None,
    process_state_id: Any = None,
) -> Dict[str, Any]:
    for feature in features:
        if not isinstance(feature, dict):
            continue
        if str(feature.get("type") or "").strip().lower() != feature_type:
            continue
        if _feature_matches(feature, material_id=material_id, phase_id=phase_id, process_state_id=process_state_id):
            return feature
    return {}


def _inject_legacy_compat_views_from_v3(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    document = _safe_dict(payload.get("document"))
    materials = [m for m in _safe_list(payload.get("materials")) if isinstance(m, dict)]
    process_states = [s for s in _safe_list(payload.get("process_states")) if isinstance(s, dict)]
    conditions = [c for c in _safe_list(payload.get("conditions")) if isinstance(c, dict)]
    models = [m for m in _safe_list(payload.get("models")) if isinstance(m, dict)]
    mechanisms = _safe_dict(payload.get("mechanisms"))
    microstructure_features = [f for f in _safe_list(payload.get("microstructure_features")) if isinstance(f, dict)]
    claims = [c for c in _safe_list(payload.get("parameter_claims")) if isinstance(c, dict)]

    primary_material = materials[0] if materials else {}
    primary_phases = [p for p in _safe_list(primary_material.get("phases")) if isinstance(p, dict)]
    primary_phase = primary_phases[0] if primary_phases else {}
    primary_model = models[0] if models else {}
    primary_condition = conditions[0] if conditions else {}
    primary_process_state = process_states[0] if process_states else {}
    primary_grain_feature = _first_feature(
        microstructure_features,
        "grain_size",
        material_id=primary_material.get("material_id"),
        phase_id=primary_phase.get("phase_id"),
        process_state_id=primary_process_state.get("process_state_id"),
    )
    primary_texture_feature = _first_feature(
        microstructure_features,
        "texture",
        material_id=primary_material.get("material_id"),
        phase_id=primary_phase.get("phase_id"),
        process_state_id=primary_process_state.get("process_state_id"),
    )
    primary_grain_structure_feature = _first_feature(
        microstructure_features,
        "grain_structure",
        material_id=primary_material.get("material_id"),
        phase_id=primary_phase.get("phase_id"),
        process_state_id=primary_process_state.get("process_state_id"),
    )
    primary_dislocation_feature = _first_feature(
        microstructure_features,
        "dislocation_density",
        material_id=primary_material.get("material_id"),
        phase_id=primary_phase.get("phase_id"),
        process_state_id=primary_process_state.get("process_state_id"),
    )
    primary_precip_feature = _first_feature(
        microstructure_features,
        "precipitates",
        material_id=primary_material.get("material_id"),
        phase_id=primary_phase.get("phase_id"),
        process_state_id=primary_process_state.get("process_state_id"),
    )

    payload["record_id"] = payload.get("record_id")
    payload["source_document"] = {
        "title": document.get("title"),
        "authors": _safe_list(document.get("authors")),
        "year": document.get("year"),
        "journal_or_venue": document.get("journal"),
        "doi": document.get("doi"),
    }
    payload["material"] = {
        "name": primary_material.get("name"),
        "chemical_formula": primary_material.get("formula"),
        "phase": ("multi" if len(primary_phases) > 1 else "single") if primary_phases else None,
        "phases": [
            {
                "phase_id": phase.get("phase_id"),
                "phase_name": phase.get("name"),
                "role": phase.get("role"),
                "crystal_structure": _safe_dict(phase.get("crystal_structure")) or None,
                "volume_fraction": {
                    "value_SI": _safe_dict(phase.get("volume_fraction")).get("value"),
                    "unit_SI": _safe_dict(phase.get("volume_fraction")).get("unit"),
                    "reported_value": _safe_dict(phase.get("volume_fraction")).get("reported_value"),
                    "reported_unit": _safe_dict(phase.get("volume_fraction")).get("reported_unit"),
                    "notes": _safe_dict(phase.get("volume_fraction")).get("notes"),
                },
                "notes": phase.get("notes"),
            }
            for phase in primary_phases
        ],
        "notes": primary_material.get("notes"),
    }
    payload["paper_profile"] = {
        "studied_materials": [
            {
                "material_id": material.get("material_id"),
                "name": material.get("name"),
                "chemical_formula": material.get("formula"),
                "material_class": material.get("material_class"),
                "phase_mode": ("multi" if len(_safe_list(material.get("phases"))) > 1 else "single") if _safe_list(material.get("phases")) else None,
                "composition": {
                    "basis": _safe_dict(material.get("composition")).get("basis"),
                    "rows": [
                        {
                            "component": component.get("component"),
                            "value": component.get("value"),
                            "unit": component.get("unit"),
                            "notes": component.get("notes"),
                        }
                        for component in _safe_list(_safe_dict(material.get("composition")).get("components"))
                        if isinstance(component, dict)
                    ],
                    "notes": _safe_dict(material.get("composition")).get("notes"),
                },
                "phase_ids": [
                    phase.get("phase_id")
                    for phase in _safe_list(material.get("phases"))
                    if isinstance(phase, dict) and phase.get("phase_id")
                ],
                "notes": material.get("notes"),
            }
            for material in materials
        ],
        "sample_profiles": [
            {
                "sample_id": process_state.get("process_state_id"),
                "material_id": process_state.get("material_id"),
                "condition_id": _safe_list(process_state.get("linked_condition_ids"))[0] if _safe_list(process_state.get("linked_condition_ids")) else None,
                "label": process_state.get("label"),
                "processing_state": process_state.get("label"),
                "temperature": None,
                "grain_size": {
                    "value": _first_feature(microstructure_features, "grain_size", material_id=process_state.get("material_id"), process_state_id=process_state.get("process_state_id")).get("value"),
                    "unit": _first_feature(microstructure_features, "grain_size", material_id=process_state.get("material_id"), process_state_id=process_state.get("process_state_id")).get("unit"),
                    "notes": _first_feature(microstructure_features, "grain_size", material_id=process_state.get("material_id"), process_state_id=process_state.get("process_state_id")).get("notes"),
                } if _first_feature(microstructure_features, "grain_size", material_id=process_state.get("material_id"), process_state_id=process_state.get("process_state_id")) else None,
                "phase_fractions": [
                    {
                        "phase_id": _safe_dict(feature.get("applies_to")).get("phase_id"),
                        "value": feature.get("value"),
                        "unit": feature.get("unit"),
                        "notes": feature.get("notes"),
                    }
                    for feature in microstructure_features
                    if isinstance(feature, dict)
                    and str(feature.get("type") or "").strip().lower() == "phase_fraction"
                    and _feature_matches(feature, material_id=process_state.get("material_id"), process_state_id=process_state.get("process_state_id"))
                ],
                "selected_grains": [
                    {
                        "grain_id": feature.get("feature_id"),
                        "label": feature.get("description"),
                        "phase_id": _safe_dict(feature.get("applies_to")).get("phase_id"),
                        "grain_size": {"value": feature.get("value"), "unit": feature.get("unit"), "notes": feature.get("notes")} if feature.get("value") not in (None, "") else None,
                        "orientation_notes": feature.get("description"),
                        "notes": feature.get("notes"),
                    }
                    for feature in microstructure_features
                    if isinstance(feature, dict)
                    and str(feature.get("type") or "").strip().lower() == "selected_grain"
                    and _feature_matches(feature, material_id=process_state.get("material_id"), process_state_id=process_state.get("process_state_id"))
                ],
                "texture_or_orientation": _first_feature(microstructure_features, "texture", material_id=process_state.get("material_id"), process_state_id=process_state.get("process_state_id")).get("description"),
                "notes": process_state.get("notes"),
            }
            for process_state in process_states
        ],
        "notes": payload.get("global_notes"),
    }
    payload["microstructure"] = {
        "grain_structure": primary_grain_structure_feature.get("value"),
        "grain_size": {
            "value": primary_grain_feature.get("value"),
            "unit": primary_grain_feature.get("unit"),
            "notes": primary_grain_feature.get("notes"),
        } if primary_grain_feature else None,
        "orientation_texture": {
            "description": primary_texture_feature.get("description"),
            "texture_type": primary_texture_feature.get("method"),
            "texture_data_available": None,
            "data_location": None,
            "notes": primary_texture_feature.get("notes"),
        },
        "initial_defect_state": {
            "dislocation_density": {
                "value": primary_dislocation_feature.get("value"),
                "unit": primary_dislocation_feature.get("unit"),
                "notes": primary_dislocation_feature.get("notes"),
            } if primary_dislocation_feature else None,
            "precipitate_state": primary_precip_feature.get("description") or primary_precip_feature.get("value"),
            "solute_state": None,
            "prestrain": None,
            "notes": _first_non_empty(primary_dislocation_feature.get("notes"), primary_precip_feature.get("notes")),
        },
        "notes": payload.get("global_notes"),
    }
    payload["constitutive_model"] = {
        "class": primary_model.get("class"),
        "framework": primary_model.get("framework"),
        "implementation": _safe_dict(primary_model.get("implementation")) or None,
        "kinematics": primary_model.get("kinematics"),
        "rate_dependence": primary_model.get("rate_dependence"),
        "single_or_poly": primary_model.get("single_or_poly"),
        "homogenization_assumption": primary_model.get("homogenization_assumption"),
        "notes": primary_model.get("notes"),
    }
    payload["parameters"] = {
        "registry": [_project_v3_claim_to_legacy_registry_item(claim) for claim in claims],
        "notes": None,
    }
    payload["parameter_bundles"] = _safe_list(payload.get("parameter_bundles"))
    payload["deformation_conditions"] = primary_condition or {}
    payload["condition_profiles"] = conditions
    payload["deformation_mechanisms"] = {
        "slip_families": [
            {
                "family_id": family.get("family_id"),
                "family_name": family.get("name"),
                "plane_direction": family.get("plane_direction"),
                "num_systems": family.get("num_systems"),
                "systems": _safe_list(family.get("systems")),
                "active": family.get("active"),
                "notes": family.get("notes"),
            }
            for family in _safe_list(mechanisms.get("slip_families"))
            if isinstance(family, dict)
        ],
        "twinning_families": [
            {
                "family_id": family.get("family_id"),
                "family_name": family.get("name"),
                "plane_direction": family.get("plane_direction"),
                "num_systems": family.get("num_systems"),
                "systems": _safe_list(family.get("systems")),
                "active": family.get("active"),
                "reorientation_rule": family.get("reorientation_rule"),
                "notes": family.get("notes"),
            }
            for family in _safe_list(mechanisms.get("twinning_families"))
            if isinstance(family, dict)
        ],
        "cleavage_families": [
            {
                "family_id": family.get("family_id"),
                "family_name": family.get("name"),
                "plane_direction": family.get("plane_direction"),
                "num_systems": family.get("num_systems"),
                "systems": _safe_list(family.get("systems")),
                "active": family.get("active"),
                "notes": family.get("notes"),
            }
            for family in _safe_list(mechanisms.get("cleavage_families"))
            if isinstance(family, dict)
        ],
        "damage_mechanisms": _safe_list(mechanisms.get("damage_mechanisms")),
        "transformation_mechanisms": _safe_list(mechanisms.get("transformation_mechanisms")),
        "other_mechanisms": _safe_list(mechanisms.get("other_mechanisms")),
        "notes": mechanisms.get("notes"),
    }
    return payload


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
        "notes": "string or null"
      }}
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
        "notes": "string or null"
      }}
    }}
  ]
}}

3 Processing suggestions
- If text says both adopted and calibrated, use adopted_then_calibrated.
- Copy citation labels (numbers) into the reference id arrays.
- Do not put this_study/present_study/current_study into reference id arrays.
- Do not rewrite or replace the existing `evidence` object. This step is only for provenance/source refinement.
- If a parameter has an explicit value but no explicit unit, keep the value and leave unit-related fields null; do not delete or downgrade the parameter solely because the unit is missing.
- If a parameter value is explicitly `0`, preserve it as a real value.
- If one table row applies to multiple families/scopes, preserve that shared semantics in existing `evidence` and `applies_to`; do not blur it in provenance notes.
- Preserve material/sample/condition identity already present in the extracted JSON; provenance refinement should not blur which material, sample, or bundle a parameter belongs to.

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


def _merge_source_payload(existing: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    existing = existing if isinstance(existing, dict) else {}
    update = update if isinstance(update, dict) else {}
    out = dict(existing)

    for key, value in update.items():
        if value in (None, "", [], {}):
            continue
        if key == "table_evidence":
            current_te = out.get("table_evidence") if isinstance(out.get("table_evidence"), dict) else {}
            new_te = value if isinstance(value, dict) else {}
            merged_te = dict(current_te)
            for te_key, te_value in new_te.items():
                if te_value in (None, "", [], {}):
                    continue
                if merged_te.get(te_key) in (None, "", [], {}):
                    merged_te[te_key] = te_value
            if merged_te:
                out["table_evidence"] = merged_te
            continue
        if out.get(key) in (None, "", [], {}):
            out[key] = value
    return out


def _build_source_enrich_prompt(context: str, extracted: Dict[str, Any]) -> str:
    prompt = SOURCE_ENRICH_USER_PROMPT_TEMPLATE.replace("{context}", context)
    return prompt.replace("{extracted_json}", json.dumps(extracted, ensure_ascii=False)[:20000])


def _merge_source_enrichment(extracted: Dict[str, Any], enrich: Dict[str, Any]) -> Dict[str, Any]:
    registry = extracted.get("parameters", {}).get("registry", [])
    claim_mode = False
    has_legacy_registry = isinstance(extracted.get("parameters"), dict) and isinstance(registry, list) and bool(registry)
    if not has_legacy_registry:
        registry = extracted.get("parameter_claims", [])
        claim_mode = isinstance(registry, list)
    if not isinstance(registry, list):
        return extracted

    elastic_idx = []
    plastic_idx = []
    for i, it in enumerate(registry):
        if not isinstance(it, dict):
            continue
        domain = str(it.get("domain") or _safe_dict(_safe_dict(it.get("parameter")).get("domain")) or "").strip().lower()
        cname = str(it.get("canonical_name") or _safe_dict(it.get("parameter")).get("canonical_name") or "").strip().lower()
        if domain == "elastic" or cname in _ELASTIC_CANONICALS:
            elastic_idx.append(i)
        else:
            plastic_idx.append(i)

    for item in enrich.get("elastic_sources", []) or []:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(elastic_idx):
            ridx = elastic_idx[idx]
            if isinstance(item.get("source"), dict):
                target_key = "provenance" if claim_mode else "source"
                registry[ridx][target_key] = _merge_source_payload(registry[ridx].get(target_key), item["source"])
            if claim_mode and item.get("confidence") not in (None, "", [], {}):
                registry[ridx]["confidence"] = _merge_source_payload(registry[ridx].get("confidence"), {"label": item.get("confidence")})

    for item in enrich.get("plastic_sources", []) or []:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(plastic_idx):
            ridx = plastic_idx[idx]
            if isinstance(item.get("source"), dict):
                target_key = "provenance" if claim_mode else "source"
                registry[ridx][target_key] = _merge_source_payload(registry[ridx].get(target_key), item["source"])
            if claim_mode and item.get("confidence") not in (None, "", [], {}):
                registry[ridx]["confidence"] = _merge_source_payload(registry[ridx].get("confidence"), {"label": item.get("confidence")})

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




def build_context(selected_sections, selected_tables, max_context_chars: int) -> Tuple[str, Dict[str, Any]]:
    parts = []
    total = 0
    context_meta: Dict[str, Any] = {
        "max_context_chars": max_context_chars,
        "included_tables": [],
        "included_sections": [],
        "truncated_tables": [],
        "truncated_sections": [],
        "omitted_tables": [],
        "omitted_sections": [],
    }

    table_budget_total = max_context_chars if not selected_sections else int(max_context_chars * 0.8)
    section_budget_total = max_context_chars - table_budget_total
    per_table_budget = table_budget_total // max(1, len(selected_tables)) if selected_tables else 0
    per_section_budget = section_budget_total // max(1, len(selected_sections)) if selected_sections else 0

    for idx, t in enumerate(selected_tables):
        remaining_budget = max_context_chars - total
        if remaining_budget <= 0:
            context_meta["omitted_tables"].extend(
                [tbl["name"] for tbl in selected_tables[idx:] if isinstance(tbl, dict) and tbl.get("name")]
            )
            break
        full_table_text = str(t.get("extract_text") or t.get("json_summary") or t.get("text") or "").strip()
        semantic_hint = _table_semantic_hint(t)
        full_table_text = f"{semantic_hint}\n{full_table_text}" if full_table_text else semantic_hint
        local_budget = max(1200, min(remaining_budget, per_table_budget or remaining_budget))
        content = trim_text(full_table_text, local_budget)
        if len(full_table_text) > local_budget:
            context_meta["truncated_tables"].append(t["name"])
        chunk = f"\n\n=== TABLE: {t['name']} ===\n{content}"
        parts.append(chunk)
        total += len(chunk)
        context_meta["included_tables"].append(t["name"])
        if total > max_context_chars:
            overflow = [tbl["name"] for tbl in selected_tables[idx + 1:] if isinstance(tbl, dict) and tbl.get("name")]
            context_meta["omitted_tables"].extend(overflow)
            context_meta["final_context_chars"] = len("\n".join(parts))
            return "\n".join(parts), context_meta

    for idx, s in enumerate(selected_sections):
        remaining_budget = max_context_chars - total
        if remaining_budget <= 0:
            context_meta["omitted_sections"].extend(
                [sec["name"] for sec in selected_sections[idx:] if isinstance(sec, dict) and sec.get("name")]
            )
            break
        local_budget = max(800, min(remaining_budget, per_section_budget or remaining_budget))
        content = trim_text(s["text"], local_budget)
        if len(str(s.get("text") or "")) > local_budget:
            context_meta["truncated_sections"].append(s["name"])
        chunk = f"\n\n=== SECTION: {s['name']} ===\n{content}"
        parts.append(chunk)
        total += len(chunk)
        context_meta["included_sections"].append(s["name"])
        if total > max_context_chars:
            overflow = [sec["name"] for sec in selected_sections[idx + 1:] if isinstance(sec, dict) and sec.get("name")]
            context_meta["omitted_sections"].extend(overflow)
            break

    context_meta["final_context_chars"] = len("\n".join(parts).strip())
    return "\n".join(parts).strip(), context_meta

def _build_extract_messages(prompt: str, selected_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    for table in selected_tables:
        if str(table.get("table_kind") or "") != "image_backed":
            continue
        image_path = str(table.get("image_local_path") or "").strip()
        if not image_path or not os.path.exists(image_path):
            continue
        table_json = table.get("table_json") or {}
        caption = str(table_json.get("caption") or "").strip()
        label = str(table_json.get("table_label") or table.get("name") or "").strip()
        table_hint = f"Image-backed table {label}."
        if caption:
            table_hint += f" Caption: {caption}"
        content.append({"type": "text", "text": sanitize_text_for_openai(table_hint)})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})

    return [
        {"role": "system", "content": sanitize_text_for_openai(EXTRACT_SYSTEM_PROMPT)},
        {"role": "user", "content": content},
    ]


def llm_extract(context: str, selected_tables: List[Dict[str, Any]], model: str, max_retries: int = 2) -> Tuple[Dict[str, Any], Any, float]:
    prompt = _build_extract_prompt(context)
    attempts = max(1, max_retries + 1)
    start_all = time.perf_counter()
    last_errors: List[str] = []
    last_usage = None

    for attempt in range(1, attempts + 1):
        resp = _chat_completion_with_retry(
            model=model,
            messages=_build_extract_messages(prompt, selected_tables),
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
    direct_image_table_input: bool = True,
    image_download_api_key: str | None = None,
    image_download_inst_token: str | None = None,
):
    sections_dir = os.path.join(paper_dir, "sections")
    tables_dir = os.path.join(paper_dir, "tables")

    sections = load_md_files(sections_dir) if os.path.exists(sections_dir) else []
    tables = load_table_files(tables_dir) if os.path.exists(tables_dir) else []
    if direct_image_table_input and tables:
        tables = ensure_image_backed_table_images(
            paper_dir,
            tables,
            api_key=image_download_api_key,
            inst_token=image_download_inst_token,
        )

    selection, sel_usage, sel_time = llm_select_files(
        sections, tables,
        model=model_select,
        max_snippet_chars=max_snippet_chars
    )

    selected_section_names = set(selection.get("selected_sections", []))
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
    selected_tables = [t for t in tables if t.get("selection_id") in selected_table_ids or t["name"] in selected_table_ids]
    used_fallback_sections = False
    used_fallback_tables = False

    # Fallback for robustness when selection stage returns empty.
    if not selected_sections and sections:
        selected_sections = sections[:2]
        used_fallback_sections = True
    if not selected_tables and tables:
        selected_tables = _fallback_select_tables(tables)
        used_fallback_tables = True

    if selected_tables and tables:
        has_material_profile_table = any(_is_material_profile_table(t) for t in selected_tables)
        if not has_material_profile_table:
            profile_candidates = [t for t in tables if _is_material_profile_table(t)]
            if profile_candidates:
                profile_candidates = sorted(
                    profile_candidates,
                    key=lambda t: (_fallback_table_score(t), -int(t.get("length") or 0)),
                    reverse=True,
                )
                best = profile_candidates[0]
                if all(best.get("name") != t.get("name") for t in selected_tables):
                    selected_tables.append(best)

    with open(os.path.join(paper_dir, "llm_selected_files.json"), "w", encoding="utf-8") as f:
        selection_out = dict(selection)
        selection_out["selected_tables"] = [t.get("selection_id") or Path(t["name"]).stem for t in selected_tables]
        selection_out["resolved_selected_table_files"] = [t["name"] for t in selected_tables]
        selection_out["used_fallback_selection"] = used_fallback_sections or used_fallback_tables
        selection_out["fallback"] = {
            "sections": used_fallback_sections,
            "tables": used_fallback_tables,
        }
        json.dump(selection_out, f, ensure_ascii=False, indent=2)

    extract_tables = selected_tables if direct_image_table_input else [
        t for t in selected_tables if str(t.get("table_kind") or "") != "image_backed"
    ]
    context, context_meta = build_context(selected_sections, extract_tables, max_context_chars=max_context_chars)
    skipped_no_explicit_parameters = _should_skip_extraction_no_explicit_parameters(selected_sections, selected_tables)
    if skipped_no_explicit_parameters:
        extracted = _empty_extraction_payload(
            "Skipped LLM extraction because the selected context did not contain explicit parameter values."
        )
        ext_usage = None
        ext_time = 0.0
    else:
        extracted, ext_usage, ext_time = llm_extract(
            context,
            extract_tables,
            model=model_extract,
            max_retries=max_extract_retries,
        )

    enrich_usage = None
    enrich_time = None
    source_enrichment_applied = False
    source_enrichment_error = None
    if enable_source_enrichment and not skipped_no_explicit_parameters:
        try:
            enrich_start = time.perf_counter()
            enrich_resp = _chat_completion_with_retry(
                model=model_extract,
                messages=[
                    {"role": "system", "content": SOURCE_ENRICH_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_source_enrich_prompt(context, extracted)},
                ],
                max_retries=3,
            )
            enrich_time = time.perf_counter() - enrich_start
            enrich_usage = enrich_resp.usage
            enrich_payload = json.loads(enrich_resp.choices[0].message.content)
            extracted = _merge_source_enrichment(extracted, enrich_payload)
            source_enrichment_applied = True
        except Exception as exc:
            # Keep pipeline robust: provenance enrichment is optional.
            source_enrichment_error = str(exc)

    extracted = _strip_internal_keys(extracted)
    extracted = _inject_legacy_compat_views_from_v3(extracted)
    extracted = _drop_legacy_parameter_blocks(extracted)
    with open(os.path.join(paper_dir, "materials_extracted.json"), "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)
    print(
        "Extraction complete with "
        f"{(getattr(ext_usage, 'total_tokens', 0) if ext_usage else 0)+sel_usage.total_tokens} total tokens, "
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
                "input_tokens": getattr(ext_usage, "prompt_tokens", 0) if ext_usage else 0,
                "output_tokens": getattr(ext_usage, "completion_tokens", 0) if ext_usage else 0,
                "total_tokens": getattr(ext_usage, "total_tokens", 0) if ext_usage else 0,
                "time_seconds": ext_time,
                "skipped_no_explicit_parameters": skipped_no_explicit_parameters,
            },
            "source_enrichment": {
                "enabled": bool(enable_source_enrichment),
                "applied": source_enrichment_applied,
                "error": source_enrichment_error,
                "input_tokens": getattr(enrich_usage, "prompt_tokens", 0) if enrich_usage else 0,
                "output_tokens": getattr(enrich_usage, "completion_tokens", 0) if enrich_usage else 0,
                "total_tokens": getattr(enrich_usage, "total_tokens", 0) if enrich_usage else 0,
                "time_seconds": enrich_time,
            },
            "selection_resolution": {
                "used_fallback_sections": used_fallback_sections,
                "used_fallback_tables": used_fallback_tables,
                "resolved_section_files": [s["name"] for s in selected_sections],
                "resolved_table_files": [t["name"] for t in selected_tables],
            },
            "context": context_meta,
        }
    }
