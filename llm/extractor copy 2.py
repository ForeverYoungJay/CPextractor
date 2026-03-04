import os, re, json, glob
from typing import List, Dict, Any
from openai import OpenAI
import time

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
You are an expert in crystal plasticity simulations.

Your task is to select the MINIMUM set of files required to extract:
- material identity and crystal structure
- slip systems and deformation mechanisms
- crystal plasticity model type (rate-dependent / CPFE / FFT / VPSC)
- elastic constants (e.g., C11, C12, C44)
- plastic parameters (tau0, h0, n, gamma0, etc.)
- calibration and parameter provenance ("adopted from", "fitted to")

Priority rules:
1. Tables containing material parameters are HIGHEST priority
2. Methodology / Simulation sections are critical
3. Abstract alone is NEVER sufficient

Return ONLY JSON:
{
  "selected_sections": [...],
  "selected_tables": [...],
}
"""


SELECTION_USER_PROMPT_TEMPLATE = """
You are given a paper split into files.

Your task:
Select the minimum set of section files and table files needed to extract:
- materials, crystal structure, composition (wt%), processing
- simulation info: software, method, model type, parameters, conditions

Here are the available sections:
{sections_catalog}

Here are the available tables:
{tables_catalog}

Return only JSON.
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
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
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
You are a crystal plasticity parameter extraction engine.

Extract information STRICTLY from the given text.
Do NOT guess.
Do NOT infer missing values.

Rules:
- Return ONLY valid JSON.
- Use null if information is not stated.
- Parameters must include units if available.
- Distinguish elastic vs plastic parameters.
- Capture parameter provenance if stated.
"""


EXTRACT_USER_PROMPT_TEMPLATE = """
Extract the following schema from the paper excerpt:

{{
  "material": {{
    "name": "string or null",
    "chemical_formula": "string or null",
    "crystal_structure": {{
  "crystal_system": "cubic / hexagonal / tetragonal / orthorhombic/ (Trigonal/Rhombohedral)/ monoclinic/ triclinic",
  "lattice_type": "FCC / BCC / HCP / null"
}},
    "phase": "single / multi / null",
    "notes": "string or null"
  }},

  "microstructure": {{
    "grain_structure": "single crystal / polycrystal / bicrystal / null",
    "grain_size": {{
      "value": "number or null",
      "unit": "micrometer / mm / null",
      "distribution": "string or null",
      "notes": "string or null"
    }},
    "orientation_texture": {{
      "description": "string or null",
      "texture_type": "ODF / pole_figure / EBSD / none / null",
      "texture_data_available": "yes / no / null",
      "data_location": "figure/table/section id or null",
      "notes": "string or null"
    }},
    "initial_defect_state": {{
      "dislocation_density": {{
        "value": "number or null",
        "unit": "m^-2 / null",
        "notes": "string or null"
      }},
      "precipitate_state": "string or null",
      "solute_state": "string or null",
      "prestrain": {{
        "value": "number or null",
        "unit": "strain / % / null",
        "notes": "string or null"
      }},
      "notes": "string or null"
    }},
    "notes": "string or null"
  }},

  "constitutive_model": {{
    "class": "crystal plasticity / phase-field / continuum damage / null",
    "framework": "CPFE / FFT / VPSC / UMAT / DAMASK / EVPFFT / null",
    "implementation": {{
      "code_name": "string or null",
      "platform": "Abaqus / DAMASK / VPSC / custom / other / null",
      "subroutine_or_solver": "UMAT / VUMAT / spectral / FEM / null",
      "version": "string or null",
      "notes": "string or null"
    }},
    "kinematics": "finite strain / small strain / null",
    "rate_dependence": "rate-dependent / rate-independent / null",
    "single_or_poly": "single_crystal / polycrystal / null",
    "homogenization_assumption": "Taylor / self-consistent / full-field / mean-field / null",
    "notes": "string or null"
  }},

  "elastic_parameters": {{
    "symmetry": "isotropic / cubic / hexagonal / orthotropic / null",
    "constants": [
      {{
        "canonical_name": "C11 / C12 / C13 / C33 / C44 / E / nu / G / K / null",
        "symbol": "string or null",
        "description": "string or null",
        "value": "number or null",
        "unit": "string or null",
        "temperature_dependent": "yes / no / null",
        "source": {{
          "type": "original / adopted / calibrated / null",
          "reference_ids": ["string"],
          "calibration_method": "string or null",
          "validation_targets": ["string"]
        }},
        "notes": "string or null"
      }}
    ],
    "notes": "string or null"
  }},

  "deformation_conditions": {{
    "loading_mode": "uniaxial tension / compression / shear / indentation / cyclic / creep / torsion / bending / null",
    "stress_state": "uniaxial / plane_strain / biaxial / triaxial / multiaxial / null",
    "loading_path": {{
      "control": "strain-controlled / stress-controlled / displacement-controlled / mixed / null",
      "loading_direction": "RD / TD / ND / crystal_axis / null",
      "crystal_axis": "e.g., [001] / [011] / null",
      "description": "string or null"
    }},
    "strain_rate": {{
      "value": "number or null",
      "unit": "s^-1 / null",
      "range": "string or null"
    }},
    "temperature": {{
      "value": "number or null",
      "unit": "K / C / null",
      "history": "isothermal / non-isothermal / null"
    }},
    "environment": {{
      "pressure": {{
        "value": "number or null",
        "unit": "Pa / MPa / bar / null",
        "notes": "string or null"
      }},
      "medium": "air / vacuum / liquid / inert gas / hydrogen / null",
      "notes": "string or null"
    }},
    "calibration_strain_range": {{
      "min": "number or null",
      "max": "number or null",
      "unit": "strain / % / null",
      "notes": "string or null"
    }},
    "notes": "string or null"
  }},

  "slip_twin_systems": {{
    "lattice": "FCC / BCC / HCP / null",
    "slip_families": [
      {{
        "family_name": "basal / prismatic / pyramidal_ca / {{111}}<110> / {{110}}<111> / null",
        "plane_direction": "string or null",
        "num_systems": "number or null",
        "active": "yes / no / null",
        "notes": "string or null"
      }}
    ],
    "twinning_families": [
      {{
        "family_name": "extension_twin / contraction_twin / null",
        "plane_direction": "string or null",
        "num_systems": "number or null",
        "active": "yes / no / null",
        "reorientation_rule": "string or null",
        "notes": "string or null"
      }}
    ],
    "notes": "string or null"
  }},

  "plastic_parameters": {{
    "flow_rule": "power-law / viscoplastic / null",
    "parameters": [
      {{
      "canonical_name": "gamma0_ref /
rate_sensitivity_m /
exponent_n /
crss_initial /
crss_saturation /
hardening_h0 /
hardening_h1 /
hardening_rate /
latent_ratio_q /
interaction_matrix_qab /
backstress_C /
backstress_gamma /
drag_stress /
drag_coefficient /
twin_crss_initial /
twin_hardening_h0 /
twin_saturation /
twin_reorientation_rate /
dislocation_density_initial /
hardening_k1 /
hardening_k2 /
activation_energy_Q /
thermal_softening_beta/null",
        "symbol": "string or null",
        "description": "string or null",
        "value": "number or string or null",
        "unit": "string or null",
        "applies_to": {{
          "mechanism": "basal_slip / prismatic_slip / pyramidal_slip / twinning / all_slip / all_twin / null",
          "family_name": "string or null",
          "system_count": "number or null"
        }},
        "temperature_dependent": "yes / no / null",
        "strain_rate_dependent": "yes / no / null",
        "valid_range": "string or null",
        "source": {{
          "type": "original / adopted / calibrated / null",
          "reference_ids": ["string"],
          "calibration_method": "manual fitting / inverse modeling / optimization / bayesian / null",
          "calibration_targets": ["stress-strain", "texture evolution", "r-value", "twin fraction", "yield surface", "null"],
          "validation_targets": ["string"]
        }},
        "notes": "string or null"
      }}
    ],
    "notes": "string or null"
  }},

  "numerical_settings": {{
    "rve_type": "single_element / taylor / voronoi_grains / fft_voxels / fe_mesh / null",
    "boundary_conditions": {{
      "type": "PBC / mixed / free_surface / null",
      "description": "string or null"
    }},
    "mesh_or_voxels": {{
      "element_type": "string or null",
      "element_count": "number or null",
      "voxel_resolution": "string or null",
      "notes": "string or null"
    }},
    "num_grains": "number or null",
    "integration_scheme": "implicit / explicit / semi-implicit / null",
    "time_step_or_increment": "string or null",
    "convergence_settings": "string or null",
    "notes": "string or null"
  }},

  "fit_quality": {{
    "reported_metrics": [
      {{
        "name": "RMSE / R2 / MAE / max_error / null",
        "value": "number or string or null",
        "unit": "string or null",
        "extraction_location": "table/figure/section id or null",
        "notes": "string or null"
      }}
    ],
    "qualitative_assessment": "good / medium / poor / null",
    "validated_on_independent_case": "yes / no / null",
    "validation_cases": ["string"],
    "notes": "string or null"
  }},
}}


Paper excerpt:
----------------
{context}
----------------

Return JSON only.
"""




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

def llm_extract(context: str, model: str) -> Dict[str, Any]:
    start = time.perf_counter()
    prompt = EXTRACT_USER_PROMPT_TEMPLATE.format(context=context)

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )
    elapsed = time.perf_counter() - start
    usage = resp.usage
    return json.loads(resp.choices[0].message.content), usage, elapsed

def run_llm_on_paper_dir(
    paper_dir: str,
    model_select: str,
    model_extract: str,
    max_snippet_chars: int,
    max_context_chars: int,
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

    with open(os.path.join(paper_dir, "llm_selected_files.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    context = build_context(selected_sections, selected_tables, max_context_chars=max_context_chars)
    extracted, ext_usage, ext_time = llm_extract(context, model=model_extract)

    with open(os.path.join(paper_dir, "materials_extracted.json"), "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

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
