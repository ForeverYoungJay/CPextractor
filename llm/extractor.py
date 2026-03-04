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
  "schema_version": "1.1.0",
  "record_id": "string or null",

  "applicability": {{
    "primary_intent": "crystal_plasticity_parameter_extraction",
    "crystal_plasticity_applicable": "yes / partial / no / null",
    "why_not_applicable": "string or null",
    "notes": "string or null"
  }},

  "source_document": {{
    "title": "string or null",
    "authors": ["string"],
    "year": "number or null",
    "journal_or_venue": "string or null",
    "doi": "string or null",
    "url": "string or null",
    "extraction_method": "manual / semi_automatic / automatic / null",
    "extraction_notes": "string or null"
  }},

  "material": {{
    "name": "string or null",
    "chemical_formula": "string or null",
    "phase": "single / multi / null",
    "phases": [
      {{
        "phase_id": "string",
        "phase_name": "string or null",
        "role": "matrix / precipitate / inclusion / transformed_product / other / null",

        "volume_fraction": {{
          "value_SI": "number or null",
          "unit_SI": "fraction",
          "reported_value": "number or null",
          "reported_unit": "fraction / % / null",
          "notes": "string or null"
        }},

        "crystal_structure": {{
          "crystal_system": "cubic / hexagonal / tetragonal / orthorhombic / trigonal_rhombohedral / monoclinic / triclinic / null",
          "lattice_type": "FCC / BCC / HCP / other / null",
          "space_group": "string or null",

          "lattice_parameters": {{
            "a": {{ "value_SI": "number or null", "unit_SI": "m", "reported_value": "number or null", "reported_unit": "m / nm / angstrom / null", "notes": "string or null" }},
            "b": {{ "value_SI": "number or null", "unit_SI": "m", "reported_value": "number or null", "reported_unit": "m / nm / angstrom / null", "notes": "string or null" }},
            "c": {{ "value_SI": "number or null", "unit_SI": "m", "reported_value": "number or null", "reported_unit": "m / nm / angstrom / null", "notes": "string or null" }},
            "alpha": {{ "value_SI": "number or null", "unit_SI": "rad", "reported_value": "number or null", "reported_unit": "deg / rad / null", "notes": "string or null" }},
            "beta":  {{ "value_SI": "number or null", "unit_SI": "rad", "reported_value": "number or null", "reported_unit": "deg / rad / null", "notes": "string or null" }},
            "gamma": {{ "value_SI": "number or null", "unit_SI": "rad", "reported_value": "number or null", "reported_unit": "deg / rad / null", "notes": "string or null" }}
          }},

          "hcp_geometry": {{
            "miller_convention": "miller / miller_bravais / null",
            "c_over_a": {{ "value": "number or null", "notes": "string or null" }},
            "notes": "string or null"
          }},

          "notes": "string or null"
        }},

        "notes": "string or null"
      }}
    ],
    "notes": "string or null"
  }},

  "microstructure": {{
    "grain_structure": "single_crystal / polycrystal / bicrystal / null",

    "grain_size": {{
      "value_SI": "number or null",
      "unit_SI": "m",
      "reported_value": "number or null",
      "reported_unit": "micrometer / mm / m / null",
      "distribution": "string or null",
      "measurement_method": "EBSD / optical / TEM / XRD / other / null",
      "extraction_location": "figure/table/section id or null",
      "notes": "string or null"
    }},

    "orientation_texture": {{
      "description": "string or null",
      "texture_type": "ODF / pole_figure / EBSD / none / null",
      "texture_data_available": "yes / no / null",
      "orientation_representation": "euler_bunge / quaternion / rodrigues / axis_angle / null",
      "sample_frame_definition": "RD/TD/ND definition or null",
      "data_location": "figure/table/section id or null",
      "notes": "string or null"
    }},

    "initial_defect_state": {{
      "dislocation_density": {{
        "value_SI": "number or null",
        "unit_SI": "m^-2",
        "reported_value": "number or null",
        "reported_unit": "m^-2 / cm^-2 / null",
        "measurement_method": "XRD / TEM / EBSD-KAM / assumed / other / null",
        "notes": "string or null"
      }},
      "precipitate_state": "string or null",
      "solute_state": "string or null",
      "prestrain": {{
        "value_SI": "number or null",
        "unit_SI": "strain",
        "reported_value": "number or null",
        "reported_unit": "strain / % / null",
        "notes": "string or null"
      }},
      "notes": "string or null"
    }},

    "notes": "string or null"
  }},

  "constitutive_model": {{
    "class": "crystal_plasticity / phase_field / continuum_damage / other / null",
    "framework": "CPFE / FFT / EVPFFT / DAMASK / VPSC / UMAT / other / null",

    "implementation": {{
      "code_name": "string or null",
      "platform": "Abaqus / DAMASK / VPSC / custom / other / null",
      "subroutine_or_solver": "UMAT / VUMAT / spectral / FEM / other / null",
      "version": "string or null",
      "repository_or_link": "string or null",
      "notes": "string or null"
    }},

    "kinematics": "finite_strain / small_strain / null",

    "flow_kinetics": {{
      "rate_dependence": "rate_dependent / rate_independent / null",
      "flow_rule_form": "power_law / overstress / arrhenius / tabulated / user_defined / null",
      "uses_m_or_n": "m / n / both / unclear / null",
      "notes": "string or null"
    }},

    "hardening": {{
      "isotropic_hardening_law": "Voce / Kocks_Mecking / MTS / Bassani_Wu / tabulated / user_defined / null",
      "kinematic_hardening_law": "none / Armstrong_Frederick / Chaboche / user_defined / null",
      "latent_hardening_form": "none / q_ratio / interaction_matrix / user_defined / null",
      "notes": "string or null"
    }},

    "homogenization_assumption": "Taylor / self_consistent / full_field / mean_field / null",

    "state_variables": {{
      "includes_dislocation_density": "yes / no / null",
      "includes_backstress": "yes / no / null",
      "includes_twin_volume_fraction": "yes / no / null",
      "includes_damage": "yes / no / null",
      "notes": "string or null"
    }},

    "notes": "string or null"
  }},

  "elastic_parameters": {{
    "context": "single_crystal / effective_polycrystal / null",
    "symmetry": "isotropic / cubic / hexagonal / tetragonal / orthorhombic / monoclinic / triclinic / null",

    "constants": [
      {{
        "phase_id": "string or null",
        "canonical_name": "C11 / C12 / C13 / C33 / C44 / C55 / C66 / E / nu / G / K / null",
        "symbol": "string or null",
        "description": "string or null",

        "value_SI": "number or null",
        "unit_SI": "Pa",
        "reported_value": "number or null",
        "reported_unit": "Pa / MPa / GPa / null",

        "temperature_dependence": {{
          "type": "none / constant / linear / arrhenius / table / piecewise / user_defined / null",
          "parameters": [
            {{
              "name": "string or null",
              "value_SI": "number or null",
              "unit_SI": "string or null",
              "reported_value": "number or null",
              "reported_unit": "string or null",
              "notes": "string or null"
            }}
          ],
          "table_location": "figure/table/section id or null",
          "notes": "string or null"
        }},

        "source": {{
          "type": "original / adopted / calibrated / null",
          "reference_ids": ["string"],
          "calibration_method": "string or null",
          "validation_targets": ["string"]
        }},

        "extraction_location": "figure/table/section id or null",
        "notes": "string or null"
      }}
    ],

    "notes": "string or null"
  }},

  "deformation_conditions": {{
    "loading_mode": "uniaxial_tension / compression / shear / indentation / cyclic / creep / torsion / bending / null",
    "stress_state": "uniaxial / plane_strain / biaxial / triaxial / multiaxial / null",

    "strain_measure": "engineering / true / logarithmic / null",
    "stress_measure": "engineering / true / cauchy / pk1 / pk2 / null",

    "loading_path": {{
      "control": "strain_controlled / stress_controlled / displacement_controlled / mixed / null",
      "loading_direction": "RD / TD / ND / crystal_axis / custom / null",
      "crystal_axis": "e.g., [001] / [011] / null",
      "description": "string or null"
    }},

    "strain_rate": {{
      "value_SI": "number or null",
      "unit_SI": "s^-1",
      "reported_value": "number or null",
      "reported_unit": "s^-1 / null",
      "range": "string or null"
    }},

    "temperature": {{
      "value_SI": "number or null",
      "unit_SI": "K",
      "reported_value": "number or null",
      "reported_unit": "K / C / null",
      "history": "isothermal / non_isothermal / null"
    }},

    "environment": {{
      "pressure": {{
        "value_SI": "number or null",
        "unit_SI": "Pa",
        "reported_value": "number or null",
        "reported_unit": "Pa / MPa / bar / null",
        "notes": "string or null"
      }},
      "medium": "air / vacuum / liquid / inert_gas / hydrogen / null",
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

  "mechanisms": {{
    "slip_twin": {{
      "by_phase": [
        {{
          "phase_id": "string",
          "lattice": "FCC / BCC / HCP / other / null",
          "crystal_frame_definition": "string or null",

          "slip_families": [
            {{
              "family_id": "string or null",
              "family_name": "basal / prismatic / pyramidal_ca / {111}<110> / {110}<111> / other / null",
              "plane_direction": "string or null",
              "num_systems": "number or null",
              "active": "yes / no / null",

              "systems_explicit": [
                {{
                  "system_id": "string or null",
                  "plane": "string or null",
                  "direction": "string or null",
                  "notes": "string or null"
                }}
              ],

              "notes": "string or null"
            }}
          ],

          "twinning_families": [
            {{
              "family_id": "string or null",
              "family_name": "extension_twin / contraction_twin / other / null",
              "plane_direction": "string or null",
              "num_systems": "number or null",
              "active": "yes / no / null",
              "reorientation_rule": "string or null",

              "twin_shear": {{
                "value_SI": "number or null",
                "unit_SI": "dimensionless",
                "reported_value": "number or null",
                "reported_unit": "dimensionless / null",
                "notes": "string or null"
              }},

              "notes": "string or null"
            }}
          ],

          "notes": "string or null"
        }}
      ],
      "notes": "string or null"
    }},

    "extensions": [
      {{
        "type": "creep / transformation / damage / diffusion_climb / irradiation / user_defined / null",
        "description": "string or null",
        "parameters": [
          {{
            "name": "string or null",
            "symbol": "string or null",
            "definition": "string or null",
            "value_SI": "number or string or null",
            "unit_SI": "string or null",
            "reported_value": "number or string or null",
            "reported_unit": "string or null",
            "applies_to": {{
              "phase_id": "string or null",
              "mechanism": "string or null"
            }},
            "source": {{
              "type": "original / adopted / calibrated / null",
              "reference_ids": ["string"],
              "calibration_method": "string or null",
              "calibration_targets": ["string"],
              "validation_targets": ["string"]
            }},
            "extraction_location": "figure/table/section id or null",
            "notes": "string or null"
          }}
        ],
        "notes": "string or null"
      }}
    ]
  }},

  "plastic_parameters": {{
    "parameters": [
      {{
        "canonical_name": "gamma0_ref / rate_sensitivity_m / exponent_n / crss_initial / crss_saturation / hardening_h0 / hardening_h1 / hardening_rate / latent_ratio_q / interaction_matrix_qab / backstress_C / backstress_gamma / drag_stress / drag_coefficient / twin_crss_initial / twin_hardening_h0 / twin_saturation / twin_reorientation_rate / dislocation_density_initial / hardening_k1 / hardening_k2 / activation_energy_Q / thermal_softening_beta / user_defined / null",
        "user_defined_name": "string or null",

        "symbol": "string or null",
        "definition": "string or null",
        "description": "string or null",

        "value_SI": "number or string or null",
        "unit_SI": "string or null",
        "reported_value": "number or string or null",
        "reported_unit": "string or null",

        "applies_to": {{
          "phase_id": "string or null",
          "mechanism": "slip / twinning / all_slip / all_twin / all_mechanisms / user_defined / null",
          "family_id": "string or null",
          "family_name": "string or null",
          "system_ids": ["string"],
          "system_count": "number or null"
        }},

        "temperature_dependence": {{
          "type": "none / constant / linear / arrhenius / table / piecewise / user_defined / null",
          "parameters": [
            {{
              "name": "string or null",
              "value_SI": "number or null",
              "unit_SI": "string or null",
              "reported_value": "number or null",
              "reported_unit": "string or null",
              "notes": "string or null"
            }}
          ],
          "table_location": "figure/table/section id or null",
          "notes": "string or null"
        }},

        "strain_rate_dependence": {{
          "type": "none / power_law / table / piecewise / user_defined / null",
          "parameters": [
            {{
              "name": "string or null",
              "value_SI": "number or null",
              "unit_SI": "string or null",
              "reported_value": "number or null",
              "reported_unit": "string or null",
              "notes": "string or null"
            }}
          ],
          "table_location": "figure/table/section id or null",
          "notes": "string or null"
        }},

        "valid_range": "string or null",

        "source": {{
          "type": "original / adopted / calibrated / null",
          "reference_ids": ["string"],
          "calibration_method": "manual_fitting / inverse_modeling / optimization / bayesian / null",
          "calibration_targets": ["stress_strain", "texture_evolution", "r_value", "twin_fraction", "yield_surface", "other", "null"],
          "validation_targets": ["string"]
        }},

        "extraction_location": "figure/table/section id or null",
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

    "discretization": {{
      "orientation_representation": "euler_bunge / quaternion / rodrigues / axis_angle / null",
      "mesh_element_type": "string or null",
      "element_count": "number or null",
      "voxel_resolution": "string or null",
      "integration_points_per_element": "number or null",
      "num_grains": "number or null",
      "notes": "string or null"
    }},

    "solver": {{
      "integration_scheme": "implicit / explicit / semi_implicit / null",
      "time_step_or_increment": "string or null",
      "max_increments": "number or null",
      "convergence_tolerance": "string or null",
      "max_iterations": "number or null",
      "line_search": "yes / no / null",
      "notes": "string or null"
    }},

    "outputs_requested": {{
      "stress_strain": "yes / no / null",
      "lattice_rotation": "yes / no / null",
      "texture_evolution": "yes / no / null",
      "twin_fraction": "yes / no / null",
      "other": ["string"],
      "notes": "string or null"
    }},

    "notes": "string or null"
  }},

  "experimental_data_links": {{
    "stress_strain_data": {{
      "available": "yes / no / null",
      "type": "engineering / true / null",
      "format": "csv / xlsx / json / digitized / null",
      "location": "supplement/figure/table/section id or null",
      "digitized_from_figure": "yes / no / null",
      "notes": "string or null"
    }},
    "texture_data": {{
      "available": "yes / no / null",
      "type": "ODF / pole_figure / EBSD / null",
      "format": "ctf / ang / odf / image / null",
      "location": "supplement/figure/table/section id or null",
      "notes": "string or null"
    }},
    "other_data": [
      {{
        "name": "string or null",
        "available": "yes / no / null",
        "format": "string or null",
        "location": "string or null",
        "notes": "string or null"
      }}
    ],
    "notes": "string or null"
  }},

  "fit_quality": {{
    "fit_targets": ["stress_strain", "texture_evolution", "r_value", "twin_fraction", "yield_surface", "other", "null"],
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

  "references": [
    {{
      "reference_id": "string",
      "type": "paper / dataset / thesis / report / code / other / null",
      "citation": "string or null",
      "doi": "string or null",
      "url": "string or null",
      "notes": "string or null"
    }}
  ],

  "global_notes": "string or null"
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
