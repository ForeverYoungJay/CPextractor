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
    "common_name": "string or null",
    "chemical_formula": "string or null",
    "material_class": "metal / ceramic / polymer / mineral / intermetallic / composite / other / null",

    "composition": {{
      "basis": "wt% / at% / mol% / fraction / null",
      "elements": [
        {{
          "element": "string",
          "reported_value": "number or null",
          "reported_unit": "wt% / at% / mol% / fraction / null",
          "value_SI": "number or null",
          "unit_SI": "fraction",
          "uncertainty": {{
            "reported_value": "number or null",
            "reported_unit": "same_as_reported / null",
            "notes": "string or null"
          }},
          "notes": "string or null"
        }}
      ],
      "notes": "string or null"
    }},

    "phase_state": "single / multi / unknown / null",

    "phases": [
      {{
        "phase_id": "string",
        "phase_name": "string or null",
        "role": "matrix / precipitate / second_phase / inclusion / pore / null",

        "volume_fraction": {{
          "reported_value": "number or null",
          "reported_unit": "fraction / % / null",
          "value_SI": "number or null",
          "unit_SI": "fraction",
          "uncertainty": {{
            "reported_value": "number or null",
            "reported_unit": "same_as_reported / null",
            "notes": "string or null"
          }},
          "notes": "string or null"
        }},

        "crystal_structure": {{
          "crystal_system": "cubic / hexagonal / tetragonal / orthorhombic / trigonal_rhombohedral / monoclinic / triclinic / unknown / null",
          "bravais_lattice": "P / I / F / R / A / B / C / unknown / null",

          "lattice_family_hint": "FCC / BCC / HCP / diamond_cubic / zincblende / rocksalt_like / fluorite_like / perovskite_like / other / unknown / null",
          "structure_prototype": "fluorite / rocksalt / perovskite / spinel / wurtzite / zincblende / corundum / olivine / laves / custom / unknown / null",
          "space_group": "string or null",

          "lattice_parameters": {{
            "a": {{ "reported_value": "number or null", "reported_unit": "angstrom / nm / m / null", "value_SI": "number or null", "unit_SI": "m" }},
            "b": {{ "reported_value": "number or null", "reported_unit": "angstrom / nm / m / null", "value_SI": "number or null", "unit_SI": "m" }},
            "c": {{ "reported_value": "number or null", "reported_unit": "angstrom / nm / m / null", "value_SI": "number or null", "unit_SI": "m" }},
            "alpha_deg": {{ "reported_value": "number or null", "reported_unit": "deg / null", "value_SI": "number or null", "unit_SI": "rad" }},
            "beta_deg": {{ "reported_value": "number or null", "reported_unit": "deg / null", "value_SI": "number or null", "unit_SI": "rad" }},
            "gamma_deg": {{ "reported_value": "number or null", "reported_unit": "deg / null", "value_SI": "number or null", "unit_SI": "rad" }},
            "c_over_a": {{ "value": "number or null", "notes": "string or null" }},
            "temperature": {{
              "reported_value": "number or null",
              "reported_unit": "K / C / null",
              "value_SI": "number or null",
              "unit_SI": "K"
            }},
            "notes": "string or null"
          }},

          "orientation_convention": {{
            "miller_indexing": "miller / miller_bravais / cartesian / unknown / null",
            "axis_definition": "string or null",
            "handedness": "right / left / unknown / null",
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
    "grain_structure": "single_crystal / polycrystal / bicrystal / oligocrystal / unknown / null",

    "specimen_geometry": {{
      "type": "bulk / sheet / foil / wire / pillar / film / powder / coated / composite / unknown / null",
      "dimensions": "string or null",
      "surface_finish": "polished / etched / as_received / unknown / null",
      "notes": "string or null"
    }},

    "grain_size": {{
      "reported_value": "number or null",
      "reported_unit": "nm / micrometer / mm / m / null",
      "value_SI": "number or null",
      "unit_SI": "m",
      "distribution": "normal / lognormal / bimodal / unknown / null",
      "std_dev": {{ "reported_value": "number or null", "reported_unit": "nm / micrometer / mm / m / null", "value_SI": "number or null", "unit_SI": "m" }},
      "measurement_method": "EBSD / optical / SEM / TEM / XRD / unknown / null",
      "notes": "string or null"
    }},

    "texture_or_orientation": {{
      "type": "single_crystal_orientation / polycrystal_texture / none / unknown / null",
      "method": "XRD / Laue / EBSD / neutron / synchrotron / unknown / null",
      "representation": "euler_bunge / quaternion / axis_angle / rodrigues / pole_figure / ODF / unknown / null",
      "accuracy_deg": "number or null",
      "data_available": "yes / no / partial / null",
      "notes": "string or null"
    }},

    "initial_defect_state": {{
      "dislocation_density": {{
        "reported_value": "number or null",
        "reported_unit": "m^-2 / null",
        "value_SI": "number or null",
        "unit_SI": "m^-2",
        "measurement_method": "TEM / XRD_line_profile / etch_pits / assumed / unknown / null",
        "notes": "string or null"
      }},
      "precipitate_state": "string or null",
      "solute_state": "string or null",
      "porosity": {{
        "reported_value": "number or null",
        "reported_unit": "fraction / % / null",
        "value_SI": "number or null",
        "unit_SI": "fraction",
        "notes": "string or null"
      }},
      "prestrain": {{
        "reported_value": "number or null",
        "reported_unit": "strain / % / null",
        "value_SI": "number or null",
        "unit_SI": "strain",
        "notes": "string or null"
      }},
      "notes": "string or null"
    }},

    "notes": "string or null"
  }},

  "constitutive_model": {{
    "class": "crystal_plasticity / viscoplasticity / continuum_damage / phase_field / cohesive_zone / null",

    "framework": "CPFE / FFT / EVPFFT / VPSC / DAMASK / UMAT / custom / null",

    "implementation": {{
      "code_name": "string or null",
      "platform": "Abaqus / DAMASK / VPSC / custom / other / null",
      "solver_type": "FEM / FFT / spectral / mean_field / null",
      "subroutine": "UMAT / VUMAT / user_material / built_in / null",
      "version": "string or null",
      "repository_or_link": "string or null",
      "element_or_grid": "string or null",
      "notes": "string or null"
    }},

    "kinematics": "finite_strain / small_strain / null",

    "flow_kinetics": {{
      "rate_dependence": "rate_dependent / rate_independent / unknown / null",
      "flow_rule_form": "power_law / overstress / sinh / custom / unknown / null",

      "parameter_convention": {{
        "uses_m": "yes / no / unknown / null",
        "uses_n": "yes / no / unknown / null",
        "notes": "string or null"
      }},

      "equation": "string or null",
      "notes": "string or null"
    }},

    "hardening": {{
      "isotropic_hardening_law": "Voce / Kocks_Mecking / MTS / Bassani_Wu / user_defined / unknown / null",
      "kinematic_hardening_law": "none / Armstrong_Frederick / Chaboche / user_defined / unknown / null",
      "latent_hardening_form": "none / q_ratio / interaction_matrix / user_defined / unknown / null",
      "equation": "string or null",
      "notes": "string or null"
    }},

    "damage_or_fracture": {{
      "included": "yes / no / partial / null",
      "type": "cleavage / cohesive / phase_field / continuum_damage / custom / null",
      "planes": ["string"],
      "equation_or_criterion": "string or null",
      "notes": "string or null"
    }},

    "state_variables": {{
      "includes_dislocation_density": "yes / no / null",
      "includes_backstress": "yes / no / null",
      "includes_twin_volume_fraction": "yes / no / null",
      "includes_damage": "yes / no / null",
      "notes": "string or null"
    }},

    "notes": "string or null"
  }},

  "mechanisms": {{
    "conventions": {{
      "indices": "miller / miller_bravais / cartesian / unknown / null",
      "sign_convention": "standard / author_defined / unknown / null",
      "string_preservation": "keep_as_written / normalize_and_keep / unknown / null",
      "notes": "string or null"
    }},

    "by_phase": [
      {{
        "phase_id": "string",

        "slip": {{
          "lattice": "FCC / BCC / HCP / cubic_other / tetragonal / orthorhombic / other / unknown / null",

          "families": [
            {{
              "family_id": "string",
              "family_name": "string or null",

              "shorthand": "basal / prismatic / pyramidal_ca / {111}<110> / {110}<111> / {100}<110> / custom / unknown / null",

              "system_count": {{
                "theoretical_full": "number or null",
                "reported": "number or null",
                "modeled": "number or null",
                "counting_rule": "unique_by_sign / unique_by_plane / unique_by_direction / author_defined / unknown / null",
                "notes": "string or null"
              }},

              "systems": [
                {{
                  "system_id": "string",
                  "plane": {{
                    "as_written": "string or null",
                    "indices": ["number"],
                    "basis": "hkl / hkil / unknown / null"
                  }},
                  "direction": {{
                    "as_written": "string or null",
                    "indices": ["number"],
                    "basis": "uvw / uvtw / unknown / null"
                  }},
                  "burgers_vector": {{
                    "reported_value": "number or null",
                    "reported_unit": "m / nm / angstrom / null",
                    "value_SI": "number or null",
                    "unit_SI": "m",
                    "notes": "string or null"
                  }},
                  "schmid_tensor_available": "yes / no / null",
                  "notes": "string or null"
                }}
              ],

              "active": "yes / no / partial / null",
              "notes": "string or null"
            }}
          ],

          "notes": "string or null"
        }},

        "twinning": {{
          "families": [
            {{
              "family_id": "string",
              "family_name": "string or null",
              "plane_direction_as_written": "string or null",
              "system_count": {{
                "theoretical_full": "number or null",
                "reported": "number or null",
                "modeled": "number or null",
                "counting_rule": "author_defined / unknown / null",
                "notes": "string or null"
              }},
              "twin_shear": {{
                "reported_value": "number or null",
                "reported_unit": "dimensionless / null",
                "value_SI": "number or null",
                "unit_SI": "dimensionless"
              }},
              "reorientation_rule": "instantaneous / kinetics_based / user_defined / unknown / null",
              "active": "yes / no / partial / null",
              "notes": "string or null"
            }}
          ],
          "notes": "string or null"
        }},

        "cleavage": {{
          "included": "yes / no / partial / null",
          "planes": ["string"],
          "criterion": "max_principal_stress / energy_release / cohesive / user_defined / unknown / null",
          "active": "yes / no / partial / null",
          "notes": "string or null"
        }},

        "notes": "string or null"
      }}
    ],

    "notes": "string or null"
  }},

  "parameters": {{
    "canonical_dictionary": {{
      "elastic": ["C11", "C12", "C13", "C33", "C44", "C55", "C66", "E", "nu", "G", "K"],
      "flow": ["gamma0_ref", "rate_sensitivity_m", "exponent_n", "drag_stress", "drag_coefficient"],
      "strength_and_hardening": ["crss_initial", "crss_saturation", "hardening_h0", "hardening_h1", "hardening_rate", "latent_ratio_q", "interaction_matrix_qab", "backstress_C", "backstress_gamma"],
      "dislocation_based": ["dislocation_density_initial", "hardening_k1", "hardening_k2", "activation_energy_Q", "thermal_softening_beta"],
      "twinning": ["twin_crss_initial", "twin_hardening_h0", "twin_saturation", "twin_reorientation_rate"],
      "fracture_or_damage": ["cleavage_strength", "cohesive_strength", "fracture_energy_Gc", "critical_energy_release_rate", "damage_threshold"],
      "numerical": ["time_step", "tolerance", "max_iterations", "viscosity_regularization"]
    }},

    "registry": [
      {{
        "parameter_id": "string",

        "domain": "elastic / plastic / twinning / damage / thermal / numerical / other",

        "canonical_name": "string or null",
        "user_defined_name": "string or null",
        "symbol": "string or null",
        "definition": "string or null",
        "equation_context": "string or null",

        "reported_value": "number or string or null",
        "reported_unit": "string or null",
        "value_SI": "number or string or null",
        "unit_SI": "string or null",

        "applies_to": {{
          "phase_id": "string or null",

          "scope": "all / mechanism_type / family / system / group / null",

          "mechanism_type": "slip / twinning / cleavage / damage / null",

          "family_id": "string or null",
          "system_ids": ["string"],

          "group_id": "string or null",

          "notes": "string or null"
        }},

        "dependence": {{
          "temperature": {{
            "type": "none / arrhenius / table / piecewise / custom / unknown / null",
            "parameters": [
              {{
                "name": "string",
                "reported_value": "number or null",
                "reported_unit": "string or null",
                "value_SI": "number or null",
                "unit_SI": "string or null",
                "notes": "string or null"
              }}
            ],
            "table_location": {{
              "kind": "figure / table / section / supplement / dataset / code / null",
              "id": "string or null",
              "page": "number or null",
              "notes": "string or null"
            }},
            "notes": "string or null"
          }},

          "strain_rate": {{
            "type": "none / power_law / table / piecewise / custom / unknown / null",
            "parameters": [
              {{
                "name": "string",
                "reported_value": "number or null",
                "reported_unit": "string or null",
                "value_SI": "number or null",
                "unit_SI": "string or null",
                "notes": "string or null"
              }}
            ],
            "table_location": {{
              "kind": "figure / table / section / supplement / dataset / code / null",
              "id": "string or null",
              "page": "number or null",
              "notes": "string or null"
            }},
            "notes": "string or null"
          }}
        }},

        "valid_range": "string or null",

        "source": {{
          "type": "original / adopted / calibrated / inferred / mixed / null",
          "reference_ids": ["string"],
          "adopted_from_reference_ids": ["string"],
          "calibrated_against_reference_ids": ["string"],
          "calibration_method": "manual_fitting / inverse_modeling / optimization / bayesian / null",
          "calibration_targets": ["load_displacement / stress_strain / texture_evolution / r_value / twin_fraction / yield_surface / lattice_strain / damage_metric / null"],
          "validation_targets": ["string"],
          "evidence": {{
            "text": "string or null",
            "location": {{
              "kind": "figure / table / section / supplement / dataset / code / null",
              "id": "string or null",
              "page": "number or null",
              "notes": "string or null"
            }}
          }},
          "notes": "string or null"
        }},

        "location": {{
          "kind": "figure / table / section / supplement / dataset / code / null",
          "id": "string or null",
          "page": "number or null",
          "notes": "string or null"
        }},

        "confidence": "high / medium / low / null",
        "notes": "string or null"
      }}
    ],

    "groups": [
      {{
        "group_id": "string",
        "name": "string or null",
        "description": "string or null",

        "mapping_rule": {{
          "type": "family_wise / system_wise / matrix / user_defined / null",
          "details": "string or null"
        }},

        "members": {{
          "phase_id": "string or null",
          "mechanism_type": "slip / twinning / null",
          "family_id": "string or null",
          "system_ids": ["string"],
          "notes": "string or null"
        }},

        "notes": "string or null"
      }}
    ],

    "notes": "string or null"
  }},

  "code_compatibility": {{
    "targets": [
      {{
        "code": "DAMASK / VPSC / Abaqus_UMAT / Abaqus_VUMAT / FFT_CP / custom / null",
        "version": "string or null",

        "parameter_mapping": [
          {{
            "parameter_id": "string",
            "code_parameter_name": "string",
            "code_scope": "global / phase / family / system / group / unknown / null",
            "expected_units": "string or null",
            "transform": {{
              "type": "identity / unit_convert / invert / scale / custom / null",
              "expression": "string or null",
              "notes": "string or null"
            }},
            "notes": "string or null"
          }}
        ],

        "mechanism_mapping": {{
          "slip_family_map": [
            {{
              "family_id": "string",
              "code_family_name": "string",
              "notes": "string or null"
            }}
          ],
          "system_map": [
            {{
              "system_id": "string",
              "code_system_index": "number or string",
              "notes": "string or null"
            }}
          ],
          "notes": "string or null"
        }},

        "known_gaps": ["string"],
        "notes": "string or null"
      }}
    ],

    "notes": "string or null"
  }},

  "loading_and_environment": {{
    "loading_mode": "uniaxial_tension / compression / shear / indentation / cyclic / creep / torsion / bending / null",
    "stress_state": "uniaxial / plane_strain / biaxial / triaxial / multiaxial / unknown / null",
    "control": "strain_controlled / stress_controlled / displacement_controlled / mixed / unknown / null",
    "strain_measure": "engineering / true / logarithmic / unknown / null",
    "stress_measure": "engineering / true / cauchy / PK1 / PK2 / unknown / null",

    "loading_path": {{
      "loading_direction": "RD / TD / ND / crystal_axis / arbitrary_vector / unknown / null",
      "crystal_axis": "string or null",
      "description": "string or null"
    }},

    "strain_rate": {{
      "reported_value": "number or null",
      "reported_unit": "s^-1 / null",
      "value_SI": "number or null",
      "unit_SI": "s^-1",
      "notes": "string or null"
    }},

    "temperature": {{
      "reported_value": "number or null",
      "reported_unit": "K / C / null",
      "value_SI": "number or null",
      "unit_SI": "K",
      "history": "isothermal / non_isothermal / unknown / null",
      "notes": "string or null"
    }},

    "environment": {{
      "medium": "air / vacuum / liquid / inert_gas / hydrogen / unknown / null",
      "pressure": {{
        "reported_value": "number or null",
        "reported_unit": "Pa / MPa / bar / null",
        "value_SI": "number or null",
        "unit_SI": "Pa"
      }},
      "humidity": {{
        "reported_value": "number or null",
        "reported_unit": "%RH / null"
      }},
      "notes": "string or null"
    }},

    "indentation": {{
      "indenter_type": "berkovich / spherical / cono_spherical / vickers / knoop / custom / unknown / null",

      "tip_radius": {{
        "reported_value": "number or null",
        "reported_unit": "nm / micrometer / null",
        "value_SI": "number or null",
        "unit_SI": "m"
      }},

      "max_depth": {{
        "reported_value": "number or null",
        "reported_unit": "nm / micrometer / null",
        "value_SI": "number or null",
        "unit_SI": "m"
      }},

      "max_load": {{
        "reported_value": "number or null",
        "reported_unit": "mN / N / null",
        "value_SI": "number or null",
        "unit_SI": "N"
      }},

      "hold_time_at_peak": {{
        "reported_value": "number or null",
        "reported_unit": "s / null",
        "value_SI": "number or null",
        "unit_SI": "s"
      }},

      "loading_rate": {{
        "type": "displacement_rate / load_rate / strain_rate / unknown / null",
        "reported_value": "number or null",
        "reported_unit": "nm/s / um/s / mN/s / N/s / s^-1 / null",
        "value_SI": "number or null",
        "unit_SI": "m/s / N/s / s^-1 / null",
        "notes": "string or null"
      }},

      "friction_coefficient": {{
        "value": "number or null",
        "notes": "string or null"
      }},

      "notes": "string or null"
    }},

    "calibration_domain": {{
      "primary_measured_response": "load_displacement / stress_strain / creep_curve / cyclic_hysteresis / null",
      "calibration_strain_range": {{
        "min": "number or null",
        "max": "number or null",
        "unit": "strain / % / null"
      }},
      "notes": "string or null"
    }},

    "location": {{
      "kind": "figure / table / section / supplement / dataset / code / null",
      "id": "string or null",
      "page": "number or null",
      "notes": "string or null"
    }},

    "notes": "string or null"
  }},

  "numerical_settings": {{
    "rve_type": "single_element / taylor / voronoi_grains / fft_voxels / fe_mesh / null",

    "boundary_conditions": {{
      "type": "PBC / mixed / free_surface / unknown / null",
      "description": "string or null"
    }},

    "discretization": {{
      "method": "FEM / FFT / mean_field / unknown / null",
      "orientation_representation": "euler_bunge / quaternion / rodrigues / unknown / null",
      "mesh_element_type": "string or null",
      "element_count": "number or null",
      "integration_points_per_element": "number or null",
      "voxel_resolution": "string or null",
      "num_grains": "number or null",
      "notes": "string or null"
    }},

    "solver": {{
      "integration_scheme": "implicit / explicit / semi_implicit / unknown / null",
      "time_step_or_increment": "string or null",
      "max_increments": "number or null",
      "convergence_tolerance": "string or null",
      "max_iterations": "number or null",
      "line_search": "yes / no / null",
      "notes": "string or null"
    }},

    "outputs_requested": {{
      "load_displacement": "yes / no / null",
      "stress_strain": "yes / no / null",
      "lattice_rotation": "yes / no / null",
      "texture_evolution": "yes / no / null",
      "twin_fraction": "yes / no / null",
      "damage_indicators": "yes / no / null",
      "other": ["string"],
      "notes": "string or null"
    }},

    "notes": "string or null"
  }},

  "data_links": {{
    "stress_strain_data": {{
      "available": "yes / no / partial / null",
      "type": "engineering / true / null",
      "format": "csv / table / image / digitized / null",
      "location": {{
        "kind": "figure / table / section / supplement / dataset / code / null",
        "id": "string or null",
        "page": "number or null",
        "notes": "string or null"
      }},
      "notes": "string or null"
    }},

    "load_displacement_data": {{
      "available": "yes / no / partial / null",
      "format": "csv / table / image / digitized / null",
      "location": {{
        "kind": "figure / table / section / supplement / dataset / code / null",
        "id": "string or null",
        "page": "number or null",
        "notes": "string or null"
      }},
      "notes": "string or null"
    }},

    "texture_data": {{
      "available": "yes / no / partial / null",
      "format": "ODF / pole_figure / EBSD / XRD / null",
      "location": {{
        "kind": "figure / table / section / supplement / dataset / code / null",
        "id": "string or null",
        "page": "number or null",
        "notes": "string or null"
      }},
      "notes": "string or null"
    }},

    "other_data": [
      {{
        "name": "string",
        "available": "yes / no / partial / null",
        "format": "string or null",
        "location": {{
          "kind": "figure / table / section / supplement / dataset / code / null",
          "id": "string or null",
          "page": "number or null",
          "notes": "string or null"
        }},
        "notes": "string or null"
      }}
    ],

    "notes": "string or null"
  }},

  "fit_quality": {{
    "fit_targets": ["load_displacement / stress_strain / texture_evolution / r_value / yield_surface / twin_fraction / lattice_strain / damage_metric / null"],
    "reported_metrics": [
      {{
        "name": "RMSE / R2 / MAE / max_error / NRMSE / null",
        "value": "number or string or null",
        "unit": "string or null",
        "location": {{
          "kind": "figure / table / section / supplement / dataset / code / null",
          "id": "string or null",
          "page": "number or null",
          "notes": "string or null"
        }},
        "notes": "string or null"
      }}
    ],
    "qualitative_assessment": "good / medium / poor / mixed / null",
    "validated_on_independent_case": "yes / no / partial / null",
    "validation_cases": ["string"],
    "notes": "string or null"
  }},

  "references": [
    {{
      "ref_id_in_paper": "string",
      "global_key": "doi_or_bibkey_or_null",
      "type": "paper / book / dataset / code / thesis / standard / null",
      "citation_string": "string or null",
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


def _schema_skeleton_from_prompt_template(template: str) -> Dict[str, Any]:
    start_marker = "Extract the following schema from the paper excerpt:"
    end_marker = "Paper excerpt:"

    start = template.find(start_marker)
    text = template[start + len(start_marker):] if start >= 0 else template
    end = text.find(end_marker)
    if end >= 0:
        text = text[:end]

    text = text.strip().replace("{{", "{").replace("}}", "}")
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


EXTRACT_SCHEMA_SKELETON = _schema_skeleton_from_prompt_template(EXTRACT_USER_PROMPT_TEMPLATE)


def _validate_extracted_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["payload is not an object"]

    for key in EXTRACT_SCHEMA_SKELETON.keys():
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    if not isinstance(payload.get("elastic_parameters", {}).get("constants", []), list):
        errors.append("elastic_parameters.constants must be a list")
    if not isinstance(payload.get("plastic_parameters", {}).get("parameters", []), list):
        errors.append("plastic_parameters.parameters must be a list")

    return errors


def _build_extract_prompt(context: str) -> str:
    # Avoid str.format() here because the template contains literal braces such as crystallographic {111}.
    return EXTRACT_USER_PROMPT_TEMPLATE.replace("{context}", context)




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

    with open(os.path.join(paper_dir, "materials_extracted.json"), "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)
    print(
        "Extraction complete with "
        f"{ext_usage.total_tokens} total tokens, "
        f"{ext_usage.completion_tokens} completion tokens, "
        f"in {ext_time:.2f} seconds."
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
