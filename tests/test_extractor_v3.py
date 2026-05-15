import unittest
import sys
import types


openai_stub = types.ModuleType("openai")


class _OpenAIStub:
    def __init__(self, *args, **kwargs):
        pass


openai_stub.OpenAI = _OpenAIStub
sys.modules.setdefault("openai", openai_stub)

fulltext_parser_stub = types.ModuleType("elsevier.fulltext_parser")


def _download_table_image_stub(*args, **kwargs):
    return None


fulltext_parser_stub.download_table_image = _download_table_image_stub
sys.modules.setdefault("elsevier.fulltext_parser", fulltext_parser_stub)

from llm.extractor import (
    EXTRACT_USER_PROMPT_TEMPLATE,
    EXTRACT_SCHEMA_SKELETON,
    _augment_selected_equations,
    _inject_legacy_compat_views_from_v3,
    _merge_source_enrichment,
    _normalize_equation_references_in_payload,
    _render_table_rows_with_alignment,
    _table_json_full_text,
)


class ExtractorV3Tests(unittest.TestCase):
    def test_render_table_rows_with_alignment_expands_multirow_header_and_inherits_grouped_cells(self):
        rows = [
            ["Temp.", "Grain size", "Model parameters"],
            ["", "", "γ ̇ 0", "f ̇ 0", "γ t w", "n", "m", "C R S S b a s a l", "C R S S R a t i o"],
            ["298 K", "9 μ m", "0.001", "0.02", "0.129", "0.08", "0.04", "9.5", "1:3:4:5.5:2"],
            ["5 μ m", "0.001", "0.02", "0.129", "0.05", "0.04", "12.5", "1:3:4:5.5:2"],
        ]

        rendered = _render_table_rows_with_alignment(rows, max_rows=len(rows), max_cells_per_row=10)

        self.assertIn("c3=Model parameters / γ ̇ 0", rendered[0])
        self.assertIn("c9=C R S S R a t i o", rendered[0])
        self.assertEqual("Header row 1: c1=Temp. | c2=Grain size | c3=Model parameters | c4=<EMPTY> | c5=<EMPTY> | c6=<EMPTY> | c7=<EMPTY> | c8=<EMPTY> | c9=<EMPTY>", rendered[1])
        self.assertEqual("Header row 2: c1=<EMPTY> | c2=<EMPTY> | c3=γ ̇ 0 | c4=f ̇ 0 | c5=γ t w | c6=n | c7=m | c8=C R S S b a s a l | c9=C R S S R a t i o", rendered[2])
        self.assertIn("Temp.=<INHERITED:298 K>", rendered[4])
        self.assertIn("Grain size=5 μ m", rendered[4])
        self.assertIn("Model parameters / γ ̇ 0=0.001", rendered[4])
        self.assertIn("f ̇ 0=0.02", rendered[4])

    def test_table_json_full_text_serializes_comparison_table_as_self_describing_rows(self):
        table_json = {
            "caption": "Elastic constants, c/a ratio (HCP), Zener ratio (FCC), and initial slip strength ( τ 0 , i ) values for different slip modes in investigated materials.",
            "rows": [
                ["Structure", "", "Material", "", "Ratio", "", "Elastic constants (GPa)", "Ref.", "", "τ 0 , i for slip modes (MPa)", "Ref."],
                ["", "", "", "", "Zener", "", "C 11", "C 12", "C 44", "", "", "{ 1 1 ¯ 1 } 〈 110 〉", ""],
                ["FCC", "", "René 88DT", "", "2.23", "", "267.1", "170.5", "107.6", "[72]", "", "525", "–"],
                ["", "Inconel 718-PS", "", "2.72", "", "259.6", "179.0", "109.6", "[73]", "", "495", "–"],
                ["", "", "", "", "c / a", "", "C 11", "C 12", "C 13", "C 33", "C 44", "", "", "Prismatic 〈 a 〉", "Basal 〈 a 〉", "Pyramidal I 〈 c + a 〉", ""],
                ["HCP", "", "Titanium Ti–6Al–4V", "", "1.588", "", "162", "", "", "92", "", "", "69", "", "", "181", "", "", "47", "", "", "[76,77]", "", "370", "420", "590", "[78]"],
            ],
        }

        rendered = _table_json_full_text(table_json)

        self.assertIn("Comparative row: Structure=FCC, Material=René 88DT, Zener=2.23", rendered)
        self.assertIn("τ0,i for { 1 1 ¯ 1 } 〈 110 〉=525 MPa", rendered)
        self.assertIn("HCP slip-mode columns: Prismatic 〈 a 〉, Basal 〈 a 〉, Pyramidal I 〈 c + a 〉", rendered)
        self.assertIn("Comparative row: Structure=HCP, Material=Titanium Ti–6Al–4V, c/a=1.588, C11=162 GPa", rendered)
        self.assertIn("Prismatic 〈 a 〉=370 MPa, Basal 〈 a 〉=420 MPa, Pyramidal I 〈 c + a 〉=590 MPa", rendered)

    def test_table_json_full_text_serializes_composition_matrix_as_material_blocks(self):
        table_json = {
            "caption": "Chemical composition of investigated materials (wt.%).",
            "rows": [
                ["Element", "CoNi-SB", "316L FP", "Copper"],
                ["Al", "5.97", "", ""],
                ["B", "0.014", "0.009", ""],
                ["Co", "bal.", "0.19", ""],
                ["Cu", "", "0.11", ""],
                ["Ni", "35.8", "11.90", ""],
            ],
        }

        rendered = _table_json_full_text(table_json)

        self.assertIn("Matrix orientation: rows are composition components/elements; columns are materials", rendered)
        self.assertIn("Material: CoNi-SB", rendered)
        self.assertIn("Entries: Al=5.97; B=0.014; Co=bal.; Ni=35.8", rendered)
        self.assertIn("Material: 316L FP", rendered)
        self.assertIn("Entries: B=0.009; Co=0.19; Cu=0.11; Ni=11.90", rendered)
        self.assertIn("Material: Copper", rendered)
        self.assertIn("Entries: <NO EXPLICIT COMPOSITION ENTRIES>", rendered)

    def test_inject_legacy_compat_views_from_v3_keeps_v3_and_projects_legacy(self):
        payload = {
            "schema_version": "3.0.0",
            "document": {
                "doi": "10.1000/example",
                "title": "Example Paper",
                "authors": ["A. Author"],
                "year": 2024,
                "journal": "Acta Materialia",
            },
            "study": {"study_type": "single_material", "notes": "test"},
            "materials": [
                {
                    "material_id": "mat_1",
                    "name": "Ti-6Al-4V",
                    "formula": None,
                    "material_class": "titanium_alloy",
                    "composition": {
                        "basis": "wt_percent",
                        "components": [{"component": "Al", "value": 6, "unit": "wt%", "notes": None}],
                        "notes": None,
                    },
                    "processing_history": [],
                    "phases": [
                        {
                            "phase_id": "phase_alpha",
                            "name": "alpha",
                            "role": "matrix",
                            "crystal_structure": {"crystal_system": "hexagonal"},
                            "volume_fraction": {"value": 0.6, "unit": "fraction"},
                            "microstructure": {
                                "grain_structure": "polycrystal",
                                "grain_size": {"value": 8.5, "unit": "um"},
                                "texture": {"description": "basal texture", "method": "ebsd"},
                                "defect_state": {"dislocation_density": {"value": 1e12, "unit": "m^-2"}},
                            },
                        }
                    ],
                    "material_level_microstructure": {"summary": "bimodal"},
                }
            ],
            "samples": [
                {
                    "sample_id": "s1",
                    "material_id": "mat_1",
                    "label": "aged",
                    "processing_state": "aged",
                    "condition_ids": ["c1"],
                    "microstructure_overrides": {
                        "phase_microstructure_overrides": [
                            {
                                "phase_id": "phase_alpha",
                                "grain_size": {"value": 7.5, "unit": "um"},
                                "phase_fraction": {"value": 0.6, "unit": "fraction"},
                            }
                        ],
                        "selected_grains": [{"grain_id": "g1", "label": "grain 1", "phase_id": "phase_alpha"}],
                    },
                }
            ],
            "conditions": [{"condition_id": "c1", "label": "RT tension", "loading_mode": "uniaxial_tension"}],
            "models": [{"model_id": "m1", "class": "crystal_plasticity", "framework": "cpfe"}],
            "mechanisms": {"slip_families": [{"family_id": "f1", "name": "basal", "phase_id": "phase_alpha"}]},
            "parameter_claims": [
                {
                    "claim_id": "claim_1",
                    "model_id": "m1",
                    "canonical_name": "tau0",
                    "symbol": "tau_0",
                    "domain": "plastic",
                    "value": 85,
                    "unit": "MPa",
                    "applies_to": {
                        "scope": "family",
                        "material_id": "mat_1",
                        "sample_id": "s1",
                        "condition_id": "c1",
                        "phase_id": "phase_alpha",
                        "family_id": "f1",
                        "system_ids": [],
                    },
                    "provenance": {"origin_type": "calibrated", "reference_ids": []},
                    "evidence": {"evidence_text": "tau0 = 85 MPa"},
                }
            ],
        }

        projected = _inject_legacy_compat_views_from_v3(payload)

        self.assertEqual("3.0.0", projected["schema_version"])
        self.assertEqual("Example Paper", projected["document"]["title"])
        self.assertEqual("Example Paper", projected["source_document"]["title"])
        self.assertEqual("Ti-6Al-4V", projected["material"]["name"])
        self.assertEqual("mat_1", projected["paper_profile"]["studied_materials"][0]["material_id"])
        self.assertEqual("s1", projected["paper_profile"]["sample_profiles"][0]["sample_id"])
        self.assertEqual("tau0", projected["parameters"]["registry"][0]["canonical_name"])
        self.assertEqual("calibrated", projected["parameters"]["registry"][0]["source"]["origin_type"])

    def test_merge_source_enrichment_updates_parameter_claim_provenance(self):
        extracted = {
            "parameter_claims": [
                {
                    "domain": "plastic",
                    "canonical_name": "tau0",
                    "provenance": {"origin_type": None, "reference_ids": []},
                }
            ]
        }
        enrich = {
            "plastic_sources": [
                {
                    "index": 0,
                    "source": {
                        "origin_type": "adopted",
                        "reference_ids": ["12"],
                    },
                }
            ]
        }

        merged = _merge_source_enrichment(extracted, enrich)
        self.assertEqual("adopted", merged["parameter_claims"][0]["provenance"]["origin_type"])
        self.assertEqual(["12"], merged["parameter_claims"][0]["provenance"]["reference_ids"])

    def test_main_schema_exposes_direct_binding_fields(self):
        self.assertEqual("5.1.1", EXTRACT_SCHEMA_SKELETON["schema_version"])
        self.assertNotIn("document", EXTRACT_SCHEMA_SKELETON)
        self.assertNotIn("study", EXTRACT_SCHEMA_SKELETON)
        self.assertIsInstance(EXTRACT_SCHEMA_SKELETON["process_states"][0]["state_type"], list)
        self.assertIn("deformation_systems", EXTRACT_SCHEMA_SKELETON)
        self.assertIn("simulation_geometries", EXTRACT_SCHEMA_SKELETON)
        self.assertNotIn("orientation_inputs", EXTRACT_SCHEMA_SKELETON)
        self.assertIn("numerical_methods", EXTRACT_SCHEMA_SKELETON)
        self.assertNotIn("simulation_outputs", EXTRACT_SCHEMA_SKELETON)
        self.assertNotIn("model_evaluations", EXTRACT_SCHEMA_SKELETON)
        self.assertIn("governing_equation_ids", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0])
        self.assertNotIn("qualifier", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["assertion"])
        self.assertIn("branch_ids", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["applies_to"])
        self.assertIn("family_id", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["applies_to"])
        self.assertIn("system_ids", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["applies_to"])
        self.assertIn("reported_value", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["assertion"])
        self.assertIn("calibration", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["provenance"])
        self.assertIn("target_type", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["provenance"]["calibration"])
        self.assertIn("constituents", EXTRACT_SCHEMA_SKELETON)
        self.assertIn("constitutive_branches", EXTRACT_SCHEMA_SKELETON["models"][0])
        self.assertIn("branch_type", EXTRACT_SCHEMA_SKELETON["models"][0]["constitutive_branches"][0])
        self.assertIn("evidence_ids", EXTRACT_SCHEMA_SKELETON["models"][0])
        self.assertNotIn("geometry_representation", EXTRACT_SCHEMA_SKELETON["models"][0]["solver_framework"])
        self.assertNotIn("constituent_id", EXTRACT_SCHEMA_SKELETON["microstructure_features"][0])
        self.assertEqual(
            ["string"],
            EXTRACT_SCHEMA_SKELETON["models"][0]["constitutive_description"]["slip_description"]["deformation_system_ids"],
        )
        self.assertEqual(
            ["string"],
            EXTRACT_SCHEMA_SKELETON["models"][0]["constitutive_description"]["twinning"]["deformation_system_ids"],
        )
        self.assertIn("all explicit equation labels", EXTRACT_SCHEMA_SKELETON["models"][0]["constitutive_branches"][0]["governing_equation_ids"][0])
        self.assertIn("all explicit equation labels", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["governing_equation_ids"][0])

    def test_prompt_requires_many_to_many_equation_binding(self):
        self.assertIn("Treat `models[].constitutive_branches[].governing_equation_ids` as a full multi-equation array", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("parameter_claims[].applies_to.branch_ids", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("Do not embed full equation objects or equation text inside `models[]`, `constitutive_branches[]`, or `parameter_claims[]`", EXTRACT_USER_PROMPT_TEMPLATE)

    def test_prompt_requires_claim_specific_table_evidence_packaging(self):
        self.assertIn("prefer claim-specific evidence packaging over whole-table summaries", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("Do not leave `locator.value=null` when the same table excerpt gives an explicit parameter value", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("A reviewer should be able to see the exact condition-value mapping without re-reading the whole table", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("Multiple branches may share the same equation, and one branch or one parameter may also bind to multiple equations", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("do not force one equation per branch or one equation per parameter", EXTRACT_USER_PROMPT_TEMPLATE)

    def test_prompt_uses_governing_equation_ids_instead_of_equation_evidence(self):
        self.assertIn("Do not create equation-only `evidence_objects[]` entries", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("Store equation support through `models[].equation_ids`, `models[].constitutive_branches[].governing_equation_ids`, and `parameter_claims[].governing_equation_ids`", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("Do not put equation evidence IDs in `materials[].evidence_ids`, `models[].evidence_ids`, `constitutive_branches[].evidence_ids`, `parameter_claims[].evidence_ids`", EXTRACT_USER_PROMPT_TEMPLATE)

    def test_prompt_removes_document_and_output_evaluation_blocks_from_extractor_schema(self):
        self.assertIn("Do not emit a `document` block", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("do not invent separate output or evaluation objects", EXTRACT_USER_PROMPT_TEMPLATE)

    def test_prompt_supports_multi_branch_parameter_binding(self):
        self.assertIn("Use `parameter_claims[].applies_to.branch_ids` for branch linkage in every case", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("same parameter claim is explicitly shared across multiple constitutive branches", EXTRACT_USER_PROMPT_TEMPLATE)

    def test_prompt_requires_direct_id_mounting(self):
        self.assertIn("fill `applies_to.material_id`, `applies_to.constituent_id`, `applies_to.process_state_id`, `applies_to.model_id`, `applies_to.condition_id`, `applies_to.branch_ids`, `applies_to.family_id`, and `applies_to.system_ids`", EXTRACT_USER_PROMPT_TEMPLATE)

    def test_prompt_requires_deterministic_semantic_ids(self):
        self.assertIn("Make IDs deterministic and semantic, not conversational", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("Prefer semantic IDs over encounter-order numbering", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("Avoid arbitrary names like `mat_1`, `cond_2`, `feat_7`, `claim_12`, or `ev_3`", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("`material_id=mat_ti6al4v`, `process_state_id=ps_as_built_annealed_773k`", EXTRACT_USER_PROMPT_TEMPLATE)

    def test_prompt_requires_zener_claims_and_explicit_constituent_handling(self):
        self.assertIn("treat explicitly reported auxiliary elastic descriptors such as `Zener ratio`, `c/a`", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("emit a dedicated `zener_ratio` claim for each material", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("If `materials[].phase_mode=multi_phase`, actively check whether the excerpt also gives explicit named constituents", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("do not leave `constituents[]` empty in that case", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("leave `constituents[]` empty, and note that the constituent-level identities were not explicitly recoverable", EXTRACT_USER_PROMPT_TEMPLATE)

    def test_inject_legacy_compat_accepts_mechanism_list_shape(self):
        payload = {
            "schema_version": "4.4.0",
            "document": {"title": "Example"},
            "materials": [{"material_id": "mat_1", "name": "Ti"}],
            "process_states": [],
            "conditions": [],
            "models": [{"model_id": "model_cp", "model_type": "crystal_plasticity"}],
            "mechanisms": [
                {
                    "mechanism_id": "mech_basal",
                    "mechanism_type": "slip",
                    "level": "family",
                    "family_name": "basal",
                    "name": "basal",
                }
            ],
            "microstructure_features": [],
            "parameter_claims": [],
        }

        projected = _inject_legacy_compat_views_from_v3(payload)
        self.assertEqual("basal", projected["deformation_mechanisms"]["slip_families"][0]["family_name"])

    def test_new_schema_does_not_require_legacy_views(self):
        self.assertNotIn("source_document", EXTRACT_SCHEMA_SKELETON)
        self.assertNotIn("paper_profile", EXTRACT_SCHEMA_SKELETON)
        self.assertNotIn("deformation_mechanisms", EXTRACT_SCHEMA_SKELETON)

    def test_augment_selected_equations_adds_relevant_equations_for_constitutive_sections(self):
        selected_sections = [
            {
                "name": "004_Constitutive law.md",
                "title": "Constitutive law",
                "selection_preview": "Flow rule and hardening equations",
            }
        ]
        equations = [
            {
                "selection_id": "eq_0003",
                "name": "eq_0003",
                "section_title": "Constitutive law",
                "text": "gamma flow rule",
                "length": 100,
            },
            {
                "selection_id": "eq_0007",
                "name": "eq_0007",
                "section_title": "Constitutive law",
                "text": "backstress evolution",
                "length": 120,
            },
        ]

        augmented = _augment_selected_equations([], equations, selected_sections, limit=8)
        self.assertEqual({"eq_0003", "eq_0007"}, {row["selection_id"] for row in augmented})

    def test_normalize_equation_references_keeps_only_numbered_labels(self):
        payload = {
            "models": [
                {
                    "equation_ids": ["eq_0005", "eq_creep_power_law", "Eq. (7)"],
                    "constitutive_branches": [
                        {"governing_equation_ids": ["equation 4", "backstress_eq"]}
                    ],
                }
            ],
            "parameter_claims": [
                {
                    "governing_equation_ids": ["Eq. (5)", "eq_hardening"],
                    "provenance": {"calibration": {}},
                }
            ],
        }

        normalized = _normalize_equation_references_in_payload(payload)

        self.assertEqual(["(5)", "(7)"], normalized["models"][0]["equation_ids"])
        self.assertEqual(["(4)"], normalized["models"][0]["constitutive_branches"][0]["governing_equation_ids"])
        self.assertEqual(["(5)"], normalized["parameter_claims"][0]["governing_equation_ids"])

if __name__ == "__main__":
    unittest.main()
