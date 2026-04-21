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
)


class ExtractorV3Tests(unittest.TestCase):
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
        self.assertEqual("5.0.2", EXTRACT_SCHEMA_SKELETON["schema_version"])
        self.assertIn("document", EXTRACT_SCHEMA_SKELETON)
        self.assertIn("study", EXTRACT_SCHEMA_SKELETON)
        self.assertIsInstance(EXTRACT_SCHEMA_SKELETON["process_states"][0]["state_type"], list)
        self.assertIn("governing_equation_ids", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0])
        self.assertIn("calibration", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["provenance"])
        self.assertIn("target_type", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["provenance"]["calibration"])
        self.assertIn("constituents", EXTRACT_SCHEMA_SKELETON)
        self.assertIn("constitutive_branches", EXTRACT_SCHEMA_SKELETON["models"][0])
        self.assertIn("branch_type", EXTRACT_SCHEMA_SKELETON["models"][0]["constitutive_branches"][0])
        self.assertIn("evidence_ids", EXTRACT_SCHEMA_SKELETON["models"][0])
        self.assertIn("all explicit equation labels", EXTRACT_SCHEMA_SKELETON["models"][0]["constitutive_branches"][0]["governing_equation_ids"][0])
        self.assertIn("all explicit equation labels", EXTRACT_SCHEMA_SKELETON["parameter_claims"][0]["governing_equation_ids"][0])

    def test_prompt_requires_many_to_many_equation_binding(self):
        self.assertIn("Treat `models[].constitutive_branches[].governing_equation_ids` as a full multi-equation array", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("Multiple branches may share the same equation, and one branch or one parameter may also bind to multiple equations", EXTRACT_USER_PROMPT_TEMPLATE)
        self.assertIn("do not force one equation per branch or one equation per parameter", EXTRACT_USER_PROMPT_TEMPLATE)

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
