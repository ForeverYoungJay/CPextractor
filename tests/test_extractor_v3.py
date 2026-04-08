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

from llm.extractor import _inject_legacy_compat_views_from_v3, _merge_source_enrichment


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


if __name__ == "__main__":
    unittest.main()
