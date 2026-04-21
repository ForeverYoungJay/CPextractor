import unittest

from postprocess.workflow import (
    run_evidence_linking,
    run_finalization,
    run_structure_normalization,
)


class WorkflowTests(unittest.TestCase):
    def test_extractor_first_structure_normalization_skips_rewrite_steps(self):
        extracted = {
            "schema_version": "4.4.0",
            "document": {"title": "Example"},
            "materials": [],
            "process_states": [],
            "models": [],
            "calibration_contexts": [],
            "microstructure_features": [],
            "conditions": [],
            "mechanisms": [],
            "parameter_claims": [],
            "evidence_objects": [],
        }

        updated, report = run_structure_normalization(
            extracted,
            paper_dir=".",
            doi_hint="10.1000/example",
            reference_map=None,
        )

        self.assertEqual(updated["schema_version"], "4.4.0")
        self.assertTrue(report["parameter_normalization"]["skipped"])
        self.assertTrue(report["material_phase_normalization"]["skipped"])
        self.assertIn("document", updated)
        self.assertNotIn("source_document", updated)

    def test_extractor_first_linking_skips_table_and_condition_rewriters(self):
        extracted = {
            "schema_version": "4.4.0",
            "models": [{"model_id": "model_001", "equation_ids": []}],
            "parameter_claims": [
                {
                    "claim_id": "claim_0001",
                    "parameter": {"canonical_name": "m", "domain": "plastic"},
                    "assertion": {"reported_value": 20, "reported_unit": None},
                    "applies_to": {"model_id": "model_001"},
                }
            ],
            "evidence_objects": [],
        }

        _, report = run_evidence_linking(extracted, paper_dir=".")

        self.assertTrue(report["provenance_normalization"]["skipped"])
        self.assertTrue(report["parameter_table_resolution"]["skipped"])
        self.assertTrue(report["condition_binding"]["skipped"])
        self.assertIn("model_equation_binding", report)

    def test_extractor_first_finalization_preserves_schema_and_skips_rebuild(self):
        extracted = {
            "schema_version": "4.4.0",
            "document": {"title": "Example"},
            "study": {"study_type": "single_material"},
            "materials": [],
            "process_states": [],
            "models": [],
            "calibration_contexts": [],
            "microstructure_features": [],
            "conditions": [],
            "mechanisms": [],
            "parameter_claims": [],
            "evidence_objects": [],
        }

        updated, report = run_finalization(
            extracted,
            evaluation_report=None,
            quality_report=None,
        )

        self.assertEqual(updated["schema_version"], "4.4.0")
        self.assertTrue(report["parameter_claims"]["skipped"])
        self.assertTrue(report["final_hierarchy"]["skipped"])


if __name__ == "__main__":
    unittest.main()
