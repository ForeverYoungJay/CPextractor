import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

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
        self.assertIn("material_phase_normalization", report)
        self.assertIn("model_setup_normalization", report)
        self.assertIn("document", updated)
        self.assertNotIn("source_document", updated)

    def test_legacy_structure_normalization_also_backfills_canonical_document(self):
        extracted = {
            "schema_version": "2.1.1",
            "materials": [],
            "process_states": [],
            "models": [],
            "microstructure_features": [],
            "conditions": [],
            "parameter_claims": [],
            "evidence_objects": [],
        }

        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "paper.xml").write_text(
                """
                <root>
                  <dc:title>Example Legacy Paper</dc:title>
                  <dc:creator>A. Author</dc:creator>
                  <prism:publicationName>Acta Materialia</prism:publicationName>
                  <prism:coverDate>2025-03-01</prism:coverDate>
                  <prism:doi>10.1000/example</prism:doi>
                </root>
                """,
                encoding="utf-8",
            )

            updated, _ = run_structure_normalization(
                extracted,
                paper_dir=tmpdir,
                doi_hint=None,
                reference_map=None,
            )

        self.assertIn("document", updated)
        self.assertIn("source_document", updated)
        self.assertEqual("Example Legacy Paper", updated["document"]["title"])
        self.assertEqual("Acta Materialia", updated["document"]["journal"])
        self.assertEqual("10.1000/example", updated["document"]["doi"])
        self.assertEqual("Acta Materialia", updated["source_document"]["journal_or_venue"])

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

        self.assertEqual(updated["schema_version"], "5.1.0")
        self.assertIn("parameter_claims", report)
        self.assertIn("final_hierarchy", report)


if __name__ == "__main__":
    unittest.main()
