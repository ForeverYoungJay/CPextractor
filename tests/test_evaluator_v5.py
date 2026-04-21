import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


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

from llm.evaluator import _build_document_summary, _build_parameter_records
from postprocess.quality_checks import run_quality_checks


class EvaluatorV5Tests(unittest.TestCase):
    def test_quality_checks_accepts_branch_and_constituent_scope(self):
        extracted = {
            "schema_version": "5.0.2",
            "parameter_claims": [
                {
                    "claim_id": "cl_tau",
                    "parameter": {
                        "canonical_name": "tau0",
                        "symbol_reported": "tau0",
                        "domain": "plastic",
                    },
                    "assertion": {
                        "reported_value": 85,
                        "reported_unit": "MPa",
                    },
                    "applies_to": {
                        "scope": "branch",
                        "model_id": "model_cp",
                        "branch_id": "branch_flow",
                    },
                    "evidence_ids": ["ev_1"],
                    "evidence": {"evidence_text": "tau0 = 85 MPa"},
                },
                {
                    "claim_id": "cl_h0",
                    "parameter": {
                        "canonical_name": "hardening_h0",
                        "symbol_reported": "h0",
                        "domain": "hardening",
                    },
                    "assertion": {
                        "reported_value": 200,
                        "reported_unit": "MPa",
                    },
                    "applies_to": {
                        "scope": "constituent",
                        "constituent_id": "const_austenite",
                        "model_id": "model_cp",
                    },
                    "evidence_ids": ["ev_2"],
                    "evidence": {"evidence_text": "h0 = 200 MPa"},
                },
            ],
        }
        _, report = run_quality_checks(extracted)
        scope_issues = [i for i in report["issues"] if i.get("type") == "scope_inconsistency"]
        self.assertEqual([], scope_issues)

    def test_build_parameter_records_include_many_to_many_equation_context(self):
        extracted = {
            "schema_version": "5.0.2",
            "materials": [{"material_id": "mat_1", "name": "316H", "phase_mode": "single_phase"}],
            "models": [
                {
                    "model_id": "model_cp",
                    "name": "CPFE",
                    "model_type": "crystal_plasticity",
                    "model_role": "primary_simulation",
                    "equation_ids": ["(3)", "(4)", "(5)"],
                    "constitutive_branches": [
                        {
                            "branch_id": "branch_flow",
                            "branch_type": "combined_flow",
                            "name": "Flow",
                            "governing_equation_ids": ["(3)", "(4)"],
                        }
                    ],
                },
                {
                    "model_id": "model_cmp",
                    "name": "J2 comparison",
                    "model_type": "other",
                    "model_role": "comparison",
                    "equation_ids": ["(8)", "(9)"],
                    "constitutive_branches": [],
                },
            ],
            "parameter_claims": [
                {
                    "claim_id": "cl_gamma",
                    "parameter": {
                        "canonical_name": "gamma0",
                        "symbol_reported": "gamma0,1",
                        "domain": "plastic",
                    },
                    "assertion": {
                        "reported_value": 0.1,
                        "reported_unit": "s^-1",
                    },
                    "applies_to": {
                        "scope": "branch",
                        "material_id": "mat_1",
                        "model_id": "model_cp",
                        "branch_id": "branch_flow",
                    },
                    "governing_equation_ids": ["(3)", "(4)"],
                    "provenance": {"origin_type": "calibrated"},
                    "evidence_ids": ["ev_1"],
                    "evidence": {},
                }
            ],
            "evidence_objects": [
                {
                    "evidence_id": "ev_1",
                    "evidence_type": "table",
                    "source_file": "tables/table_001.json",
                    "locator": {
                        "excerpt": "gamma0,1 | 0.1 | s^-1",
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir)
            (paper_dir / "sections").mkdir()
            (paper_dir / "tables").mkdir()
            (paper_dir / "equations").mkdir()
            (paper_dir / "tables" / "table_001.json").write_text(
                json.dumps(
                    {
                        "table_label": "Table 1",
                        "caption": "Material parameters",
                        "rows": [
                            ["parameter", "value", "unit"],
                            ["gamma0,1", "0.1", "s^-1"],
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (paper_dir / "llm_selected_files.json").write_text(
                json.dumps(
                    {
                        "selected_sections": [],
                        "selected_tables": ["table_001"],
                        "selected_equations": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            summary = _build_document_summary(extracted)
            self.assertEqual("5.0.2", summary["schema_version"])
            self.assertEqual(2, summary["model_count"])
            self.assertEqual("comparison", summary["models"][1]["model_role"])

            records = _build_parameter_records(str(paper_dir), extracted, per_evidence_chars=500, limit=10)
            self.assertEqual(1, len(records))
            row = records[0]
            self.assertEqual(["(3)", "(4)"], row["governing_equation_ids"])
            self.assertEqual(["(3)", "(4)"], row["branch_context"]["governing_equation_ids"])
            self.assertEqual(["(3)", "(4)", "(5)"], row["model_context"]["equation_ids"])
            self.assertEqual("extractor_raw_document_backfill_v5", row["evaluator_mode"])
            self.assertEqual(["ev_1"], row["source"]["evidence_ids"])
            self.assertTrue(row["inferred_support"])
            self.assertEqual("table", row["inferred_support"][0]["source_type"])


if __name__ == "__main__":
    unittest.main()
