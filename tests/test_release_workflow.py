import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval.benchmark_protocol import PROTOCOL_VERSION
from scripts.eval.run_ablations import VARIANTS, create_plan, variant_config, run_job
from scripts.eval.prepare_reliable_pilot import prepare, collect
from scripts.eval.evaluate_release import run
from scripts.eval.summarize_ablations import summarize


CONFIG = {"llm": {"model_select": "selector", "model_extract": "extractor", "model_evaluate": "judge",
                  "max_snippet_chars": 500, "max_context_chars": 10000, "enable_source_enrichment": False},
          "pipeline": {}, "elsevier": {"api_key": "test-secret"}, "db": {"password": "test-secret"}}


class ReleaseWorkflowTests(unittest.TestCase):
    def fixture(self, root):
        paper = Path(root) / "10.1000_example"
        (paper / "sections").mkdir(parents=True)
        (paper / "sections/s.md").write_text("tau0 = 20 MPa")
        payload = {"schema_version": "6.0.0", "document": {"doi": "10.1000/example"},
                   "materials": [], "parameter_claims": [
                       {"claim_id": "a", "parameter": {"canonical_name": "tau0", "symbol_reported": "t0"},
                        "assertion": {"reported_value": 20, "reported_unit": "MPa"},
                        "evidence": {"source_file": "s.md", "text": "tau0 = 20 MPa"}}]}
        (paper / "materials_extracted.json").write_text(json.dumps(payload))
        manifest = {"protocol_version": PROTOCOL_VERSION, "papers": [{"doi": "10.1000/example", "paper_dir": str(paper), "split": "development"}]}
        return paper, payload, manifest

    def test_pilot_roundtrip_remains_pending(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            paper, _, _ = self.fixture(root)
            out = Path(tmp) / "pilot"
            prepare(root, out, 1)
            rows = collect(out, out / "reviewed.jsonl")
            self.assertEqual(rows[0]["annotation"]["status"], "pending")
            result = run(out / "reviewed.jsonl", root, Path(tmp) / "eval", manifest_path=out / "manifest.json")
            self.assertFalse(result["publication_ready"])
            self.assertIsNone(result["claim_detection"]["precision"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                prepare(root, out, 1)

    def test_single_factor_config_and_secrets(self):
        baseline = variant_config(CONFIG, "baseline")
        for name, overrides in VARIANTS.items():
            expected = copy.deepcopy(baseline)
            for key, value in overrides.items():
                section, field = key.split(".")
                expected[section][field] = value
            self.assertEqual(variant_config(CONFIG, name), expected)
        self.assertIsNone(baseline["elsevier"]["api_key"])
        self.assertIsNone(baseline["db"]["password"])
        self.assertEqual(CONFIG["elsevier"]["api_key"], "test-secret")

    def test_frozen_plan_rejects_changed_sources_and_reports_not_run(self):
        with TemporaryDirectory() as tmp:
            paper, _, manifest = self.fixture(Path(tmp) / "corpus")
            out = Path(tmp) / "runs"
            create_plan(CONFIG, manifest, out, ["baseline"], "development")
            gold = Path(tmp) / "gold.jsonl"
            gold.write_text("")
            rows = summarize(out / "plan.json", gold, Path(tmp) / "compare", diagnostic=True)
            self.assertEqual(rows[0]["not_run_papers"], 1)
            self.assertNotIn("f1", rows[0])
            (paper / "sections/s.md").write_text("changed")
            with self.assertRaisesRegex(ValueError, "different inputs"):
                create_plan(CONFIG, manifest, out, ["baseline"], "development")

    def test_mocked_job_forwards_variant_and_never_changes_source(self):
        with TemporaryDirectory() as tmp:
            paper, payload, manifest = self.fixture(Path(tmp) / "corpus")
            plan = create_plan(CONFIG, manifest, Path(tmp) / "runs", ["double_pass"], "development")
            before = (paper / "materials_extracted.json").read_bytes()
            with patch("llm.extractor.run_llm_on_paper_dir", return_value={"extracted": payload, "metrics": {}}) as extract, \
                 patch("llm.evaluator.run_llm_evaluation", return_value=({"verdict": "accepted", "overall_score": 90}, {})):
                result = run_job(plan["jobs"][0])
                self.assertEqual(result["status"], "success", result)
                self.assertTrue(extract.call_args.kwargs["two_pass_extraction"])
            self.assertEqual((paper / "materials_extracted.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
