import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval.benchmark_protocol import (
    PROTOCOL_VERSION, normalize_row, prepare_universe, match_rows, value_equal,
)
from scripts.eval.evaluate_release import build_report, classification, run, field_equal
from scripts.eval.calibrate_gate import precision_coverage, choose_threshold, calibrate
from scripts.eval.export_annotation_draft import _claim_rows_for_paper


def row(**kwargs):
    return {"doi": "10.1000/example", "claim_id": "old_1", "canonical_name": "tau0",
            "symbol": "t0", "value": 20, "unit": "MPa", "context": {"material": "Ti"},
            "annotation": {"status": "correct"}, **kwargs}


def manifest(**kwargs):
    return {"protocol_version": PROTOCOL_VERSION, "papers": [{
        "doi": "10.1000/example", "split": "test", "annotation_status": "adjudicated",
        "exhaustive": True, "reviewer_type": "human_expert", "reviewer": "expert A",
        "adjudicator": "expert B", **kwargs}]}


class ProtocolTests(unittest.TestCase):
    def test_doi_and_usable_schema(self):
        r = normalize_row({"doi": "https://doi.org/10.1000/EXAMPLE", "parameter_body": {"value": 0, "unit": "Pa"},
                           "material_object": {"material_name": "Ti"}})
        self.assertEqual(r["doi"], "10.1000/example")
        self.assertEqual(r["value"], 0)
        self.assertEqual(r["context"]["material"], "Ti")

    def test_value_and_unit_errors_do_not_break_identity(self):
        result = match_rows([row()], [row(claim_id="new_99", value=200, unit="GPa")])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))

    def test_reused_id_does_not_match_different_parameter(self):
        result = match_rows([row()], [row(canonical_name="c11", symbol="C11")])
        self.assertEqual(result["tp"], 0)

    def test_scope_separates_same_parameter(self):
        result = match_rows([row()], [row(context={"material": "Al"})])
        self.assertEqual(result["tp"], 0)

    def test_cell_anchor_allows_scope_error_scoring(self):
        evidence = {"file": "table_001.json", "row_name": "tau0", "column_name": "Ti"}
        result = match_rows([row(evidence=evidence)], [row(evidence=evidence, context={"material": "Al"})])
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["match_log"][0]["reason"], "evidence_anchor")

    def test_row_anchor_disambiguates_grain_size_without_column(self):
        a = row(evidence={"file": "t.json", "row_name": "298K 5um"})
        b = row(evidence={"file": "t.json", "row_name": "298K 9um"})
        result = match_rows([a, b], [b, a])
        self.assertEqual(result["tp"], 2)
        self.assertFalse(result["ambiguities"])

    def test_unreviewed_claim_does_not_become_false_positive(self):
        pending = row(canonical_name="c11", symbol="C11", annotation={"status": "pending"})
        gold, pred, report = prepare_universe([row(), pending], [row(), pending])
        self.assertEqual(len(pred), 1)
        self.assertEqual(report["excluded_pending_predictions"], 1)

    def test_duplicate_ids_do_not_drop_rows(self):
        gold = [row(), row(canonical_name="c11", symbol="C11")]
        result = match_rows(gold, copy.deepcopy(gold))
        self.assertEqual(result["tp"], 2)

    def test_ambiguity_is_not_arbitrary_assignment(self):
        result = match_rows([row()], [row(), row(claim_id="other")])
        self.assertEqual(len(result["ambiguities"]), 1)
        self.assertEqual(result["fp"], 2)
        self.assertEqual(result["fn"], 1)

    def test_unannotated_papers_excluded(self):
        gold, pred, report = prepare_universe([row()], [row(), row(doi="10.1000/other")])
        self.assertEqual(len(pred), 1)
        self.assertEqual(report["excluded_prediction_rows"], 1)

    def test_missing_prediction_paper_stays_in_universe(self):
        gold, pred, report = prepare_universe([row()], [])
        self.assertEqual(match_rows(gold, pred)["fn"], 1)
        self.assertEqual(report["missing_prediction_papers"], ["10.1000/example"])

    def test_pending_is_not_spurious_or_gold(self):
        gold, pred, report = prepare_universe([row(annotation={"status": "pending"})], [row()])
        self.assertEqual((len(gold), len(pred)), (0, 0))
        self.assertEqual(report["pending_gold_rows"], 1)

    def test_missing_doi_fails_instead_of_one_empty_paper(self):
        with self.assertRaisesRegex(ValueError, "valid DOI"):
            prepare_universe([row(doi="")], [])

    def test_ai_cannot_be_expert_gold(self):
        with self.assertRaisesRegex(ValueError, "expert-adjudicated"):
            prepare_universe([row()], [], manifest(reviewer_type="ai_assistant"), strict=True)

    def test_strict_empty_gold_paper_needs_declaration(self):
        with self.assertRaisesRegex(ValueError, "zero-claim"):
            prepare_universe([], [row()], manifest(), strict=True)
        gold, pred, _ = prepare_universe([], [row()], manifest(gold_claim_count=0), strict=True)
        self.assertEqual(match_rows(gold, pred)["fp"], 1)

    def test_strict_requires_row_reviewer(self):
        with self.assertRaisesRegex(ValueError, "Every gold row"):
            prepare_universe([row()], [], manifest(), strict=True)

    def test_scientific_notation_fraction_and_unit_case(self):
        self.assertTrue(value_equal("3.4 × 10^8", 340000000))
        self.assertTrue(value_equal("3/2", 1.5))
        self.assertFalse(field_equal("unit", "mPa", "MPa"))
        self.assertFalse(field_equal("symbol", "m", "M"))

    def test_nan_and_zero_and_array_values(self):
        self.assertFalse(value_equal(float("nan"), float("nan")))
        self.assertFalse(value_equal(None, 0))
        self.assertTrue(value_equal([0, 1], [0, 1.00001]))

    def test_field_errors_and_missing_annotations(self):
        gold, pred, universe = prepare_universe([row()], [row(value=21)])
        summary, artifacts = build_report(gold, pred, universe)
        self.assertEqual(summary["fields"]["value"]["accuracy"], 0)
        self.assertIsNone(summary["fields"]["context.condition"]["accuracy"])
        self.assertEqual(summary["error_taxonomy"]["value_unit"], 1)

    def test_unknown_gate_labels_not_good_papers(self):
        gold, pred, universe = prepare_universe([row()], [row()])
        summary, _ = build_report(gold, pred, universe, manifest(), [{"doi": "10.1000/example", "blocked": True}])
        self.assertEqual(summary["gate"]["paper_count"], 0)
        self.assertEqual(len(summary["gate"]["unlabeled_papers"]), 1)

    def test_real_export_v6_value_scope_and_evidence(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            payload = {"schema_version": "6.0.0", "document": {"doi": "10.1000/example"},
                       "materials": [{"material_id": "m2", "name": "Ti"}],
                       "parameter_claims": [{"claim_id": "c1", "parameter": {"canonical_name": "tau0"},
                                             "assertion": {"reported_value": 0, "reported_unit": "MPa"},
                                             "applies_to": {"material_id": "m2"}, "evidence_ids": ["ev1"],
                                             "quality_assessment": {"final_confidence_score": 88}}],
                       "evidence_objects": [{"evidence_id": "ev1", "evidence_type": "text", "source_file": "s.md", "snippet": "tau0=0"}]}
            (p / "materials_extracted.json").write_text(json.dumps(payload))
            out = _claim_rows_for_paper(p, "materials_extracted.json")[0]
            self.assertEqual(out["value"], 0)
            self.assertEqual(out["context"]["material"], "Ti")
            self.assertEqual(out["evidence"]["file"], "s.md")
            self.assertEqual(out["confidence_score"], 88)
            self.assertEqual(out["annotation"]["status"], "pending")
            payload["parameter_claims"][0].pop("quality_assessment")
            (p / "materials_extracted.json").write_text(json.dumps(payload))
            (p / "postprocess_report.json").write_text(json.dumps({"confidence_fusion": {"parameter_confidence": [{"location": "claim:c1", "canonical_name": "tau0", "score": 0}]}}))
            self.assertEqual(_claim_rows_for_paper(p, "materials_extracted.json")[0]["confidence_score"], 0)


class CalibrationTests(unittest.TestCase):
    def test_rates_have_explicit_denominators(self):
        result = classification([{"blocked": False, "should_block": True}, {"blocked": True, "should_block": False}])
        self.assertEqual(result["false_admission_rate"], 1)
        self.assertEqual(result["false_block_rate"], 1)

    def test_missing_confidence_is_not_zero(self):
        result = precision_coverage([{"score": None, "correct": True}, {"score": 0, "correct": False}])
        self.assertEqual(result["missing_score"], 1)
        self.assertEqual(result["curve"][0]["coverage"], .5)

    def test_no_feasible_threshold_is_not_success(self):
        curve = precision_coverage([{"score": 90, "correct": False}])["curve"]
        self.assertIsNone(choose_threshold(curve, .95, 1))

    def test_fit_rejects_test_split(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "metrics"
            p.mkdir()
            (p / "benchmark_claims.json").write_text(json.dumps({"publication_ready": True, "universe": {"split": "test"}}))
            with self.assertRaisesRegex(ValueError, "calibration"):
                calibrate(tmp, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
