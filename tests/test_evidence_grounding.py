import copy
import unittest
from pathlib import Path

from postprocess.evidence_grounding import _direct_table_match_from_source, verify_evidence_grounding


class EvidenceGroundingTests(unittest.TestCase):
    def test_direct_table_match_can_use_table_evidence_indices_without_internal_match(self):
        file_path = Path("table_001.json")
        source = {
            "evidence_location": {"kind": "table", "id": "table_001.json"},
            "table_evidence": {
                "row_index": 2,
                "column_index": 2,
                "value": "85",
            },
        }

        # Use a fake table by monkeypatching the module-local loader through direct import behavior.
        import postprocess.evidence_grounding as mod

        original = mod._table_rows_from_json
        mod._table_rows_from_json = lambda _: [(1, "Symbol | Value", ["Symbol", "Value"]), (2, "tau0 | 85", ["tau0", "85"])]
        try:
            hit = _direct_table_match_from_source(file_path, source)
        finally:
            mod._table_rows_from_json = original

        self.assertIsNotNone(hit)
        self.assertEqual(hit["table_cell"]["row_index"], 2)
        self.assertEqual(hit["table_cell"]["column_index"], 2)
        self.assertEqual(hit["value"], "85")

    def test_verify_evidence_grounding_does_not_rebuild_evidence_objects(self):
        extracted = {
            "schema_version": "4.4.0",
            "parameter_claims": [
                {
                    "claim_id": "claim_0001",
                    "parameter": {"canonical_name": "tau0", "domain": "plastic", "symbol_reported": "tau0"},
                    "assertion": {"reported_value": 85, "reported_unit": "MPa"},
                    "applies_to": {"model_id": "model_001"},
                    "provenance": {"origin_type": "calibrated", "evidence_ids": ["ev_existing"]},
                    "evidence_ids": ["ev_existing"],
                }
            ],
            "evidence_objects": [
                {
                    "evidence_id": "ev_existing",
                    "evidence_type": "table_cell",
                    "source_file": "table_001.json",
                    "source_id": "table_001",
                    "locator": {
                        "row_name": "tau0",
                        "column_name": "Value",
                        "value": "85",
                        "excerpt": "tau0 | 85 MPa",
                    },
                    "snippet": "tau0 | 85 MPa",
                }
            ],
        }
        original_evidence_objects = copy.deepcopy(extracted["evidence_objects"])

        import postprocess.evidence_grounding as mod

        original = mod._table_rows_from_json
        mod._table_rows_from_json = lambda _: [(1, "Symbol | Value", ["Symbol", "Value"]), (2, "tau0 | 85 MPa", ["tau0", "85 MPa"])]
        try:
            updated, report = verify_evidence_grounding(extracted, ".")
        finally:
            mod._table_rows_from_json = original

        self.assertEqual(updated["evidence_objects"], original_evidence_objects)
        self.assertEqual(report["evidence_object_count"], 1)
        self.assertEqual(report["rows"][0]["evidence_ids"], ["ev_existing"])


if __name__ == "__main__":
    unittest.main()
