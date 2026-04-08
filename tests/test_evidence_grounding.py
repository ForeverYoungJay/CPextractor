import unittest
from pathlib import Path

from postprocess.evidence_grounding import _direct_table_match_from_source


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


if __name__ == "__main__":
    unittest.main()
