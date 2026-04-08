import json
import tempfile
import unittest
from pathlib import Path

from db.ingest import _extract_vector_dim, _table_rows_from_json_file


class IngestSchemaTests(unittest.TestCase):
    def test_extract_vector_dim(self):
        self.assertEqual(_extract_vector_dim("vector(1536)"), 1536)
        self.assertIsNone(_extract_vector_dim("text"))

    def test_table_rows_from_json_file_builds_row_vectors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table_001.json"
            path.write_text(
                json.dumps(
                    {
                        "table_label": "Table 1",
                        "caption": "Parameters",
                        "rows": [["Symbol", "Value"], ["tau0", "85"], ["h0", "3555"]],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            rows = _table_rows_from_json_file(str(path))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0][0], "row_0001")
            self.assertIn("Table 1", rows[0][1])


if __name__ == "__main__":
    unittest.main()
