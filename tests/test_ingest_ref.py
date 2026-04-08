import unittest

from db.ingest_ref import ingest_references


class FakeConn:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((" ".join(str(sql).split()), params))
        return self


class IngestReferenceTests(unittest.TestCase):
    def test_deletes_stale_paper_and_parameter_reference_links(self):
        conn = FakeConn()
        extracted_json = {
            "references": [
                {"reference_id": "12", "doi": "10.1000/ref", "title": "Prior work"},
            ],
            "parameters": {
                "registry": [
                    {
                        "symbol": "tau0",
                        "source": {
                            "origin_type": "adopted",
                            "reference_ids": ["12"],
                        },
                    }
                ]
            },
        }

        ingest_references(conn, "10.1000/paper", extracted_json)

        deletes = [sql for sql, _ in conn.calls if sql.startswith("DELETE FROM")]
        self.assertTrue(any("DELETE FROM parameter_references" in sql for sql in deletes))
        self.assertTrue(any("DELETE FROM paper_references" in sql for sql in deletes))


if __name__ == "__main__":
    unittest.main()
