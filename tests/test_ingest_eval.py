import unittest

from db.ingest_eval import ingest_evaluation


class FakeConn:
    def __init__(self) -> None:
        self.calls = []
        self.fail_primary_insert = True
        self.fail_fallback_insert = True

    def execute(self, sql, params=None):
        normalized = " ".join(str(sql).split())
        self.calls.append((normalized, params))
        if "INSERT INTO evaluation_runs" in normalized and "llm_evaluate_input_tokens" in normalized and self.fail_primary_insert:
            raise RuntimeError("primary insert failed")
        if "INSERT INTO evaluation_runs" in normalized and "evaluation_json, confidence_json" in normalized and self.fail_fallback_insert:
            raise RuntimeError("fallback insert failed")
        return self


class IngestEvaluationTests(unittest.TestCase):
    def test_raises_when_primary_and_fallback_ingest_fail(self):
        conn = FakeConn()
        with self.assertRaises(RuntimeError):
            ingest_evaluation(
                conn,
                doi="10.1234/example",
                evaluation={"verdict": "pass", "parameter_audits": []},
                confidence={"document_confidence": "high"},
                model_evaluate="gpt-test",
                metrics={},
            )


if __name__ == "__main__":
    unittest.main()
