import unittest

from pipelines.decision_layer import build_ingest_gate_report


class DecisionLayerTests(unittest.TestCase):
    def test_accepts_high_confidence_accepted_document(self):
        gate = build_ingest_gate_report(
            evaluation_report={
                "verdict": "accepted",
                "review_escalation": {"required": False},
            },
            confidence_report={
                "document_confidence_score": 91.0,
                "rejected_parameter_count": 0,
                "flagged_parameter_count": 1,
                "review_required_parameter_count": 1,
                "review_recommended": False,
            },
            extracted_json={
                "materials": [{"material_id": "mat_1", "name": "Ti-6Al-4V"}],
                "parameter_claims": [{"claim_id": "c1"}],
            },
            enabled=True,
            blocked_verdicts={"rejected"},
            min_document_confidence_score=65.0,
        )
        self.assertFalse(gate["blocked"])
        self.assertEqual(gate["verdict"], "accepted")

    def test_blocks_rejected_document_and_rejected_parameters(self):
        gate = build_ingest_gate_report(
            evaluation_report={
                "verdict": "rejected",
                "review_escalation": {"required": False},
            },
            confidence_report={
                "document_confidence_score": 92.0,
                "rejected_parameter_count": 2,
                "flagged_parameter_count": 0,
                "review_required_parameter_count": 0,
                "review_recommended": True,
            },
            extracted_json={"parameter_claims": [{"claim_id": "c1"}]},
            enabled=True,
            blocked_verdicts={"rejected"},
            min_document_confidence_score=65.0,
        )
        self.assertTrue(gate["blocked"])
        self.assertTrue(gate["gate_reasons"]["blocked_verdict"])
        self.assertTrue(gate["gate_reasons"]["has_rejected_parameters"])


if __name__ == "__main__":
    unittest.main()
