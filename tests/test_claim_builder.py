import unittest

from postprocess.claim_builder import build_parameter_claims


class ClaimBuilderTests(unittest.TestCase):
    def test_assigns_claim_id_to_matching_evidence_objects_by_evidence_id(self):
        extracted_json = {
            "parameters": {
                "registry": [
                    {
                        "claim_id": "claim_0001",
                        "canonical_name": "crss_initial",
                        "symbol": "tau0",
                        "value": 85,
                        "unit": "MPa",
                        "source": {"evidence_ids": ["ev-2"]},
                    }
                ]
            },
            "evidence_objects": [
                {"evidence_id": "ev-1"},
                {"evidence_id": "ev-2"},
            ],
        }

        updated, _ = build_parameter_claims(extracted_json)
        evidence_objects = updated["evidence_objects"]
        self.assertIsNone(evidence_objects[0].get("claim_id"))
        self.assertEqual(evidence_objects[1].get("claim_id"), "claim_0001")


if __name__ == "__main__":
    unittest.main()
