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

    def test_preserves_nested_v4_claim_fields(self):
        extracted_json = {
            "parameter_claims": [
                {
                    "claim_id": "claim_0001",
                    "parameter": {
                        "canonical_name": "m",
                        "parameter_family": "slip_kinetics",
                        "symbol_reported": "m",
                        "domain": "plastic",
                    },
                    "assertion": {
                        "reported_value": 20,
                        "reported_unit": None,
                        "normalized_value": 20,
                        "normalized_unit": None,
                    },
                    "applies_to": {
                        "model_id": "model_cp",
                        "mechanism_id": "mech_basal",
                        "scope": "family",
                    },
                    "provenance": {
                        "origin_type": "calibrated",
                        "calibration_in_this_study": "yes",
                    },
                    "governing_equation_ids": ["eq_0003"],
                    "evidence_ids": ["ev_eq_model_cp_eq_0003"],
                }
            ],
            "evidence_objects": [
                {
                    "evidence_id": "ev_eq_model_cp_eq_0003",
                    "snippet": "m appears in the flow rule",
                }
            ],
        }

        updated, report = build_parameter_claims(extracted_json)

        claim = updated["parameter_claims"][0]
        self.assertEqual(claim["parameter"]["canonical_name"], "m")
        self.assertEqual(claim["assertion"]["reported_value"], 20)
        self.assertIsNone(claim["assertion"]["reported_unit"])
        self.assertEqual(claim["governing_equation_ids"], ["eq_0003"])
        self.assertEqual(claim["evidence_ids"], ["ev_eq_model_cp_eq_0003"])
        self.assertEqual(claim["evidence"]["evidence_text"], "m appears in the flow rule")
        self.assertEqual(report["claims_built"], 1)


if __name__ == "__main__":
    unittest.main()
