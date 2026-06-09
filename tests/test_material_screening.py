import unittest

from pipelines.material_screening import (
    detect_concrete_material_signal,
    material_identity_report_from_extracted_json,
)


class MaterialScreeningTests(unittest.TestCase):
    def test_detects_named_material_grade_from_metadata(self):
        report = detect_concrete_material_signal(
            "Crystal plasticity analysis of Ti-6Al-4V under dwell fatigue loading.",
            min_score=2,
        )
        self.assertTrue(report["matched"])
        self.assertGreaterEqual(report["score"], 2)

    def test_rejects_generic_material_only_metadata(self):
        report = detect_concrete_material_signal(
            "A generic crystal plasticity framework for metallic materials and alloys.",
            min_score=2,
        )
        self.assertFalse(report["matched"])

    def test_material_identity_requires_concrete_material_record_for_claims(self):
        report = material_identity_report_from_extracted_json({
            "materials": [],
            "parameter_claims": [{"claim_id": "c1"}],
        })
        self.assertFalse(report["has_concrete_materials"])
        self.assertEqual(1, report["parameter_count"])

    def test_material_identity_accepts_specific_material_name(self):
        report = material_identity_report_from_extracted_json({
            "materials": [
                {
                    "material_id": "mat_1",
                    "name": "Ti-6Al-4V",
                    "composition": {"components": []},
                }
            ],
            "parameter_claims": [{"claim_id": "c1"}],
        })
        self.assertTrue(report["has_concrete_materials"])
        self.assertEqual(1, report["concrete_material_count"])

    def test_material_identity_accepts_composition_even_if_name_is_generic(self):
        report = material_identity_report_from_extracted_json({
            "materials": [
                {
                    "material_id": "mat_1",
                    "name": "titanium alloy",
                    "composition": {
                        "components": [
                            {"component": "Ti", "value": "bal."},
                            {"component": "Al", "value": "6"},
                            {"component": "V", "value": "4"},
                        ]
                    },
                }
            ],
            "parameter_claims": [{"claim_id": "c1"}],
        })
        self.assertTrue(report["has_concrete_materials"])


if __name__ == "__main__":
    unittest.main()
