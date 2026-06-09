import unittest

from pipelines.material_screening import detect_concrete_material_signal


class ScopusMaterialMetadataTests(unittest.TestCase):
    def test_metadata_rule_keeps_specific_material_abstract(self):
        report = detect_concrete_material_signal(
            "We calibrate a crystal plasticity model for 316L stainless steel and compare it with Ti-6Al-4V.",
            min_score=2,
        )
        self.assertTrue(report["matched"])

    def test_metadata_rule_drops_generic_framework_abstract(self):
        report = detect_concrete_material_signal(
            "This paper proposes a generic crystal plasticity framework for polycrystalline metals.",
            min_score=2,
        )
        self.assertFalse(report["matched"])


if __name__ == "__main__":
    unittest.main()
