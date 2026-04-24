import unittest

from postprocess.material_phase_normalizer import normalize_material_phases


class MaterialPhaseNormalizerTests(unittest.TestCase):
    def test_preserves_empty_constituents_from_extractor(self):
        extracted = {
            "schema_version": "5.0.2",
            "materials": [
                {
                    "name": "15-5PH stainless steel",
                    "phase_mode": "multi_phase",
                }
            ],
            "constituents": [],
        }

        updated, report = normalize_material_phases(extracted)

        self.assertEqual(updated["materials"][0]["material_id"], "mat_001")
        self.assertEqual(updated["materials"][0]["phase_mode"], "multi_phase")
        self.assertEqual(updated["constituents"], [])
        self.assertEqual(report["constituents_checked"], 0)
        self.assertEqual(report["constituent_ids_filled"], 0)
        self.assertEqual(report["inferred_single_phase_constituents"], 0)

    def test_preserves_constituents_without_postprocess_mutation(self):
        extracted = {
            "schema_version": "5.0.2",
            "materials": [
                {
                    "material_id": "mat_001",
                    "name": "Duplex steel",
                    "phase_mode": "multi_phase",
                }
            ],
            "constituents": [
                {
                    "constituent_id": None,
                    "material_id": None,
                    "name": "austenite",
                }
            ],
        }

        updated, report = normalize_material_phases(extracted)

        self.assertIsNone(updated["constituents"][0]["constituent_id"])
        self.assertIsNone(updated["constituents"][0]["material_id"])
        self.assertEqual(updated["constituents"][0]["name"], "austenite")
        self.assertEqual(report["constituents_checked"], 1)
        self.assertEqual(report["constituent_ids_filled"], 0)


if __name__ == "__main__":
    unittest.main()
