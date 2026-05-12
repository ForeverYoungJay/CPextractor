import unittest

from postprocess.final_hierarchy import build_final_hierarchy


class FinalHierarchyTests(unittest.TestCase):
    def test_builds_v5_1_hierarchy_and_enriches_claim_scope(self):
        extracted_json = {
            "schema_version": "2.1.1",
            "source_document": {
                "doi": "10.1000/example",
                "title": "Example Paper",
                "authors": ["A. Author"],
                "year": 2025,
                "journal_or_venue": "Acta Materialia",
            },
            "material": {
                "name": "Ti-6Al-4V",
                "chemical_formula": None,
                "phase": "single",
                "phases": [
                    {
                        "phase_id": "phase_001",
                        "phase_name": "alpha",
                        "role": "matrix",
                        "crystal_structure": {"lattice_type": "hcp"},
                    }
                ],
            },
            "paper_profile": {
                "studied_materials": [
                    {
                        "material_id": "mat_001",
                        "name": "Ti-6Al-4V",
                        "material_class": "titanium_alloy",
                        "phase_ids": ["phase_001"],
                    }
                ],
                "sample_profiles": [
                    {
                        "sample_id": "samp_001",
                        "material_id": "mat_001",
                        "condition_id": "cond_001",
                        "label": "aged",
                        "processing_state": "solution treated + aged",
                    }
                ],
            },
            "condition_profiles": [
                {
                    "condition_id": "cond_001",
                    "label": "room temperature tension",
                    "temperature": {"value": 298, "unit": "K"},
                    "strain_rate": {"value": 0.001, "unit": "s^-1"},
                    "loading_mode": "uniaxial_tension",
                    "stress_state": "uniaxial",
                }
            ],
            "constitutive_model": {
                "class": "crystal_plasticity",
                "framework": "cpfe",
                "kinematics": "finite_strain",
                "rate_dependence": "rate_dependent",
            },
            "deformation_mechanisms": {"slip_families": [{"family_id": "fam_001", "family_name": "basal"}]},
            "parameter_claims": [
                {
                    "claim_id": "claim_0001",
                    "canonical_name": "tau0",
                    "symbol": "tau_0",
                    "value": 85,
                    "unit": "MPa",
                    "applies_to": {
                        "sample_id": "samp_001",
                        "phase_id": "phase_001",
                        "family_id": "fam_001",
                        "scope": "family",
                    },
                    "source": {"origin_type": "calibrated"},
                    "evidence": {"evidence_text": "tau0 = 85 MPa"},
                }
            ],
        }

        updated, report = build_final_hierarchy(extracted_json)

        self.assertEqual(updated["schema_version"], "5.1.0")
        self.assertEqual(updated["document"]["doi"], "10.1000/example")
        self.assertNotIn("study", updated)
        self.assertEqual(len(updated["materials"]), 1)
        self.assertEqual(updated["constituents"][0]["constituent_id"], "phase_001")
        self.assertEqual(updated["constituents"][0]["material_id"], "mat_001")
        self.assertIn("process_states", updated)
        self.assertEqual(updated["conditions"][0]["condition_id"], "cond_001")
        self.assertEqual(updated["models"][0]["model_id"], "model_001")
        self.assertEqual(updated["parameter_claims"][0]["applies_to"]["model_id"], "model_001")
        self.assertEqual(updated["parameter_claims"][0]["applies_to"]["material_id"], "mat_001")
        self.assertIn("provenance", updated["parameter_claims"][0])
        self.assertEqual([], updated["parameter_claims"][0]["provenance"]["reference_ids"])
        self.assertEqual([], updated["parameter_claims"][0]["provenance"]["adopted_from_reference_ids"])
        self.assertEqual([], updated["parameter_claims"][0]["provenance"]["calibration_based_on_reference_ids"])
        self.assertEqual(report["parameter_claims"], 1)


if __name__ == "__main__":
    unittest.main()
