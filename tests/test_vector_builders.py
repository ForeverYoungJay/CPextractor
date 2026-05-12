import unittest

from db.vector_builders import build_parameter_vector_rows


class VectorBuilderTests(unittest.TestCase):
    def test_parameter_vector_rows_include_hierarchy_context(self):
        extracted_json = {
            "document": {"title": "Example Paper"},
            "materials": [
                {
                    "material_id": "mat_001",
                    "name": "Ti-6Al-4V",
                    "formula": None,
                    "composition": {
                        "components": [
                            {"component": "Al", "value": 6, "unit": "wt%"},
                            {"component": "V", "value": 4, "unit": "wt%"},
                        ]
                    },
                    "phases": [
                        {
                            "phase_id": "phase_001",
                            "name": "alpha",
                            "crystal_structure": {"lattice_type": "hcp"},
                        }
                    ],
                }
            ],
            "samples": [
                {
                    "sample_id": "samp_001",
                    "material_id": "mat_001",
                    "label": "aged",
                    "processing_state": "solution treated + aged",
                    "condition_ids": ["cond_001"],
                }
            ],
            "conditions": [
                {
                    "condition_id": "cond_001",
                    "label": "room temperature tension",
                    "temperature": {"value": 298, "unit": "K"},
                    "strain_rate": {"value": 0.001, "unit": "s^-1"},
                }
            ],
            "models": [{"model_id": "model_001", "framework": "cpfe"}],
            "deformation_systems": [
                {
                    "system_id": "sys_basal",
                    "model_id": "model_001",
                    "system_type": "slip",
                    "family_name": "basal",
                    "plane": "{0001}",
                    "direction": "<11-20>",
                }
            ],
            "simulation_geometries": [
                {
                    "geometry_id": "geom_001",
                    "model_id": "model_001",
                    "geometry_type": "grain_aggregate",
                    "mesh_type": "voxel",
                    "periodic_geometry": "yes",
                }
            ],
            "orientation_inputs": [
                {
                    "orientation_id": "ori_001",
                    "model_id": "model_001",
                    "source": "ebsd",
                    "representation": "euler_angles",
                }
            ],
            "numerical_methods": [
                {
                    "numerical_method_id": "num_001",
                    "model_id": "model_001",
                    "time_integration": "implicit",
                }
            ],
            "parameter_claims": [
                {
                    "claim_id": "claim_0001",
                    "model_id": "model_001",
                    "canonical_name": "tau0",
                    "symbol": "tau_0",
                    "domain": "plastic",
                    "value": 85,
                    "unit": "MPa",
                    "applies_to": {
                        "material_id": "mat_001",
                        "sample_id": "samp_001",
                        "condition_id": "cond_001",
                        "phase_id": "phase_001",
                        "scope": "phase",
                        "system_ids": ["sys_basal"],
                    },
                    "provenance": {"origin_type": "calibrated"},
                    "evidence": {"file": "table_003.json", "evidence_text": "tau0 = 85 MPa"},
                    "claim_class": "material_constitutive_parameter",
                }
            ],
        }

        rows = build_parameter_vector_rows("10.1000/example", extracted_json)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["material_id"], "mat_001")
        self.assertEqual(row["sample_id"], "samp_001")
        self.assertEqual(row["condition_id"], "cond_001")
        self.assertEqual(row["phase_name"], "alpha")
        self.assertEqual(row["model_id"], "model_001")
        self.assertEqual(row["claim_class"], "material_constitutive_parameter")
        self.assertIn("Claim class: material_constitutive_parameter", row["retrieval_text"])
        self.assertIn("Material: Ti-6Al-4V", row["retrieval_text"])
        self.assertIn("Process state: aged", row["retrieval_text"])
        self.assertIn("Condition: room temperature tension", row["retrieval_text"])
        self.assertIn("Constituent: alpha", row["retrieval_text"])
        self.assertIn("Geometry: grain_aggregate; voxel; periodic yes", row["retrieval_text"])
        self.assertIn("Orientation input: ebsd; euler_angles", row["retrieval_text"])
        self.assertIn("Systems: slip; basal; {0001}; <11-20>", row["retrieval_text"])


if __name__ == "__main__":
    unittest.main()
