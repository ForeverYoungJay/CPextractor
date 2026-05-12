import unittest

from postprocess.model_setup_normalizer import normalize_model_setup


class ModelSetupNormalizerTests(unittest.TestCase):
    def test_backfills_new_model_setup_sections_from_explicit_context(self):
        extracted = {
            "models": [
                {
                    "model_id": "model_cp",
                    "solver_framework": {
                        "grain_resolution": "grain_resolved",
                        "discretization": "fft",
                        "boundary_condition_style": "periodic",
                        "geometry_representation": "voxelized",
                    },
                    "implementation": {
                        "subroutine": "vumat",
                    },
                    "constitutive_description": {
                        "slip_description": {"slip_families_defined": "yes"},
                        "twinning": {"enabled": "yes"},
                    },
                }
            ],
            "mechanisms": {
                "slip_families": [
                    {
                        "family_id": "fam_basal",
                        "name": "basal",
                        "num_systems": 3,
                        "phase_id": "const_alpha",
                    }
                ],
                "twinning_families": [
                    {
                        "family_id": "fam_twin",
                        "name": "extension_twin",
                        "num_systems": 6,
                        "phase_id": "const_alpha",
                    }
                ],
            },
            "microstructure_features": [
                {
                    "feature_id": "feat_tex",
                    "feature_family": "texture",
                    "feature_name": "texture",
                    "method": "ebsd",
                    "description": "Measured Euler angles used as model input.",
                    "evidence_ids": ["ev_tex"],
                }
            ],
            "parameter_claims": [
                {
                    "claim_id": "claim_tau",
                    "applies_to": {
                        "model_id": "model_cp",
                        "condition_id": "cond_rt",
                        "family_id": "fam_basal",
                    },
                    "provenance": {
                        "calibration": {
                            "target_type": "stress_strain_curve",
                            "observation_scope": "macroscopic",
                            "target_description": "Fit to macroscopic tensile curve",
                        }
                    },
                    "evidence_ids": ["ev_1"],
                }
            ],
        }

        updated, report = normalize_model_setup(extracted)

        self.assertEqual(2, report["deformation_systems_added"])
        self.assertEqual(1, report["simulation_geometries_added"])
        self.assertEqual(1, report["orientation_inputs_added"])
        self.assertEqual(0, report["numerical_methods_added"])
        self.assertEqual(0, report["simulation_outputs_added"])
        self.assertEqual(0, report["model_evaluations_added"])
        self.assertEqual(["fam_basal"], updated["models"][0]["constitutive_description"]["slip_description"]["deformation_system_ids"])
        self.assertEqual(["fam_twin"], updated["models"][0]["constitutive_description"]["twinning"]["deformation_system_ids"])
        self.assertEqual("voxel", updated["simulation_geometries"][0]["mesh_type"])
        self.assertEqual("yes", updated["simulation_geometries"][0]["periodic_geometry"])
        self.assertEqual("ebsd", updated["orientation_inputs"][0]["source"])
        self.assertEqual("euler_angles", updated["orientation_inputs"][0]["representation"])
        self.assertEqual([], updated.get("numerical_methods", []))
        self.assertEqual([], updated.get("simulation_outputs", []))
        self.assertEqual([], updated.get("model_evaluations", []))

    def test_infers_fcc_placeholder_deformation_system_when_only_fcc_slip_is_stated(self):
        extracted = {
            "materials": [
                {
                    "material_id": "mat_001",
                    "name": "Austenitic steel",
                }
            ],
            "constituents": [
                {
                    "constituent_id": "const_gamma",
                    "material_id": "mat_001",
                    "name": "austenite",
                    "crystal_structure": {"bravais_lattice": "fcc"},
                }
            ],
            "models": [
                {
                    "model_id": "model_cp",
                    "constituent_scope": ["const_gamma"],
                    "constitutive_description": {
                        "slip_description": {
                            "slip_families_defined": "yes",
                            "notes": "Slip systems for FCC materials were used in the simulation.",
                        }
                    },
                }
            ],
            "parameter_claims": [],
        }

        updated, report = normalize_model_setup(extracted)

        self.assertEqual(1, report["deformation_systems_added"])
        placeholder = updated["deformation_systems"][0]
        self.assertEqual("fcc_12", placeholder["family_name"])
        self.assertEqual("{111}", placeholder["plane"])
        self.assertEqual("<110>", placeholder["direction"])
        self.assertEqual(12, placeholder["number_of_systems"])
        self.assertIn("Inferred placeholder", placeholder["notes"])
        self.assertEqual(
            [placeholder["system_id"]],
            updated["models"][0]["constitutive_description"]["slip_description"]["deformation_system_ids"],
        )


if __name__ == "__main__":
    unittest.main()
