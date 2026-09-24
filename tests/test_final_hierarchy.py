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

        self.assertEqual(updated["schema_version"], "6.0.0")
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

    def test_preserves_v6_extractor_fields_but_drops_evaluator_owned_assessments(self):
        extracted_json = {
            "schema_version": "6.0.0",
            "materials": [{"material_id": "mat_ti", "name": "Ti", "normalized_name": "ti"}],
            "models": [{"model_id": "model_cp", "constitutive_branches": []}],
            "deformation_families": [{"family_id": "family_basal", "family_name": "basal <a>"}],
            "deformation_systems": [{"system_id": "sys_basal", "family_id": "family_basal"}],
            "equations": [{"equation_id": "eq_1", "equation_label": "(1)", "equation_type": "flow_rule"}],
            "parameter_claims": [
                {
                    "claim_id": "claim_tau0",
                    "parameter": {"canonical_name": "tau0", "symbol_normalized": "tau_0"},
                    "assertion": {"reported_value": "85", "reported_unit": "MPa", "normalized_value": 85, "normalized_unit": "MPa"},
                    "provenance": {"origin_type": "reported_in_current_paper", "source_scope": "current_paper"},
                    "simulation_role": {"is_required_for_simulation": "yes"},
                }
            ],
            "simulation_readiness": [
                {
                    "readiness_id": "ready_model_cp",
                    "model_id": "model_cp",
                    "status": "partial",
                    "missing_required_items": ["boundary_conditions"],
                }
            ],
            "quality_control": {
                "extraction_status": "partial",
                "detected_warnings": [{"warning_type": "incomplete_model", "message": "Boundary conditions are not explicit.", "severity": "medium"}],
            },
        }

        updated, report = build_final_hierarchy(extracted_json)

        self.assertEqual("6.0.0", updated["schema_version"])
        self.assertEqual("ti", updated["materials"][0]["normalized_name"])
        self.assertEqual("family_basal", updated["deformation_families"][0]["family_id"])
        self.assertEqual("eq_1", updated["equations"][0]["equation_id"])
        self.assertEqual(85, updated["parameter_claims"][0]["assertion"]["normalized_value"])
        self.assertEqual("tau_0", updated["parameter_claims"][0]["parameter"]["symbol_normalized"])
        self.assertEqual("yes", updated["parameter_claims"][0]["simulation_role"]["is_required_for_simulation"])
        self.assertNotIn("simulation_readiness", updated)
        self.assertNotIn("quality_control", updated)
        self.assertEqual(1, report["deformation_families"])
        self.assertEqual(1, report["equations"])

    def test_coerces_scientific_notation_strings_in_claim_values(self):
        extracted_json = {
            "schema_version": "6.0.0",
            "materials": [{"material_id": "mat_1", "name": "steel"}],
            "models": [{"model_id": "model_cp", "constitutive_branches": []}],
            "parameter_claims": [
                {
                    "claim_id": "claim_rho",
                    "parameter": {"canonical_name": "rho0"},
                    "assertion": {
                        "reported_value": "10 4",
                        "reported_unit": "mm^-2",
                        "normalized_value": "10 4",
                        "normalized_unit": "mm^-2",
                    },
                },
                {
                    "claim_id": "claim_kb",
                    "parameter": {"canonical_name": "kb"},
                    "assertion": {"reported_value": "10 −6", "normalized_value": "10 −6"},
                },
                {
                    "claim_id": "claim_dl",
                    "parameter": {"canonical_name": "DL"},
                    "assertion": {"reported_value": "1.56 × 10 −8", "normalized_value": "1.56 × 10 −8"},
                },
            ],
        }

        updated, _ = build_final_hierarchy(extracted_json)
        claims = {row["claim_id"]: row for row in updated["parameter_claims"]}

        self.assertEqual(10000, claims["claim_rho"]["assertion"]["reported_value"])
        self.assertEqual(1e-6, claims["claim_kb"]["assertion"]["reported_value"])
        self.assertEqual(1.56e-8, claims["claim_dl"]["assertion"]["reported_value"])
        self.assertEqual(1.56e-8, claims["claim_dl"]["assertion"]["normalized_value"])

    def test_preserves_raw_like_top_level_order_and_empty_geometry_array(self):
        extracted_json = {
            "schema_version": "6.0.0",
            "materials": [],
            "process_states": [],
            "constituents": [],
            "microstructure_features": [],
            "deformation_systems": [],
            "deformation_families": [],
            "models": [],
            "equations": [],
            "simulation_geometries": [],
            "numerical_methods": [],
            "conditions": [],
            "parameter_claims": [],
            "evidence_objects": [],
            "global_notes": None,
        }

        updated, _ = build_final_hierarchy(extracted_json)
        keys = list(updated.keys())

        self.assertIn("simulation_geometries", updated)
        self.assertEqual([], updated["simulation_geometries"])
        self.assertLess(keys.index("deformation_systems"), keys.index("deformation_families"))
        self.assertLess(keys.index("equations"), keys.index("simulation_geometries"))
        self.assertLess(keys.index("simulation_geometries"), keys.index("numerical_methods"))
        self.assertLess(keys.index("evidence_objects"), keys.index("global_notes"))


if __name__ == "__main__":
    unittest.main()
