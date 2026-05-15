import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.eval.export_usable_parameter_annotation import build_usable_parameter_rows, export_usable_parameter_packets


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class UsableParameterAnnotationExportTests(unittest.TestCase):
    def test_build_usable_parameter_rows_exports_minimum_blocks(self):
        with TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir) / "10.1000_example"
            _write_json(
                paper_dir / "materials_extracted.json",
                {
                    "document": {"doi": "10.1000/example", "title": "Example"},
                    "materials": [{"material_id": "mat_1", "name": "Ti alloy", "material_class": "alloy", "phase_mode": "single_phase"}],
                    "constituents": [{"constituent_id": "const_1", "material_id": "mat_1", "process_state_id": "ps_1", "constituent_type": "phase", "name": "alpha", "crystal_structure": {"lattice_type": "HCP"}}],
                    "process_states": [{"process_state_id": "ps_1", "material_ids": ["mat_1"], "label": "heat treated"}],
                    "conditions": [{"condition_id": "cond_1", "label": "RT tension", "temperature": {"reported_value": 298, "reported_unit": "K"}, "strain_rate": {"reported_value": 0.001, "reported_unit": "s^-1"}}],
                    "models": [{
                        "model_id": "model_1",
                        "name": "CPFE",
                        "model_type": "crystal_plasticity",
                        "model_role": "primary_simulation",
                        "solver_framework": {"scale": "polycrystal", "discretization": "fem"},
                        "implementation": {"software": "Abaqus", "code_name": "UMAT"},
                        "constitutive_description": {
                            "kinematics": "finite_strain",
                            "flow_kinetics": {"flow_rule_form": "power_law", "rate_dependence": "rate_dependent"},
                            "hardening": {"slip_hardening_law": "Voce"}
                        },
                        "constitutive_branches": [{"branch_id": "branch_1", "name": "slip"}]
                    }],
                    "deformation_systems": [{
                        "system_id": "sys_1",
                        "family_id": "fam_1",
                        "model_id": "model_1",
                        "constituent_id": "const_1",
                        "family_name": "basal",
                        "system_type": "slip",
                        "plane": "{0001}",
                        "direction": "<11-20>",
                        "number_of_systems": 3,
                    }],
                    "evidence_objects": [{
                        "evidence_id": "ev_1",
                        "evidence_type": "table_row",
                        "source_file": "table_001.json",
                        "section_heading": "Methods",
                        "locator": {"table_id": "Table 1", "row_name": "tau0", "column_name": "Value", "value": "85 MPa", "excerpt": "tau0 85 MPa"},
                    }],
                    "parameter_claims": [{
                        "claim_id": "claim_1",
                        "parameter": {"canonical_name": "tau0", "parameter_family": "crss", "raw_name": "tau0", "symbol_reported": "tau0", "domain": "plastic"},
                        "assertion": {"reported_value": 85, "reported_unit": "MPa"},
                        "applies_to": {
                            "material_id": "mat_1",
                            "constituent_id": "const_1",
                            "process_state_id": "ps_1",
                            "model_id": "model_1",
                            "condition_id": "cond_1",
                            "branch_ids": ["branch_1"],
                            "mechanism": "slip",
                            "family_id": "fam_1",
                            "family_name": "basal",
                            "system_ids": ["sys_1"],
                            "scope": "family",
                            "notes": "Basal slip parameter."
                        },
                        "provenance": {"origin_type": "calibrated", "reference_ids": [], "adopted_from_reference_ids": [], "calibration_based_on_reference_ids": ["12"], "calibration": {"method": "manual_fitting", "target_type": "stress_strain_curve"}},
                        "evidence_ids": ["ev_1"],
                    }],
                },
            )
            _write_json(
                paper_dir / "llm_evaluation.json",
                {"parameter_audits": [{"location": "claim:claim_1", "verdict": "warning", "review_required": True}]},
            )

            rows = build_usable_parameter_rows(paper_dir)
            self.assertEqual(1, len(rows))
            row = rows[0]
            self.assertEqual("Ti alloy", row["material_object"]["material_name"])
            self.assertEqual("crystal_plasticity", row["cp_model"]["model_type"])
            self.assertIn("CPFE", row["cp_model"]["model_label"])
            self.assertIn("power_law", row["cp_model"]["model_label"])
            self.assertEqual("tau0", row["parameter_body"]["canonical_name"])
            self.assertEqual("tau0", row["parameter_body"]["symbol"])
            self.assertEqual("plastic", row["parameter_body"]["domain"])
            self.assertEqual("family:basal ({0001}<11-20>, n=3)", row["parameter_scope"]["scope_target"])
            self.assertNotIn("file", row["evidence"])
            self.assertNotIn("row_name", row["evidence"])
            self.assertNotIn("column_name", row["evidence"])
            self.assertNotIn("value_text", row["evidence"])
            self.assertEqual("calibrated", row["provenance"]["origin_type"])
            self.assertNotIn("source_scope", row["provenance"])
            self.assertEqual({"status": "correct"}, row["annotation"])

    def test_export_usable_parameter_packets_writes_combined_outputs(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "data"
            paper_dir = root / "10.1000_example"
            _write_json(
                paper_dir / "materials_extracted.json",
                {
                    "document": {"doi": "10.1000/example"},
                    "materials": [],
                    "constituents": [],
                    "process_states": [],
                    "conditions": [],
                    "models": [],
                    "deformation_systems": [],
                    "evidence_objects": [],
                    "parameter_claims": [{
                        "claim_id": "claim_1",
                        "parameter": {"canonical_name": "m", "symbol_reported": "m"},
                        "assertion": {"reported_value": 0.02, "reported_unit": None},
                        "applies_to": {"scope": "global"},
                        "provenance": {"origin_type": "adopted"},
                        "evidence_ids": [],
                    }],
                },
            )
            _write_json(paper_dir / "llm_evaluation.json", {})

            outdir = Path(tmpdir) / "out"
            stats = export_usable_parameter_packets(root, outdir)

            self.assertEqual(1, stats["papers"])
            self.assertEqual(1, stats["records"])
            self.assertTrue((outdir / "usable_parameter_annotation_draft.jsonl").exists())
            self.assertTrue((outdir / "usable_parameter_annotation_sheet.csv").exists())
            self.assertTrue((outdir / "packets" / "10.1000_example" / "usable_parameters.csv").exists())


if __name__ == "__main__":
    unittest.main()
