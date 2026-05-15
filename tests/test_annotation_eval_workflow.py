import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

from scripts.eval.prepare_annotation_pilot import (
    collect_pilot_candidates,
    export_pilot_packets,
    select_pilot_candidates,
)
from scripts.eval.run_annotation_benchmarks import build_annotation_benchmark_summary


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _paper_payload(*, doi: str, claim_id: str, value: float, unit: str | None, review_required: bool, verdict: str, table_kind: str) -> dict:
    table_file = "table_001.json"
    return {
        "document": {"doi": doi, "title": f"Title for {doi}"},
        "materials": [{"material_id": "mat_001", "name": "Ti-6Al-4V", "material_class": "alloy", "phase_mode": "single_phase"}],
        "constituents": [{
            "constituent_id": "const_001",
            "material_id": "mat_001",
            "process_state_id": "ps_001",
            "constituent_type": "phase",
            "name": "alpha",
            "crystal_structure": {"lattice_type": "HCP"},
        }],
        "process_states": [{"process_state_id": "ps_001", "material_ids": ["mat_001"], "label": "as-built"}],
        "conditions": [{"condition_id": "cond_001", "label": "room temperature", "temperature": {"reported_value": 298, "reported_unit": "K"}, "strain_rate": {"reported_value": 0.001, "reported_unit": "s^-1"}}],
        "models": [{
            "model_id": "model_001",
            "name": "CPFE model",
            "model_type": "crystal_plasticity",
            "model_role": "primary_simulation",
            "solver_framework": {"scale": "polycrystal", "discretization": "fem"},
            "implementation": {"software": "Abaqus", "code_name": "UMAT"},
            "constitutive_description": {
                "kinematics": "finite_strain",
                "flow_kinetics": {"flow_rule_form": "power_law", "rate_dependence": "rate_dependent"},
                "hardening": {"slip_hardening_law": "Voce"},
            },
            "constitutive_branches": [],
        }],
        "parameter_claims": [
            {
                "claim_id": claim_id,
                "parameter": {
                    "canonical_name": "tau0",
                    "parameter_family": "crss",
                    "raw_name": "tau0",
                    "symbol_reported": "tau0",
                    "domain": "plastic",
                },
                "assertion": {"reported_value": value, "reported_unit": unit},
                "applies_to": {
                    "material_id": "mat_001",
                    "constituent_id": "const_001",
                    "process_state_id": "ps_001",
                    "model_id": "model_001",
                    "condition_id": "cond_001",
                    "branch_ids": [],
                    "mechanism": "slip",
                    "family_id": None,
                    "family_name": None,
                    "system_ids": [],
                    "scope": "global",
                },
                "provenance": {"origin_type": "calibrated", "reference_ids": []},
                "evidence_ids": ["ev_001"],
            }
        ],
        "evidence_objects": [
            {
                "evidence_id": "ev_001",
                "evidence_type": "table_row",
                "source_file": table_file,
                "locator": {"row_name": "tau0", "column_name": "Value", "value": str(value), "excerpt": "tau0 | value"},
            }
        ],
        "_test_table_kind": table_kind,
        "_test_review_required": review_required,
        "_test_verdict": verdict,
    }


def _write_paper_dir(root: Path, folder: str, payload: dict) -> Path:
    paper_dir = root / folder
    _write_json(paper_dir / "materials_extracted.json", {k: v for k, v in payload.items() if not k.startswith("_test_")})
    _write_json(
        paper_dir / "llm_evaluation.json",
        {
            "parameter_audits": [
                {
                    "location": f"claim:{payload['parameter_claims'][0]['claim_id']}",
                    "verdict": payload["_test_verdict"],
                    "review_required": payload["_test_review_required"],
                    "policy_adjustments": [],
                }
            ]
        },
    )
    _write_json(paper_dir / "tables" / "table_001.json", {"table_kind": payload["_test_table_kind"], "rows": [["tau0", "85"]]})
    (paper_dir / "paper.md").write_text(f"# {payload['document']['title']}\n", encoding="utf-8")
    return paper_dir


class AnnotationEvalWorkflowTests(unittest.TestCase):
    def test_select_pilot_candidates_balances_journals(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_paper_dir(
                root,
                "10.1016_j.actamat.2024.000001",
                _paper_payload(
                    doi="10.1016/j.actamat.2024.000001",
                    claim_id="claim_0001",
                    value=85,
                    unit="MPa",
                    review_required=True,
                    verdict="warning",
                    table_kind="image_backed",
                ),
            )
            _write_paper_dir(
                root,
                "10.1016_j.msea.2024.000002",
                _paper_payload(
                    doi="10.1016/j.msea.2024.000002",
                    claim_id="claim_0002",
                    value=90,
                    unit="MPa",
                    review_required=False,
                    verdict="accepted",
                    table_kind="digital",
                ),
            )
            _write_paper_dir(
                root,
                "10.1016_j.actamat.2024.000003",
                _paper_payload(
                    doi="10.1016/j.actamat.2024.000003",
                    claim_id="claim_0003",
                    value=0,
                    unit=None,
                    review_required=False,
                    verdict="warning",
                    table_kind="digital",
                ),
            )

            candidates = collect_pilot_candidates(root, "materials_extracted.json")
            selected = select_pilot_candidates(candidates, n=2, strategy="journal_balanced")

            self.assertEqual(2, len(selected))
            self.assertEqual(
                {"actamat", "msea"},
                {row["journal_key"] for row in selected},
            )
            self.assertGreater(selected[0]["difficulty_score"], 0)

    def test_export_pilot_packets_writes_manifest_and_packet_metadata(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_paper_dir(
                root,
                "10.1016_j.actamat.2024.000001",
                _paper_payload(
                    doi="10.1016/j.actamat.2024.000001",
                    claim_id="claim_0001",
                    value=85,
                    unit="MPa",
                    review_required=True,
                    verdict="warning",
                    table_kind="image_backed",
                ),
            )
            candidates = collect_pilot_candidates(root, "materials_extracted.json")
            output_root = root / "pilot_packets"
            export_pilot_packets(
                candidates,
                output_root=output_root,
                csv_profile="compact",
                source_name="materials_extracted.json",
                total_candidates=len(candidates),
                strategy="journal_balanced",
            )

            manifest_path = output_root / "manifest.csv"
            meta_path = output_root / "10.1016_j.actamat.2024.000001" / "packet_meta.json"
            readme_path = output_root / "README.md"
            csv_path = output_root / "10.1016_j.actamat.2024.000001" / "usable_parameters.csv"
            jsonl_path = output_root / "10.1016_j.actamat.2024.000001" / "usable_parameters.jsonl"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(meta_path.exists())
            self.assertTrue(readme_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(jsonl_path.exists())

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual("journal_balanced", meta["selection_strategy"])
            self.assertIn("difficulty_signals", meta)
            self.assertEqual(1, meta["difficulty_signals"]["image_backed_claim_count"])
            self.assertEqual("usable_parameter_annotation", meta["schema_type"])

    def test_build_annotation_benchmark_summary_extracts_primary_metrics(self):
        summary = build_annotation_benchmark_summary(
            {
                "gold_rows": 20,
                "gold_positive_rows": 18,
                "pred_rows": 22,
                "claim_detection": {"precision": 0.8, "recall": 0.9, "f1": 0.847, "tp": 16, "fp": 4, "fn": 2},
                "field_accuracy": {
                    "canonical_name_accuracy": 0.95,
                    "value_accuracy": 0.9,
                    "unit_accuracy": 0.85,
                    "grounding_accuracy": 0.75,
                    "grounding_annotation_coverage": 0.6,
                },
                "bundle": {"macro_bundle_completeness": 0.88},
            },
            {
                "paper_count": 5,
                "blocked_rate": 0.2,
                "gate_vs_gold_review_recommended": {"f1": 0.7},
                "gate_vs_gold_block_recommended": {"f1": 0.6},
            },
            {
                "slices": [
                    {"slice": "image_backed_table", "gold_rows": 3, "f1": 0.5, "value_accuracy": 0.33, "unit_accuracy": 0.66},
                    {"slice": "review_required", "gold_rows": 6, "f1": 0.7, "value_accuracy": 0.8, "unit_accuracy": 0.9},
                ]
            },
        )

        self.assertEqual(20, summary["gold_rows"])
        self.assertEqual(0.847, summary["claim_detection"]["f1"])
        self.assertEqual("review_required", summary["largest_slice"]["slice"])
        self.assertEqual(0.2, summary["gate"]["blocked_rate"])


if __name__ == "__main__":
    unittest.main()
