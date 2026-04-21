import json
import tempfile
import unittest
from pathlib import Path

from postprocess.model_equation_binding import bind_model_equations


class ModelEquationBindingTests(unittest.TestCase):
    def test_bind_model_equations_resolves_model_branch_and_claim_equations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eq_dir = Path(tmpdir) / "equations"
            eq_dir.mkdir(parents=True, exist_ok=True)
            (eq_dir / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "equation_id": "eq_0003",
                            "label": "(3)",
                            "section_title": "Constitutive law",
                            "paragraph_text": "Plastic flow rule.",
                            "text": "gamma_dot_0,1 term",
                            "latex": "\\dot{\\gamma}_{0,1}",
                            "text_file": "equation_003.txt",
                        },
                        {
                            "equation_id": "eq_0007",
                            "label": "(7)",
                            "section_title": "Constitutive law",
                            "paragraph_text": "Backstress evolution.",
                            "text": "h gamma_dot - hD X |gamma_dot|",
                            "latex": "\\dot{X}=h\\dot{\\gamma}-h_DX|\\dot{\\gamma}|",
                            "text_file": "equation_007.txt",
                        },
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            extracted = {
                "models": [
                    {
                        "model_id": "model_cp",
                        "equation_ids": ["eq_0003", "eq_0007"],
                        "constitutive_branches": [
                            {
                                "branch_id": "branch_backstress",
                                "branch_type": "backstress_evolution",
                                "governing_equation_ids": ["eq_0007"],
                            }
                        ],
                    }
                ],
                "parameter_claims": [
                    {
                        "claim_id": "claim_h",
                        "governing_equation_ids": ["eq_0007"],
                        "provenance": {"calibration": {}},
                    }
                ],
                "evidence_objects": [],
            }

            updated, report = bind_model_equations(extracted, paper_dir=tmpdir)

            self.assertEqual(report["bound_models"], 1)
            self.assertGreaterEqual(report["bound_equations"], 4)
            model = updated["models"][0]
            self.assertEqual(["eq_0003", "eq_0007"], model["equation_ids"])
            self.assertEqual(2, len(model["equations"]))
            self.assertEqual(["eq_0007"], model["constitutive_branches"][0]["governing_equation_ids"])
            self.assertEqual(1, len(model["constitutive_branches"][0]["governing_equations"]))
            claim = updated["parameter_claims"][0]
            self.assertEqual(["eq_0007"], claim["governing_equation_ids"])
            self.assertEqual(1, len(claim["governing_equations"]))

    def test_bind_model_equations_promotes_branch_equations_into_model_union(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eq_dir = Path(tmpdir) / "equations"
            eq_dir.mkdir(parents=True, exist_ok=True)
            (eq_dir / "index.json").write_text(
                json.dumps(
                    [
                        {"equation_id": "eq_0003", "label": "(3)", "text": "plastic", "latex": "", "text_file": "equation_003.txt"},
                        {"equation_id": "eq_0004", "label": "(4)", "text": "creep", "latex": "", "text_file": "equation_004.txt"},
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            extracted = {
                "models": [
                    {
                        "model_id": "model_cp",
                        "equation_ids": ["eq_0003"],
                        "constitutive_branches": [
                            {"branch_type": "plastic_flow", "governing_equation_ids": ["eq_0003"]},
                            {"branch_type": "creep_flow", "governing_equation_ids": ["eq_0004"]},
                        ],
                    }
                ],
                "evidence_objects": [],
            }

            updated, _ = bind_model_equations(extracted, paper_dir=tmpdir)

            self.assertEqual(["eq_0003", "eq_0004"], updated["models"][0]["equation_ids"])

    def test_bind_model_equations_resolves_inline_equation_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eq_dir = Path(tmpdir) / "equations"
            eq_dir.mkdir(parents=True, exist_ok=True)
            (eq_dir / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "equation_id": "eq_0003",
                            "equation_index": 3,
                            "label": "(3)",
                            "section_title": "Constitutive law",
                            "text": "plastic flow rule",
                            "latex": "\\dot{\\gamma}_{0,1}",
                            "text_file": "equation_003.txt",
                        },
                        {
                            "equation_id": "eq_0007",
                            "equation_index": 7,
                            "label": "(7)",
                            "section_title": "Constitutive law",
                            "text": "backstress evolution",
                            "latex": "\\dot{X}",
                            "text_file": "equation_007.txt",
                        },
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            extracted = {
                "models": [{"model_id": "model_cp", "equation_ids": ["Eq. (3)", "(7)"]}],
                "parameter_claims": [
                    {
                        "claim_id": "claim_h",
                        "governing_equation_ids": ["(7)"],
                        "provenance": {"calibration": {}},
                    }
                ],
                "evidence_objects": [],
            }

            updated, _ = bind_model_equations(extracted, paper_dir=tmpdir)

            self.assertEqual(["eq_0003", "eq_0007"], updated["models"][0]["equation_ids"])
            claim = updated["parameter_claims"][0]
            self.assertEqual(["eq_0007"], claim["governing_equation_ids"])

    def test_bind_model_equations_backfills_from_inline_section_symbols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sections_dir = Path(tmpdir) / "sections"
            sections_dir.mkdir(parents=True, exist_ok=True)
            (sections_dir / "004_Constitutive law.md").write_text(
                "\n".join(
                    [
                        "## Constitutive law",
                        "",
                        "(3) gamma_dot_alpha = gamma_dot_0,1 * (...) where gamma_dot_0,1 is the reference strain rate and m1 is the strain rate sensitivity.",
                        "",
                        "(4) gamma_dot_alpha = gamma_dot_0,1 * (...) + gamma_dot_0,2 * (...) where gamma_dot_0,2 and m2 are the reference strain rate and strain rate sensitivity for the creep regime.",
                        "",
                        "(5) tau_c_dot_alpha = h0 * (...) - A * tau_c_alpha^d * exp(-QRT) where h0 is the initial hardening modulus, tau_c0 is the initial slip hardening, n, A, and d are fitting parameters, Q is the creep activation energy, R is the gas constant, and T is the temperature.",
                        "",
                        "(7) X_dot_alpha = h * gamma_dot_alpha - hD * X_alpha * |gamma_dot_alpha| where h is the hardening coefficient and hD is related to the dynamic recovery.",
                    ]
                ),
                encoding="utf-8",
            )

            extracted = {
                "models": [
                    {
                        "model_id": "model_cp",
                        "equation_ids": [],
                        "constitutive_branches": [
                            {"branch_id": "branch_plastic", "branch_type": "plastic_flow", "governing_equation_ids": []},
                            {"branch_id": "branch_creep", "branch_type": "creep_flow", "governing_equation_ids": []},
                            {"branch_id": "branch_hardening", "branch_type": "hardening", "governing_equation_ids": []},
                        ],
                    }
                ],
                "parameter_claims": [
                    {
                        "claim_id": "claim_gamma01",
                        "parameter": {"canonical_name": "reference_shear_rate_plastic", "symbol_reported": "gamma_dot_0,1"},
                        "applies_to": {"model_id": "model_cp", "branch_id": "branch_plastic"},
                        "governing_equation_ids": [],
                    },
                    {
                        "claim_id": "claim_h0",
                        "parameter": {"canonical_name": "hardening_h0", "symbol_reported": "h0"},
                        "applies_to": {"model_id": "model_cp", "branch_id": "branch_hardening"},
                        "governing_equation_ids": [],
                    },
                    {
                        "claim_id": "claim_h",
                        "parameter": {"canonical_name": "hardening_coefficient_h", "symbol_reported": "h"},
                        "applies_to": {"model_id": "model_cp", "branch_id": "branch_hardening"},
                        "governing_equation_ids": [],
                    },
                ],
                "evidence_objects": [],
            }

            updated, report = bind_model_equations(extracted, paper_dir=tmpdir)

            self.assertEqual(4, report["equation_sources"])
            self.assertGreaterEqual(report["branches_backfilled"], 2)
            self.assertGreaterEqual(report["claims_backfilled"], 3)
            model = updated["models"][0]
            self.assertIn("(3)", model["equation_ids"])
            self.assertIn("(5)", model["equation_ids"])
            self.assertIn("(7)", model["equation_ids"])
            branches = {row["branch_id"]: row for row in model["constitutive_branches"]}
            self.assertIn("(5)", branches["branch_hardening"]["governing_equation_ids"])
            claims = {row["claim_id"]: row for row in updated["parameter_claims"]}
            self.assertEqual(["(5)"], claims["claim_h0"]["governing_equation_ids"])
            self.assertEqual(["(7)"], claims["claim_h"]["governing_equation_ids"])
            self.assertGreaterEqual(len(claims["claim_h"]["governing_equations"]), 1)

    def test_bind_model_equations_uses_equation_index_when_equation_id_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eq_dir = Path(tmpdir) / "equations"
            eq_dir.mkdir(parents=True, exist_ok=True)
            (eq_dir / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "equation_index": 3,
                            "label": "(3)",
                            "section_title": "Constitutive law",
                            "text": "gamma_dot_0,1 term",
                            "latex": "\\dot{\\gamma}_{0,1}",
                            "text_file": "equation_003.txt",
                        },
                        {
                            "equation_index": 5,
                            "label": "(5)",
                            "section_title": "Constitutive law",
                            "text": "h0 tau_c0 Q",
                            "latex": "\\dot{\\tau}_c",
                            "text_file": "equation_005.txt",
                        },
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            extracted = {
                "models": [
                    {
                        "model_id": "model_cp",
                        "equation_ids": ["(3)"],
                        "constitutive_branches": [
                            {"branch_id": "branch_hardening", "branch_type": "hardening", "governing_equation_ids": ["(5)"]},
                        ],
                    }
                ],
                "parameter_claims": [
                    {
                        "claim_id": "claim_h0",
                        "parameter": {"canonical_name": "hardening_h0", "symbol_reported": "h0"},
                        "applies_to": {"model_id": "model_cp", "branch_id": "branch_hardening"},
                        "governing_equation_ids": [],
                    }
                ],
                "evidence_objects": [],
            }

            updated, report = bind_model_equations(extracted, paper_dir=tmpdir)

            self.assertEqual(2, report["equation_sources"])
            model = updated["models"][0]
            self.assertEqual(["eq_0003", "eq_0005"], model["equation_ids"])
            self.assertEqual(["eq_0005"], model["constitutive_branches"][0]["governing_equation_ids"])
            claim = updated["parameter_claims"][0]
            self.assertEqual(["eq_0005"], claim["governing_equation_ids"])
            self.assertEqual("eq_0005", claim["governing_equations"][0]["equation_id"])


if __name__ == "__main__":
    unittest.main()
