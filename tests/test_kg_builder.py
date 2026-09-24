import unittest

from kg.builder import build_graph_from_extraction


class KnowledgeGraphBuilderTests(unittest.TestCase):
    def test_builds_claim_binding_graph(self):
        extracted = {
            "document": {"title": "Example", "doi": "10.1000/example"},
            "materials": [
                {
                    "material_id": "mat_1",
                    "name": "Ti-6Al-4V",
                    "phases": [{"phase_id": "phase_alpha", "name": "alpha"}],
                }
            ],
            "process_states": [{"process_state_id": "ps_1", "material_id": "mat_1", "name": "annealed"}],
            "conditions": [{"condition_id": "cond_1", "label": "298 K tension"}],
            "models": [{"model_id": "model_1", "framework": "CPFE"}],
            "deformation_systems": [{"system_id": "sys_1", "model_id": "model_1", "family_name": "basal"}],
            "evidence_objects": [{"evidence_id": "ev_1", "snippet": "tau0 = 85 MPa"}],
            "references": [{"reference_id": "ref_1", "doi": "10.1000/ref", "title": "Source"}],
            "parameter_claims": [
                {
                    "claim_id": "claim_1",
                    "canonical_name": "initial_slip_resistance",
                    "symbol": "tau0",
                    "value": 85,
                    "unit": "MPa",
                    "applies_to": {
                        "material_id": "mat_1",
                        "process_state_id": "ps_1",
                        "condition_id": "cond_1",
                        "phase_id": "phase_alpha",
                        "model_id": "model_1",
                        "system_ids": ["sys_1"],
                        "mechanism": "slip",
                    },
                    "evidence_ids": ["ev_1"],
                    "provenance": {"adopted_from_reference_ids": ["ref_1"]},
                    "confidence": {"score": 92.5},
                }
            ],
        }

        graph = build_graph_from_extraction("", extracted)
        node_ids = {node["node_id"] for node in graph["nodes"]}
        edge_types = {(edge["source_id"], edge["edge_type"], edge["target_id"]) for edge in graph["edges"]}

        self.assertIn("paper:10.1000/example", node_ids)
        self.assertIn("claim:10.1000/example:claim_1", node_ids)
        self.assertIn("parameter:initial_slip_resistance", node_ids)
        self.assertIn(
            (
                "claim:10.1000/example:claim_1",
                "APPLIES_TO_MATERIAL",
                "material:10.1000/example:mat_1",
            ),
            edge_types,
        )
        self.assertIn(
            (
                "claim:10.1000/example:claim_1",
                "APPLIES_TO_DEFORMATION_SYSTEM",
                "deformationsystem:10.1000/example:sys_1",
            ),
            edge_types,
        )
        self.assertIn(
            (
                "claim:10.1000/example:claim_1",
                "SUPPORTED_BY",
                "evidence:10.1000/example:ev_1",
            ),
            edge_types,
        )
        self.assertIn(
            (
                "claim:10.1000/example:claim_1",
                "ADOPTED_FROM",
                "reference:10.1000/ref",
            ),
            edge_types,
        )

    def test_legacy_claim_inline_evidence_gets_synthetic_evidence_node(self):
        extracted = {
            "source_document": {"doi": "10.1000/legacy", "title": "Legacy"},
            "material": {"name": "Al"},
            "constitutive_model": {"framework": "CPFEM"},
            "parameter_claims": [
                {
                    "claim_id": "h0_range",
                    "canonical_name": "hardening_h0",
                    "value": "[245, 260]",
                    "unit": "MPa",
                    "applies_to": {"mechanism": "all_slip"},
                    "source": {"evidence_text": "h0 | [245, 260] MPa"},
                }
            ],
        }

        graph = build_graph_from_extraction("", extracted)
        node_ids = {node["node_id"] for node in graph["nodes"]}
        edge_types = {(edge["source_id"], edge["edge_type"], edge["target_id"]) for edge in graph["edges"]}

        self.assertIn("evidence:10.1000/legacy:h0_range:inline", node_ids)
        self.assertIn(
            (
                "claim:10.1000/legacy:h0_range",
                "SUPPORTED_BY",
                "evidence:10.1000/legacy:h0_range:inline",
            ),
            edge_types,
        )


if __name__ == "__main__":
    unittest.main()
