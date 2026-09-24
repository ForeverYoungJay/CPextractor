from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape


JSONDict = dict[str, Any]


def _safe_dict(value: Any) -> JSONDict:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _first(*values: Any) -> str:
    for value in values:
        text = _clean(value)
        if text:
            return text
    return ""


def _stable_hash(*parts: Any) -> str:
    raw = "\x1f".join(_clean(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _node_id(node_type: str, *parts: Any) -> str:
    cleaned = [_clean(part) for part in parts if _clean(part)]
    return f"{node_type.lower()}:" + ":".join(cleaned)


def _doi_from_extraction(extracted_json: JSONDict, fallback: str = "") -> str:
    document = _safe_dict(extracted_json.get("document"))
    source_document = _safe_dict(extracted_json.get("source_document"))
    return _first(
        fallback,
        document.get("doi"),
        source_document.get("doi"),
        extracted_json.get("doi"),
        extracted_json.get("record_id"),
    )


def _label_for_reference(ref: JSONDict) -> str:
    return _first(ref.get("doi"), ref.get("reference_doi"), ref.get("title"), ref.get("citation"), ref.get("reference_id"))


def _reference_id(doi: str, ref: Any) -> str:
    if isinstance(ref, dict):
        ref_doi = _first(ref.get("doi"), ref.get("reference_doi"))
        if ref_doi:
            return _node_id("reference", ref_doi.lower())
        local_id = _first(ref.get("reference_id"), ref.get("id"), ref.get("label"), ref.get("citation"), ref.get("title"))
        return _node_id("reference", doi, local_id or _stable_hash(ref))
    return _node_id("reference", doi, ref)


def _entity_id(entity: JSONDict, keys: Iterable[str]) -> str:
    for key in keys:
        value = _clean(entity.get(key))
        if value:
            return value
    return ""


@dataclass
class GraphBuilder:
    nodes: dict[str, JSONDict] = field(default_factory=dict)
    edges: dict[str, JSONDict] = field(default_factory=dict)

    def add_node(self, node_id: str, node_type: str, label: str = "", properties: JSONDict | None = None) -> str:
        if not node_id:
            return ""
        props = dict(properties or {})
        existing = self.nodes.get(node_id)
        if existing:
            existing_props = existing.setdefault("properties", {})
            for key, value in props.items():
                if value not in ("", None, [], {}):
                    existing_props[key] = value
            if label and not existing.get("label"):
                existing["label"] = label
            return node_id
        self.nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "label": label or node_id,
            "properties": props,
        }
        return node_id

    def add_edge(
        self,
        source_id: str,
        edge_type: str,
        target_id: str,
        *,
        doi: str = "",
        claim_id: str = "",
        confidence_score: float | None = None,
        evidence_ids: list[str] | None = None,
        properties: JSONDict | None = None,
    ) -> str:
        if not source_id or not target_id:
            return ""
        edge_id = f"edge:{_stable_hash(source_id, edge_type, target_id, doi, claim_id, properties or {})}"
        self.edges[edge_id] = {
            "edge_id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "doi": doi,
            "claim_id": claim_id,
            "confidence_score": confidence_score,
            "evidence_ids": evidence_ids or [],
            "properties": properties or {},
        }
        return edge_id

    def as_dict(self) -> JSONDict:
        return {
            "nodes": sorted(self.nodes.values(), key=lambda row: row["node_id"]),
            "edges": sorted(self.edges.values(), key=lambda row: row["edge_id"]),
        }


def _iter_references(extracted_json: JSONDict) -> list[JSONDict]:
    refs = extracted_json.get("references")
    if isinstance(refs, dict):
        out = []
        for key, value in refs.items():
            if isinstance(value, dict):
                out.append({"reference_id": key, **value})
            else:
                out.append({"reference_id": key, "citation": value})
        return out
    return [ref for ref in _safe_list(refs) if isinstance(ref, dict)]


def _iter_reference_mentions(provenance: JSONDict, source: JSONDict) -> list[tuple[str, Any]]:
    for key, edge_type in (
        ("adopted_from_reference_ids", "ADOPTED_FROM"),
        ("calibration_based_on_reference_ids", "CALIBRATED_FROM"),
        ("reference_ids", "MENTIONS_REFERENCE"),
        ("adopted_from_references", "ADOPTED_FROM"),
        ("calibration_based_on_references", "CALIBRATED_FROM"),
        ("references", "MENTIONS_REFERENCE"),
    ):
        for ref in _safe_list(provenance.get(key)):
            yield edge_type, ref
    for key, edge_type in (
        ("references", "MENTIONS_REFERENCE"),
        ("adopted_from_references", "ADOPTED_FROM"),
        ("calibration_based_on_references", "CALIBRATED_FROM"),
    ):
        for ref in _safe_list(source.get(key)):
            yield edge_type, ref


def _claim_confidence(claim: JSONDict) -> float | None:
    confidence = claim.get("confidence")
    if isinstance(confidence, (int, float)):
        return float(confidence)
    if isinstance(confidence, dict):
        for key in ("score", "confidence_score", "final_score"):
            value = confidence.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    for key in ("confidence_score", "final_confidence_score"):
        value = claim.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _add_evidence_for_claim(builder: GraphBuilder, doi: str, paper_id: str, claim: JSONDict, claim_node_id: str) -> list[str]:
    claim_id = _clean(claim.get("claim_id"))
    evidence_ids: list[str] = []
    for raw_id in _safe_list(claim.get("evidence_ids")):
        evidence_id = _clean(raw_id)
        if evidence_id:
            evidence_ids.append(evidence_id)
            evidence_node_id = _node_id("evidence", doi, evidence_id)
            builder.add_edge(claim_node_id, "SUPPORTED_BY", evidence_node_id, doi=doi, claim_id=claim_id)

    evidence = _safe_dict(claim.get("evidence"))
    source = _safe_dict(claim.get("source"))
    source_text = _first(
        evidence.get("text"),
        evidence.get("evidence_text"),
        source.get("evidence_text"),
        _safe_dict(evidence.get("table_evidence")).get("excerpt"),
        _safe_dict(source.get("table_evidence")).get("excerpt"),
    )
    if source_text and not evidence_ids:
        synthetic_id = f"{claim_id}:inline"
        evidence_ids.append(synthetic_id)
        evidence_node_id = _node_id("evidence", doi, synthetic_id)
        builder.add_node(
            evidence_node_id,
            "Evidence",
            label=f"Evidence for {claim_id}",
            properties={
                "doi": doi,
                "evidence_id": synthetic_id,
                "snippet": source_text,
                "source_file": _first(evidence.get("source_file"), evidence.get("file"), source.get("source_file")),
                "section_heading": _first(evidence.get("section_heading"), source.get("section_heading")),
                "table_evidence": _safe_dict(evidence.get("table_evidence")) or _safe_dict(source.get("table_evidence")),
            },
        )
        builder.add_edge(paper_id, "HAS_EVIDENCE", evidence_node_id, doi=doi)
        builder.add_edge(claim_node_id, "SUPPORTED_BY", evidence_node_id, doi=doi, claim_id=claim_id)
    return evidence_ids


def build_graph_from_extraction(doi: str, extracted_json: JSONDict) -> JSONDict:
    """Project one finalized CPextractor JSON document into KG nodes and edges."""
    doi = _doi_from_extraction(extracted_json, doi)
    if not doi:
        raise ValueError("Cannot build graph without a DOI or record id")

    builder = GraphBuilder()
    document = _safe_dict(extracted_json.get("document"))
    source_document = _safe_dict(extracted_json.get("source_document"))
    paper_id = _node_id("paper", doi)
    builder.add_node(
        paper_id,
        "Paper",
        label=_first(document.get("title"), source_document.get("title"), doi),
        properties={
            "doi": doi,
            "title": _first(document.get("title"), source_document.get("title")),
            "year": document.get("year") or source_document.get("year"),
            "journal": _first(document.get("journal"), source_document.get("journal_or_venue")),
            "quality_tier": extracted_json.get("quality_tier"),
            "document_confidence": extracted_json.get("document_confidence"),
            "document_confidence_score": extracted_json.get("document_confidence_score"),
        },
    )

    material_node_by_id: dict[str, str] = {}
    constituent_node_by_id: dict[str, str] = {}
    for material in [m for m in _safe_list(extracted_json.get("materials")) if isinstance(m, dict)]:
        material_id = _entity_id(material, ("material_id", "id", "name"))
        node_id = _node_id("material", doi, material_id)
        material_node_by_id[material_id] = builder.add_node(
            node_id,
            "Material",
            label=_first(material.get("name"), material_id),
            properties={**material, "doi": doi},
        )
        builder.add_edge(paper_id, "HAS_MATERIAL", node_id, doi=doi)
        for phase in [p for p in _safe_list(material.get("phases")) if isinstance(p, dict)]:
            phase_id = _entity_id(phase, ("phase_id", "constituent_id", "id", "name"))
            phase_node_id = _node_id("constituent", doi, phase_id)
            constituent_node_by_id[phase_id] = builder.add_node(
                phase_node_id,
                "Constituent",
                label=_first(phase.get("name"), phase_id),
                properties={**phase, "doi": doi, "material_id": material_id},
            )
            builder.add_edge(node_id, "HAS_CONSTITUENT", phase_node_id, doi=doi)

    legacy_material = _safe_dict(extracted_json.get("material"))
    if legacy_material and not material_node_by_id:
        material_id = _entity_id(legacy_material, ("material_id", "id", "name")) or "material"
        node_id = _node_id("material", doi, material_id)
        material_node_by_id[material_id] = builder.add_node(
            node_id,
            "Material",
            label=_first(legacy_material.get("name"), material_id),
            properties={**legacy_material, "doi": doi},
        )
        builder.add_edge(paper_id, "HAS_MATERIAL", node_id, doi=doi)

    for constituent in [c for c in _safe_list(extracted_json.get("constituents")) if isinstance(c, dict)]:
        constituent_id = _entity_id(constituent, ("constituent_id", "phase_id", "id", "name"))
        node_id = _node_id("constituent", doi, constituent_id)
        constituent_node_by_id[constituent_id] = builder.add_node(
            node_id,
            "Constituent",
            label=_first(constituent.get("name"), constituent_id),
            properties={**constituent, "doi": doi},
        )
        material_id = _clean(constituent.get("material_id"))
        if material_id and material_id in material_node_by_id:
            builder.add_edge(material_node_by_id[material_id], "HAS_CONSTITUENT", node_id, doi=doi)

    process_node_by_id: dict[str, str] = {}
    for state in [s for s in _safe_list(extracted_json.get("process_states")) + _safe_list(extracted_json.get("samples")) if isinstance(s, dict)]:
        state_id = _entity_id(state, ("process_state_id", "sample_id", "id", "label", "name"))
        node_id = _node_id("processstate", doi, state_id)
        process_node_by_id[state_id] = builder.add_node(
            node_id,
            "ProcessState",
            label=_first(state.get("name"), state.get("label"), state_id),
            properties={**state, "doi": doi},
        )
        builder.add_edge(paper_id, "HAS_PROCESS_STATE", node_id, doi=doi)
        material_id = _clean(state.get("material_id"))
        if material_id and material_id in material_node_by_id:
            builder.add_edge(node_id, "STATE_OF", material_node_by_id[material_id], doi=doi)

    condition_node_by_id: dict[str, str] = {}
    for condition in [c for c in _safe_list(extracted_json.get("conditions")) if isinstance(c, dict)]:
        condition_id = _entity_id(condition, ("condition_id", "id", "label", "name"))
        node_id = _node_id("condition", doi, condition_id)
        condition_node_by_id[condition_id] = builder.add_node(
            node_id,
            "Condition",
            label=_first(condition.get("label"), condition.get("name"), condition_id),
            properties={**condition, "doi": doi},
        )
        builder.add_edge(paper_id, "HAS_CONDITION", node_id, doi=doi)

    model_node_by_id: dict[str, str] = {}
    for model in [m for m in _safe_list(extracted_json.get("models")) if isinstance(m, dict)]:
        model_id = _entity_id(model, ("model_id", "id", "name", "framework"))
        node_id = _node_id("model", doi, model_id)
        model_node_by_id[model_id] = builder.add_node(
            node_id,
            "Model",
            label=_first(model.get("name"), model.get("framework"), model_id),
            properties={**model, "doi": doi},
        )
        builder.add_edge(paper_id, "USES_MODEL", node_id, doi=doi)

    legacy_model = _safe_dict(extracted_json.get("constitutive_model"))
    if legacy_model and not model_node_by_id:
        model_id = _entity_id(legacy_model, ("model_id", "id", "framework", "name")) or "model"
        node_id = _node_id("model", doi, model_id)
        model_node_by_id[model_id] = builder.add_node(
            node_id,
            "Model",
            label=_first(legacy_model.get("framework"), model_id),
            properties={**legacy_model, "doi": doi},
        )
        builder.add_edge(paper_id, "USES_MODEL", node_id, doi=doi)

    system_node_by_id: dict[str, str] = {}
    for system in [s for s in _safe_list(extracted_json.get("deformation_systems")) if isinstance(s, dict)]:
        system_id = _entity_id(system, ("system_id", "id", "family_id", "family_name"))
        node_id = _node_id("deformationsystem", doi, system_id)
        system_node_by_id[system_id] = builder.add_node(
            node_id,
            "DeformationSystem",
            label=_first(system.get("family_name"), system.get("system_type"), system_id),
            properties={**system, "doi": doi},
        )
        model_id = _clean(system.get("model_id"))
        if model_id and model_id in model_node_by_id:
            builder.add_edge(model_node_by_id[model_id], "HAS_DEFORMATION_SYSTEM", node_id, doi=doi)

    evidence_node_by_id: dict[str, str] = {}
    for evidence in [e for e in _safe_list(extracted_json.get("evidence_objects")) if isinstance(e, dict)]:
        evidence_id = _entity_id(evidence, ("evidence_id", "id", "source_id"))
        node_id = _node_id("evidence", doi, evidence_id)
        evidence_node_by_id[evidence_id] = builder.add_node(
            node_id,
            "Evidence",
            label=_first(evidence.get("label"), evidence.get("source_file"), evidence_id),
            properties={**evidence, "doi": doi},
        )
        builder.add_edge(paper_id, "HAS_EVIDENCE", node_id, doi=doi)

    reference_node_by_key: dict[str, str] = {}
    for ref in _iter_references(extracted_json):
        ref_node_id = _reference_id(doi, ref)
        builder.add_node(ref_node_id, "Reference", label=_label_for_reference(ref), properties=ref)
        builder.add_edge(paper_id, "CITES", ref_node_id, doi=doi, properties={"reference_id": ref.get("reference_id")})
        for key in ("reference_id", "id", "label", "doi", "reference_doi"):
            value = _clean(ref.get(key))
            if value:
                reference_node_by_key[value] = ref_node_id
                reference_node_by_key[value.lower()] = ref_node_id

    def resolve_reference_node(ref: Any) -> str:
        if isinstance(ref, dict):
            for key in ("reference_id", "id", "label", "doi", "reference_doi"):
                value = _clean(ref.get(key))
                if value and (value in reference_node_by_key or value.lower() in reference_node_by_key):
                    return reference_node_by_key.get(value) or reference_node_by_key[value.lower()]
        else:
            value = _clean(ref)
            if value and (value in reference_node_by_key or value.lower() in reference_node_by_key):
                return reference_node_by_key.get(value) or reference_node_by_key[value.lower()]
        return _reference_id(doi, ref)

    for claim in [c for c in _safe_list(extracted_json.get("parameter_claims")) if isinstance(c, dict)]:
        claim_id = _clean(claim.get("claim_id")) or _stable_hash(claim)
        claim_node_id = _node_id("claim", doi, claim_id)
        parameter = _safe_dict(claim.get("parameter"))
        assertion = _safe_dict(claim.get("assertion"))
        canonical_name = _first(claim.get("canonical_name"), parameter.get("canonical_name"))
        symbol = _first(claim.get("symbol"), parameter.get("symbol_reported"), parameter.get("symbol_normalized"))
        value = _first(claim.get("value"), claim.get("reported_value"), assertion.get("reported_value"))
        unit = _first(claim.get("unit"), claim.get("reported_unit"), assertion.get("reported_unit"))
        confidence_score = _claim_confidence(claim)
        builder.add_node(
            claim_node_id,
            "ParameterClaim",
            label=_first(canonical_name, symbol, claim_id),
            properties={
                "doi": doi,
                "claim_id": claim_id,
                "claim_class": claim.get("claim_class"),
                "domain": _first(claim.get("domain"), parameter.get("domain")),
                "canonical_name": canonical_name,
                "symbol": symbol,
                "value": value,
                "unit": unit,
                "value_SI": _first(claim.get("value_SI"), assertion.get("normalized_value")),
                "unit_SI": _first(claim.get("unit_SI"), assertion.get("normalized_unit")),
                "simulation_role": claim.get("simulation_role"),
                "confidence_score": confidence_score,
                "audit_verdict": claim.get("audit_verdict"),
            },
        )
        builder.add_edge(paper_id, "HAS_PARAMETER_CLAIM", claim_node_id, doi=doi, claim_id=claim_id, confidence_score=confidence_score)

        if canonical_name:
            parameter_type_id = _node_id("parameter", canonical_name.lower())
            builder.add_node(
                parameter_type_id,
                "ParameterType",
                label=canonical_name,
                properties={"canonical_name": canonical_name, "symbol": symbol, "domain": _first(claim.get("domain"), parameter.get("domain"))},
            )
            builder.add_edge(claim_node_id, "ASSERTS", parameter_type_id, doi=doi, claim_id=claim_id, confidence_score=confidence_score)

        evidence_ids = _add_evidence_for_claim(builder, doi, paper_id, claim, claim_node_id)
        binding = _safe_dict(claim.get("applies_to"))
        bindings = [
            ("material_id", material_node_by_id, "APPLIES_TO_MATERIAL"),
            ("process_state_id", process_node_by_id, "APPLIES_TO_PROCESS_STATE"),
            ("sample_id", process_node_by_id, "APPLIES_TO_PROCESS_STATE"),
            ("condition_id", condition_node_by_id, "APPLIES_TO_CONDITION"),
            ("constituent_id", constituent_node_by_id, "APPLIES_TO_CONSTITUENT"),
            ("phase_id", constituent_node_by_id, "APPLIES_TO_CONSTITUENT"),
            ("model_id", model_node_by_id, "APPLIES_TO_MODEL"),
        ]
        for key, lookup, edge_type in bindings:
            target_key = _clean(binding.get(key))
            if not target_key and key == "model_id":
                target_key = _clean(claim.get("model_id"))
            if target_key and target_key in lookup:
                builder.add_edge(
                    claim_node_id,
                    edge_type,
                    lookup[target_key],
                    doi=doi,
                    claim_id=claim_id,
                    confidence_score=confidence_score,
                    evidence_ids=evidence_ids,
                )
        for system_id in [_clean(v) for v in _safe_list(binding.get("system_ids")) if _clean(v)]:
            if system_id in system_node_by_id:
                builder.add_edge(
                    claim_node_id,
                    "APPLIES_TO_DEFORMATION_SYSTEM",
                    system_node_by_id[system_id],
                    doi=doi,
                    claim_id=claim_id,
                    confidence_score=confidence_score,
                    evidence_ids=evidence_ids,
                )

        mechanism = _first(binding.get("mechanism"), claim.get("mechanism"))
        if mechanism:
            mechanism_id = _node_id("mechanism", mechanism.lower())
            builder.add_node(mechanism_id, "Mechanism", label=mechanism, properties={"mechanism": mechanism})
            builder.add_edge(claim_node_id, "APPLIES_TO_MECHANISM", mechanism_id, doi=doi, claim_id=claim_id)

        family_id = _first(binding.get("family_id"), binding.get("family_name"))
        if family_id:
            family_node_id = _node_id("family", doi, family_id)
            builder.add_node(family_node_id, "DeformationFamily", label=_first(binding.get("family_name"), family_id), properties={"doi": doi, **binding})
            builder.add_edge(claim_node_id, "APPLIES_TO_FAMILY", family_node_id, doi=doi, claim_id=claim_id)

        provenance = _safe_dict(claim.get("provenance"))
        source = _safe_dict(claim.get("source"))
        seen_ref_edges: set[tuple[str, str]] = set()
        for edge_type, ref in _iter_reference_mentions(provenance, source):
            ref_node_id = resolve_reference_node(ref)
            key = (edge_type, ref_node_id)
            if key in seen_ref_edges:
                continue
            seen_ref_edges.add(key)
            label = _label_for_reference(ref) if isinstance(ref, dict) else _clean(ref)
            builder.add_node(ref_node_id, "Reference", label=label or ref_node_id, properties=ref if isinstance(ref, dict) else {"reference_id": ref})
            builder.add_edge(claim_node_id, edge_type, ref_node_id, doi=doi, claim_id=claim_id)

    return builder.as_dict()


def _infer_doi_from_path(path: Path, extracted_json: JSONDict) -> str:
    doi = _doi_from_extraction(extracted_json)
    if doi:
        return doi
    dirname = path.parent.name
    return dirname.replace("_", "/") if dirname.startswith("10.") else dirname


def build_graph_from_root(root: str | Path, source_name: str = "materials_extracted.json") -> JSONDict:
    builder = GraphBuilder()
    for path in sorted(Path(root).glob(f"*/{source_name}")):
        try:
            extracted_json = json.loads(path.read_text(encoding="utf-8"))
            doc_graph = build_graph_from_extraction(_infer_doi_from_path(path, extracted_json), extracted_json)
        except Exception:
            continue
        for node in doc_graph["nodes"]:
            builder.add_node(node["node_id"], node["node_type"], node.get("label") or "", node.get("properties") or {})
        for edge in doc_graph["edges"]:
            builder.add_edge(
                edge["source_id"],
                edge["edge_type"],
                edge["target_id"],
                doi=edge.get("doi") or "",
                claim_id=edge.get("claim_id") or "",
                confidence_score=edge.get("confidence_score"),
                evidence_ids=edge.get("evidence_ids") or [],
                properties=edge.get("properties") or {},
            )
    return builder.as_dict()


def write_graphml(graph: JSONDict, path: str | Path) -> None:
    """Write a conservative Gephi/yEd-readable GraphML file.

    Some GraphML importers are fussy about XML id-like values. CP graph ids
    intentionally contain DOI punctuation, so GraphML gets compact safe ids and
    keeps the original CP ids as attributes.
    """
    node_xml_ids = {
        node["node_id"]: f"n{idx}"
        for idx, node in enumerate(graph.get("nodes", []))
        if node.get("node_id")
    }
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="node_label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="node_type" for="node" attr.name="node_type" attr.type="string"/>',
        '  <key id="node_original_id" for="node" attr.name="original_id" attr.type="string"/>',
        '  <key id="node_properties" for="node" attr.name="properties" attr.type="string"/>',
        '  <key id="edge_label" for="edge" attr.name="label" attr.type="string"/>',
        '  <key id="edge_type" for="edge" attr.name="edge_type" attr.type="string"/>',
        '  <key id="edge_original_id" for="edge" attr.name="original_id" attr.type="string"/>',
        '  <key id="edge_properties" for="edge" attr.name="properties" attr.type="string"/>',
        '  <graph edgedefault="directed">',
    ]
    for node in graph.get("nodes", []):
        props = json.dumps(node.get("properties") or {}, ensure_ascii=False, sort_keys=True)
        xml_id = node_xml_ids.get(node["node_id"], f"n{len(node_xml_ids)}")
        lines.extend(
            [
                f'    <node id="{xml_id}">',
                f'      <data key="node_label">{escape(_clean(node.get("label")))}</data>',
                f'      <data key="node_type">{escape(_clean(node.get("node_type")))}</data>',
                f'      <data key="node_original_id">{escape(_clean(node.get("node_id")))}</data>',
                f'      <data key="node_properties">{escape(props)}</data>',
                "    </node>",
            ]
        )
    for idx, edge in enumerate(graph.get("edges", [])):
        source_id = node_xml_ids.get(edge.get("source_id"))
        target_id = node_xml_ids.get(edge.get("target_id"))
        if not source_id or not target_id:
            continue
        props = json.dumps(edge.get("properties") or {}, ensure_ascii=False, sort_keys=True)
        lines.extend(
            [
                f'    <edge id="e{idx}" source="{source_id}" target="{target_id}">',
                f'      <data key="edge_label">{escape(_clean(edge.get("edge_type")))}</data>',
                f'      <data key="edge_type">{escape(_clean(edge.get("edge_type")))}</data>',
                f'      <data key="edge_original_id">{escape(_clean(edge.get("edge_id")))}</data>',
                f'      <data key="edge_properties">{escape(props)}</data>',
                "    </edge>",
            ]
        )
    lines.extend(["  </graph>", "</graphml>"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cypher(graph: JSONDict, path: str | Path) -> None:
    """Write idempotent Cypher for Neo4j import."""
    out: list[str] = [
        "CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.node_id IS UNIQUE;",
    ]
    for node in graph.get("nodes", []):
        props = {
            "node_id": node["node_id"],
            "label": node.get("label") or node["node_id"],
            "node_type": node.get("node_type") or "Node",
            **(node.get("properties") or {}),
        }
        out.append(f"MERGE (n:KGNode {{node_id: {json.dumps(node['node_id'])}}}) SET n += {json.dumps(props, ensure_ascii=False)};")
    for edge in graph.get("edges", []):
        props = {
            "edge_id": edge["edge_id"],
            "edge_type": edge["edge_type"],
            "doi": edge.get("doi"),
            "claim_id": edge.get("claim_id"),
            "confidence_score": edge.get("confidence_score"),
            "evidence_ids": edge.get("evidence_ids") or [],
            **(edge.get("properties") or {}),
        }
        out.append(
            "MATCH (a:KGNode {node_id: "
            + json.dumps(edge["source_id"])
            + "}), (b:KGNode {node_id: "
            + json.dumps(edge["target_id"])
            + "}) MERGE (a)-[r:KG_EDGE {edge_id: "
            + json.dumps(edge["edge_id"])
            + " }]->(b) SET r += "
            + json.dumps(props, ensure_ascii=False)
            + ";"
        )
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a CP-specific knowledge graph from finalized CPextractor JSON.")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="Folder containing */materials_extracted.json paper folders.")
    source.add_argument("--input-json", help="Single materials_extracted.json file.")
    ap.add_argument("--doi", default="", help="Optional DOI override for --input-json.")
    ap.add_argument("--source-name", default="materials_extracted.json")
    ap.add_argument("--output-json", default="output/kg/cp_kg.json")
    ap.add_argument("--output-graphml", default="")
    ap.add_argument("--output-cypher", default="")
    args = ap.parse_args()

    if args.input_json:
        path = Path(args.input_json)
        extracted_json = json.loads(path.read_text(encoding="utf-8"))
        graph = build_graph_from_extraction(args.doi or _infer_doi_from_path(path, extracted_json), extracted_json)
    else:
        graph = build_graph_from_root(args.root, args.source_name)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_graphml:
        path = Path(args.output_graphml)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_graphml(graph, path)
    if args.output_cypher:
        path = Path(args.output_cypher)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_cypher(graph, path)
    print(f"Wrote {len(graph['nodes'])} nodes and {len(graph['edges'])} edges to {output_json}")


if __name__ == "__main__":
    main()
