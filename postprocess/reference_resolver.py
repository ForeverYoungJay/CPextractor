import json
from typing import Dict, List


def load_references(ref_path: str) -> Dict[str, dict]:
    """
    Load references.json and build label -> reference dict.
    """
    with open(ref_path, "r", encoding="utf-8") as f:
        refs = json.load(f)

    return {
        r["label"]: r
        for r in refs
        if r.get("label")
    }


def resolve_references(extracted_json: dict, reference_map: Dict[str, dict]) -> tuple[dict, dict]:
    """
    Attach resolved citations (with DOIs) to extracted parameters.
    """
    report = {
        "total_reference_ids": 0,
        "resolved_reference_ids": 0,
        "unresolved_reference_ids": 0,
        "unresolved_labels": [],
    }

    def resolve_items(items: List[dict]):
        for item in items:
            src = item.get("source", {})
            ids = src.get("reference_ids", [])
            resolved = []
            unresolved = []

            for rid in ids:
                report["total_reference_ids"] += 1
                if rid in reference_map:
                    resolved.append(reference_map[rid])
                    report["resolved_reference_ids"] += 1
                else:
                    unresolved.append(rid)
                    report["unresolved_reference_ids"] += 1

            if resolved:
                src["citations"] = resolved
            if unresolved:
                src["unresolved_reference_ids"] = unresolved
                report["unresolved_labels"].extend(unresolved)

    # Plastic parameters
    resolve_items(
        extracted_json
        .get("plastic_parameters", {})
        .get("parameters", [])
    )

    # Elastic parameters
    resolve_items(
        extracted_json
        .get("elastic_parameters", {})
        .get("constants", [])
    )

    report["unresolved_labels"] = sorted(set(report["unresolved_labels"]))
    return extracted_json, report
