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


def resolve_references(extracted_json: dict, reference_map: Dict[str, dict]) -> dict:
    """
    Attach resolved citations (with DOIs) to extracted parameters.
    """
    def resolve_items(items: List[dict]):
        for item in items:
            src = item.get("source", {})
            ids = src.get("reference_ids", [])
            resolved = []

            for rid in ids:
                if rid in reference_map:
                    resolved.append(reference_map[rid])

            if resolved:
                src["citations"] = resolved

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

    return extracted_json

