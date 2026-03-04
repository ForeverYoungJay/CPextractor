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
        "by_role": {
            "adopted": {"total": 0, "resolved": 0},
            "calibration": {"total": 0, "resolved": 0},
            "legacy": {"total": 0, "resolved": 0},
        },
    }

    def _unique_keep_order(values: List[str]) -> List[str]:
        out = []
        seen = set()
        for v in values:
            s = str(v).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def resolve_items(items: List[dict]):
        for item in items:
            src = item.get("source", {})
            adopted_ids = _unique_keep_order(src.get("adopted_from_reference_ids", []) or [])
            calibration_ids = _unique_keep_order(src.get("calibration_based_on_reference_ids", []) or [])
            legacy_ids = _unique_keep_order(src.get("reference_ids", []) or [])
            ids = _unique_keep_order(adopted_ids + calibration_ids + legacy_ids)

            resolved = []
            unresolved = []
            adopted_citations = []
            calibration_citations = []

            for rid in ids:
                report["total_reference_ids"] += 1
                if rid in reference_map:
                    c = reference_map[rid]
                    resolved.append(c)
                    report["resolved_reference_ids"] += 1
                    if rid in adopted_ids:
                        adopted_citations.append(c)
                    if rid in calibration_ids:
                        calibration_citations.append(c)
                else:
                    unresolved.append(rid)
                    report["unresolved_reference_ids"] += 1

            report["by_role"]["adopted"]["total"] += len(adopted_ids)
            report["by_role"]["adopted"]["resolved"] += sum(1 for rid in adopted_ids if rid in reference_map)
            report["by_role"]["calibration"]["total"] += len(calibration_ids)
            report["by_role"]["calibration"]["resolved"] += sum(1 for rid in calibration_ids if rid in reference_map)
            report["by_role"]["legacy"]["total"] += len(legacy_ids)
            report["by_role"]["legacy"]["resolved"] += sum(1 for rid in legacy_ids if rid in reference_map)

            if resolved:
                src["citations"] = resolved
                if adopted_citations:
                    src["adopted_citations"] = adopted_citations
                if calibration_citations:
                    src["calibration_citations"] = calibration_citations
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
