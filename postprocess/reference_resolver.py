import json
from typing import Dict, List
from postprocess.param_iter import iter_parameter_items

_NON_REFERENCE_IDS = {
    "this_study",
    "this study",
    "present_study",
    "present study",
    "current_study",
    "current study",
    "our_work",
    "our work",
}


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
    Resolve parameter-level reference IDs against references.json.
    Keep source compact: IDs + flags only (no title/doi objects in source).
    """
    report = {
        "total_reference_ids": 0,
        "resolved_reference_ids": 0,
        "unresolved_reference_ids": 0,
        "unresolved_labels": [],
        "top_level_references_backfilled": 0,
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

    def _is_non_reference_id(v: str) -> bool:
        return str(v or "").strip().lower() in _NON_REFERENCE_IDS

    def resolve_items(items: List[dict]):
        for item in items:
            src = item.get("source", {})
            adopted_ids = _unique_keep_order(src.get("adopted_from_reference_ids", []) or [])
            calibration_ids = _unique_keep_order(src.get("calibration_based_on_reference_ids", []) or [])
            if any(_is_non_reference_id(x) for x in calibration_ids):
                src["calibration_in_this_study"] = True
            calibration_ids = [x for x in calibration_ids if not _is_non_reference_id(x)]
            legacy_ids = _unique_keep_order(src.get("reference_ids", []) or [])
            legacy_ids = [x for x in legacy_ids if not _is_non_reference_id(x)]
            overlap = set(adopted_ids).intersection(set(calibration_ids))
            if overlap:
                calibration_ids = [x for x in calibration_ids if x not in overlap]
            role_ids = _unique_keep_order(adopted_ids + calibration_ids)
            residual_ids = [x for x in legacy_ids if x not in role_ids]
            ids = _unique_keep_order(adopted_ids + calibration_ids + residual_ids)
            src["adopted_from_reference_ids"] = adopted_ids
            src["calibration_based_on_reference_ids"] = calibration_ids
            src["reference_ids"] = residual_ids

            unresolved = []

            for rid in ids:
                report["total_reference_ids"] += 1
                if rid in reference_map:
                    report["resolved_reference_ids"] += 1
                else:
                    unresolved.append(rid)
                    report["unresolved_reference_ids"] += 1

            report["by_role"]["adopted"]["total"] += len(adopted_ids)
            report["by_role"]["adopted"]["resolved"] += sum(1 for rid in adopted_ids if rid in reference_map)
            report["by_role"]["calibration"]["total"] += len(calibration_ids)
            report["by_role"]["calibration"]["resolved"] += sum(1 for rid in calibration_ids if rid in reference_map)
            report["by_role"]["legacy"]["total"] += len(legacy_ids)
            report["by_role"]["legacy"]["resolved"] += sum(1 for rid in legacy_ids if rid in reference_map)

            if unresolved:
                src["unresolved_reference_ids"] = unresolved
                report["unresolved_labels"].extend(unresolved)
            else:
                src.pop("unresolved_reference_ids", None)

            # Keep source compact and deduplicated.
            src.pop("adopted_references", None)
            src.pop("calibration_references", None)
            src.pop("references", None)
            src.pop("citations", None)
            src.pop("adopted_citations", None)
            src.pop("calibration_citations", None)

    all_items = [it for _, it in iter_parameter_items(extracted_json)]
    resolve_items(all_items)

    # Backfill top-level references[] with title/doi from references.json.
    for ref in extracted_json.get("references", []) or []:
        rid = str(ref.get("reference_id") or "").strip()
        if not rid:
            continue
        mapped = reference_map.get(rid)
        if not mapped:
            continue
        changed = False
        if not ref.get("doi") and mapped.get("doi"):
            ref["doi"] = mapped.get("doi")
            changed = True
        if not ref.get("citation") and mapped.get("title"):
            ref["citation"] = mapped.get("title")
            changed = True
        if changed:
            report["top_level_references_backfilled"] += 1

    report["unresolved_labels"] = sorted(set(report["unresolved_labels"]))
    return extracted_json, report
