from __future__ import annotations

from typing import Any, Dict, List, Tuple
from postprocess.param_iter import iter_parameter_items


_ORIGIN_TYPE_MAP = {
    "original": "original",
    "adopted": "adopted",
    "calibrated": "calibrated",
    "adopted_then_calibrated": "adopted_then_calibrated",
    "mixed_adopted_and_calibrated": "adopted_then_calibrated",
    "mixed": "adopted_then_calibrated",
}

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


def _unique_strings(values: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _is_non_reference_id(v: Any) -> bool:
    s = str(v or "").strip().lower()
    return s in _NON_REFERENCE_IDS


def _norm_origin_type(v: Any) -> str | None:
    key = str(v or "").strip().lower()
    if not key:
        return None
    return _ORIGIN_TYPE_MAP.get(key, key)


def _normalize_source(src: Dict[str, Any], report: Dict[str, int]) -> None:
    before = dict(src)

    origin = _norm_origin_type(src.get("origin_type") or src.get("type"))
    if origin is not None:
        src["origin_type"] = origin
    src.pop("type", None)

    adopted_ids = _unique_strings(src.get("adopted_from_reference_ids", []) or [])
    calib_ids_raw = _unique_strings(src.get("calibration_based_on_reference_ids", []) or [])
    legacy_ids = _unique_strings(src.get("reference_ids", []) or [])
    this_study_flag = src.get("calibration_in_this_study")
    if isinstance(this_study_flag, str):
        this_study_flag = this_study_flag.strip().lower() in {"yes", "true", "1"}
    else:
        this_study_flag = bool(this_study_flag)
    this_study_calibrated = this_study_flag or any(_is_non_reference_id(x) for x in calib_ids_raw)
    calib_ids = [x for x in calib_ids_raw if not _is_non_reference_id(x)]
    legacy_ids = [x for x in legacy_ids if not _is_non_reference_id(x)]
    overlap = set(adopted_ids).intersection(set(calib_ids))
    if overlap:
        calib_ids = [x for x in calib_ids if x not in overlap]
        report["role_id_overlap_collapsed"] += len(overlap)

    role_ids = _unique_strings(adopted_ids + calib_ids)
    residual_ids = [x for x in legacy_ids if x not in role_ids]

    src["adopted_from_reference_ids"] = adopted_ids
    src["calibration_based_on_reference_ids"] = calib_ids
    src["reference_ids"] = residual_ids
    if this_study_calibrated:
        src["calibration_in_this_study"] = True

    # Enforce compact source; reference metadata belongs to top-level references.
    src.pop("adopted_references", None)
    src.pop("calibration_references", None)
    src.pop("references", None)
    src.pop("citations", None)
    src.pop("adopted_citations", None)
    src.pop("calibration_citations", None)

    # Keep source compact.
    src.pop("calibration_targets", None)
    src.pop("validation_targets", None)

    if src != before:
        report["sources_normalized"] += 1


def normalize_provenance(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report = {
        "sources_seen": 0,
        "sources_normalized": 0,
        "non_reference_calibration_marked": 0,
        "role_id_overlap_collapsed": 0,
    }

    for _, item in iter_parameter_items(extracted_json):
        src = item.get("source")
        if not isinstance(src, dict):
            continue
        report["sources_seen"] += 1
        before_flag = bool(src.get("calibration_in_this_study"))
        _normalize_source(src, report)
        if (not before_flag) and bool(src.get("calibration_in_this_study")):
            report["non_reference_calibration_marked"] += 1

    return extracted_json, report
