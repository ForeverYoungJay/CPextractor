from __future__ import annotations

from typing import Any, Dict, List, Tuple
from postprocess.param_iter import iter_parameter_items


_KEY_SYMBOLS = {"tau0", "τ0", "n", "h0", "g0", "crss0"}


def _iter_params(extracted_json: Dict[str, Any]):
    counts = {"elastic": 0, "plastic": 0}
    for block, item in iter_parameter_items(extracted_json):
        idx = counts.get(block, 0)
        counts[block] = idx + 1
        yield block, idx, item


def _has_evidence(src: Dict[str, Any]) -> bool:
    txt = str(src.get("evidence_text") or "").strip()
    sec = str(src.get("evidence_section") or "").strip()
    loc = src.get("evidence_location", {})
    has_loc = isinstance(loc, dict) and any(loc.get(k) not in (None, "") for k in ("kind", "id", "page"))
    return bool(txt or sec or has_loc)


def run_quality_checks(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    total = 0
    evidence_missing = 0
    key_param_missing = 0
    source_conflict = 0

    for block, idx, p in _iter_params(extracted_json):
        total += 1
        symbol = str(p.get("symbol") or "").strip()
        src = p.get("source", {}) if isinstance(p.get("source"), dict) else {}
        origin = str(src.get("origin_type") or "").strip()
        adopted_ids = src.get("adopted_from_reference_ids", []) or []
        calib_ids = src.get("calibration_based_on_reference_ids", []) or []
        this_study = bool(src.get("calibration_in_this_study"))

        if not _has_evidence(src):
            evidence_missing += 1
            issues.append({
                "type": "missing_evidence",
                "block": block,
                "index": idx,
                "symbol": symbol or None,
            })

        if symbol.lower() in _KEY_SYMBOLS and (p.get("value") in (None, "")):
            key_param_missing += 1
            issues.append({
                "type": "missing_key_parameter_value",
                "block": block,
                "index": idx,
                "symbol": symbol,
            })

        if origin == "adopted" and (calib_ids or this_study):
            source_conflict += 1
            issues.append({
                "type": "source_conflict_origin_vs_calibration",
                "block": block,
                "index": idx,
                "symbol": symbol or None,
                "origin_type": origin,
            })
        if origin == "calibrated" and adopted_ids:
            source_conflict += 1
            issues.append({
                "type": "source_conflict_origin_vs_adopted",
                "block": block,
                "index": idx,
                "symbol": symbol or None,
                "origin_type": origin,
            })

    report = {
        "total_parameters": total,
        "missing_evidence": evidence_missing,
        "missing_key_parameter_value": key_param_missing,
        "source_conflicts": source_conflict,
        "issue_count": len(issues),
        "issues": issues,
    }
    return extracted_json, report
