from __future__ import annotations

from typing import Any, Dict, List, Tuple
from postprocess.param_iter import iter_parameter_items


_STRESS_FACTORS = {
    "pa": 1.0,
    "kpa": 1e3,
    "mpa": 1e6,
    "gpa": 1e9,
}


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_param_items(items: List[Dict[str, Any]], block: str) -> Dict[str, int]:
    converted = 0
    skipped = 0

    for p in items:
        # Normalize both legacy fields (value/unit) and schema-v2 style (reported_value/reported_unit).
        value = _to_float(p.get("value"))
        unit = str(p.get("unit") or "").strip().lower()
        if value is None:
            value = _to_float(p.get("reported_value"))
        if not unit:
            unit = str(p.get("reported_unit") or "").strip().lower()

        if value is not None and p.get("value") is None:
            p["value"] = value
        if unit and p.get("unit") is None:
            p["unit"] = unit

        if value is None or unit not in _STRESS_FACTORS:
            skipped += 1
            continue

        p["value"] = float(value)
        p["value_SI"] = value * _STRESS_FACTORS[unit]
        p["unit_SI"] = "Pa"
        converted += 1

    return {
        "block": block,
        "converted": converted,
        "skipped": skipped,
    }


def normalize_extracted_units(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Add SI fields for commonly seen stress-like parameters.
    This preserves original value/unit while adding value_SI/unit_SI.
    """
    report: Dict[str, Any] = {"items": []}

    by_block = {"elastic": [], "plastic": []}
    for block, item in iter_parameter_items(extracted_json):
        by_block.setdefault(block, []).append(item)

    for block in ("elastic", "plastic"):
        report["items"].append(_normalize_param_items(by_block.get(block, []), block))

    report["converted_total"] = sum(i["converted"] for i in report["items"])
    report["skipped_total"] = sum(i["skipped"] for i in report["items"])
    return extracted_json, report
