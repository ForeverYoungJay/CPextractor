from __future__ import annotations

from typing import Any, Dict, List, Tuple

from postprocess.param_iter import iter_parameter_items_with_index


_KEY_SYMBOLS = {"tau0", "τ0", "n", "h0", "g0", "crss0"}
_POSITIVE_CANONICAL_HINTS = (
    "crss",
    "hardening",
    "drag",
    "stress",
    "energy",
    "modulus",
    "density",
    "gamma0",
    "rate_sensitivity",
    "activation_energy",
    "fracture_energy",
    "cleavage_strength",
    "burgers_vector",
)
_ALLOWED_SCOPE_RULES = {
    "global": {"phase_id": False, "family_id": False, "system_ids": False},
    "phase": {"phase_id": True, "family_id": False, "system_ids": False},
    "family": {"phase_id": True, "family_id": False, "system_ids": False},
    "system": {"phase_id": True, "family_id": False, "system_ids": True},
}


def _has_evidence(item: Dict[str, Any], src: Dict[str, Any]) -> bool:
    evidence = item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
    txt = str(evidence.get("evidence_text") or src.get("evidence_text") or "").strip()
    table_evidence = evidence.get("table_evidence") if isinstance(evidence.get("table_evidence"), dict) else src.get("table_evidence")
    has_table = isinstance(table_evidence, dict) and any(
        table_evidence.get(k) not in (None, "") for k in ("row_name", "column_name", "value", "excerpt")
    )
    return bool(txt or has_table)


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        if isinstance(v, str) and v.strip():
            return float(v.strip())
    except Exception:
        return None
    return None


def _param_path(block: str, idx: int) -> str:
    return f"parameters.registry[{idx}]" if block in {"elastic", "plastic"} else f"{block}[{idx}]"


def _is_positive_expected(item: Dict[str, Any]) -> bool:
    canonical = str(item.get("canonical_name") or "").strip().lower()
    symbol = str(item.get("symbol") or "").strip().lower()
    return any(h in canonical for h in _POSITIVE_CANONICAL_HINTS) or symbol in {"n", "m", "q", "h0", "g0", "τ0", "tau0"}


def _scope_consistency_issues(app: Dict[str, Any], block: str, idx: int) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    scope = str(app.get("scope") or "").strip().lower()
    if scope not in _ALLOWED_SCOPE_RULES:
        return issues

    rule = _ALLOWED_SCOPE_RULES[scope]
    phase_id = app.get("phase_id")
    family_id = app.get("family_id")
    family_name = app.get("family_name")
    system_ids = app.get("system_ids") or []

    if rule["phase_id"] and not phase_id:
        issues.append({
            "type": "scope_inconsistency",
            "severity": "medium",
            "path": _param_path(block, idx) + ".applies_to.phase_id",
            "message": f"scope={scope} requires phase_id",
        })
    if rule["system_ids"] and not system_ids:
        issues.append({
            "type": "scope_inconsistency",
            "severity": "medium",
            "path": _param_path(block, idx) + ".applies_to.system_ids",
            "message": f"scope={scope} requires system_ids",
        })
    return issues


def run_quality_checks(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    total = 0
    evidence_missing = 0
    key_param_missing = 0
    source_conflict = 0
    invalid_units = 0
    negative_when_positive = 0
    scope_inconsistency = 0
    missing_phase_bindings = 0

    for idx, block, p in iter_parameter_items_with_index(extracted_json):
        total += 1
        path = _param_path(block, idx)
        symbol = str(p.get("symbol") or "").strip()
        src = p.get("source", {}) if isinstance(p.get("source"), dict) else {}
        app = p.get("applies_to", {}) if isinstance(p.get("applies_to"), dict) else {}

        if not _has_evidence(p, src):
            evidence_missing += 1
            issues.append({
                "type": "missing_evidence",
                "severity": "low",
                "path": path + ".source",
                "message": "parameter lacks evidence.evidence_text/table_evidence",
                "symbol": symbol or None,
            })

        if symbol.lower() in _KEY_SYMBOLS and (p.get("value") in (None, "")):
            key_param_missing += 1
            issues.append({
                "type": "missing_key_parameter_value",
                "severity": "high",
                "path": path + ".value",
                "message": f"key parameter {symbol} is missing value",
                "symbol": symbol,
            })

        if p.get("value") not in (None, "") and p.get("unit") in ("", None) and block == "elastic":
            invalid_units += 1
            issues.append({
                "type": "missing_unit",
                "severity": "low",
                "path": path + ".unit",
                "message": "elastic parameter has value but missing unit",
                "symbol": symbol or None,
            })

        value = _to_float(p.get("value"))
        if value is not None and _is_positive_expected(p) and value < 0:
            negative_when_positive += 1
            issues.append({
                "type": "implausible_negative_value",
                "severity": "high",
                "path": path + ".value",
                "message": f"value={value} looks invalid for positive-only parameter",
                "symbol": symbol or None,
            })

        if app.get("scope") in {"phase", "family", "system"} and not app.get("phase_id"):
            missing_phase_bindings += 1

        scope_issues = _scope_consistency_issues(app, block, idx)
        scope_inconsistency += len(scope_issues)
        issues.extend(scope_issues)

    high = sum(1 for i in issues if i.get("severity") == "high")
    medium = sum(1 for i in issues if i.get("severity") == "medium")
    low = sum(1 for i in issues if i.get("severity") == "low")

    total_weight = (high * 20) + (medium * 5) + (low * 1.5)
    rule_score = max(0.0, 100.0 - float(total_weight))

    report = {
        "total_parameters": total,
        "missing_evidence": evidence_missing,
        "missing_key_parameter_value": key_param_missing,
        "source_conflicts": source_conflict,
        "invalid_units": invalid_units,
        "implausible_negative_values": negative_when_positive,
        "scope_inconsistencies": scope_inconsistency,
        "missing_phase_bindings": missing_phase_bindings,
        "issue_count": len(issues),
        "severity_counts": {
            "high": high,
            "medium": medium,
            "low": low,
        },
        "rule_score": round(rule_score, 2),
        "issues": issues,
    }
    return extracted_json, report
