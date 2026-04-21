from __future__ import annotations

from typing import Any, Dict, List, Tuple

from postprocess.location_ids import claim_location
from postprocess.param_iter import iter_parameter_items_with_index
from postprocess.record_links import resolve_evidence_objects


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
_DIMENSIONLESS_ELASTIC_CANONICALS = {
    "zener_ratio",
    "hcp_c_over_a_ratio",
    "poisson_ratio",
}
_ALLOWED_SCOPE_RULES = {
    "global": {"constituent_id": False, "branch_id": False, "system_ids": False},
    "constituent": {"constituent_id": True, "branch_id": False, "system_ids": False},
    "family": {"constituent_id": True, "branch_id": False, "system_ids": False},
    "system": {"constituent_id": True, "branch_id": False, "system_ids": True},
    "branch": {"constituent_id": False, "branch_id": True, "system_ids": False},
    "local_region": {"constituent_id": False, "branch_id": False, "system_ids": False},
    "other": {"constituent_id": False, "branch_id": False, "system_ids": False},
}


def _evidence_ids(item: Dict[str, Any], src: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for raw in (item.get("evidence_ids") or []):
        text = str(raw or "").strip()
        if text and text not in ids:
            ids.append(text)
    for raw in (src.get("evidence_ids") or []):
        text = str(raw or "").strip()
        if text and text not in ids:
            ids.append(text)
    return ids


def _has_evidence(extracted_json: Dict[str, Any], item: Dict[str, Any], src: Dict[str, Any]) -> bool:
    evidence = item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
    txt = str(evidence.get("evidence_text") or src.get("evidence_text") or "").strip()
    table_evidence = evidence.get("table_evidence") if isinstance(evidence.get("table_evidence"), dict) else src.get("table_evidence")
    has_table = isinstance(table_evidence, dict) and any(
        table_evidence.get(k) not in (None, "") for k in ("row_name", "column_name", "value", "excerpt")
    )
    if txt or has_table:
        return True
    evidence_ids = _evidence_ids(item, src)
    if not evidence_ids:
        return False
    for obj in resolve_evidence_objects(extracted_json, evidence_ids):
        if not isinstance(obj, dict):
            continue
        locator = obj.get("locator") if isinstance(obj.get("locator"), dict) else {}
        if any(
            obj.get(k) not in (None, "", [], {})
            for k in ("evidence_type", "source_file", "source_id", "quote", "text")
        ):
            return True
        if any(locator.get(k) not in (None, "", [], {}) for k in ("row_name", "column_name", "value", "excerpt", "section_heading")):
            return True
    return False


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
    return f"parameter_claims[{idx}]"


def _issue_location(item: Dict[str, Any], idx: int) -> str:
    return claim_location(item, idx)


def _is_positive_expected(item: Dict[str, Any]) -> bool:
    canonical = str(item.get("canonical_name") or "").strip().lower()
    symbol = str(item.get("symbol") or "").strip().lower()
    return any(h in canonical for h in _POSITIVE_CANONICAL_HINTS) or symbol in {"n", "m", "q", "h0", "g0", "τ0", "tau0"}


def _scope_consistency_issues(extracted_json: Dict[str, Any], app: Dict[str, Any], block: str, idx: int) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    scope = str(app.get("scope") or "").strip().lower()
    if scope not in _ALLOWED_SCOPE_RULES:
        return issues

    rule = _ALLOWED_SCOPE_RULES[scope]
    constituent_id = app.get("constituent_id")
    branch_id = app.get("branch_id")
    system_ids = app.get("system_ids") or []
    constituents = extracted_json.get("constituents") if isinstance(extracted_json.get("constituents"), list) else []

    require_constituent = rule["constituent_id"]
    if scope == "family" and not constituents:
        require_constituent = False

    if require_constituent and not constituent_id:
        issues.append({
            "type": "scope_inconsistency",
            "severity": "medium",
            "path": _param_path(block, idx) + ".applies_to.constituent_id",
            "message": f"scope={scope} requires constituent_id",
        })
    if rule["branch_id"] and not branch_id:
        issues.append({
            "type": "scope_inconsistency",
            "severity": "medium",
            "path": _param_path(block, idx) + ".applies_to.branch_id",
            "message": f"scope={scope} requires branch_id",
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
    missing_constituent_bindings = 0
    empty_claim_shells = 0

    for idx, block, p in iter_parameter_items_with_index(extracted_json):
        total += 1
        path = _param_path(block, idx)
        location = _issue_location(p, idx)
        symbol = str(p.get("symbol") or "").strip()
        src = p.get("source", {}) if isinstance(p.get("source"), dict) else {}
        app = p.get("applies_to", {}) if isinstance(p.get("applies_to"), dict) else {}
        evidence_ids = _evidence_ids(p, src)

        if not _has_evidence(extracted_json, p, src):
            evidence_missing += 1
            issues.append({
                "type": "missing_evidence",
                "severity": "low",
                "path": path + ".source",
                "location": location,
                "message": "parameter lacks evidence.evidence_text/table_evidence",
                "symbol": symbol or None,
            })

        if symbol.lower() in _KEY_SYMBOLS and (p.get("value") in (None, "")):
            key_param_missing += 1
            issues.append({
                "type": "missing_key_parameter_value",
                "severity": "high",
                "path": path + ".value",
                "location": location,
                "message": f"key parameter {symbol} is missing value",
                "symbol": symbol,
            })

        canonical_name = str(p.get("canonical_name") or "").strip().lower()
        if (
            p.get("value") not in (None, "")
            and p.get("unit") in ("", None)
            and block == "elastic"
            and canonical_name not in _DIMENSIONLESS_ELASTIC_CANONICALS
        ):
            invalid_units += 1
            issues.append({
                "type": "missing_unit",
                "severity": "low",
                "path": path + ".unit",
                "location": location,
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
                "location": location,
                "message": f"value={value} looks invalid for positive-only parameter",
                "symbol": symbol or None,
            })

        has_identity = bool(str(p.get("canonical_name") or "").strip() or symbol)
        has_assertion = any(p.get(k) not in (None, "") for k in ("value", "value_SI", "unit", "unit_SI"))
        if not has_identity and not has_assertion and not evidence_ids:
            empty_claim_shells += 1
            issues.append({
                "type": "empty_claim_shell",
                "severity": "high",
                "path": path,
                "location": location,
                "message": "claim lacks parameter identity, numeric assertion, and evidence_ids",
                "symbol": symbol or None,
            })

        constituent_id = app.get("constituent_id")
        if app.get("scope") in {"constituent", "family", "system"} and not constituent_id:
            missing_constituent_bindings += 1

        scope_issues = _scope_consistency_issues(extracted_json, app, block, idx)
        for issue in scope_issues:
            issue["location"] = location
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
        "missing_constituent_bindings": missing_constituent_bindings,
        "empty_claim_shells": empty_claim_shells,
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
