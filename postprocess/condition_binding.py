from __future__ import annotations

from typing import Any, Dict, List, Tuple

from postprocess.location_ids import claim_location
from postprocess.param_iter import iter_parameter_items_with_index


def _safe_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _safe_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def _family_name_from_mechanism(mechanism: Any) -> str | None:
    text = str(mechanism or "").strip().lower()
    if not text:
        return None
    if "prism" in text:
        return "prism"
    if "basal" in text:
        return "basal"
    if "pyramidal" in text:
        return "pyramidal"
    if "hydride" in text or "delta" in text:
        return "{111}<110>"
    return None


def resolve_condition_bindings(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report_rows: List[Dict[str, Any]] = []
    resolved = 0
    ambiguous = 0
    downgraded = 0
    binding_records: List[Dict[str, Any]] = []
    binding_keys: Dict[str, str] = {}

    for idx, _, item in iter_parameter_items_with_index(extracted_json):
        applies_to = _safe_dict(item.get("applies_to"))

        issues: List[str] = []
        scope = str(applies_to.get("scope") or "").strip().lower()
        phase_id = applies_to.get("phase_id")
        family_id = applies_to.get("family_id")
        system_ids = _safe_list(applies_to.get("system_ids"))
        family_name = applies_to.get("family_name")
        inferred_family_name = family_name or _family_name_from_mechanism(applies_to.get("mechanism"))

        # Be conservative for family-level table rows: only keep system scope when
        # the evidence explicitly supports per-system assignment. Family rows often
        # get synthetic system_ids upstream, but they should still remain family-scoped.
        if scope == "system" and (family_id or inferred_family_name):
            scope = "family"
            applies_to["scope"] = "family"
            if inferred_family_name and not applies_to.get("family_name"):
                applies_to["family_name"] = inferred_family_name
            applies_to.pop("system_ids", None)
            applies_to.pop("system_count", None)
            item["applies_to"] = applies_to
            system_ids = []
            family_name = applies_to.get("family_name")
            downgraded += 1

        if scope in {"phase", "family", "system"} and not phase_id:
            issues.append("missing_phase_binding")
        if scope == "global" and phase_id:
            issues.append("unexpected_phase_binding")
        if scope == "family" and not family_id:
            issues.append("missing_family_binding")
        if scope in {"global", "phase", "system"} and family_id:
            issues.append("unexpected_family_binding")
        if scope == "system" and not system_ids:
            issues.append("missing_system_binding")
        if scope in {"global", "phase", "family"} and system_ids:
            issues.append("unexpected_system_binding")
        if not scope:
            issues.append("missing_scope")

        binding_record = {
            "phase_id": phase_id,
            "scope": scope or None,
            "mechanism": applies_to.get("mechanism"),
            "family_id": family_id,
            "family_name": applies_to.get("family_name"),
            "system_ids": system_ids,
            "system_count": applies_to.get("system_count"),
            "notes": applies_to.get("notes"),
        }
        binding_key_record = dict(binding_record)
        binding_key_record.pop("notes", None)
        binding_key = str(sorted(binding_key_record.items(), key=lambda kv: kv[0]))
        binding_id = binding_keys.get(binding_key)
        if not binding_id:
            binding_id = f"bind_{len(binding_records) + 1:04d}"
            binding_keys[binding_key] = binding_id
            binding_records.append({
                "binding_id": binding_id,
                **binding_record,
            })
        elif binding_record.get("notes"):
            for record in binding_records:
                if record.get("binding_id") == binding_id and not record.get("notes"):
                    record["notes"] = binding_record.get("notes")
                    break
        item["binding_id"] = binding_id
        item.pop("binding_context", None)

        status = "resolved" if not issues else "ambiguous"
        if status == "resolved":
            resolved += 1
        else:
            ambiguous += 1

        report_rows.append({
            "location": claim_location(item, idx),
            "canonical_name": item.get("canonical_name"),
            "symbol": item.get("symbol"),
            "binding_id": binding_id,
            "binding_status": status,
            "binding_issues": issues,
            "binding_record": binding_record,
        })

    extracted_json["binding_contexts"] = binding_records
    report = {
            "parameters_checked": len(report_rows),
            "bindings_resolved": resolved,
            "bindings_ambiguous": ambiguous,
            "system_scope_downgraded_to_family": downgraded,
            "binding_contexts_built": len(binding_records),
            "rows": report_rows,
    }
    return extracted_json, report
