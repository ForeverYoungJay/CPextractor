from __future__ import annotations

from typing import Any, Dict


def claim_location(item: Dict[str, Any], idx: int | None = None) -> str:
    claim_id = str((item or {}).get("claim_id") or "").strip()
    if claim_id:
        return f"claim:{claim_id}"
    if idx is not None:
        return f"parameters.registry[{idx}]"
    return ""


def legacy_registry_location(idx: int) -> str:
    return f"parameters.registry[{idx}]"


def location_candidates(item: Dict[str, Any], idx: int | None = None) -> list[str]:
    out: list[str] = []
    new_loc = claim_location(item, idx)
    if new_loc:
        out.append(new_loc)
    if idx is not None:
        out.append(legacy_registry_location(idx))
    seen: set[str] = set()
    deduped: list[str] = []
    for loc in out:
        if loc and loc not in seen:
            deduped.append(loc)
            seen.add(loc)
    return deduped


def is_claim_location(loc: str) -> bool:
    return str(loc or "").startswith("claim:")


def claim_id_from_location(loc: str) -> str | None:
    text = str(loc or "").strip()
    if text.startswith("claim:"):
        claim_id = text.split(":", 1)[1].strip()
        return claim_id or None
    return None
