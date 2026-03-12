from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple


_ELASTIC_CANONICALS = {"c11", "c12", "c13", "c33", "c44", "c55", "c66", "e", "nu", "g", "k"}


def _infer_block_from_registry_item(item: Dict[str, Any]) -> str:
    domain = str(item.get("domain") or "").strip().lower()
    if domain == "elastic":
        return "elastic"
    if domain in {"plastic", "twinning", "damage", "thermal", "numerical"}:
        return "plastic"
    cname = str(item.get("canonical_name") or "").strip().lower()
    if cname in _ELASTIC_CANONICALS:
        return "elastic"
    return "plastic"


def iter_parameter_items(extracted_json: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yield (block, item) with block in {"elastic","plastic"}.
    Prefer schema-v2.1.1 parameters.registry; fallback to legacy blocks.
    """
    registry = (extracted_json.get("parameters", {}) or {}).get("registry", [])
    if isinstance(registry, list):
        for it in registry:
            if isinstance(it, dict):
                yield _infer_block_from_registry_item(it), it
        return

    for it in ((extracted_json.get("elastic_parameters", {}) or {}).get("constants", []) or []):
        if isinstance(it, dict):
            yield "elastic", it
    for it in ((extracted_json.get("plastic_parameters", {}) or {}).get("parameters", []) or []):
        if isinstance(it, dict):
            yield "plastic", it
