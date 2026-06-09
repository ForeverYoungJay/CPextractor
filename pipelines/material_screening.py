from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


_GENERIC_MATERIAL_NAMES = {
    "alloy",
    "alloys",
    "material",
    "materials",
    "metal",
    "metals",
    "steel",
    "stainless steel",
    "titanium",
    "titanium alloy",
    "magnesium",
    "magnesium alloy",
    "aluminum",
    "aluminium",
    "aluminum alloy",
    "aluminium alloy",
    "nickel",
    "nickel alloy",
    "nickel superalloy",
    "nickel-based superalloy",
    "cobalt alloy",
    "zirconium alloy",
    "copper",
    "copper alloy",
    "fcc metal",
    "hcp metal",
    "bcc metal",
    "polycrystal",
    "single crystal",
    "bicrystal",
}

_STRONG_SIGNAL_PATTERNS: List[Tuple[re.Pattern[str], str, int]] = [
    (
        re.compile(
            r"\b(?:ti-?6al-?4v|ti64|tc4|az31|az91|we43|ze10|inconel\s*718|in718|in625|rene\s*88dt|gh4169|316l|316h|304l|17-4ph|2205)\b",
            re.IGNORECASE,
        ),
        "named_material_grade",
        3,
    ),
    (
        re.compile(
            r"\b(?:ti|al|ni|co|fe|mg|zr|cu|nb|ta|cr|mo|w|v|mn|si|sn|zn)(?:[-–][a-z0-9.+]+){1,4}\b",
            re.IGNORECASE,
        ),
        "composition_like_grade",
        3,
    ),
    (
        re.compile(r"\b(?:wt\.?\s*%|at\.?\s*%)\b", re.IGNORECASE),
        "composition_percent",
        2,
    ),
    (
        re.compile(r"\bchemical composition\b", re.IGNORECASE),
        "chemical_composition_phrase",
        2,
    ),
    (
        re.compile(
            r"\b(?:commercially pure|cp[-\s]?(?:ti|titanium|ni|nickel|cu|copper|al|aluminum|aluminium)|pure\s+(?:titanium|magnesium|nickel|copper|aluminum|aluminium|iron))\b",
            re.IGNORECASE,
        ),
        "purity_qualified_material",
        2,
    ),
    (
        re.compile(
            r"\b(?:austenitic|martensitic|ferritic|duplex|nickel-based|cobalt-based|alpha\+beta|beta)\s+(?:stainless steel|steel|superalloy|titanium alloy|magnesium alloy|zirconium alloy|aluminum alloy|aluminium alloy|copper alloy)\b",
            re.IGNORECASE,
        ),
        "qualified_material_family",
        2,
    ),
]

_WEAK_SIGNAL_PATTERNS: List[Tuple[re.Pattern[str], str, int]] = [
    (
        re.compile(
            r"\b(?:stainless steel|titanium alloy|magnesium alloy|nickel(?:-based)? superalloy|zirconium alloy|aluminum alloy|aluminium alloy|copper alloy)\b",
            re.IGNORECASE,
        ),
        "material_family",
        1,
    ),
]


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def detect_concrete_material_signal(text: Any, *, min_score: int = 2) -> Dict[str, Any]:
    haystack = _normalize_space(text)
    lowered = haystack.lower()
    score = 0
    matched: List[Dict[str, Any]] = []

    for pattern, label, weight in _STRONG_SIGNAL_PATTERNS + _WEAK_SIGNAL_PATTERNS:
        m = pattern.search(haystack)
        if not m:
            continue
        snippet = _normalize_space(m.group(0))
        matched.append({
            "label": label,
            "weight": weight,
            "snippet": snippet[:160],
        })
        score += weight

    exact_name = lowered.strip(" .,:;()[]{}")
    generic_only = exact_name in _GENERIC_MATERIAL_NAMES
    if generic_only and score < 3:
        return {
            "matched": False,
            "score": score,
            "signals": matched,
            "generic_only": True,
        }

    return {
        "matched": score >= min_score,
        "score": score,
        "signals": matched,
        "generic_only": generic_only,
    }


def _material_components_count(material: Dict[str, Any]) -> int:
    composition = _safe_dict(material.get("composition"))
    components = _safe_list(composition.get("components"))
    return sum(
        1
        for row in components
        if isinstance(row, dict) and _normalize_space(row.get("component") or row.get("element") or row.get("name"))
    )


def material_identity_report_from_extracted_json(extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    materials = [m for m in _safe_list(extracted_json.get("materials")) if isinstance(m, dict)]
    concrete_materials: List[Dict[str, Any]] = []

    for idx, material in enumerate(materials):
        name = _normalize_space(material.get("name"))
        formula = _normalize_space(material.get("chemical_formula") or material.get("formula"))
        text_report = detect_concrete_material_signal(" ".join(part for part in (name, formula) if part), min_score=2)
        component_count = _material_components_count(material)
        has_concrete_identity = bool(text_report.get("matched")) or component_count >= 2
        if not has_concrete_identity:
            continue
        concrete_materials.append({
            "index": idx,
            "material_id": material.get("material_id"),
            "name": name or None,
            "formula": formula or None,
            "composition_component_count": component_count,
            "signals": text_report.get("signals") or [],
        })

    parameter_claim_count = len([c for c in _safe_list(extracted_json.get("parameter_claims")) if isinstance(c, dict)])
    registry_count = len([r for r in _safe_list(_safe_dict(extracted_json.get("parameters")).get("registry")) if isinstance(r, dict)])
    parameter_count = parameter_claim_count or registry_count

    return {
        "has_concrete_materials": bool(concrete_materials),
        "concrete_material_count": len(concrete_materials),
        "concrete_materials": concrete_materials,
        "material_count": len(materials),
        "parameter_count": parameter_count,
        "requires_material_identity_for_claims": parameter_count > 0,
    }


def _load_text_files(root: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for path in sorted(root.glob("*.md")):
        try:
            out.append((str(path.name), path.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            continue
    return out


def _load_table_texts(root: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        parts: List[str] = []
        for key in ("table_label", "caption", "title"):
            value = _normalize_space(payload.get(key))
            if value:
                parts.append(value)
        rows = payload.get("rows")
        if isinstance(rows, list):
            for row in rows[:12]:
                if isinstance(row, list):
                    parts.append(" | ".join(_normalize_space(cell) for cell in row if _normalize_space(cell)))
        out.append((str(path.name), "\n".join(p for p in parts if p)))
    return out


def screen_paper_dir_for_concrete_material(paper_dir: str, *, min_score: int = 2) -> Dict[str, Any]:
    root = Path(paper_dir)
    scanned_files: List[Dict[str, Any]] = []
    best_score = 0

    text_sources: List[Tuple[str, str]] = []
    paper_md = root / "paper.md"
    if paper_md.exists():
        try:
            text_sources.append((paper_md.name, paper_md.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            pass
    sections_dir = root / "sections"
    if sections_dir.exists():
        text_sources.extend(_load_text_files(sections_dir))
    tables_dir = root / "tables"
    if tables_dir.exists():
        text_sources.extend(_load_table_texts(tables_dir))

    hits: List[Dict[str, Any]] = []
    for name, text in text_sources:
        report = detect_concrete_material_signal(text, min_score=min_score)
        scanned_files.append({
            "file": name,
            "score": report.get("score", 0),
            "matched": bool(report.get("matched")),
        })
        best_score = max(best_score, int(report.get("score") or 0))
        if report.get("matched"):
            hits.append({
                "file": name,
                "score": report.get("score", 0),
                "signals": report.get("signals") or [],
            })

    return {
        "has_concrete_material_signal": bool(hits),
        "best_score": best_score,
        "hit_count": len(hits),
        "hits": hits[:10],
        "files_scanned": scanned_files,
    }

