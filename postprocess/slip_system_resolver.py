from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _parse_hkl(text: str | None) -> List[int] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    content = raw.strip("()[]{}<> ")
    compact = re.sub(r"\s+", "", content)
    if re.fullmatch(r"[-+0-9]+", compact):
        out: List[int] = []
        sign = 1
        for ch in compact:
            if ch == "-":
                sign = -1
            elif ch == "+":
                sign = 1
            elif ch.isdigit():
                out.append(sign * int(ch))
                sign = 1
        if out:
            return out
    nums = [int(tok) for tok in re.findall(r"[-+]?\d+", content)]
    return nums if nums else None


def _direction_indices(system: Dict[str, Any]) -> List[int] | None:
    direction = _safe_dict(system.get("direction"))
    indices = direction.get("indices")
    if isinstance(indices, list) and all(isinstance(v, (int, float)) for v in indices):
        return [int(v) for v in indices]
    return None


def _dot(a: List[int], b: List[int]) -> int:
    return sum(int(x) * int(y) for x, y in zip(a, b))


def _load_table_jsons(paper_dir: str) -> List[Dict[str, Any]]:
    tables_dir = Path(paper_dir) / "tables"
    out: List[Dict[str, Any]] = []
    if not tables_dir.exists():
        return out
    for path in sorted(tables_dir.glob("table_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload["_path"] = path.name
            out.append(payload)
    return out


def _row_map(rows: List[Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        label = _norm(row[0])
        values = [str(v).strip() for v in row[1:] if str(v).strip()]
        if label and values:
            out[label] = values
    return out


def _candidate_slip_tables(paper_dir: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for table in _load_table_jsons(paper_dir):
        caption = _norm(table.get("caption"))
        rows = _safe_list(table.get("rows"))
        row_map = _row_map(rows)
        if "slip system" not in caption and "slip system" not in " ".join(row_map.keys()):
            continue
        planes = row_map.get("slip plane normal", [])
        directions = row_map.get("slip direction", [])
        systems = row_map.get("slip system", [])
        if planes and directions:
            candidates.append({
                "path": table.get("_path"),
                "caption": table.get("caption"),
                "planes": planes,
                "directions": directions,
                "systems": systems,
            })
    return candidates


def _best_table_for_family(family: Dict[str, Any], tables: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    family_notes = _norm(family.get("notes"))
    plane_direction = _norm(family.get("plane_direction"))
    for table in tables:
        path = str(table.get("path") or "")
        if path and path in family_notes:
            return table
    for table in tables:
        planes = " ".join(table.get("planes", []))
        if plane_direction and any(token in _norm(planes) for token in re.findall(r"\([^)]+\)|\{[^}]+\}", plane_direction)):
            return table
    return tables[0] if tables else None


def _plane_candidates(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, plane_text in enumerate(table.get("planes", []), start=1):
        indices = _parse_hkl(plane_text)
        if indices:
            out.append({
                "candidate_id": idx,
                "as_written": plane_text,
                "indices": indices,
                "basis": "hkl",
            })
    return out


def resolve_slip_systems(extracted_json: Dict[str, Any], paper_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mechanisms = _safe_dict(extracted_json.get("deformation_mechanisms"))
    slip_families = _safe_list(mechanisms.get("slip_families"))
    tables = _candidate_slip_tables(paper_dir)

    report_rows: List[Dict[str, Any]] = []
    families_updated = 0
    systems_updated = 0

    for fam_idx, family in enumerate(slip_families):
        if not isinstance(family, dict):
            continue
        systems = _safe_list(family.get("systems"))
        if not systems:
            continue

        table = _best_table_for_family(family, tables)
        if not table:
            report_rows.append({
                "family_index": fam_idx,
                "family_id": family.get("family_id"),
                "status": "no_table_found",
                "updated_systems": 0,
            })
            continue

        plane_candidates = _plane_candidates(table)
        direction_texts = [str(v).strip() for v in table.get("directions", [])]
        if len(plane_candidates) >= len(systems):
            report_rows.append({
                "family_index": fam_idx,
                "family_id": family.get("family_id"),
                "table": table.get("path"),
                "status": "no_merged_header_pattern",
                "updated_systems": 0,
            })
            continue

        changed_here = 0
        for sys_idx, system in enumerate(systems):
            if not isinstance(system, dict):
                continue
            dir_indices = _direction_indices(system)
            if dir_indices is None:
                continue
            compatible = [p for p in plane_candidates if len(p["indices"]) == len(dir_indices) and _dot(p["indices"], dir_indices) == 0]
            if len(compatible) != 1:
                continue
            plane = compatible[0]
            current_plane = _safe_dict(system.get("plane"))
            current_indices = current_plane.get("indices")
            if current_indices == plane["indices"] and current_plane.get("as_written") == plane["as_written"]:
                if sys_idx < len(direction_texts):
                    system.setdefault("direction", {})
                    if not _safe_dict(system.get("direction")).get("as_written"):
                        system["direction"]["as_written"] = direction_texts[sys_idx]
                continue
            system["plane"] = {
                "as_written": plane["as_written"],
                "indices": plane["indices"],
                "basis": plane["basis"],
            }
            if sys_idx < len(direction_texts):
                system.setdefault("direction", {})
                system["direction"]["as_written"] = direction_texts[sys_idx]
            changed_here += 1

        if changed_here:
            family["systems"] = systems
            note = str(family.get("notes") or "").strip()
            resolver_note = f" Slip-system planes reconciled from merged-header table {table.get('path')}."
            if resolver_note.strip() not in note:
                family["notes"] = (note + resolver_note).strip()
            families_updated += 1
            systems_updated += changed_here

        report_rows.append({
            "family_index": fam_idx,
            "family_id": family.get("family_id"),
            "table": table.get("path"),
            "status": "updated" if changed_here else "unchanged",
            "updated_systems": changed_here,
        })

    mechanisms["slip_families"] = slip_families
    extracted_json["deformation_mechanisms"] = mechanisms
    return extracted_json, {
        "candidate_tables": [t.get("path") for t in tables],
        "families_checked": len(slip_families),
        "families_updated": families_updated,
        "systems_updated": systems_updated,
        "rows": report_rows,
    }
