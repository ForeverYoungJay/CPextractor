from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from postprocess.location_ids import claim_location
from postprocess.param_iter import iter_parameter_items_with_index
from postprocess.parameter_table_resolver import _split_value_parts


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _candidate_files(paper_dir: str, location: Dict[str, Any]) -> List[Path]:
    root = Path(paper_dir)
    loc_id = str(location.get("id") or "").strip()
    kind = str(location.get("kind") or "").strip().lower()
    files: List[Path] = []

    if loc_id:
        if kind == "table":
            files.append(root / "tables" / Path(loc_id).with_suffix(".json"))
            files.append(root / "tables" / loc_id)
        elif kind == "section":
            files.append(root / "sections" / loc_id)
        else:
            files.append(root / loc_id)
            files.append(root / "sections" / loc_id)
            files.append(root / "tables" / loc_id)

    if kind == "table":
        files.extend(sorted((root / "tables").glob("*.json")) if (root / "tables").exists() else [])
        files.extend(sorted((root / "tables").glob("*.md")) if (root / "tables").exists() else [])
    elif kind == "section":
        files.extend(sorted((root / "sections").glob("*.md")) if (root / "sections").exists() else [])
    else:
        files.extend(sorted((root / "sections").glob("*.md")) if (root / "sections").exists() else [])
        files.extend(sorted((root / "tables").glob("*.md")) if (root / "tables").exists() else [])

    deduped: List[Path] = []
    seen = set()
    for f in files:
        if f.exists() and str(f) not in seen:
            deduped.append(f)
            seen.add(str(f))
    return deduped


def _table_rows_from_json(file_path: Path) -> List[Tuple[int, str, List[str]]]:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    out: List[Tuple[int, str, List[str]]] = []
    for i, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            continue
        cells = [str(c).strip() for c in row if str(c).strip()]
        if not cells:
            continue
        out.append((i, " | ".join(cells), cells))
    return out


def _table_payload(file_path: Path) -> Dict[str, Any] | None:
    if file_path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _is_image_backed_table_file(file_path: Path) -> bool:
    payload = _table_payload(file_path)
    if not payload:
        return False
    return str(payload.get("table_kind") or "").strip().lower() == "image_backed"


def _table_lookup(file_path: Path) -> Dict[int, Dict[str, Any]]:
    return {
        row_idx: {"row_text": row_text, "cells": cells}
        for row_idx, row_text, cells in _table_rows_from_json(file_path)
    }


def _evidence_payload(item: Dict[str, Any], source: Dict[str, Any] | None = None) -> Dict[str, Any]:
    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        return evidence
    source = source if isinstance(source, dict) else {}
    legacy: Dict[str, Any] = {}
    if source.get("evidence_text") not in (None, "", []):
        legacy["evidence_text"] = source.get("evidence_text")
    if isinstance(source.get("table_evidence"), dict):
        legacy["table_evidence"] = source.get("table_evidence")
    return legacy


def _table_row_name_and_column_name(file_path: Path, row_index: int | None, column_index: int | None) -> Tuple[str | None, str | None]:
    if row_index is None:
        return None, None
    lookup = _table_lookup(file_path)
    row = lookup.get(row_index)
    if not row:
        return None, None
    cells = row.get("cells") or []
    row_name = str(cells[0]).strip() if cells else None
    column_name = None
    if isinstance(column_index, int) and column_index >= 1:
        for header_idx in range(row_index - 1, 0, -1):
            header_row = lookup.get(header_idx)
            if not header_row:
                continue
            header_cells = header_row.get("cells") or []
            if len(header_cells) < column_index:
                continue
            first = str(header_cells[0]).strip() if header_cells else ""
            if len([c for c in header_cells if str(c).strip()]) == 1 and " - " in first:
                continue
            alpha_cells = sum(1 for c in header_cells if re.search(r"[A-Za-zα-ωΑ-Ω]", str(c or "")))
            if alpha_cells == 0:
                continue
            candidate = str(header_cells[column_index - 1]).strip()
            if candidate and not re.search(r"[A-Za-zα-ωΑ-Ω()]", candidate):
                continue
            if candidate:
                column_name = candidate
                break
    return row_name or None, column_name


def _build_normalized_with_map(text: str) -> Tuple[str, List[int]]:
    chars: List[str] = []
    raw_map: List[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if chars and not prev_space:
                chars.append(" ")
                raw_map.append(i)
            prev_space = True
            continue
        chars.append(ch.lower())
        raw_map.append(i)
        prev_space = False
    while chars and chars[-1] == " ":
        chars.pop()
        raw_map.pop()
    return "".join(chars), raw_map


def _find_span(raw_text: str, evidence_text: str) -> Tuple[str, int | None, int | None]:
    target = _norm(evidence_text)
    if not target:
        return "missing_evidence_text", None, None

    raw_start = raw_text.lower().find(evidence_text.strip().lower())
    if raw_start >= 0:
        return "exact_match", raw_start, raw_start + len(evidence_text.strip())

    normalized_text, raw_map = _build_normalized_with_map(raw_text)
    idx = normalized_text.find(target)
    if idx >= 0:
        start_raw = raw_map[idx]
        end_raw = raw_map[min(len(raw_map) - 1, idx + len(target) - 1)] + 1
        return "normalized_match", start_raw, end_raw

    if len(target) >= 40:
        prefix = target[:40]
        idx = normalized_text.find(prefix)
        if idx >= 0:
            start_raw = raw_map[idx]
            end_raw = raw_map[min(len(raw_map) - 1, idx + len(prefix) - 1)] + 1
            return "prefix_match", start_raw, end_raw

    target_tokens = [t for t in set(target.split(" ")) if t]
    if target_tokens:
        best_ratio = 0.0
        best_range: Tuple[int | None, int | None] = (None, None)
        lines = raw_text.splitlines()
        cursor = 0
        for line in lines:
            line_norm = _norm(line)
            if not line_norm:
                cursor += len(line) + 1
                continue
            matched = sum(1 for tok in target_tokens if tok in line_norm)
            ratio = matched / max(1, len(target_tokens))
            if ratio > best_ratio:
                best_ratio = ratio
                best_range = (cursor, cursor + len(line))
            cursor += len(line) + 1
        if best_ratio >= 0.6:
            return "line_fuzzy_match", best_range[0], best_range[1]

    return "not_found", None, None


def _line_numbers(text: str, start: int | None, end: int | None) -> Tuple[int | None, int | None]:
    if start is None or end is None:
        return None, None
    start_line = text.count("\n", 0, start) + 1
    end_line = text.count("\n", 0, end) + 1
    return start_line, end_line


def _context_window(text: str, start: int | None, end: int | None, radius: int = 120) -> str | None:
    if start is None or end is None:
        return None
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return text[lo:hi].strip() or None


def _table_cell_info(text: str, start_line: int | None) -> Dict[str, Any] | None:
    if start_line is None:
        return None
    lines = text.splitlines()
    line_idx = start_line - 1
    if line_idx < 0 or line_idx >= len(lines):
        return None
    line = lines[line_idx]
    if "|" not in line:
        return None
    row_cells = [c.strip() for c in line.strip().strip("|").split("|")]
    best_col = None
    best_len = -1
    for col_idx, cell in enumerate(row_cells):
        if cell and len(cell) > best_len:
            best_len = len(cell)
            best_col = col_idx + 1
    return {
        "row_index": start_line,
        "column_index": best_col,
        "row_text": line.strip(),
    }


def _find_value_in_table(raw_text: str, symbol: str | None, value: Any) -> Dict[str, Any] | None:
    value_str = str(value).strip() if value not in (None, "") else ""
    symbol_str = str(symbol or "").strip()
    if not value_str and not symbol_str:
        return None
    target_float = _to_float(value)
    lines = raw_text.splitlines()
    best_hit = None
    best_score = -1
    for i, line in enumerate(lines, start=1):
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for j, cell in enumerate(cells, start=1):
            cell_norm = _norm(cell)
            value_hit = False
            if value_str:
                if target_float is not None:
                    numeric_tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cell)
                    for token in numeric_tokens:
                        token_float = _to_float(token)
                        if token_float is not None and abs(token_float - target_float) <= max(1e-9, abs(target_float) * 1e-6):
                            value_hit = True
                            break
                else:
                    value_hit = len(value_str) >= 3 and value_str.lower() in cell_norm

            symbol_hit = False
            if symbol_str:
                symbol_norm = symbol_str.lower()
                if re.search(r"[a-zA-Z]", symbol_norm) and len(symbol_norm) == 1:
                    symbol_hit = cell_norm == symbol_norm
                elif re.fullmatch(r"[a-zA-Z0-9_]+", symbol_norm):
                    symbol_hit = re.search(rf"(?<![a-zA-Z0-9_]){re.escape(symbol_norm)}(?![a-zA-Z0-9_])", cell_norm) is not None
                else:
                    symbol_hit = symbol_norm in cell_norm
            if value_hit or symbol_hit:
                score = 2 if value_hit else 1
                hit = {
                    "status": "table_cell_match",
                    "char_start": None,
                    "char_end": None,
                    "line_start": i,
                    "line_end": i,
                    "matched_span": cell,
                    "context_window": line.strip(),
                    "table_cell": {
                        "row_index": i,
                        "column_index": j,
                        "row_text": line.strip(),
                    },
                }
                if score > best_score:
                    best_hit = hit
                    best_score = score
    return best_hit


def _find_value_in_table_json(file_path: Path, symbol: str | None, value: Any) -> Dict[str, Any] | None:
    value_str = str(value).strip() if value not in (None, "") else ""
    symbol_str = str(symbol or "").strip()
    if not value_str and not symbol_str:
        return None
    target_float = _to_float(value)
    best_hit = None
    best_score = -1
    for i, row_text, cells in _table_rows_from_json(file_path):
        for j, cell in enumerate(cells, start=1):
            cell_norm = _norm(cell)
            value_hit = False
            if value_str:
                if target_float is not None:
                    numeric_tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cell)
                    for token in numeric_tokens:
                        token_float = _to_float(token)
                        if token_float is not None and abs(token_float - target_float) <= max(1e-9, abs(target_float) * 1e-6):
                            value_hit = True
                            break
                else:
                    value_hit = len(value_str) >= 3 and value_str.lower() in cell_norm

            symbol_hit = False
            if symbol_str:
                symbol_norm = symbol_str.lower()
                if re.search(r"[a-zA-Z]", symbol_norm) and len(symbol_norm) == 1:
                    symbol_hit = cell_norm == symbol_norm
                elif re.fullmatch(r"[a-zA-Z0-9_]+", symbol_norm):
                    symbol_hit = re.search(rf"(?<![a-zA-Z0-9_]){re.escape(symbol_norm)}(?![a-zA-Z0-9_])", cell_norm) is not None
                else:
                    symbol_hit = symbol_norm in cell_norm
            if value_hit or symbol_hit:
                score = 2 if value_hit else 1
                hit = {
                    "status": "table_cell_match",
                    "char_start": None,
                    "char_end": None,
                    "line_start": i,
                    "line_end": i,
                    "matched_span": cell,
                    "context_window": row_text,
                    "table_cell": {
                        "row_index": i,
                        "column_index": j,
                        "row_text": row_text,
                    },
                }
                row_name, column_name = _table_row_name_and_column_name(file_path, i, j)
                hit["row_name"] = row_name
                hit["column_name"] = column_name
                hit["value"] = cell
                if score > best_score:
                    best_hit = hit
                    best_score = score
    return best_hit


def _direct_table_match_from_source(
    file_path: Path,
    source: Dict[str, Any],
    evidence: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    evidence = evidence if isinstance(evidence, dict) else {}
    table_match = source.get("_table_match") if isinstance(source.get("_table_match"), dict) else {}
    if not evidence and isinstance(source.get("table_evidence"), dict):
        evidence = {"table_evidence": source.get("table_evidence")}
    table_evidence = evidence.get("table_evidence") if isinstance(evidence.get("table_evidence"), dict) else {}
    if not table_match and table_evidence:
        table_match = {
            "row_index": table_evidence.get("row_index"),
            "column_index": table_evidence.get("column_index"),
            "raw_value": table_evidence.get("value"),
        }
    if not table_match:
        return None
    target_table = str(table_match.get("table") or "").strip()
    if target_table and file_path.name != target_table:
        return None
    row_index = table_match.get("row_index")
    if not isinstance(row_index, int):
        return None

    rows = _table_rows_from_json(file_path)
    row_data = next((row for row in rows if row[0] == row_index), None)
    if not row_data:
        return None
    _, row_text, cells = row_data
    segment_index = table_match.get("segment_index")
    raw_value = str(table_match.get("raw_value") or "").strip()
    snippet = raw_value or str(evidence.get("evidence_text") or "").strip()
    column_index = table_match.get("column_index") if isinstance(table_match.get("column_index"), int) else (len(cells) if cells else None)
    if isinstance(column_index, int) and 1 <= column_index <= len(cells):
        value_cell = cells[column_index - 1]
    else:
        value_cell = cells[-1] if cells else ""
        column_index = len(cells) if cells else None
    if isinstance(segment_index, int) and value_cell:
        value_parts = _split_value_parts(value_cell)
        if 1 <= segment_index <= len(value_parts):
            snippet = value_parts[segment_index - 1].strip()
        else:
            snippet = value_cell
    elif value_cell:
        snippet = value_cell

    if not snippet:
        snippet = row_text
    row_name, column_name = _table_row_name_and_column_name(file_path, row_index, column_index)
    return {
        "status": "table_cell_match",
        "matched_file": file_path.name,
        "char_start": None,
        "char_end": None,
        "line_start": row_index,
        "line_end": row_index,
        "matched_span": snippet,
        "context_window": row_text,
        "row_name": row_name,
        "column_name": column_name,
        "value": snippet,
        "table_cell": {
            "row_index": row_index,
            "column_index": column_index,
            "row_text": row_text,
        },
    }


def _source_backed_match(location: Dict[str, Any], source: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(source, dict):
        return None

    table_evidence = evidence.get("table_evidence") if isinstance(evidence.get("table_evidence"), dict) else {}
    evidence_text = str(evidence.get("evidence_text") or "").strip()
    row_name = str(table_evidence.get("row_name") or "").strip() or None
    column_name = str(table_evidence.get("column_name") or "").strip() or None
    row_index = table_evidence.get("row_index") if isinstance(table_evidence.get("row_index"), int) else None
    column_index = table_evidence.get("column_index") if isinstance(table_evidence.get("column_index"), int) else None
    value = str(table_evidence.get("value") or "").strip() or None
    excerpt = str(table_evidence.get("excerpt") or "").strip() or None

    snippet = excerpt or evidence_text or value or row_name or column_name
    if not snippet:
        refs = [
            str(ref).strip()
            for ref in (
                (source.get("reference_ids") or [])
                + (source.get("adopted_from_reference_ids") or [])
                + (source.get("calibration_based_on_reference_ids") or [])
            )
            if str(ref).strip()
        ]
        if refs:
            snippet = f"Source-backed claim from references [{', '.join(refs)}]"
        else:
            return None

    loc_kind = str(location.get("kind") or "").strip().lower()
    loc_id = str(location.get("id") or "").strip()
    matched_file = None
    if loc_id:
        matched_file = Path(loc_id).with_suffix(".json").name if loc_kind == "table" else Path(loc_id).name

    return {
        "status": "source_backed",
        "matched_file": matched_file,
        "char_start": None,
        "char_end": None,
        "line_start": None,
        "line_end": None,
        "matched_span": snippet,
        "context_window": evidence_text or excerpt,
        "row_name": row_name,
        "column_name": column_name,
        "value": value or snippet,
        "table_cell": {
            "row_index": row_index,
            "column_index": column_index,
            "row_text": None,
        } if row_index or column_index else None,
    }


def _image_backed_table_source_match(files: List[Path], source: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(source, dict):
        return None

    table_evidence = evidence.get("table_evidence") if isinstance(evidence.get("table_evidence"), dict) else {}
    row_name = str(table_evidence.get("row_name") or "").strip() or None
    column_name = str(table_evidence.get("column_name") or "").strip() or None
    value = str(table_evidence.get("value") or "").strip() or None
    excerpt = str(table_evidence.get("excerpt") or "").strip() or None
    evidence_text = str(evidence.get("evidence_text") or "").strip() or None

    if not any([row_name, column_name, value, excerpt, evidence_text]):
        return None

    table_match = source.get("_table_match") if isinstance(source.get("_table_match"), dict) else {}
    target_name = Path(str(table_match.get("table") or "")).name if table_match.get("table") else None
    target_file = None
    for file_path in files:
        if file_path.suffix.lower() != ".json":
            continue
        if target_name and file_path.name != target_name:
            continue
        if _is_image_backed_table_file(file_path):
            target_file = file_path
            break

    if target_file is None:
        return None

    snippet = value or excerpt or evidence_text or column_name or row_name
    context_parts = [part for part in [row_name, column_name, value, excerpt or evidence_text] if part]
    context_window = " | ".join(context_parts) if context_parts else (excerpt or evidence_text)

    return {
        "status": "table_cell_match",
        "matched_file": target_file.name,
        "char_start": None,
        "char_end": None,
        "line_start": None,
        "line_end": None,
        "matched_span": snippet,
        "context_window": context_window,
        "row_name": row_name,
        "column_name": column_name,
        "value": value or snippet,
        "table_cell": None,
        "match_strategy": "image_backed_table_source",
    }


def _locate_evidence(evidence_text: str, file_path: Path, symbol: str | None, value: Any, source: Dict[str, Any] | None = None) -> Dict[str, Any]:
    source = source if isinstance(source, dict) else {}
    evidence = _evidence_payload({"evidence": source.get("__evidence__")}, source)
    if file_path.suffix.lower() == ".json":
        direct_hit = _direct_table_match_from_source(file_path, source, evidence)
        if direct_hit:
            return direct_hit
        table_hit = _find_value_in_table_json(file_path, symbol, value)
        if table_hit:
            table_hit["matched_file"] = file_path.name
            return table_hit
        return {
            "status": "not_found",
            "matched_file": file_path.name,
        }

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except Exception:
        return {
            "status": "read_error",
            "matched_file": file_path.name,
        }

    status, start, end = _find_span(raw_text, evidence_text)
    line_start, line_end = _line_numbers(raw_text, start, end)
    cell_info = _table_cell_info(raw_text, line_start) if "table" in str(file_path.parent).lower() else None
    hit = {
        "status": status,
        "matched_file": file_path.name,
        "char_start": start,
        "char_end": end,
        "line_start": line_start,
        "line_end": line_end,
        "matched_span": raw_text[start:end].strip() if start is not None and end is not None else None,
        "context_window": _context_window(raw_text, start, end),
        "table_cell": cell_info,
    }
    if status == "not_found":
        table_hit = _find_value_in_table(raw_text, symbol, value)
        if table_hit:
            table_hit["matched_file"] = file_path.name
            return table_hit
    return hit


def _best_match(evidence_text: str, files: List[Path], symbol: str | None, value: Any, source: Dict[str, Any] | None = None) -> Dict[str, Any]:
    source = source if isinstance(source, dict) else {}
    evidence = _evidence_payload({"evidence": source.get("__evidence__")}, source)
    location: Dict[str, Any] = {}
    source_backed = _source_backed_match(location, source, evidence)
    image_backed_source = _image_backed_table_source_match(files, source, evidence)
    if image_backed_source:
        return image_backed_source
    if not evidence_text.strip():
        for file_path in files:
            direct_hit = _direct_table_match_from_source(file_path, source, evidence) if file_path.suffix.lower() == ".json" else None
            if direct_hit:
                return direct_hit
            try:
                raw_text = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            table_hit = _find_value_in_table(raw_text, symbol, value)
            if table_hit:
                table_hit["matched_file"] = file_path.name
                return table_hit
        if source_backed:
            return source_backed
        return {"status": "missing_evidence_text", "matched_file": None}

    priority = {
        "table_cell_match": 6,
        "exact_match": 5,
        "normalized_match": 4,
        "prefix_match": 3,
        "source_backed": 3,
        "line_fuzzy_match": 2,
        "not_found": 1,
        "read_error": 0,
    }
    best = source_backed or {"status": "not_found", "matched_file": None}
    best_score = priority.get(str(best.get("status") or ""), -1)
    for file_path in files:
        hit = _locate_evidence(evidence_text, file_path, symbol, value, source)
        score = priority.get(str(hit.get("status") or ""), -1)
        if score > best_score:
            best = hit
            best_score = score
            if score == priority["exact_match"]:
                break
    return best


def _evidence_object_map(extracted_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for record in extracted_json.get("evidence_objects") or []:
        if not isinstance(record, dict):
            continue
        evidence_id = str(record.get("evidence_id") or "").strip()
        if evidence_id:
            out[evidence_id] = record
    return out


def _candidate_files_from_evidence_object(paper_dir: str, evidence_object: Dict[str, Any]) -> List[Path]:
    root = Path(paper_dir)
    files: List[Path] = []
    source_file = str(evidence_object.get("source_file") or "").strip()
    source_id = str(evidence_object.get("source_id") or "").strip()

    if source_file:
        candidates = [
            root / source_file,
            root / "sections" / source_file,
            root / "tables" / source_file,
            root / "equations" / source_file,
        ]
        for path in candidates:
            if path.exists() and path not in files:
                files.append(path)

    if source_id:
        candidates = [
            root / source_id,
            root / "sections" / source_id,
            root / "tables" / source_id,
            root / "equations" / source_id,
            root / "tables" / Path(source_id).with_suffix(".json"),
            root / "sections" / Path(source_id).with_suffix(".md"),
        ]
        for path in candidates:
            if path.exists() and path not in files:
                files.append(path)

    return files


def _evidence_from_object(evidence_object: Dict[str, Any]) -> Dict[str, Any]:
    locator = evidence_object.get("locator") if isinstance(evidence_object.get("locator"), dict) else {}
    table_coord = evidence_object.get("table_coord") if isinstance(evidence_object.get("table_coord"), dict) else {}
    excerpt = str(locator.get("excerpt") or "").strip() or None
    snippet = str(evidence_object.get("snippet") or "").strip() or None
    value = str(locator.get("value") or evidence_object.get("value") or "").strip() or None
    row_name = str(locator.get("row_name") or evidence_object.get("row_name") or "").strip() or None
    column_name = str(locator.get("column_name") or evidence_object.get("column_name") or "").strip() or None
    row_index = table_coord.get("row")
    column_index = table_coord.get("col")
    table_evidence = {
        "row_name": row_name,
        "column_name": column_name,
        "value": value,
        "excerpt": excerpt or snippet,
        "row_index": row_index,
        "column_index": column_index,
    }
    table_evidence = {k: v for k, v in table_evidence.items() if v not in (None, "", [])}
    out = {
        "evidence_text": snippet or excerpt or value,
        "table_evidence": table_evidence or None,
    }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def verify_evidence_grounding(extracted_json: Dict[str, Any], paper_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    exact = normalized = prefix = source_backed = fuzzy = table_cell = missing = not_found = 0
    existing_evidence = _evidence_object_map(extracted_json)
    missing_evidence_object = 0
    claims_without_evidence_ids = 0

    for idx, _, item in iter_parameter_items_with_index(extracted_json):
        src = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
        evidence_ids = src.get("evidence_ids") if isinstance(src.get("evidence_ids"), list) else []
        if not evidence_ids and isinstance(item.get("evidence_ids"), list):
            evidence_ids = item.get("evidence_ids")

        evidence_object = None
        resolved_ids: List[str] = []
        for evidence_id in evidence_ids:
            key = str(evidence_id or "").strip()
            if not key:
                continue
            if key in existing_evidence:
                resolved_ids.append(key)
                if evidence_object is None:
                    evidence_object = existing_evidence[key]

        if not evidence_ids:
            claims_without_evidence_ids += 1
        if evidence_ids and not resolved_ids:
            missing_evidence_object += 1

        evidence = _evidence_from_object(evidence_object) if isinstance(evidence_object, dict) else _evidence_payload(item, src)
        source_for_match = dict(src)
        source_for_match["__evidence__"] = evidence

        files = _candidate_files_from_evidence_object(paper_dir, evidence_object or {})
        if not files:
            location: Dict[str, Any] = {}
            files = _candidate_files(paper_dir, location)

        if isinstance(evidence_object, dict) and str(evidence_object.get("evidence_type") or "").strip().lower() == "equation":
            hit = {
                "status": "source_backed",
                "matched_file": evidence_object.get("source_file"),
                "char_start": None,
                "char_end": None,
                "line_start": None,
                "line_end": None,
                "matched_span": evidence.get("evidence_text"),
                "context_window": evidence.get("evidence_text"),
                "row_name": None,
                "column_name": None,
                "value": evidence.get("evidence_text"),
                "table_cell": None,
            }
        else:
            hit = _best_match(str(evidence.get("evidence_text") or ""), files, item.get("symbol"), item.get("value"), source_for_match)

        status = hit.get("status")
        if status == "exact_match":
            exact += 1
        elif status == "normalized_match":
            normalized += 1
        elif status == "prefix_match":
            prefix += 1
        elif status == "source_backed":
            source_backed += 1
        elif status == "table_cell_match":
            table_cell += 1
        elif status == "line_fuzzy_match":
            fuzzy += 1
        elif status == "missing_evidence_text":
            missing += 1
        else:
            not_found += 1

        row = {
            "location": claim_location(item, idx),
            "canonical_name": item.get("canonical_name"),
            "symbol": item.get("symbol"),
            "evidence_kind": "table" if isinstance(evidence.get("table_evidence"), dict) else "text",
            "evidence_location_id": None,
            "evidence_ids": resolved_ids,
            "missing_evidence_object": bool(evidence_ids and not resolved_ids),
            **hit,
        }
        rows.append(row)

        if isinstance(src, dict):
            src.pop("_table_match", None)
            src.pop("grounding_status", None)
            src.pop("grounding_span", None)
            item["source"] = src
        if evidence:
            item["evidence"] = evidence
        item.pop("grounding_status", None)
        item.pop("grounding", None)

    total = len(rows)
    report = {
        "total_parameters_checked": total,
        "exact_match": exact,
        "normalized_match": normalized,
        "prefix_match": prefix,
        "source_backed": source_backed,
        "line_fuzzy_match": fuzzy,
        "table_cell_match": table_cell,
        "missing_evidence_text": missing,
        "not_found": not_found,
        "claims_without_evidence_ids": claims_without_evidence_ids,
        "missing_evidence_object": missing_evidence_object,
        "evidence_grounding_score": round(
            (
                (exact + (0.9 * normalized) + (0.7 * prefix) + (0.75 * source_backed) + (0.8 * table_cell) + (0.4 * fuzzy))
                / total
                * 100.0
            ) if total else 0.0,
            2,
        ),
        "evidence_object_count": len(existing_evidence),
        "rows": rows,
    }
    return extracted_json, report
