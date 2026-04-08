from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from postprocess.location_ids import claim_location
from postprocess.param_iter import iter_parameter_items_with_index


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _token_norm(text: str) -> str:
    text = str(text or "")
    text = text.replace("τ", "tau").replace("γ", "gamma").replace("˙", "dot")
    text = text.replace("̇", "dot")
    text = text.replace(" ", "")
    return re.sub(r"[^a-zA-Z0-9]+", "", text).lower()


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


def _merge_label_comma_fragments(parts: List[str]) -> List[str]:
    merged: List[str] = []
    for part in parts:
        token = str(part or "").strip()
        if not token:
            continue
        # Merge subscript/index-like numeric fragments back to the previous symbol:
        # e.g. "γ̇ 0, 1, γ̇ 0, 2" -> ["γ̇ 0,1", "γ̇ 0,2"]
        if merged and re.fullmatch(r"[-+]?[\d\s]+", token):
            merged[-1] = f"{merged[-1]},{token.replace(' ', '')}"
        else:
            merged.append(token)
    return merged


def _split_label_parts(text: Any) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = re.split(r"\s*,\s*", raw)
    return _merge_label_comma_fragments(parts)


def _split_value_parts(text: Any) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    return [part.strip() for part in re.split(r"\s*,\s*", raw) if str(part).strip()]


def _parse_scientific_text(text: str) -> float | None:
    s = str(text or "").strip()
    if not s:
        return None
    s = s.replace("−", "-").replace("–", "-").replace("×", "x")
    s = re.sub(r"\s+", " ", s)
    m = re.match(r"^([-+]?\d*\.?\d+)\s*[x\*]\s*10\s*([+-]?\d+)$", s, flags=re.IGNORECASE)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return base * (10 ** exp)
    return _to_float(s)


def _extract_leading_numeric_text(raw: str) -> Tuple[str, str | None]:
    text = str(raw or "").strip()
    if not text:
        return "", None
    text = re.sub(r"\s+", " ", text.replace("−", "-").replace("–", "-")).strip()

    sci = re.match(r"^([-+]?\d*\.?\d+(?:\s*[x×\*]\s*10\s*[+-]?\d+))(?:\s+(.*))?$", text, flags=re.IGNORECASE)
    if sci:
        return sci.group(1).strip(), (sci.group(2).strip() if sci.group(2) else None)

    dec = re.match(r"^([-+]?\d*\.?\d+)(?:\s+(.*))?$", text)
    if dec:
        return dec.group(1).strip(), (dec.group(2).strip() if dec.group(2) else None)

    return text, None


def _extract_value_unit_from_segment(segment: Any) -> Tuple[Any, str | None]:
    raw = str(segment or "").strip()
    if not raw:
        return None, None
    value_part, unit_part = _extract_leading_numeric_text(raw)
    value = _parse_scientific_text(value_part)
    if value is None:
        value = raw
    if unit_part:
        unit_part = unit_part.replace("−", "-").replace("⋅", "·")
    return value, unit_part


def _raw_value_segment(segment: Any) -> str:
    return str(segment or "").strip().replace("−", "-").replace("⋅", "·")


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


def _is_parameter_table(table: Dict[str, Any]) -> bool:
    caption = _norm(table.get("caption"))
    label = _norm(table.get("table_label"))
    rows = table.get("rows")
    haystack_parts = [caption, label]
    if isinstance(rows, list):
        for row in rows[:6]:
            if isinstance(row, list):
                haystack_parts.append(" | ".join(str(c) for c in row[:8]))
    haystack = _norm(" ".join(haystack_parts))
    parameter_tokens = (
        "parameter",
        "parameters",
        "constant",
        "constants",
        "hardening",
        "elastic",
        "plastic",
        "slip",
        "twin",
        "crss",
        "tau0",
        "theta0",
        "theta1",
        "g0",
        "g1",
        "c11",
        "c12",
        "c44",
    )
    has_parameter_signal = any(tok in haystack for tok in parameter_tokens)
    has_numeric_signal = bool(re.search(r"\d", haystack))
    if (
        "material parameter" in caption
        or "parameters in the cpfe simulation" in caption
        or "calibrated model parameters" in caption
        or "model parameters" in caption
        or ("properties" in caption and "elastic" in caption and "plastic" in caption)
    ):
        return True
    if str(table.get("table_kind") or "").strip().lower() == "image_backed" and has_parameter_signal:
        return True
    if not isinstance(rows, list) or not rows:
        return has_parameter_signal and has_numeric_signal
    header = rows[0]
    if not isinstance(header, list):
        return has_parameter_signal and has_numeric_signal
    joined = " | ".join(str(c) for c in header)
    header_norm = _norm(joined)
    has_value_col = "value" in header_norm
    has_name_col = ("parameter" in header_norm) or ("constant" in header_norm)
    if has_name_col and has_value_col:
        return True
    return has_parameter_signal and has_numeric_signal


def _extract_unit(label: str) -> str | None:
    m = re.search(r"\(([^()]+)\)\s*$", str(label or "").strip())
    return m.group(1).strip() if m else None


def _row_value(row: List[Any]) -> Any:
    if len(row) < 2:
        return None
    for cell in reversed(row[1:]):
        if str(cell or "").strip():
            return cell
    return None


def _split_section_title(cell: Any) -> Tuple[str | None, str | None]:
    text = str(cell or "").strip()
    if not text:
        return None, None
    parts = [p.strip() for p in text.split(" - ", 1)]
    if len(parts) == 2:
        return parts[0], parts[1].lower()
    return text, None


def _family_tokens(item: Dict[str, Any]) -> List[str]:
    applies = _safe_dict(item.get("applies_to"))
    values = [
        applies.get("family_name"),
        applies.get("family_id"),
        applies.get("mechanism"),
        item.get("notes"),
    ]
    tokens: List[str] = []
    mapping = {
        "prism": ["prism", "prismatic"],
        "basal": ["basal"],
        "pyramidal": ["pyramidal", "c+a"],
        "delta_hydride": ["{111}<110>", "delta", "hydride"],
        "hydride": ["{111}<110>", "delta", "hydride"],
    }
    for raw in values:
        text = _norm(str(raw or ""))
        if not text:
            continue
        for key, aliases in mapping.items():
            if key in text:
                tokens.extend(aliases)
        if "prism" in text or "prismatic" in text:
            tokens.extend(["prism", "prismatic"])
        if "basal" in text:
            tokens.append("basal")
        if "pyramidal" in text or "c+a" in text or "ca" in text:
            tokens.extend(["pyramidal", "c+a"])
        if "hydride" in text or "delta" in text:
            tokens.extend(["hydride", "delta", "{111}<110>"])
    deduped: List[str] = []
    seen = set()
    for tok in tokens:
        norm = _norm(tok)
        if norm and norm not in seen:
            deduped.append(norm)
            seen.add(norm)
    return deduped


def _section_phase_tokens(title: str | None) -> List[str]:
    t = _norm(title or "")
    out: List[str] = []
    if "alpha-zirconium" in t or "α-zirconium" in (title or "").lower():
        out.extend(["alpha", "zirconium"])
    if "delta-hydride" in t or "δ-hydride" in (title or "").lower():
        out.extend(["delta", "hydride"])
    return out


def _item_phase_tokens(item: Dict[str, Any], extracted_json: Dict[str, Any] | None = None) -> List[str]:
    applies = _safe_dict(item.get("applies_to"))
    vals = [
        applies.get("phase_id"),
        applies.get("mechanism"),
        item.get("claim_id"),
        item.get("notes"),
    ]
    out: List[str] = []
    for raw in vals:
        text = _norm(str(raw or ""))
        if not text:
            continue
        if "alpha" in text or "zirconium" in text:
            out.extend(["alpha", "zirconium"])
        if "delta" in text or "hydride" in text:
            out.extend(["delta", "hydride"])
    deduped: List[str] = []
    seen = set()
    for tok in out:
        if tok not in seen:
            deduped.append(tok)
            seen.add(tok)
    return deduped


def _column_keywords(item: Dict[str, Any]) -> List[str]:
    canonical = _norm(str(item.get("canonical_name") or ""))
    symbol = str(item.get("symbol") or "").strip()
    symbol_token = _token_norm(symbol)
    out: List[str] = []
    if canonical in {"strain_rate_reference", "reference_shear_strain_rate", "gamma0_ref"}:
        out.extend(["γ̇₀", "gammadot0", "reference shear rate", "γ̇0"])
    elif canonical in {"critical_resolved_shear_stress", "initial_crss"}:
        out.extend(["g₀", "g0"])
    elif canonical in {"asymptotic_hardening", "saturation_hardening"}:
        out.extend(["g₁", "g1"])
    elif canonical == "initial_hardening_rate":
        out.extend(["θ₀", "theta0"])
    elif canonical in {"saturation_hardening_rate", "asymptotic_hardening_rate"}:
        out.extend(["θ₁", "theta1"])
    elif canonical == "exponent_n":
        out.extend(["n"])
    elif canonical.startswith("elastic_stiffness_constant_"):
        suffix = canonical.rsplit("_", 1)[-1]
        out.extend([suffix.upper(), suffix.lower()])
    elif symbol:
        out.append(symbol)
    if "g1" in symbol_token or "θ1" in symbol or "theta1" in symbol_token:
        out.extend(["g₁", "g1", "θ₁", "theta1"])
    if "g0" in symbol_token or "θ0" in symbol or "theta0" in symbol_token:
        out.extend(["g₀", "g0", "θ₀", "theta0"])
    if symbol:
        out.append(symbol)
    deduped: List[str] = []
    seen = set()
    for tok in out:
        norm = _norm(str(tok or ""))
        if norm and norm not in seen:
            deduped.append(norm)
            seen.add(norm)
    return deduped


def _match_matrix_table(item: Dict[str, Any], table: Dict[str, Any], extracted_json: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) < 3:
        return None

    target_float = _to_float(item.get("value"))
    family_tokens = _family_tokens(item)
    phase_tokens = _item_phase_tokens(item, extracted_json)
    col_tokens = _column_keywords(item)
    if not family_tokens and not col_tokens:
        return None

    current_section_title = None
    current_section_kind = None
    header_row = None
    best = None
    best_score = -1.0

    for row_idx, row in enumerate(rows, start=1):
        if not isinstance(row, list) or not row:
            continue
        first = str(row[0] or "").strip()
        non_empty = [str(c or "").strip() for c in row if str(c or "").strip()]
        if len(non_empty) == 1 and " - " in first:
            current_section_title, current_section_kind = _split_section_title(first)
            header_row = None
            continue
        if current_section_kind and not header_row:
            header_row = [str(c or "").strip() for c in row]
            continue
        if not current_section_kind or not header_row:
            continue
        if current_section_kind == "plastic parameters":
            row_label = str(row[0] or "").strip()
            row_norm = _norm(row_label)
            row_score = 0.0
            for tok in family_tokens:
                if tok and tok in row_norm:
                    row_score += 2.0
            section_score = 0.0
            section_tokens = _section_phase_tokens(current_section_title)
            if phase_tokens and section_tokens:
                overlap = len(set(phase_tokens).intersection(section_tokens))
                section_score += overlap * 1.5
            if row_score <= 0:
                continue
            for col_idx in range(2, min(len(row), len(header_row)) + 1):
                header = str(header_row[col_idx - 1] or "").strip()
                header_norm = _norm(header)
                header_score = 0.0
                for tok in col_tokens:
                    if not tok:
                        continue
                    if tok == header_norm:
                        header_score += 3.0
                    elif tok in header_norm:
                        header_score += 2.0
                if header_score <= 0:
                    continue
                cell = row[col_idx - 1]
                parsed_value, parsed_unit = _extract_value_unit_from_segment(cell)
                value_score = 0.0
                parsed_float = _to_float(parsed_value)
                if target_float is not None and parsed_float is not None and abs(target_float - parsed_float) <= max(1e-9, abs(target_float) * 1e-6):
                    value_score += 3.0
                score = row_score + header_score + section_score + value_score
                if score > best_score:
                    best_score = score
                    best = {
                        "row_index": row_idx,
                        "label": row_label,
                        "value": parsed_value,
                        "unit": parsed_unit,
                        "score": score,
                        "column_index": col_idx,
                        "column_name": header,
                        "section_title": current_section_title,
                    }
        elif current_section_kind == "elastic constants":
            # Header row contains elastic symbols; next numeric row contains values.
            is_value_row = all(re.search(r"\d", str(c or "")) for c in row[: max(1, len(non_empty))] if str(c or "").strip())
            if not is_value_row:
                continue
            section_score = 0.0
            section_tokens = _section_phase_tokens(current_section_title)
            if phase_tokens and section_tokens:
                overlap = len(set(phase_tokens).intersection(section_tokens))
                section_score += overlap * 1.5
            for col_idx in range(1, min(len(row), len(header_row)) + 1):
                header = str(header_row[col_idx - 1] or "").strip()
                header_norm = _norm(header)
                header_score = 0.0
                for tok in col_tokens:
                    if not tok:
                        continue
                    if tok == header_norm:
                        header_score += 3.0
                    elif tok in header_norm:
                        header_score += 2.0
                if header_score <= 0:
                    continue
                cell = row[col_idx - 1]
                parsed_value, parsed_unit = _extract_value_unit_from_segment(cell)
                parsed_float = _to_float(parsed_value)
                value_score = 0.0
                if target_float is not None and parsed_float is not None and abs(target_float - parsed_float) <= max(1e-9, abs(target_float) * 1e-6):
                    value_score += 3.0
                score = header_score + section_score + value_score
                if score > best_score:
                    best_score = score
                    best = {
                        "row_index": row_idx,
                        "label": header,
                        "value": parsed_value,
                        "unit": parsed_unit or _extract_unit(header),
                        "score": score,
                        "column_index": col_idx,
                        "column_name": header,
                        "section_title": current_section_title,
                    }
    if best and best_score >= 4.0:
        return best
    return None


def _parameter_keywords(item: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    symbol = item.get("symbol")
    if symbol:
        out.append(_token_norm(symbol))
    canonical = str(item.get("canonical_name") or "").strip().lower()
    if canonical:
        out.append(_token_norm(canonical))
    description = str(item.get("description") or "").strip().lower()
    if description:
        out.append(_token_norm(description))

    alias_map = {
        "elastic_stiffness_constant_c11": ["c11", "elasticstiffnessconstantc11"],
        "elastic_stiffness_constant_c12": ["c12", "elasticstiffnessconstantc12"],
        "elastic_stiffness_constant_c44": ["c44", "elasticstiffnessconstantc44"],
        "crss_initial": ["tau0", "criticalshearstrength"],
        "latent_ratio_q": ["q", "latenthardeningcoefficient"],
        "hardening_h0": ["h0", "initialhardeningmodulus"],
        "exponent_n": ["n", "ratesensitivity"],
        "gamma0_ref": ["gamma0", "gammadot0", "referenceshearrate"],
    }
    for alias in alias_map.get(canonical, []):
        out.append(_token_norm(alias))

    deduped: List[str] = []
    seen = set()
    for token in out:
        if token and token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped


def _match_row(item: Dict[str, Any], table: Dict[str, Any], extracted_json: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    matrix_match = _match_matrix_table(item, table, extracted_json)
    if matrix_match:
        return matrix_match
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return None
    keywords = _parameter_keywords(item)
    if not keywords:
        return None
    target_float = _to_float(item.get("value"))

    best: Dict[str, Any] | None = None
    best_score = -1.0
    for row_idx, row in enumerate(rows[1:], start=2):
        if not isinstance(row, list) or not row:
            continue
        label = str(row[0]).strip()
        label_norm = _token_norm(label)
        if not label_norm:
            continue
        value = _row_value(row)
        parsed_row_value, parsed_row_unit = _extract_value_unit_from_segment(value)
        value_float = _to_float(parsed_row_value)
        score = 0.0
        segmented_match: Dict[str, Any] | None = None

        label_parts = _split_label_parts(label)
        value_parts = _split_value_parts(value)
        if len(label_parts) > 1 and len(value_parts) == len(label_parts):
            best_seg_score = -1.0
            best_seg: Dict[str, Any] | None = None
            for seg_idx, (label_part, value_part) in enumerate(zip(label_parts, value_parts), start=1):
                seg_label_norm = _token_norm(label_part)
                seg_score = 0.0
                for kw in keywords:
                    if not kw:
                        continue
                    if kw == seg_label_norm:
                        seg_score += 4.0
                    elif len(kw) > 1 and kw in seg_label_norm:
                        seg_score += 2.5
                parsed_value, parsed_unit = _extract_value_unit_from_segment(value_part)
                parsed_float = _to_float(parsed_value)
                if target_float is not None and parsed_float is not None and abs(target_float - parsed_float) <= max(1e-9, abs(target_float) * 1e-6):
                    seg_score += 2.5
                if seg_score > best_seg_score:
                    best_seg_score = seg_score
                    best_seg = {
                        "row_index": row_idx,
                        "label": label_part,
                        "value": parsed_value,
                        "unit": parsed_unit,
                        "raw_value": _raw_value_segment(value_part),
                        "score": seg_score,
                        "segmented_match": True,
                        "segment_index": seg_idx,
                        "column_index": len(row),
                        "full_row_label": label,
                        "full_row_value": value,
                    }
            if best_seg and best_seg_score >= 2.5:
                segmented_match = best_seg

        for kw in keywords:
            if not kw:
                continue
            if kw == label_norm:
                score += 3.0
            elif len(kw) > 1 and kw in label_norm:
                score += 2.0
        if target_float is not None and value_float is not None and abs(target_float - value_float) <= max(1e-9, abs(target_float) * 1e-6):
            score += 2.5
        if segmented_match and segmented_match["score"] >= score:
            return segmented_match
        if score > best_score:
            best_score = score
            best = {
                "row_index": row_idx,
                "label": label,
                "value": parsed_row_value,
                "unit": parsed_row_unit or _extract_unit(label),
                "score": score,
                "column_index": len(row),
            }
    if best and best_score >= 2.0:
        return best
    return None


def resolve_parameter_tables(extracted_json: Dict[str, Any], paper_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tables = [t for t in _load_table_jsons(paper_dir) if _is_parameter_table(t)]
    rows: List[Dict[str, Any]] = []
    matched = 0
    values_filled = 0
    units_filled = 0

    for idx, _, item in iter_parameter_items_with_index(extracted_json):
        matched_row = None
        matched_table = None
        for table in tables:
            row = _match_row(item, table, extracted_json)
            if row:
                matched_row = row
                matched_table = table
                break

        row_report = {
            "location": claim_location(item, idx),
            "canonical_name": item.get("canonical_name"),
            "symbol": item.get("symbol"),
            "status": "no_match",
        }
        if not matched_row or not matched_table:
            rows.append(row_report)
            continue

        matched += 1
        source = _safe_dict(item.get("source"))
        evidence = _safe_dict(item.get("evidence"))

        existing_table_evidence = _safe_dict(evidence.get("table_evidence"))
        has_raw_table_evidence = any(
            existing_table_evidence.get(key) not in (None, "", [])
            for key in ("row_name", "column_name", "value", "excerpt")
        )
        if not evidence.get("evidence_text"):
            evidence["evidence_text"] = (
                matched_row.get("raw_value")
                or matched_row.get("full_row_value")
                or matched_row.get("full_row_label")
                or matched_row["label"]
            )
        if not has_raw_table_evidence:
            evidence["table_evidence"] = {
                "row_name": matched_row.get("full_row_label") or matched_row.get("label"),
                "column_name": matched_row.get("column_name"),
                "value": matched_row.get("raw_value") or matched_row.get("value"),
                "excerpt": evidence.get("evidence_text"),
            }
        else:
            evidence["table_evidence"] = {
                "row_name": existing_table_evidence.get("row_name"),
                "column_name": existing_table_evidence.get("column_name"),
                "value": existing_table_evidence.get("value"),
                "excerpt": existing_table_evidence.get("excerpt"),
            }
        source["_table_match"] = {
            "table": matched_table.get("_path"),
            "row_index": matched_row.get("row_index"),
            "column_index": matched_row.get("column_index"),
            "segment_index": matched_row.get("segment_index"),
            "label": matched_row.get("label"),
            "full_row_label": matched_row.get("full_row_label") or matched_row.get("label"),
            "raw_value": matched_row.get("raw_value") or matched_row.get("value"),
            "full_row_value": matched_row.get("full_row_value") or matched_row.get("value"),
        }
        item["source"] = source
        item["evidence"] = evidence

        if matched_row.get("segmented_match"):
            item["value"] = matched_row["value"]
            item["unit"] = matched_row["unit"]
            if matched_row["unit"] in (None, ""):
                item.pop("value_SI", None)
                item.pop("unit_SI", None)
            values_filled += 1
            if matched_row["unit"] not in (None, ""):
                units_filled += 1
        elif item.get("value") in (None, "") and matched_row["value"] not in (None, ""):
            item["value"] = matched_row["value"]
            values_filled += 1
        if not matched_row.get("segmented_match") and item.get("unit") in (None, "") and matched_row["unit"]:
            item["unit"] = matched_row["unit"]
            units_filled += 1

        row_report.update({
            "status": "matched",
            "table": matched_table.get("_path"),
            "row_index": matched_row["row_index"],
            "label": matched_row["label"],
            "row_value": matched_row["value"],
            "row_unit": matched_row["unit"],
            "match_score": matched_row["score"],
            "segmented_match": bool(matched_row.get("segmented_match")),
            "segment_index": matched_row.get("segment_index"),
        })
        rows.append(row_report)

    return extracted_json, {
        "candidate_tables": [t.get("_path") for t in tables],
        "parameters_checked": len(rows),
        "matched_rows": matched,
        "values_filled": values_filled,
        "units_filled": units_filled,
        "rows": rows,
    }
