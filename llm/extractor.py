import os, re, json, glob, base64
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import time
from elsevier.fulltext_parser import download_table_image
from llm.openai_sanitize import sanitize_text_for_openai, validate_openai_json_payload

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _is_retryable_llm_error(exc: Exception) -> bool:
    name = exc.__class__.__name__
    if name in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}:
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) in {408, 409, 429, 500, 502, 503, 504}:
        return True

    return False


def _chat_completion_with_retry(*, model: str, messages: List[Dict[str, Any]], max_retries: int = 4):
    delay = 1.0
    last_exc: Exception | None = None
    request_payload = validate_openai_json_payload({
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": messages,
    })

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**request_payload)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable_llm_error(exc):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 20.0)

    raise RuntimeError(f"LLM request failed after retries: {last_exc}")

def trim_text(text: str, max_chars: int) -> str:
    text = sanitize_text_for_openai(text).strip()
    return text[:max_chars] + ("...[TRUNCATED]..." if len(text) > max_chars else "")


def safe_filename(text: str) -> str:
    text = re.sub(r"[\\/*?:\"<>|]+", "_", str(text or ""))
    text = re.sub(r"\s+", "_", text.strip())
    return text[:120] or "untitled"


def image_to_data_url(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".") or "jpeg"
    mime = "image/jpeg"
    if ext == "png":
        mime = "image/png"
    elif ext == "gif":
        mime = "image/gif"
    elif ext == "webp":
        mime = "image/webp"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _section_selection_preview(text: str, max_lines: int = 3) -> str:
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if not line:
            continue
        lines.append(line)
    if len(lines) <= 1:
        return ""
    preview_lines = lines[1:max_lines + 1]
    return " || ".join(preview_lines)

def load_md_files(folder: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(folder, "*.md")))
    out = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            txt = sanitize_text_for_openai(f.read())
        title = None
        for line in txt.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip() or None
            else:
                title = stripped[:200]
            break
        out.append({
            "name": os.path.basename(path),
            "path": path,
            "text": txt,
            "length": len(txt),
            "title": title,
            "selection_preview": _section_selection_preview(txt),
        })
    return out


def _is_comparative_parameter_table_json(table_json: Dict[str, Any]) -> bool:
    caption = str(table_json.get("caption") or "").lower()
    rows = table_json.get("rows")
    if "elastic constants" in caption and ("slip strength" in caption or "zener ratio" in caption):
        return True
    if not isinstance(rows, list) or not rows:
        return False
    first_row = " ".join(str(cell or "") for cell in rows[0]).lower()
    return "material" in first_row and ("elastic constants" in first_row or "slip modes" in first_row)


def _numeric_token(text: Any) -> str | None:
    token = str(text or "").strip().replace("−", "-").replace("–", "-")
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?", token):
        return token
    return None


def _citation_token(text: Any) -> str | None:
    token = str(text or "").strip()
    if re.fullmatch(r"\[[^\]]+\]", token):
        return token
    return None


def _render_comparative_parameter_matrix(table_json: Dict[str, Any], max_rows: int | None = None) -> str:
    rows = table_json.get("rows")
    caption = str(table_json.get("caption") or "").strip()
    if not isinstance(rows, list):
        return caption

    lines: List[str] = []
    if caption:
        lines.append(f"Caption: {caption}")

    fcc_slip_label = None
    hcp_slip_labels: List[str] = []
    current_structure = None
    rendered_data_rows = 0

    for row in rows:
        if max_rows is not None and rendered_data_rows >= max_rows:
            break
        if not isinstance(row, list):
            continue
        cells = [str(cell or "").strip() for cell in row]
        if not any(cells):
            continue
        joined = " ".join(cells)

        if "Zener" in joined and "C 11" in joined:
            fcc_slip_label = next((cell for cell in cells if "{ 1 1" in cell or "110" in cell), None)
            continue
        if "Prismatic" in joined or "Basal" in joined or "Pyramidal" in joined:
            current_structure = "HCP"
            hcp_slip_labels = [cell for cell in cells if any(tok in cell for tok in ("Prismatic", "Basal", "Pyramidal"))]
            lines.append(
                "HCP slip-mode columns: " + ", ".join(hcp_slip_labels)
                if hcp_slip_labels else "HCP slip-mode columns detected."
            )
            continue

        explicit_structure = next((cell for cell in cells[:2] if cell in {"FCC", "HCP", "BCC"}), None)
        if explicit_structure:
            current_structure = explicit_structure

        material = next(
            (
                cell for cell in cells[:4]
                if cell
                and cell not in {"FCC", "HCP", "BCC"}
                and _numeric_token(cell) is None
                and _citation_token(cell) is None
            ),
            None,
        )
        if not material or current_structure not in {"FCC", "HCP"}:
            continue

        numeric_values = [token for token in (_numeric_token(cell) for cell in cells) if token is not None]
        citations = [token for token in (_citation_token(cell) for cell in cells) if token is not None]

        if current_structure == "FCC" and len(numeric_values) >= 5:
            slip_label = fcc_slip_label or "active slip mode"
            lines.append(
                f"Comparative row: Structure=FCC, Material={material}, Zener={numeric_values[0]}, "
                f"C11={numeric_values[1]} GPa, C12={numeric_values[2]} GPa, C44={numeric_values[3]} GPa, "
                f"τ0,i for {slip_label}={numeric_values[4]} MPa"
                + (f", Ref={citations[0]}" if citations else "")
            )
            rendered_data_rows += 1
            continue

        if current_structure == "HCP" and hcp_slip_labels and len(numeric_values) >= len(hcp_slip_labels):
            slip_values = numeric_values[-len(hcp_slip_labels):]
            prefix_parts = [f"Structure=HCP", f"Material={material}"]
            if len(numeric_values) > len(hcp_slip_labels):
                prefix_parts.append(f"c/a={numeric_values[0]}")
            if len(numeric_values) > len(hcp_slip_labels) + 1:
                prefix_parts.append(f"C11={numeric_values[1]} GPa")
            slip_parts = [f"{label}={value} MPa" for label, value in zip(hcp_slip_labels, slip_values)]
            line = "Comparative row: " + ", ".join(prefix_parts + slip_parts)
            if citations:
                line += f", Ref={citations[-1]}"
            lines.append(line)
            rendered_data_rows += 1

    return "\n".join(lines).strip()


def _table_json_summary(table_json: Dict[str, Any], max_rows: int = 8, max_cells_per_row: int = 16) -> str:
    caption = str(table_json.get("caption") or "").strip()
    rows = table_json.get("rows")
    if table_json.get("table_kind") == "image_backed":
        label = str(table_json.get("table_label") or "").strip()
        image = table_json.get("image") or {}
        local_path = str(image.get("local_path") or "").strip()
        parts = []
        if label:
            parts.append(f"Label: {label}")
        if caption:
            parts.append(f"Caption: {caption}")
        if local_path:
            parts.append("This table has a local image and can be passed directly to the extraction model.")
        return "\n".join(parts).strip()
    if _is_comparative_parameter_table_json(table_json):
        return _render_comparative_parameter_matrix(table_json, max_rows=max_rows)
    if not isinstance(rows, list):
        return caption

    lines: List[str] = []
    if caption:
        lines.append(f"Caption: {caption}")
    lines.extend(_render_table_rows_with_alignment(rows, max_rows=max_rows, max_cells_per_row=max_cells_per_row))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines).strip()


def _table_json_full_text(table_json: Dict[str, Any], max_cells_per_row: int = 40) -> str:
    caption = str(table_json.get("caption") or "").strip()
    rows = table_json.get("rows")
    if table_json.get("table_kind") == "image_backed":
        return _table_json_summary(table_json, max_rows=1000, max_cells_per_row=max_cells_per_row)
    if _is_comparative_parameter_table_json(table_json):
        return _render_comparative_parameter_matrix(table_json, max_rows=None)
    if not isinstance(rows, list):
        return caption

    lines: List[str] = []
    if caption:
        lines.append(f"Caption: {caption}")
    lines.extend(_render_table_rows_with_alignment(rows, max_rows=len(rows), max_cells_per_row=max_cells_per_row))
    return "\n".join(lines).strip()


def _render_table_rows_with_alignment(
    rows: List[Any],
    *,
    max_rows: int,
    max_cells_per_row: int,
) -> List[str]:
    rendered: List[str] = []
    raw_rows: List[List[str]] = []
    for row in rows[:max_rows]:
        if not isinstance(row, list):
            continue
        normalized = [str(cell).strip() for cell in row[:max_cells_per_row]]
        if not any(cell for cell in normalized):
            continue
        raw_rows.append(normalized)

    normalized_rows, header_row_count = _normalize_table_rows_for_render(raw_rows)

    if not normalized_rows:
        return rendered

    base_header_rows = normalized_rows[:header_row_count]
    active_header_rows = list(base_header_rows)
    header_labels = _build_render_header_labels(active_header_rows)
    rendered.append("Columns: " + " | ".join(f"c{idx + 1}={label}" for idx, label in enumerate(header_labels)))

    for idx, row in enumerate(normalized_rows, start=1):
        if idx <= header_row_count:
            rendered.append(
                f"Header row {idx}: " + " | ".join(
                    f"c{col_idx + 1}={cell or '<EMPTY>'}" for col_idx, cell in enumerate(row)
                )
            )
            continue

        if _looks_like_headerish_row(row):
            active_header_rows = base_header_rows + [row]
            header_labels = _build_render_header_labels(active_header_rows)
            rendered.append(
                f"Subheader row {idx}: " + " | ".join(
                    f"c{col_idx + 1}={cell or '<EMPTY>'}" for col_idx, cell in enumerate(row)
                )
            )
            rendered.append(
                f"Columns after row {idx}: " + " | ".join(
                    f"c{col_idx + 1}={label}" for col_idx, label in enumerate(header_labels)
                )
            )
            continue

        cells: List[str] = []
        row_key = _row_key_from_cells(row)
        if row_key:
            cells.append(f"row_key={row_key}")
        for col_idx, header_label in enumerate(header_labels):
            value = row[col_idx] if col_idx < len(row) else ""
            cells.append(f"{header_label}={value or '<EMPTY>'}")
        rendered.append(f"Row {idx}: " + " | ".join(cells))
    return rendered


def _looks_like_data_row(row: List[str]) -> bool:
    numeric_like = 0
    for cell in row:
        cell_norm = str(cell or "").strip()
        if not cell_norm:
            continue
        compact = cell_norm.replace("−", "-").replace("–", "-").replace(" ", "")
        if re.fullmatch(r"\[[^\]]+\]", compact):
            continue
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?::[-+]?(?:\d+(?:\.\d+)?|\.\d+))+?", compact):
            numeric_like += 1
            continue
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?", compact):
            numeric_like += 1
    return numeric_like >= 3


def _looks_like_headerish_row(row: List[str]) -> bool:
    non_empty = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
    if len(non_empty) < 2 or _looks_like_data_row(row):
        return False
    header_tokens = 0
    for token in non_empty:
        token_norm = token.lower()
        if re.search(r"(material|structure|ratio|ref|elastic|constant|slip|mode|prismatic|basal|pyramidal|c11|c12|c13|c33|c44|zener|c/a|τ|tau)", token_norm):
            header_tokens += 1
            continue
        if re.search(r"[A-Za-zα-ωΑ-Ω]", token):
            header_tokens += 1
    return header_tokens >= max(2, len(non_empty) // 2)


def _row_key_from_cells(row: List[str]) -> str:
    for cell in row:
        token = str(cell or "").strip()
        if token and not token.startswith("<INHERITED:"):
            return token
    for cell in row:
        token = str(cell or "").strip()
        if token.startswith("<INHERITED:"):
            return token
    return ""


def _pad_row(row: List[str], width: int, *, align: str) -> List[str]:
    if len(row) >= width:
        return row[:width]
    missing = width - len(row)
    if align == "right":
        return ([""] * missing) + row
    return row + ([""] * missing)


def _fill_header_row(row: List[str]) -> List[str]:
    filled: List[str] = []
    last_non_empty = max((idx for idx, cell in enumerate(row) if str(cell or "").strip()), default=-1)
    for idx, cell in enumerate(row):
        value = str(cell or "").strip()
        if value:
            filled.append(value)
            continue
        if idx > last_non_empty:
            filled.append("")
            continue
        carry = ""
        for back_idx in range(idx - 1, -1, -1):
            if row[back_idx]:
                carry = str(row[back_idx]).strip()
                break
        filled.append(carry)
    return filled


def _build_render_header_labels(header_rows: List[List[str]]) -> List[str]:
    if not header_rows:
        return []
    width = max(len(row) for row in header_rows)
    layered: List[List[str]] = []
    for row_idx, row in enumerate(header_rows):
        padded = _pad_row(row, width, align="left")
        if row_idx == 0:
            layered.append(_fill_header_row(padded))
            continue

        filled: List[str] = []
        last_non_empty = max((idx for idx, cell in enumerate(padded) if str(cell or "").strip()), default=-1)
        for col_idx, cell in enumerate(padded):
            value = str(cell or "").strip()
            if value:
                filled.append(value)
                continue
            if col_idx > last_non_empty:
                filled.append("")
                continue
            if col_idx > 0 and all(
                str(parent[col_idx] or "").strip() == str(parent[col_idx - 1] or "").strip()
                for parent in layered
            ):
                filled.append(filled[-1] if filled else "")
            else:
                filled.append("")
        layered.append(filled)

    labels: List[str] = []
    for col_idx in range(width):
        parts: List[str] = []
        for row in layered:
            part = str(row[col_idx] or "").strip()
            if part and (not parts or parts[-1] != part):
                parts.append(part)
        labels.append(" / ".join(parts) if parts else f"col_{col_idx + 1}")
    return labels


def _normalize_table_rows_for_render(rows: List[List[str]]) -> Tuple[List[List[str]], int]:
    if not rows:
        return [], 0

    width = max(len(row) for row in rows)
    header_row_count = 1
    for idx, row in enumerate(rows[1:], start=1):
        if _looks_like_data_row(row):
            header_row_count = idx
            break
    else:
        header_row_count = min(len(rows), 2)

    normalized: List[List[str]] = []
    data_context = [""] * width
    for idx, row in enumerate(rows):
        if idx < header_row_count:
            normalized.append(_pad_row(row, width, align="left"))
            continue

        padded = _pad_row(row, width, align=_infer_data_row_alignment(row, width))
        first_explicit = next((col_idx for col_idx, cell in enumerate(padded) if cell), width)
        expanded: List[str] = []
        for col_idx, cell in enumerate(padded):
            value = str(cell or "").strip()
            if value:
                data_context[col_idx] = value
                expanded.append(value)
            elif col_idx < first_explicit and data_context[col_idx]:
                expanded.append(f"<INHERITED:{data_context[col_idx]}>")
            else:
                expanded.append("")
        normalized.append(expanded)

    return normalized, max(1, header_row_count)


def _infer_data_row_alignment(row: List[str], width: int) -> str:
    if len(row) >= width:
        return "left"
    if not row:
        return "left"
    if any(not str(cell or "").strip() for cell in row[: min(3, len(row))]):
        return "left"
    if any(not str(cell or "").strip() for cell in row[:-1]):
        return "left"
    return "right"


def _fallback_table_score(table: Dict[str, Any]) -> int:
    haystack = " ".join(
        str(table.get(k) or "")
        for k in ("display_label", "json_summary", "extract_text", "text", "name", "table_kind")
    ).lower()
    keywords = (
        "parameter",
        "hardening",
        "elastic",
        "plastic",
        "constitutive",
        "slip",
        "twin",
        "calibration",
        "moduli",
        "strength",
        "chemical composition",
        "composition",
        "phase fraction",
        "phase constitution",
        "grain size",
        "texture",
        "microstructure",
        "material input",
    )
    score = sum(1 for kw in keywords if kw in haystack)
    if str(table.get("table_kind") or "").strip().lower() == "image_backed":
        score += 1
    return score


def _fallback_select_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        tables,
        key=lambda t: (_fallback_table_score(t), -int(t.get("length") or 0)),
        reverse=True,
    )
    if not ranked:
        return []
    top_score = _fallback_table_score(ranked[0])
    if top_score <= 0:
        return ranked[:1]
    return [t for t in ranked[:2] if _fallback_table_score(t) == top_score] or ranked[:1]


def _is_material_profile_table(table: Dict[str, Any]) -> bool:
    haystack = " ".join(
        str(table.get(k) or "")
        for k in ("display_label", "json_summary", "extract_text", "text", "name")
    ).lower()
    keywords = (
        "chemical composition",
        "composition",
        "phase fraction",
        "phase constitution",
        "grain size",
        "texture",
        "microstructure",
        "material input",
        "euler angles",
    )
    return any(kw in haystack for kw in keywords)


def _microstructure_section_score(section: Dict[str, Any]) -> int:
    haystack = " ".join(
        str(section.get(k) or "")
        for k in ("name", "title", "selection_preview", "text")
    ).lower()
    keywords = (
        "microstructure",
        "texture",
        "grain size",
        "grain boundary",
        "grain boundaries",
        "gbs",
        "grain boundary sliding",
        "ebsd",
        "tkd",
        "tem",
        "stem",
        "dislocation",
        "kam",
        "misorientation",
        "twin volume fraction",
        "twinning",
        "shear band",
        "recrystallized",
        "equiaxed",
        "bimodal",
        "sub-boundar",
        "lattice distortion",
    )
    score = sum(1 for kw in keywords if kw in haystack)
    title = str(section.get("title") or "").lower()
    if any(
        kw in title
        for kw in (
            "microstructure",
            "texture",
            "dislocation",
            "material",
            "ebsd",
            "tem",
            "results",
        )
    ):
        score += 2
    return score


def _has_selected_microstructure_section(selected_sections: List[Dict[str, Any]]) -> bool:
    return any(_microstructure_section_score(section) >= 3 for section in selected_sections)


def _table_semantic_type(table: Dict[str, Any]) -> str:
    haystack = " ".join(
        str(table.get(k) or "")
        for k in ("display_label", "json_summary", "extract_text", "text", "name")
    ).lower()
    if any(kw in haystack for kw in ("chemical composition", "composition", "wt.%", "at.%")):
        return "composition_matrix"
    if any(kw in haystack for kw in ("phase fraction", "phase constitution", "volume fraction")):
        return "phase_fraction_matrix"
    if any(kw in haystack for kw in ("temperature", "grain size", "method")) and any(
        kw in haystack for kw in ("τ", "tau", "h0", "h 0", "parameter", "model parameters", "crss", "gamma", "γ")
    ):
        return "parameter_bundle_table"
    if any(kw in haystack for kw in ("grain size", "texture", "euler angle", "orientation", "microstructure")):
        return "sample_profile_matrix"
    if any(kw in haystack for kw in ("elastic constants", "slip strength", "zener ratio")):
        return "comparative_parameter_matrix"
    if any(kw in haystack for kw in ("parameter", "hardening", "elastic", "plastic", "constitutive")):
        return "parameter_table"
    return "generic_table"


def _table_semantic_hint(table: Dict[str, Any]) -> str:
    semantic_type = _table_semantic_type(table)
    hints = {
        "composition_matrix": "Interpret this as a material-composition matrix. Prefer populating materials[].composition rather than creating parameter claims.",
        "phase_fraction_matrix": "Interpret this as a process-state or condition profile table. Prefer populating microstructure_features[] and linking them to process_states[] or conditions[].",
        "sample_profile_matrix": "Interpret this as a process-state profile table. Prefer populating process_states[] and microstructure_features[] with grain size, texture, orientation, or processing-state facts.",
        "comparative_parameter_matrix": "Interpret this as a comparative parameter matrix spanning multiple materials or constituents. Keep material and process-state identity explicit and avoid collapsing all columns into one material.",
        "parameter_bundle_table": "Interpret this as a multi-condition parameter table. Expand rows or columns into separate parameter claims and link them through process_state_id and condition_id when conditions are explicit.",
        "parameter_table": "Interpret this as a parameter table. Extract all explicit parameter values completely into parameter_claims[].",
        "generic_table": "Interpret this conservatively and only extract explicit, well-supported facts.",
    }
    return f"Table semantic type: {semantic_type}. {hints.get(semantic_type, '')}".strip()


_EXPLICIT_PARAMETER_NAME_RE = re.compile(
    r"\b(c11|c12|c13|c33|c44|c55|c66|tau0|tau_0|tau1|theta0|theta1|h0|h1|g0|g1|xi0|xi_inf|gamma0|gammadot0|qab|gamma|crss|elastic constant|hardening|slip resistance|reference shear rate)\b",
    re.IGNORECASE,
)
_EXPLICIT_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def _table_has_explicit_parameter_values(table: Dict[str, Any]) -> bool:
    semantic_type = _table_semantic_type(table)
    if semantic_type not in {"parameter_table", "comparative_parameter_matrix", "parameter_bundle_table"}:
        return False
    haystack = " ".join(
        str(table.get(k) or "")
        for k in ("display_label", "extract_text", "json_summary", "text", "name")
    )
    return bool(_EXPLICIT_PARAMETER_NAME_RE.search(haystack) and _EXPLICIT_NUMBER_RE.search(haystack))


def _section_has_explicit_parameter_values(section: Dict[str, Any]) -> bool:
    text = str(section.get("text") or "")
    return bool(_EXPLICIT_PARAMETER_NAME_RE.search(text) and _EXPLICIT_NUMBER_RE.search(text))


def _should_skip_extraction_no_explicit_parameters(
    selected_sections: List[Dict[str, Any]],
    selected_tables: List[Dict[str, Any]],
) -> bool:
    if any(_table_has_explicit_parameter_values(table) for table in selected_tables):
        return False
    if any(_section_has_explicit_parameter_values(section) for section in selected_sections):
        return False
    return True


def _empty_extraction_payload(reason: str) -> Dict[str, Any]:
    payload = _coerce_to_schema_shape(EXTRACT_SCHEMA_SKELETON, {})
    payload["global_notes"] = reason
    return payload


def _equation_relevance_score(equation: Dict[str, Any]) -> int:
    haystack = " ".join(
        str(equation.get(k) or "")
        for k in ("display_label", "text", "extract_text", "name", "section_title", "paragraph_text", "role_hint")
    ).lower()
    keywords = (
        "dot{\\gamma",
        "dot{\\tau",
        "dot{\\g",
        "hardening",
        "constitutive",
        "slip",
        "twin",
        "crss",
        "tau",
        "gamma",
        "flow rule",
        "yield",
        "backstress",
        "kinematic",
        "latent",
        "resistance",
        "creep",
        "relaxation",
        "armstrong",
        "frederick",
        "superposition",
        "evolution",
    )
    return sum(1 for kw in keywords if kw in haystack)


def _fallback_select_equations(equations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        equations,
        key=lambda e: (_equation_relevance_score(e), -int(e.get("length") or 0)),
        reverse=True,
    )
    ranked = [e for e in ranked if _equation_relevance_score(e) > 0]
    return ranked[:8]


def _section_equation_relevance_score(section: Dict[str, Any]) -> int:
    haystack = " ".join(
        str(section.get(k) or "")
        for k in ("name", "title", "selection_preview", "text")
    ).lower()
    keywords = (
        "constitutive",
        "equation",
        "model",
        "calibration",
        "hardening",
        "backstress",
        "creep",
        "relaxation",
        "slip",
        "twinning",
    )
    return sum(1 for kw in keywords if kw in haystack)


def _augment_selected_equations(
    selected_equations: List[Dict[str, Any]],
    equations: List[Dict[str, Any]],
    selected_sections: List[Dict[str, Any]],
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    if not equations:
        return selected_equations

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add(rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            eq_id = str(row.get("selection_id") or row.get("name") or "").strip()
            if not eq_id or eq_id in seen:
                continue
            out.append(row)
            seen.add(eq_id)
            if len(out) >= limit:
                return

    _add(selected_equations)
    if len(out) >= limit:
        return out

    section_names = {
        str(s.get("name") or "").strip().lower()
        for s in selected_sections
        if isinstance(s, dict)
    }
    section_titles = {
        str(s.get("title") or "").strip().lower()
        for s in selected_sections
        if isinstance(s, dict) and str(s.get("title") or "").strip()
    }
    has_equation_rich_section = any(_section_equation_relevance_score(s) > 0 for s in selected_sections)

    matched_by_section: List[Dict[str, Any]] = []
    for eq in equations:
        section_title = str(eq.get("section_title") or "").strip().lower()
        if not section_title:
            continue
        if section_title in section_titles:
            matched_by_section.append(eq)
            continue
        safe_name = safe_filename(section_title)
        if any(safe_name in name for name in section_names):
            matched_by_section.append(eq)

    matched_by_section = sorted(
        matched_by_section,
        key=lambda e: (_equation_relevance_score(e), -int(e.get("length") or 0)),
        reverse=True,
    )
    _add(matched_by_section)
    if len(out) >= limit:
        return out

    if has_equation_rich_section:
        _add(_fallback_select_equations(equations))

    return out


def load_table_files(folder: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(folder, "table_*.json")))
    base_dir = os.path.dirname(folder)
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                table_json = json.load(f)
        except Exception:
            continue
        if not isinstance(table_json, dict):
            continue
        json_summary = _table_json_summary(table_json)
        extract_text = _table_json_full_text(table_json)
        selection_id = Path(path).stem
        out.append({
            "name": os.path.basename(path),
            "selection_id": selection_id,
            "display_label": str(table_json.get("table_label") or selection_id),
            "path": path,
            "text": json_summary,
            "length": len(json_summary),
            "table_json": table_json,
            "json_summary": json_summary,
            "extract_text": extract_text,
            "table_kind": table_json.get("table_kind") or "text",
            "image_local_path": (
                os.path.join(base_dir, str((table_json.get("image") or {}).get("local_path")))
                if str((table_json.get("image") or {}).get("local_path") or "").strip()
                else None
            ),
        })
    return out


def load_equation_files(folder: str) -> List[Dict[str, Any]]:
    index_path = os.path.join(folder, "index.json")
    if not os.path.exists(index_path):
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return []
    if not isinstance(records, list):
        return []

    out: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        equation_id = _normalize_equation_catalog_id(record)
        if not equation_id:
            continue
        label = _equation_record_label(record, equation_id)
        section = str(record.get("section_title") or "").strip()
        text = str(record.get("text") or "").strip()
        latex = str(record.get("latex") or "").strip()
        role_source = " ".join(x for x in [label, section, text, latex] if x).lower()
        role_hint: List[str] = []
        if any(tok in role_source for tok in ("backstress", "\\dot{x}", "dot{x}", "x^", "x^{")):
            role_hint.append("backstress_evolution")
        if any(tok in role_source for tok in ("tau_c", "crss", "\\tau_{c}", "resistance")):
            role_hint.append("crss_evolution")
        if any(tok in role_source for tok in ("gamma_0,2", "\\dot{\\gamma}_{0,2}", "m_2", "creep", "relaxation")):
            role_hint.append("creep_flow")
        if any(tok in role_source for tok in ("gamma_0,1", "\\dot{\\gamma}_{0,1}", "m_1", "sgn", "\\dot{\\gamma}")):
            role_hint.append("plastic_flow")
        if any(tok in role_source for tok in ("0,1", "0,2", "superposition")):
            role_hint.append("combined_flow")
        if any(tok in role_source for tok in ("\\Gamma", "gamma", "cumulative")):
            role_hint.append("cumulative_slip")
        preview_parts = [f"label={label}"]
        if section:
            preview_parts.append(f"section={section}")
        if text:
            preview_parts.append(f"text={trim_text(text, 240)}")
        if latex:
            preview_parts.append(f"latex={trim_text(latex, 240)}")
        if role_hint:
            preview_parts.append(f"role_hint={','.join(dict.fromkeys(role_hint))}")
        out.append({
            "name": equation_id,
            "selection_id": equation_id,
            "path": index_path,
            "text": " | ".join(preview_parts),
            "length": len(text) + len(latex),
            "equation_record": record,
            "display_label": label,
            "section_title": section,
            "role_hint": ",".join(dict.fromkeys(role_hint)) if role_hint else "",
            "extract_text": "\n".join(
                part for part in [
                    f"Equation ID: {equation_id}",
                    f"Label: {label}" if label else "",
                    f"Section: {section}" if section else "",
                    f"Role hint: {','.join(dict.fromkeys(role_hint))}" if role_hint else "",
                    f"LaTeX: {latex}" if latex else "",
                    f"Plain text: {text}" if text else "",
                ] if part
            ),
        })
    return out


def _normalize_equation_catalog_id(record: Dict[str, Any]) -> str:
    equation_id = str(record.get("equation_id") or "").strip()
    if equation_id:
        return equation_id

    eq_index = record.get("equation_index")
    try:
        if eq_index not in (None, ""):
            return f"eq_{int(eq_index):04d}"
    except Exception:
        pass

    label = str(record.get("label") or "").strip()
    if label:
        return label

    return ""


def _equation_record_label(record: Dict[str, Any], equation_id: str) -> str:
    label = str(record.get("label") or "").strip()
    if label:
        return label
    eq_index = record.get("equation_index")
    try:
        if eq_index not in (None, ""):
            return f"({int(eq_index)})"
    except Exception:
        pass
    return equation_id


def ensure_image_backed_table_images(
    paper_dir: str,
    tables: List[Dict[str, Any]],
    *,
    api_key: str | None = None,
    inst_token: str | None = None,
) -> List[Dict[str, Any]]:
    tables_dir = os.path.join(paper_dir, "tables")
    image_dir = os.path.join(tables_dir, "images")

    for table in tables:
        if str(table.get("table_kind") or "") != "image_backed":
            continue

        image_path = str(table.get("image_local_path") or "").strip()
        if image_path and os.path.exists(image_path):
            continue

        table_json = table.get("table_json") or {}
        image_info = table_json.get("image") or {}
        if not isinstance(image_info, dict):
            continue

        downloaded = download_table_image(
            image_info,
            image_dir,
            Path(table["name"]).stem,
            api_key=api_key,
            inst_token=inst_token,
        )
        if not downloaded:
            continue

        rel = os.path.relpath(downloaded, paper_dir)
        table_json.setdefault("image", {})["local_path"] = rel
        table["image_local_path"] = downloaded
        table["table_json"] = table_json
        try:
            Path(table["path"]).write_text(
                json.dumps(table_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    return tables

# ----------------------------
# Stage 1: LLM selection
# ----------------------------
SELECTION_SYSTEM_PROMPT = """
You are a crystal-plasticity literature triage assistant.
Follow the 4-section protocol in the user prompt.
Return JSON only.
"""


SELECTION_USER_PROMPT_TEMPLATE = """
1 Task description
Select the MINIMUM section and table files needed to reliably extract crystal-plasticity parameters and the core material-profile context needed to interpret them.

2 Task requirements
- Prioritize files containing: constitutive equations, parameter tables, calibration/validation details, slip/twin systems.
- If standalone equations are available in the catalog, select the governing constitutive equations explicitly instead of assuming the matching section alone is enough.
- Also include files needed to recover essential material-profile context when present:
  - chemical composition
  - phase fractions / phase constitution
  - grain size / texture / microstructure descriptors
  - deformation-condition sections and tables (fatigue, tension/compression, strain rate, temperature, sample geometry) when they define how parameters are used or calibrated
  - sample or material-input tables for multi-material or multi-sample studies
- Avoid over-selection; include only files needed to extract:
  material identity, composition or microstructure context, model type/framework, elastic constants, plastic parameters, and parameter provenance.
- Output JSON schema exactly:
{{
  "selected_sections": ["filename.md"],
  "selected_tables": ["table_001"],
  "selected_equations": ["eq_0001"],
  "why_selected": "short reason"
}}

3 Processing suggestions
- Tables with parameters are highest priority.
- Governing equations are also first-class extraction inputs when available, especially for attaching parameters to explicit equation IDs.
- Composition, phase-fraction, grain-size, texture, and material-input tables are second priority and should be included when they define the studied material system.
- If a paper contains dedicated results sections for microstructure, texture evolution, EBSD/TKD, KAM, dislocation structure, twinning, or grain-boundary-mediated deformation, include at least one of those sections when they provide explicit descriptors used to interpret the parameterization.
- Fatigue-test, loading-condition, temperature, strain-rate, and calibration-target sections are also high priority when they define deformation conditions.
- Methods/simulation sections are next priority.
- Results/discussion sections are included when they contain either calibration/validation targets or explicit microstructure/texture evidence needed to bind parameters to grain size, deformation mode, or representation assumptions.
- Abstract alone is never sufficient.
- For multi-material or comparative papers, do not select only the parameter table if a separate composition or material table is needed to identify which material each parameter bundle belongs to.
- Prefer specific subsections such as `Fatigue test`, `Material`, `Microstructural characterization`, or `Loading conditions` over relying only on a broad parent methods section.
- If a constitutive-law or calibration section is selected and relevant equations exist, usually select those equations too.

4 Few-shot examples
Example A:
- Input cues: method section + parameter table exist.
- Output behavior: select those two first, add one results section only if calibration details appear there.

Example B:
- Input cues: no explicit table, parameters embedded in text.
- Output behavior: select relevant method/result sections, leave selected_tables empty.

Example C:
- Input cues: one table has CP parameters, another table has chemical composition or phase fractions.
- Output behavior: select both, because the second table is needed to recover material identity or sample context.

Example D:
- Input cues: one parameter table exists, but grain-size dependence, texture evolution, EBSD/KAM observations, or dislocation structures are described only in results subsections.
- Output behavior: include the parameter table and at least one microstructure-rich results subsection, because those descriptors belong in `microstructure_features[]` and may define process-state or condition bindings.

Current paper file catalog:
Sections:
{sections_catalog}

Tables:
{tables_catalog}

Equations:
{equations_catalog}
"""

def build_catalog(files: List[Dict[str, Any]], max_snippet_chars: int) -> str:
    parts = []
    for f in files:
        if "selection_id" in f:
            label = str(f.get("display_label") or f.get("selection_id") or "").strip()
            caption = str((f.get("table_json") or {}).get("caption") or "").strip()
            caption = trim_text(caption, max_snippet_chars).replace("\n", " ") if caption else ""
            kind = str(f.get("table_kind") or "text")
            preview = str(f.get("json_summary") or "").strip()
            preview_lines = [line.strip() for line in preview.splitlines() if line.strip()]
            preview_lines = [line for line in preview_lines if not line.lower().startswith("caption:")]
            row_preview = ""
            if preview_lines:
                row_preview = " || ".join(preview_lines[:3])
                row_preview = trim_text(row_preview, max_snippet_chars).replace("\n", " ")
            item = f"- {f['selection_id']} | label=\"{label}\" | kind={kind} | caption=\"{caption}\""
            if row_preview:
                item += f" | preview=\"{row_preview}\""
            parts.append(item)
        else:
            title = str(f.get("title") or "").strip()
            title = trim_text(title, max_snippet_chars).replace("\n", " ") if title else ""
            preview = str(f.get("selection_preview") or "").strip()
            preview = trim_text(preview, max_snippet_chars).replace("\n", " ") if preview else ""
            item = f"- {f['name']} | title=\"{title}\""
            if preview:
                item += f" | preview=\"{preview}\""
            parts.append(item)
    return "\n".join(parts)

def llm_select_files(sections, tables, equations, model: str, max_snippet_chars: int) -> Dict[str, Any]:
    prompt = SELECTION_USER_PROMPT_TEMPLATE.format(
        sections_catalog=build_catalog(sections, max_snippet_chars),
        tables_catalog=build_catalog(tables, max_snippet_chars),
        equations_catalog=build_catalog(equations, max_snippet_chars),
    )
    start = time.perf_counter()
    resp = _chat_completion_with_retry(
        model=model,
        messages=[
            {"role": "system", "content": SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )
    elapsed = time.perf_counter() - start
    usage = resp.usage

    return json.loads(resp.choices[0].message.content), usage, elapsed


# ----------------------------
# Stage 2: LLM Extraction
# ----------------------------
EXTRACT_SYSTEM_PROMPT = """
You are a crystal-plasticity parameter extraction engine.
Follow the 4-section protocol in the user prompt.
Extract only stated facts; do not guess.
Do not emit placeholder parameter records that lack an explicit value.
For selected parameter tables, optimize for completeness: extract every explicit parameter value that can be bound to a symbol/name/scope, including zero values and auxiliary constants.
Separate paper-level material/sample profile facts from parameter-level facts whenever possible.
Return JSON only.
"""

EXTRACT_SCHEMA_JSON_TEMPLATE = r"""
{
  "schema_version": "5.1.0",
  "document": {
    "doi": "string or null",
    "title": "string or null",
    "authors": ["string"],
    "year": "number or null",
    "journal": "string or null",
    "notes": "string or null"
  },
  "materials": [
    {
      "material_id": "string or null",
      "name": "string or null",
      "chemical_formula": "string or null",
      "material_class": "steel / titanium_alloy / nickel_superalloy / magnesium_alloy / zirconium_alloy / aluminum_alloy / copper_alloy / ceramic / intermetallic / polymer / composite / other / null",
      "phase_mode": "single_phase / multi_phase / null",
      "crystal_aggregate": "single_crystal / polycrystal / bicrystal / oligocrystal / null",
      "composition": {
        "basis": "wt_percent / at_percent / mol_percent / fraction / null",
        "components": [
          {
            "component": "string or null",
            "value": "number or string or null",
            "notes": "string or null"
          }
        ],
        "notes": "string or null"
      },
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "process_states": [
    {
      "process_state_id": "string or null",
      "material_id": "string or null",
      "label": "string or null",
      "state_type": [
        "as_received / as_cast / as_built / annealed / solution_treated / aged / quenched / cold_worked / hot_worked / rolled / forged / extruded / irradiated / hydrogen_charged / fatigue_damaged / other / string or null"
      ],
      "processing_route": "string or null",
      "processing_steps": [
        {
          "step_type": "casting / additive_manufacturing / rolling / forging / extrusion / annealing / solution_treatment / aging / quenching / machining / polishing / coating / charging / irradiation / other / string or null",
          "description": "string or null",
          "temperature": {
            "value": "number or null",
            "unit": "K / C / null",
            "reported_value": "number or string or null",
            "reported_unit": "string or null"
          },
          "time": {
            "value": "number or null",
            "unit": "s / min / h / null",
            "reported_value": "number or string or null",
            "reported_unit": "string or null"
          },
          "strain": {
            "value": "number or null",
            "unit": "fraction / % / null"
          },
          "deformation_amount": {
            "value": "number or null",
            "unit": "fraction / % / true_strain / engineering_strain / null",
            "description": "string or null"
          },
          "notes": "string or null"
        }
      ],
      "state_descriptors": [
        {
          "name": "string or null",
          "value": "number or string or null",
          "unit": "string or null",
          "notes": "string or null"
        }
      ],
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "constituents": [
    {
      "constituent_id": "string or null",
      "material_id": "string or null",
      "process_state_id": "string or null",
      "constituent_type": "phase / precipitate / inclusion / pore / grain_boundary_region / matrix_region / other / string or null",
      "name": "string or null",
      "aliases": ["string"],
      "role": "matrix / precipitate / inclusion / transformed_product / pore / parent / product / interface_region / other / string or null",
      "fraction": {
        "value": "number or null",
        "unit": "fraction / % / null",
        "reported_value": "number or string or null",
        "reported_unit": "string or null",
        "basis": "volume / area / weight / unknown / null",
        "notes": "string or null"
      },
      "crystal_structure": {
        "crystal_system": "string or null",
        "bravais_lattice": "string or null",
        "lattice_type": "string or null",
        "space_group": "string or null",
        "notes": "string or null"
      },
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "deformation_systems": [
    {
      "system_id": "string or null",
      "model_id": "string or null",
      "constituent_id": "string or null",
      "system_type": "slip / twin / transformation / other / null",
      "family_name": "basal / prismatic / pyramidal_a / pyramidal_ca / octahedral / cube / other / string or null",
      "plane": "string or null",
      "direction": "string or null",
      "number_of_systems": "number or null",
      "schmid_tensor_defined": "yes / no / null",
      "non_schmid_effects": "yes / no / null",
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "models": [
    {
      "model_id": "string or null",
      "name": "string or null",
      "model_type": "crystal_plasticity / crystal_plasticity_damage / phase_field / continuum_damage / cohesive_zone / thermal / coupled_multiphysics / other / string or null",
      "model_role": "primary_simulation / calibration / validation / comparison / auxiliary / other / string or null",
      "constituent_scope": ["string"],
      "material_scope": ["string"],
      "mechanism_scope": {
        "includes_slip": "yes / no / null",
        "includes_twinning": "yes / no / null",
        "includes_phase_transformation": "yes / no / null",
        "includes_damage": "yes / no / null"
      },
      "solver_framework": {
        "representation_mode": "homogeneous / heterogeneous",
        "parameter_assignment_mode": "global_shared / phase_specific / family_specific / region_specific / grain_specific / stochastic",
        "scale": "single_crystal / bicrystal / oligocrystal / polycrystal / aggregate / other / string or null",
        "discretization": "fem / fft / mean_field / ode_based / analytical / other / string or null",
        "homogenization": "taylor / self_consistent / full_field / mean_field / none / other / string or null",
        "grain_resolution": "homogeneous / grain_resolved / mean_field / mixed / null",
        "interface_treatment": "none / implicit / explicit_interface / grain_boundary_affected / cohesive_interface / diffuse_interface / other / null",
        "boundary_condition_style": "periodic / displacement_controlled / traction_controlled / mixed / other / string or null",
        "notes": "string or null"
      },
      "implementation": {
        "software": "string or null",
        "subroutine": "umat / vumat / user_element / spectral_solver / built_in / other / string or null",
        "solver_name": "string or null",
        "code_name": "string or null",
        "version": "string or null",
        "repository_or_link": "string or null",
        "notes": "string or null"
      },
      "constitutive_description": {
        "kinematics": "small_strain / finite_strain / other / string or null",
        "elasticity": {
          "symmetry": "anisotropic / isotropic / cubic / transversely_isotropic / orthotropic / other / string or null",
          "compressibility": "compressible / incompressible / nearly_incompressible / other / string or null",
          "notes": "string or null"
        },
        "flow_kinetics": {
          "rate_dependence": "rate_dependent / rate_independent / mixed / other / string or null",
          "flow_rule_form": "power_law / overstress / thermal_activation / arrhenius / sinh / tabulated / user_defined / other / string or null",
          "reference_shear_rate_used": "yes / no / null",
          "activation_energy_used": "yes / no / null",
          "notes": "string or null"
        },
        "hardening": {
          "slip_hardening_law": "string or null",
          "latent_hardening_form": "none / identity / interaction_matrix / q_matrix / user_defined / unclear / other / string or null",
          "kinematic_hardening": "none / prager / armstrong_frederick / chaboche / ohno_wang / backstress_based / user_defined / unclear / other / null",
          "hardening_state_basis": "crss_based / dislocation_density_based / backstress_based / slip_resistance_based / user_defined / other / string or null",
          "notes": "string or null"
        },
        "slip_description": {
          "slip_families_defined": "yes / no / null",
          "deformation_system_ids": ["string"],
          "notes": "string or null"
        },
        "twinning": {
          "enabled": "yes / no / null",
          "form": "ptr / twinning_detwinning / reorientation / volume_fraction_based / user_defined / other / string or null",
          "reorientation_treated": "yes / no / null",
          "detwinning_treated": "yes / no / null",
          "deformation_system_ids": ["string"],
          "notes": "string or null"
        },
        "damage": {
          "enabled": "yes / no / null",
          "form": "phenomenological / continuum_damage / cohesive / phase_field / user_defined / other / string or null",
          "coupling_style": "uncoupled / weakly_coupled / fully_coupled / other / string or null",
          "notes": "string or null"
        },
        "thermal_coupling": {
          "enabled": "yes / no / null",
          "temperature_dependent_parameters": "yes / no / null",
          "self_heating_considered": "yes / no / null",
          "notes": "string or null"
        },
        "internal_variable_summary": {
          "includes_crss_or_slip_resistance": "yes / no / null",
          "includes_dislocation_density": "yes / no / null",
          "includes_backstress": "yes / no / null",
          "includes_twin_volume_fraction": "yes / no / null",
          "includes_phase_fraction": "yes / no / null",
          "includes_damage": "yes / no / null",
          "other_internal_variables": ["string"],
          "notes": "string or null"
        }
      },
      "constitutive_branches": [
        {
          "branch_id": "string or null",
          "branch_type": "plastic_flow / creep_flow / combined_flow / crss_evolution / hardening / latent_hardening / backstress_evolution / twinning_evolution / damage_evolution / thermal_activation / other / string or null",
          "name": "string or null",
          "description": "string or null",
          "governing_equation_ids": ["string; required array of all explicit equation labels that govern this branch, not just one primary equation; if the branch uses multiple equations keep every explicit relevant label such as '(4)', '(5)', '(6)', '(7)'"],
          "parameter_families": ["string"],
          "evidence_ids": ["string"],
          "notes": "string or null"
        }
      ],
      "equation_ids": ["string; required model-level union of all explicit governing equation labels used by this model; preserve every explicit relevant label instead of truncating to the first visible equation"],
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "simulation_geometries": [
    {
      "geometry_id": "string or null",
      "model_id": "string or null",
      "geometry_type": "rve / unit_cell / grain_aggregate / single_element / specimen_mesh / other / null",
      "dimensions": "string or null",
      "number_of_grains": "number or null",
      "number_of_elements": "number or null",
      "mesh_type": "tetrahedral / hexahedral / voxel / spectral_grid / other / null",
      "element_type": "string or null",
      "grid_size": "string or null",
      "periodic_geometry": "yes / no / null",
      "grain_shape_assumption": "equiaxed / columnar / elongated / measured / voronoi / other / null",
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "orientation_inputs": [
    {
      "orientation_id": "string or null",
      "model_id": "string or null",
      "geometry_id": "string or null",
      "source": "ebsd / xrd_odf / random_texture / ideal_texture / synthetic / literature / other / null",
      "representation": "euler_angles / quaternion / orientation_matrix / pole_figure / odf / ipf_map / other / null",
      "texture_type": "random / measured / ideal / fiber / rolling / extrusion / other / null",
      "number_of_orientations": "number or null",
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "numerical_methods": [
    {
      "numerical_method_id": "string or null",
      "model_id": "string or null",
      "time_integration": "explicit / implicit / semi_implicit / return_mapping / forward_euler / backward_euler / other / null",
      "nonlinear_solver": "newton_raphson / fixed_point / explicit_update / other / null",
      "tolerance": "string or null",
      "time_step": "string or null",
      "increment_control": "fixed / adaptive / load_increment / strain_increment / other / null",
      "regularization": "viscoplastic / gradient / nonlocal / length_scale / none / other / null",
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "conditions": [
    {
      "condition_id": "string or null",
      "label": "string or null",
      "condition_role": "calibration / validation / characterization / simulation_case / experiment / service_like / unspecified / other / string or null",
      "loading_mode": "uniaxial_tension / compression / shear / cyclic / fatigue / creep / torsion / bending / indentation / relaxation / hold / other / string or null",
      "control_mode": "strain_controlled / stress_controlled / displacement_controlled / mixed / other / string or null",
      "stress_state": "uniaxial / biaxial / triaxial / plane_strain / multiaxial / other / string or null",
      "temperature": {
        "value": "number or null",
        "unit": "K / C / null",
        "reported_value": "number or string or null",
        "reported_unit": "string or null",
        "history": "isothermal / non_isothermal / null"
      },
      "strain_rate": {
        "value": "number or null",
        "unit": "string or null",
        "reported_value": "number or string or null",
        "reported_unit": "string or null",
        "range": "string or null"
      },
      "fatigue": {
        "mode": "lcf / hcf / vhcf / strain_controlled / stress_controlled / other / string or null",
        "load_ratio": "string or null",
        "frequency": {
          "value": "number or null",
          "unit": "Hz / null"
        },
        "notes": "string or null"
      },
      "indentation": {
        "indenter_type": "berkovich / spherical / cono_spherical / vickers / knoop / custom / other / string or null",
        "tip_radius": {
          "value": "number or null",
          "unit": "string or null"
        },
        "max_load": {
          "value": "number or null",
          "unit": "string or null"
        },
        "depth": {
          "value": "number or null",
          "unit": "string or null"
        },
        "notes": "string or null"
      },
      "environment": {
        "medium": "vacuum / air / inert_gas / hydrogen / liquid / other / string or null",
        "pressure": {
          "value": "number or null",
          "unit": "string or null"
        },
        "notes": "string or null"
      },
      "duration": {
        "value": "number or null",
        "unit": "s / min / h / null",
        "reported_value": "number or string or null",
        "reported_unit": "string or null",
        "notes": "string or null"
      },
      "linked_process_state_ids": ["string"],
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "simulation_outputs": [
    {
      "output_id": "string or null",
      "model_id": "string or null",
      "condition_id": "string or null",
      "output_quantity": "stress_strain_curve / slip_activity / crss_evolution / texture_evolution / lattice_strain / strain_localization / damage_field / twin_fraction / phase_fraction / other / null",
      "scale": "macroscopic / grain / element / slip_system / phase / local_region / other / null",
      "reported_as": "curve / field / map / table / figure / scalar / other / null",
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "model_evaluations": [
    {
      "evaluation_id": "string or null",
      "model_id": "string or null",
      "condition_id": "string or null",
      "output_id": "string or null",
      "evaluation_role": "calibration_fit / validation / prediction / sensitivity / comparison / null",
      "target_observable": "stress_strain / texture / lattice_strain / strain_map / slip_activity / fatigue_life / crack_growth / other / null",
      "metric_name": "rmse / r2 / error_percent / qualitative / other / null",
      "metric_value": "number or string or null",
      "compared_against": "experiment / another_model / analytical_solution / literature / null",
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "microstructure_features": [
    {
      "feature_id": "string or null",
      "feature_family": "grain_structure / texture / precipitates / defects / porosity / interfaces / morphology / local_region / other / string or null",
      "feature_name": "string or null",
      "parameterization_scope": "shared / constituent_specific / region_specific / interface_specific / other / string or null",
      "value_type": "scalar / vector / range / categorical / text / other / string or null",
      "value": "number or string or null",
      "unit": "string or null",
      "description": "string or null",
      "method": "ebsd / xrd / sem / tem / om / narrative / table / figure / other / string or null",
      "constituent_id": "string or null",
      "applies_to": {
        "material_id": "string or null",
        "process_state_id": "string or null",
        "condition_id": "string or null"
      },
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "parameter_claims": [
    {
      "claim_id": "string or null",
      "claim_class": "material_constitutive_parameter / experimental_condition_parameter / numerical_model_parameter / null",
      "parameter": {
        "canonical_name": "string or null",
        "parameter_family": "elastic_constants / slip_kinetics / hardening / backstress / latent_hardening / twinning / damage / thermal / numerical / geometry / other / string or null",
        "raw_name": "string or null",
        "symbol_reported": "string or null",
        "domain": "elastic / plastic / creep / hardening / twinning / damage / thermal / numerical / other / string or null",
        "description": "string or null"
      },
      "assertion": {
        "value_type": "scalar / range / categorical / text / expression / other / null",
        "reported_value": "number or string or null",
        "reported_unit": "string or null",
        "valid_range": "string or null"
      },
      "applies_to": {
        "material_id": "string or null",
        "constituent_id": "string or null",
        "process_state_id": "string or null",
        "model_id": "string or null",
        "condition_id": "string or null",
        "branch_ids": ["string; use this array for one or more constitutive branches; for a single explicit branch keep one item, and for shared parameters keep every relevant branch ID"],
        "scope": "global / constituent / family / system / branch / local_region / other / string or null",
        "notes": "string or null"
      },
      "provenance": {
        "origin_type": "original / adopted / calibrated / adopted_then_calibrated / null",
        "reference_ids": ["string"],
        "adopted_from_reference_ids": ["string"],
        "calibration_based_on_reference_ids": ["string"],
        "calibration": {
          "method": "manual_fitting / inverse_modeling / optimization / bayesian / machine_learning / other / string or null",
          "target_type": "stress_strain_curve / creep_curve / relaxation_curve / cyclic_hysteresis / diffraction_lattice_strain / grain_family_response / indentation_curve / texture_fit / multi_objective / other / string or null",
          "target_description": "string or null",
          "observation_scope": "macroscopic / phase / grain_family / slip_family / local_region / other / string or null",
          "notes": "string or null"
        }
      },
      "governing_equation_ids": ["string; required array of all explicit equation labels that directly define, use, or evolve this parameter claim; if multiple equations use or evolve the parameter, keep every explicit relevant label rather than choosing only one"],
      "evidence_ids": ["string"],
      "notes": "string or null"
    }
  ],
  "evidence_objects": [
    {
      "evidence_id": "string or null; for table-backed parameter claims, prefer a claim-specific evidence ID rather than one whole-table summary ID reused across many unrelated claims",
      "evidence_type": "section_span / table_cell / table_row / figure_caption / equation / mixed / other / string or null; use table_cell or a claim-specific table_row for parameter-level table grounding",
      "extraction_method": "manual / text_llm / table_text_llm / table_image_ocr_llm / figure_caption_llm / equation_parse / other / string or null",
      "source_file": "string or null",
      "source_id": "string or null",
      "section_heading": "string or null",
      "page": "number or null",
      "locator": {
        "table_id": "string or null",
        "figure_id": "string or null",
        "equation_label": "string or null",
        "row_name": "string or null; for table-backed parameter claims this should identify the active row or grouped-row segment supporting that specific claim",
        "column_name": "string or null; for table-backed parameter claims this should identify the active value column/header for that specific claim",
        "cell_ref": "string or null; best-effort spreadsheet-style or table-local cell reference when recoverable, especially for parameter tables",
        "value": "string or null; for table-backed parameter claims do not leave null when the explicit value is visible in the table",
        "excerpt": "string or null; smallest self-contained snippet that directly shows the claim-level table grounding, not just the whole table caption"
      },
      "snippet": "string or null",
      "confidence": "number or null",
      "notes": "string or null"
    }
  ],
  "global_notes": "string or null"
}
"""

EXTRACT_USER_PROMPT_TEMPLATE = """
1 Task description
Extract crystal-plasticity information from the provided paper excerpt into the v5.1 hierarchical CP schema.

2 Task requirements
- Use only explicit evidence in the excerpt.
- If unknown, return null or empty list.
- Keep parameter provenance carefully (adopted references vs calibration references).
- For enum-like fields, prefer the listed common options when they fit.
  If the paper uses a more specific label that is not listed, keep a short schema-compatible freeform string instead of forcing a bad enum choice.
- Do not use `samples[]`.
- Use `process_states[]` for material processing-state variants and `conditions[]` for loading/testing context.
- Use `process_states[].state_type` as an ordered list when multiple state descriptors are explicitly true at the same time.
  Example: `["as_received", "forged", "solution_treated"]`.
- Do not create a top-level `phases[]` block. Use `constituents[]` for phases, precipitates, pores, and other constituent-level entities.
- Use `deformation_systems[]` for explicit slip, twin, and transformation systems or families used by the model.
- Use `simulation_geometries[]` for explicit RVE, unit-cell, grain-aggregate, voxel, spectral-grid, mesh, or specimen geometry setup.
- Use `orientation_inputs[]` for explicit simulation-input orientation or texture data such as EBSD-derived orientations, Euler angles, pole figures, ODFs, random textures, or ideal textures.
- Use `numerical_methods[]` for explicit integration, nonlinear solver, increment-control, or regularization choices.
- Use `simulation_outputs[]` only for explicit modeled observables linked to a model and condition.
- Use `model_evaluations[]` only for explicit calibration, validation, prediction, sensitivity, or comparison roles linked to those outputs.
  If the excerpt does not support them directly, leave these arrays empty rather than inferring them from provenance.
- Keep `parameter_claims[]` claim-centric: separate `parameter`, `assertion`, `applies_to`, `provenance`, and `evidence_ids`.
- Classify every `parameter_claims[]` item into one of three roles when explicit or strongly implied:
  `material_constitutive_parameter`, `experimental_condition_parameter`, or `numerical_model_parameter`.
- Use `material_constitutive_parameter` for elastic constants, slip/creep/hardening/backstress/twinning/damage/thermal constitutive quantities.
- Use `experimental_condition_parameter` for setup quantities that are really test or loading conditions, such as imposed temperature, strain rate, hold time, load ratio, frequency, environment pressure, or similar condition-setting values when they are still represented as parameter claims.
- Use `numerical_model_parameter` for solver or discretization settings such as tolerances, time step controls, iteration limits, regularization lengths, mesh-related numerical settings, or other explicit model-solution parameters.
- Do not rely on postprocessing to split, remap, or sharpen evidence. The extractor output itself must already contain final claim-level evidence bindings.
- Fill `document` directly when the excerpt explicitly contains that information; otherwise leave fields null or empty.
- Prefer final-ready bindings now rather than leaving them for later normalization or postprocessing.
- Assign stable, reusable IDs whenever the excerpt supports them: `material_id`, `process_state_id`, `constituent_id`, `condition_id`, `model_id`, `branch_id`, `feature_id`, `claim_id`, `evidence_id`.
- Use evidence links beyond parameters as well: populate `evidence_ids` for materials, process states, constituents, conditions, models, constitutive branches, and microstructure features when direct support is available.
- Prioritize five linked questions for every extraction when possible:
  1. what material or material state is being discussed
  2. under which deformation, test, or calibration condition it applies
  3. which model/framework the parameter belongs to
  4. which phase / mechanism / family / system the parameter acts on
  5. where the direct evidence is located
- Prioritize three additional linkage questions whenever possible:
  6. which process states are linked to which loading/testing conditions
  7. which calibration targets are distinct even under the same physical temperature/strain-rate condition
  8. which equations govern the model versus specific parameter claims
- Distinguish three different scopes and do not collapse them:
  physical condition in `conditions[]`,
  fitting target inside `parameter_claims[].provenance.calibration`,
  material constituent in `models[].constituent_scope` / `constituents[]`.
- Distinguish material constituents, constitutive branches, and observation targets.
  Use `constituents[]` and `models[].constituent_scope` for named phases, matrix regions, precipitate regions, pore regions, and other constituent-level targets.
  Use `models[].constitutive_branches[]` for equation-level components such as flow, hardening, damage, twinning, thermal activation, or other explicit sub-laws.
  Use `parameter_claims[].provenance.calibration` and `microstructure_features[]` for observation targets such as grain families, diffraction families, selected grains, local regions, or imaging-derived subsets.
- `models[].constituent_scope` is only for named phases or constituent regions.
  Do not put measurement labels, observation subsets, Miller-index families, or slip-family labels there unless the paper explicitly treats them as constituents.
- Grain families, diffraction families, selected grains, or local measurement regions are not themselves constituents or constitutive branches unless the paper explicitly says they are.
  Treat them as calibration or observation targets instead.
- Distinguish physical microstructure descriptors from model inputs.
  Grain size, morphology, defects, and measured texture descriptors belong in `microstructure_features[]`.
  Orientation or texture data explicitly supplied to the simulation belongs in `orientation_inputs[]`, even if related descriptive texture facts also appear in `microstructure_features[]`.
- Distinguish constitutive behavior from numerical solution strategy.
  Flow rules, hardening, twinning, damage, and thermal coupling belong in `models[].constitutive_description` and `constitutive_branches[]`.
  Explicit integration, nonlinear solver, increment control, and regularization choices belong in `numerical_methods[]`.
- Distinguish simulation setup from simulation result.
  Mesh, grid, RVE, grain count, and periodic geometry belong in `simulation_geometries[]`.
  Predicted stress-strain curves, lattice strain, texture evolution, slip activity, twin fraction, and related observables belong in `simulation_outputs[]`.
  Calibration, validation, prediction, sensitivity, and comparison roles belong in `model_evaluations[]`.
- Split calibration descriptions claim-by-claim whenever the paper calibrates different parameter subsets against different observables, even under the same temperature and strain-rate.
  One physical condition can legitimately support multiple distinct calibration targets across parameter claims.
- Use `parameter_claims[].provenance.calibration.target_type`, `target_description`, and `observation_scope` to preserve what data stream was used for fitting.
  Example: a macroscopic stress-strain dataset and a subset-specific relaxation or diffraction dataset should remain structurally distinct rather than being merged into one generic note.
- Distinguish calibrated material parameters from condition-setting quantities and physical constants.
  Temperatures, gas constants, imposed total strain, dwell stress, test duration, and similar setup values may appear inside parameter tables because they are used by the equations, but they are not automatically calibrated material parameters.
  If the excerpt treats such a quantity as a loading/testing/model-setting value rather than a fitted parameter, place it in `conditions[]` or keep it as supporting context in `notes`; do not mark its claim provenance as `calibrated` unless the paper explicitly says that value itself was fitted or optimized.
- If such a setup quantity is nevertheless extracted as a parameter claim because the paper treats it as part of a reported parameter table, classify it as `experimental_condition_parameter` or `numerical_model_parameter` rather than `material_constitutive_parameter`.
- Use `parameter_claims[].provenance.origin_type=calibrated` only when the paper explicitly indicates that the specific quantity was fitted, optimized, identified, or recalibrated.
  Do not inherit `calibrated` just because the quantity appears in the same table or sentence as genuinely calibrated parameters.
- Use `models[].equation_ids` only as the model-level summary list of all equations used by the model.
  In practice this should be the union of all `models[].constitutive_branches[].governing_equation_ids`, plus any additional model-wide equations that are explicit but not branch-specific.
- Do not embed full equation objects or equation text inside `models[]`, `constitutive_branches[]`, or `parameter_claims[]`.
  Keep only `models[].equation_ids`, `models[].constitutive_branches[].governing_equation_ids`, and `parameter_claims[].governing_equation_ids` for structured equation linkage.
- Populate `models[].constitutive_branches[]` whenever the excerpt clearly separates multiple equation branches or evolution laws.
  Typical examples are plastic branch, creep branch, combined plasticity-plus-creep branch, CRSS evolution, and backstress evolution.
- Use `parameter_claims[].applies_to.branch_ids` for branch linkage in every case.
  If the claim belongs to one explicit constitutive branch, keep one branch ID in the array.
  If the same parameter claim is explicitly shared across multiple constitutive branches, keep every relevant branch ID in the same array instead of choosing only one.
- Treat `models[].constitutive_branches[].governing_equation_ids` as a full multi-equation array, not a single best-match field.
  If one branch is defined by a flow equation plus one or more evolution or auxiliary equations, keep all explicit governing labels for that branch.
- Use `parameter_claims[].governing_equation_ids` for the equations that directly govern, use, or evolve a specific parameter claim.
  A parameter may legitimately reference multiple equations, so keep all explicit relevant equation numbers rather than forcing a single primary equation.
  The same parameter can appear in one flow law and one evolution law; preserve both labels when the text makes both roles explicit.
- When referring to equations in extracted JSON, use only the explicit equation number/label from the paper text, such as `(3)` or `(7)`.
  Postprocessing will resolve numbered labels to canonical equation IDs later.
- Bind equations by reading the constitutive text semantically, not by relying on parameter names alone.
  If the paper explains that one equation is the flow rule, another is the creep branch, another is the hardening or CRSS evolution law, and another is the backstress law, reflect those roles directly in `constitutive_branches[]`, `models[].equation_ids`, and `parameter_claims[].governing_equation_ids`.
  Do not stop after attaching the first visible equation if later numbered equations in the same constitutive block are also explicitly part of the same formulation.
  Multiple branches may share the same equation, and one branch or one parameter may also bind to multiple equations; preserve those many-to-many relationships directly in the output arrays.
  When a single parameter claim is shared across more than one branch, express that explicitly through `parameter_claims[].applies_to.branch_ids`.
- Treat phase constitution and crystal aggregation as separate dimensions.
  Use `materials[].phase_mode` for `single_phase` versus `multi_phase`.
  Use `materials[].crystal_aggregate` for `single_crystal`, `polycrystal`, `bicrystal`, or `oligocrystal`.
  Do not infer one from the other.
- Treat physical microstructure facts and modeling choices as separate dimensions.
  A paper can be physically polycrystalline yet still use one shared parameter set or a homogenized constitutive description.
  Record the physical microstructure in `materials[]`, `constituents[]`, `process_states[]`, and `microstructure_features[]`.
  Record modeling choices such as shared versus constituent-specific parameterization through `models[]`, `constitutive_branches[]`, `parameter_claims[].applies_to`, and `microstructure_features[].parameterization_scope` when explicit.
- If the paper explicitly names basal, prismatic, pyramidal, octahedral, cube, twinning, or transformation systems or families used by the simulation, emit `deformation_systems[]` entries instead of keeping those details only in notes.
- If the paper gives explicit system-level non-Schmid behavior, Schmid-tensor statements, plane/direction notation, or number of systems, store them in `deformation_systems[]`.
- If the paper gives explicit geometry or mesh setup, emit one `simulation_geometries[]` entry per explicit setup tied to the relevant `model_id`.
- If the paper states that the simulation used measured EBSD orientations, random orientations, ideal texture, Euler angles, ODFs, or pole figures as model input, emit `orientation_inputs[]`.
- If the paper explicitly states validation or calibration against stress-strain, diffraction lattice strain, texture evolution, strain maps, slip activity, or similar observables, connect those through `simulation_outputs[]` and `model_evaluations[]`.
  If the text only supports a claim-level calibration note and does not explicitly define a modeled output or evaluation object, keep the information only inside `parameter_claims[].provenance.calibration`.
- Extract the following schema from the paper excerpt:
__SCHEMA_JSON__

3 Processing suggestions
- Prefer table values over narrative values when both are present.
- Use `provenance` only for provenance and origin tracing.
- Use `evidence_objects[]` plus `evidence_ids` for evidence storage.
- For table-backed parameter claims, prefer claim-specific evidence packaging over whole-table summaries.
- Do not rely on later cleanup to infer obvious scope. If a table row or sentence makes the binding explicit, encode it directly now.
- When a selected equation directly defines the model, flow rule, hardening law, yield function, or evolution law, attach it to the relevant branch first; `models[].equation_ids` should then act as the union summary across branches.
- If the excerpt presents a constitutive subsection with several numbered equations belonging to the same CP formulation, attach all governing equation IDs that are explicitly part of that formulation rather than only the first equation.
- If those equations play different constitutive roles, also distribute them into `models[].constitutive_branches[]` with the right `branch_type`.
- Never collapse a multi-equation branch or parameter into a single equation label merely for simplicity.
  If the text explicitly supports multiple labels, the corresponding `governing_equation_ids` array should keep all of them.
- Do not create equation-only `evidence_objects[]` entries just because an equation is relevant.
  Store equation support through `models[].equation_ids`, `models[].constitutive_branches[].governing_equation_ids`, and `parameter_claims[].governing_equation_ids` instead of duplicating it in `evidence_ids`.
- Do not put equation evidence IDs in `materials[].evidence_ids`, `models[].evidence_ids`, `constitutive_branches[].evidence_ids`, `parameter_claims[].evidence_ids`, or other record-level evidence links unless the equation itself is being quoted as narrative evidence beyond the equation label.
- If equations appear only inline inside section text, still bind parameters and branches to the explicit numbered labels from the text rather than dropping the relationship.
- If a parameter subset is calibrated from one target and another subset from a different target, keep their `provenance.calibration` blocks distinct even if they share one `condition_id`.
- If a calibration target is tied to a named grain family, diffraction family, subset, orientation family, or local region, use the most specific compatible `observation_scope` and preserve the label in `target_description`.
- If a calibration condition is described as uniaxial stress-strain, uniaxial tensile response, uniaxial cyclic stress-strain, or displacement-controlled loading along one axis, prefer a specific compatible `loading_mode` such as `uniaxial_tension` or `cyclic` instead of falling back to `other`.
- Use `loading_mode=other` only when the physical loading path truly cannot be placed into one of the listed common modes.
- If both adopted and calibrated are stated, use origin_type as adopted_then_calibrated.
- Put bracketed citation labels like 12, 60, 61 into reference id arrays.
- For provenance, only output reference ID arrays; do not fabricate reference title/doi.
- Do not use placeholders like this_study/present_study as reference IDs; put such info in `provenance.notes`.
- For comparative or multi-material papers, populate `materials[]` instead of collapsing everything into one material name.
- For papers with multiple heat treatments, processing routes, or initial states, populate `process_states[]`.
- Use `process_states[].processing_steps[]` when the paper gives structured route information such as anneal/age/quench sequences.
- Use `processing_steps[].deformation_amount` for rolling reduction, extrusion reduction, prestrain, or other deformation magnitudes when reported more naturally than generic strain.
- When the excerpt explicitly links a process state to one or more testing conditions, fill `conditions[].linked_process_state_ids` and use the same process-state IDs consistently from the relevant claims and features.
- Use `conditions[]` for temperature, strain rate, fatigue mode, indentation settings, environment, and calibration/validation role.
- Use `conditions[]` for the physical test or loading setup; use `parameter_claims[].provenance.calibration` for which observable or dataset was used to fit the model.
- If a calibration or validation target corresponds to a concrete modeled observable, also emit a compatible `simulation_outputs[]` entry and `model_evaluations[]` entry when the excerpt explicitly supports that structure.
- Do not emit `simulation_outputs[]` or `model_evaluations[]` solely because a calibration target exists in provenance. These sections should remain empty unless the modeled output or evaluation role is explicit.
- Use `constituents[]` instead of embedding constituent details inside `microstructure_features[]`.
- Do not encode calibration target differences only in free-text notes when the `provenance.calibration` structure can represent them directly.
- Represent the common physical combinations explicitly by combining separate fields rather than creating one fused enum:
  `single_phase + single_crystal`,
  `single_phase + polycrystal`,
  `multi_phase + single_crystal`,
  `multi_phase + polycrystal`.
- Represent phase-like or region-like microstructural entities through `constituents[]` and `microstructure_features[]`, not through a separate legacy phase block.
- Use `microstructure_features[].parameterization_scope` to capture whether one parameter set is `shared` across the microstructure or is `constituent_specific`, `region_specific`, or `interface_specific`.
- If the paper uses one shared constitutive description, even for a heterogeneous microstructure, do not artificially split parameter claims by constituent unless the excerpt explicitly gives constituent-specific values.
- If grain-boundary effects, interface regions, or local zones are modeled separately, represent them as `microstructure_features[]` or `constituents[]` only when the excerpt explicitly distinguishes them.
- For equation-rich constitutive sections, extract reusable `constitutive_branches[]` so different parameter subsets can bind to the right branch or evolution law rather than all sharing one generic model-level association.
- If the paper explicitly reports geometry setup or numerical solution details, do not collapse them into one free-text `models[].notes` field. Prefer `simulation_geometries[]` and `numerical_methods[]`.
- When explicit, fill `applies_to.material_id`, `applies_to.constituent_id`, `applies_to.process_state_id`, `applies_to.model_id`, `applies_to.condition_id`, and `applies_to.branch_ids`.
- If a claim is global or shared, keep `scope` broad and leave narrower target IDs null rather than inventing unsupported constituent or branch specificity.
- If a claim is tied to a fitting target rather than only a physical condition, fill `provenance.calibration` rather than inventing a synthetic scope ID.
- Keep the parameter identity in `parameter`, the numeric statement in `assertion`, and the applicability in `applies_to`.
- Put range/bounds or qualifiers inside `assertion`, not at top level.
- If the claim or microstructure fact comes from a table or image-backed table, create an `evidence_objects[]` entry with row/column/value/excerpt and link it through `evidence_ids`.
- Always try to fill `evidence_objects[].source_file`, `source_id`, `section_heading`, and `extraction_method` when identifiable.
- Tables may be row-oriented, column-oriented, transposed, matrix-like, multi-level-header, grouped-row, or image-backed. Interpret all of these as valid parameter sources.
- Table serialization may include explicit `<EMPTY>` cell markers to preserve original column alignment.
  Treat `<EMPTY>` as a true blank cell, not as missing text to be skipped.
  Never shift neighboring values left or right to fill an `<EMPTY>` position.
  When a header row is present, bind each value to its header strictly by column position.
- Resolve units from the nearest reliable source: the value cell, row label, column header, section header, or shared table note. Do not duplicate or invent units.
- Preserve table structure instead of flattening it. Keep row/column identity, grouped headers, section headers, and shared values explicit in `evidence_objects[].locator`.
- `evidence_objects[].locator.row_name/column_name/value` should be human-readable labels. Do not emit row_index or column_index.
- `evidence_objects[].locator.excerpt` should be the smallest self-contained supporting snippet: usually one table row or one grouped-row segment with the parameter label and value together.
- For table-backed parameter claims, use claim-specific evidence packaging, not whole-table summaries.
  The extractor output must make each claim auditable without any later evidence-splitting pass.
- A parameter claim must not use a whole-table summary evidence object as its only table evidence when the table exposes a narrower row/cell grounding.
- For table-backed parameter claims, include the concrete row segment and the explicit value for that claim in `locator.value` and/or `snippet` whenever the table provides it.
- Do not leave `locator.value=null` when the same table excerpt gives an explicit parameter value that can be copied verbatim.
- Avoid evidence objects whose excerpt only says a table has certain columns or ranges of rows if a narrower row/cell snippet is available.
- If one table row supports multiple claims, reuse one evidence object only when its `locator.value` or `snippet` explicitly enumerates every mapped claim-value pair and a reviewer can verify each mapping directly from that one object.
  Otherwise create smaller evidence objects so each claim has direct visible support.
- For grouped rows such as `c11, c12, c44 -> 183.9 GPa, 123.4 GPa, 91.5 GPa`, the preferred output is one evidence object per claim with the local row/column/value context for that claim.
- Fill `locator.cell_ref` whenever a deterministic table-local reference can be recovered; if not recoverable, keep it null rather than inventing one.
- When a table has a dedicated value column such as `Value`, keep `locator.column_name` anchored to that header rather than copying a neighboring numeric cell.
- For condition-rich tables with repeated temperatures, grain sizes, phases, families, or slip modes, make the active row context explicit in `locator.row_name`, `locator.column_name`, `locator.value`, or `snippet`.
  A reviewer should be able to see the exact condition-value mapping without re-reading the whole table.
- Reuse the same `evidence_id` from `evidence_objects[]` across all records supported by the same snippet instead of inventing near-duplicate evidence objects.
- `parameter.canonical_name` should already use the project-standard canonical name when it is clear from symbol/description/context. Avoid verbose phrase-like names if a stable canonical label is available.
- Use `parameter.parameter_family` for the stable middle layer between coarse domain and specific canonical name.
- Use `parameter.symbol_reported` for the symbol as written in the paper. Do not treat it as a normalized identifier.
- Use snake_case enum values exactly.
- Only emit a parameter claim when you can bind it to an explicit numeric/string value in the excerpt or to a concrete grouped table row with one-to-one value mapping.
- If the excerpt only gives a definition, equation form, literature provenance, or says a parameter was calibrated/adopted without stating its value, omit that parameter from `parameter_claims`.
- For grouped rows such as `c11, c12, c44 -> 183.9 GPa, 123.4 GPa, 91.5 GPa`, emit separate parameter claims for each value instead of one null-valued grouped shell.
- For comparison tables spanning many materials or slip modes, only extract entries that can be bound to the focal studied material/phase/slip family from context; otherwise omit them rather than producing a generic null-valued parameter.
- In comparative parameter matrices, treat explicitly reported auxiliary elastic descriptors such as `Zener ratio`, `c/a`, or similarly named anisotropy ratios as first-class parameter claims when they are given a concrete value for a specific material or phase.
  Do not leave such quantities only inside evidence text if the table gives a one-to-one material-value mapping.
- If a grouped row gives a numeric value but omits a unit for one segment, still emit that parameter with the explicit value and set unit to null; do not fabricate a unit.
- Missing unit alone is not a reason to omit a parameter when the value itself is explicit and the parameter identity is clear.
- For rows like `h, hD -> 3555 MPa, 245`, emit both parameters. Keep `h=3555 MPa`; keep `hD=245` with unit null if no unit is explicitly shown for `hD`.
- For selected parameter tables, scan the whole table from top to bottom. Do not stop after extracting only the most salient plastic parameters.
- Treat `0` as a valid explicit parameter value. Never omit a parameter solely because its value is zero.
- Extract elastic constants from parameter tables with the same priority as plastic parameters.
- Extract auxiliary constants, fitted coefficients, and numerical/material constants from selected parameter tables when they have explicit values, but keep setup values and fixed physical constants distinct from genuinely calibrated quantities when the paper makes that distinction.
- If one table row covers multiple scopes, expand it into multiple parameter claims when the mapping is explicit.
- Example: a row that lists two named targets with one shared value should become two claims if the value is explicitly shared by both targets.
- Example: a row that lists two named targets with one shared exponent or coefficient should become two claims, not one ambiguous shared record.
- For grouped rows with multiple labels and multiple values, preserve order and map each label to the corresponding value segment.
- For comparative multi-family or multi-system tables with multi-row headers, inherit the active header semantics correctly and assign any shared citation or provenance to each corresponding claim.
- If continuation rows under the same structure block omit repeated constants but still show explicit parameter values, extract those explicit values instead of dropping the row.
- For chemical-composition matrices, each column/material should become a separate entry in `materials[]`; do not flatten the entire table into one material string.
- For grain-size / phase-fraction / texture tables, populate `microstructure_features[]` even if the table contains no CP parameters.
- For phase constitution tables, create `constituents[]` entries rather than a separate `phases[]` block. Add `microstructure_features[]` only for explicit measured descriptors tied to those constituents.
- If `materials[].phase_mode=multi_phase`, actively check whether the excerpt also gives explicit named constituents such as matrix, precipitate, parent/product phases, pores, or interface regions.
  When those named constituents are explicit, create `constituents[]` entries instead of leaving the material as multi-phase with an empty constituent list.
  When the excerpt supports only a broad `multi_phase` statement but gives no explicit constituent identities or fractions, keep `constituents[]` empty and record that limitation in notes rather than inventing unnamed phases.
- Treat the following as first-class `microstructure_features[]` when explicitly stated: grain size, grain-size distribution, bimodal versus uniform grain structure, recrystallized/equiaxed morphology, texture type or intensity, texture-component volume fraction, EBSD/TKD/KAM-derived lattice-distortion patterns, grain-boundary character, twinning fraction or twin-dominated texture change, grain-boundary sliding, shear bands, sub-grains/sub-boundaries, dislocation density regime, and named dislocation-network descriptions.
- Do not stop at one generic grain-size feature if the paper gives multiple explicit microstructure states or descriptors. Emit multiple `microstructure_features[]` items when different grain sizes, temperatures, deformation stages, or microstructure modes are explicitly distinguished.
- When the paper compares several grain sizes or microstructure states, prefer separate `process_states[]` or separately bound `microstructure_features[]` rather than collapsing them into one broad mixed state.
- When a microstructure fact has direct support, create an evidence object and link it via `microstructure_features[].evidence_ids`.
- When a microstructure fact is tied to a testing or loading scenario, fill `microstructure_features[].applies_to.condition_id`.
- When a microstructure fact is specific to a testing or loading scenario, fill `microstructure_features[].applies_to.condition_id` as well as any material/phase/process-state bindings that are explicit.
- Keep material-state hierarchy explicit: detailed per-material composition belongs in `materials[]`; constituent or phase-like organization belongs in `constituents[]`; per-process-state variation belongs in `process_states[]`; loading/test variation belongs in `conditions[]`.
- For multiple deformation conditions in one paper, populate `conditions[]` and link `parameter_claims[]` through `applies_to.condition_id` when explicit.
- For parameter tables organized by temperature, grain size, process state, or method, expand each condition row into distinct parameter claims and link them to the right `process_state_id` / `condition_id`.
- Distinguish calibration bounds from final calibrated parameters. If a table gives bounds or search ranges, do not convert the bound itself into a standalone calibrated parameter. Preserve it in `assertion.valid_range` when the mapping is explicit.
- For image-backed parameter tables, apply the same completeness rule as text tables: recover all explicit numeric entries, including zeros, shared rows, and elastic blocks.
- Do not omit a tail row of a selected parameter table merely because earlier rows already provided more prominent parameters.

4 Few-shot examples
Example A: Text says parameters adopted from [12,13] and calibrated against stress-strain curves.
Expected behavior: provenance.origin_type is adopted_then_calibrated, adopted_from_reference_ids includes 12 and 13, and the claim links to an evidence object mentioning the calibration target.

Example B: Text states a parameter value but no source citation.
Expected behavior: parameter extracted with null/empty provenance reference arrays.

Example C: A grouped table row lists multiple parameter labels and multiple corresponding values.
Expected behavior: emit one parameter claim per label-value pair with explicit values and inherited units when the mapping is clear.

Example D: Section says `Slip shear rate equation from Ref. [47]` but gives no numeric parameter value.
Expected behavior: do not emit a parameter record for slip shear rate.

Example E: A grouped row lists two parameters and two values, but only one value has an explicit unit.
Expected behavior: emit both parameter claims. Keep the explicit unit only where it is actually supported, and leave the other unit null if necessary.

Example F: A row states that one explicit value is shared across two named targets.
Expected behavior: emit one claim per target with the shared explicit value.

Example G: A row reports an explicit parameter value of `0`.
Expected behavior: emit the parameter with value `0`; do not treat zero as missing.

Example H: A selected parameter table contains elastic constants at the top and additional constants or coefficients near the bottom.
Expected behavior: extract the explicit values that are truly treated as model parameters in the paper, but do not automatically classify table-listed setup values or physical constants as calibrated parameters. If `R` or `T` is used as a constant or test setting rather than a fitted parameter, keep that distinction explicit through `conditions[]` or non-calibrated provenance.

Example I: A composition table lists materials as columns and elements as rows.
Expected behavior: populate `materials[]` with one material entry per column and keep composition rows under each material.

Example J: A parameter table is indexed by temperature and grain size for one alloy after different heat treatments.
Expected behavior: create one `process_states[]` entry per heat-treatment state when explicit, store the grain-size fact in `microstructure_features[]`, and link the corresponding parameter claims with `process_state_id`.

Example K: A table lists process states as rows and phase fractions as columns for room temperature and cryogenic temperature.
Expected behavior: populate `conditions[]` for room temperature and cryogenic temperature, store phase-fraction observations in `microstructure_features[]`, and do not convert phase fractions into CP parameter claims.

Example L: A comparative table lists many materials as rows or columns, with elastic constants and slip strengths for each material.
Expected behavior: preserve one material entry per material and keep parameter claims tied to the correct material_id or microstructure_id instead of mixing all values into one generic target.

Example L2: A comparative table lists one material per row and includes Zener ratio, elastic constants, and slip strengths in the same row.
Expected behavior: emit a dedicated `zener_ratio` claim for each material whenever the table gives an explicit one-to-one value, instead of keeping Zener only inside evidence text.

Example M: A selected-grain table lists grain IDs, phase labels, and grain sizes for a nanoindentation study.
Expected behavior: populate `microstructure_features[]` entries representing the selected local regions/grains instead of collapsing the information into a free-text note.

Example N: One table gives GA calibration bounds, and later tables give GA/T&E final calibrated parameters.
Expected behavior: keep the bounds as `assertion.valid_range` for the corresponding parameter when the mapping is explicit; do not create standalone `*_bound` parameters unless the paper explicitly treats the bound as a model quantity.

Example O: A structured table row contains multiple named families or systems with explicit values and one shared literature reference.
Expected behavior: emit one parameter claim per explicit family or system value and attach the same adopted provenance reference to each claim.

Example P: The paper models a heterogeneous material with one shared parameter set for the whole aggregate.
Expected behavior: keep the shared parameterization explicit through `parameterization_scope=shared` or broad applicability, but do not split parameter claims by constituent unless explicit constituent-specific values are given.

Example P2: The excerpt says an alloy is multi-phase and also explicitly names matrix and precipitate phases or parent and product phases.
Expected behavior: keep `materials[].phase_mode=multi_phase` and also create explicit `constituents[]` entries for the named constituents; do not leave `constituents[]` empty in that case.

Example P3: The excerpt says a material is multi-phase but gives no explicit constituent names, fractions, or roles.
Expected behavior: keep `materials[].phase_mode=multi_phase`, leave `constituents[]` empty, and note that the constituent-level identities were not explicitly recoverable from the excerpt.

Example Q: A paper gives one parameter table, a texture table listing ED-oriented volume fraction by temperature and grain size, and results text describing bimodal versus uniform grains, KAM heterogeneity, grain-boundary sliding, and dislocation-density changes.
Expected behavior: keep the parameter claims from the parameter table, and also emit multiple `microstructure_features[]` entries for texture fraction, bimodal or uniform morphology, heterogeneous KAM or lattice distortion, grain-boundary sliding, and low-versus-high dislocation density where each fact is explicitly supported.

Example R: One constitutive section states the active constitutive channels and then gives several numbered equations for flow and one or more evolution laws.
Expected behavior: attach all governing constitutive equations to `models[].equation_ids` and create separate `constitutive_branches[]` entries for each explicit branch or evolution law.

Example R2: The model uses basal and prismatic slip, and the paper explicitly states the family names, plane/direction notation, and number of systems.
Expected behavior: emit separate `deformation_systems[]` entries and link them from `models[].constitutive_description.slip_description.deformation_system_ids`.

Example R3: The paper states that a CPFE simulation used an RVE with 300 grains, periodic boundary conditions, and a voxel grid.
Expected behavior: emit a `simulation_geometries[]` entry linked to the model instead of keeping those details only in `solver_framework.notes`.

Example S: A paper says one parameter subset is calibrated from a macroscopic mechanical response, while another subset is calibrated from a subset-specific relaxation, diffraction, or local response, both under the same temperature.
Expected behavior: create one physical `condition` if appropriate, but keep distinct `provenance.calibration` descriptions across the affected parameter claims; the subset-based claims should use the most specific compatible `observation_scope` and preserve the subset label in `target_description`.

Example S2: The paper says the simulation used measured EBSD orientations as input and compares predicted lattice strain against diffraction data.
Expected behavior: emit `orientation_inputs[]` for the EBSD-derived input, `simulation_outputs[]` for lattice strain, and `model_evaluations[]` describing the calibration or validation role.

Example T: A paper discusses a named orientation family, subset label, diffraction family, or local-region response during calibration.
Expected behavior: treat that label as an observation or calibration target, not as a constituent or constitutive branch unless the text explicitly defines it that way.

Example U: The paper explicitly names a standard kinematic-hardening law or another standard evolution law.
Expected behavior: normalize the named law into the closest supported structured field when possible, create a `constitutive_branches[]` entry for the explicit evolution law when appropriate, and bind the related parameters to that branch or its governing equations where possible.

Example V: A constitutive subsection introduces several numbered equations with distinct constitutive roles.
Expected behavior: read the constitutive narrative and attach all of the explicit governing equation numbers across `models[].equation_ids`, the relevant `constitutive_branches[]`, and any affected `parameter_claims[].governing_equation_ids`; do not rely only on whether the parameter name resembles a hard-coded canonical label.

Example W: A branch is governed by a coupled flow equation and separate hardening or backstress evolution equations, and one parameter appears in more than one of those equations.
Expected behavior: keep every explicit related equation label in the branch and parameter `governing_equation_ids` arrays; do not force one equation per branch or one equation per parameter.

Example X: A parameter such as `γ̇0,1` or `m1` appears in both a plastic-flow branch and a combined plastic-creep branch.
Expected behavior: keep all relevant `governing_equation_ids` and set `parameter_claims[].applies_to.branch_ids` to every explicit matching branch rather than choosing only one branch.

Paper excerpt:
----------------
{context}
----------------

Return JSON only.
"""


def _schema_skeleton_from_json_template(template: str) -> Dict[str, Any]:
    text = template.strip().replace("{{", "{").replace("}}", "}")
    l = text.find("{")
    r = text.rfind("}")
    if l < 0 or r < 0 or r <= l:
        raise RuntimeError("Failed to parse extraction schema template")
    return json.loads(text[l:r + 1])


def _coerce_to_schema_shape(schema_node: Any, payload_node: Any) -> Any:
    if isinstance(schema_node, dict):
        src = payload_node if isinstance(payload_node, dict) else {}
        out: Dict[str, Any] = {}
        for k, sv in schema_node.items():
            out[k] = _coerce_to_schema_shape(sv, src.get(k))
        return out

    if isinstance(schema_node, list):
        if not isinstance(payload_node, list):
            return []
        if not schema_node:
            return payload_node
        item_schema = schema_node[0]
        return [_coerce_to_schema_shape(item_schema, item) for item in payload_node]

    # Leaf placeholder in template (e.g., "string or null"): prefer payload value, else null.
    if payload_node is None:
        return None
    return payload_node


EXTRACT_SCHEMA_SKELETON = _schema_skeleton_from_json_template(EXTRACT_SCHEMA_JSON_TEMPLATE)


def _validate_extracted_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["payload is not an object"]

    for key in EXTRACT_SCHEMA_SKELETON.keys():
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    def _walk(schema_node: Any, value_node: Any, path: str):
        if isinstance(schema_node, dict):
            if not isinstance(value_node, dict):
                errors.append(f"{path or 'root'} must be an object")
                return
            for k, sv in schema_node.items():
                _walk(sv, value_node.get(k), f"{path}.{k}" if path else k)
            return

        if isinstance(schema_node, list):
            if not isinstance(value_node, list):
                errors.append(f"{path} must be a list")
                return
            if schema_node:
                for i, item in enumerate(value_node):
                    _walk(schema_node[0], item, f"{path}[{i}]")
            return

    _walk(EXTRACT_SCHEMA_SKELETON, payload, "")
    return errors


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        if value not in (None, "", [], {}):
            return value
    return None


def _project_v3_claim_to_legacy_registry_item(claim: Dict[str, Any]) -> Dict[str, Any]:
    claim = _safe_dict(claim)
    parameter = _safe_dict(claim.get("parameter"))
    assertion = _safe_dict(claim.get("assertion"))
    context = _safe_dict(claim.get("context"))
    mechanism_scope = _safe_dict(context.get("mechanism_scope"))
    provenance = _safe_dict(claim.get("provenance"))
    evidence = _safe_dict(claim.get("evidence"))
    legacy_applies = _safe_dict(claim.get("applies_to"))
    if not parameter:
        parameter = {
            "canonical_name": claim.get("canonical_name"),
            "symbol": claim.get("symbol"),
            "domain": claim.get("domain"),
            "description": claim.get("description"),
            "units": {
                "reported_unit": claim.get("unit"),
                "si_unit": claim.get("unit_SI"),
            },
        }
    if not assertion:
        assertion = {
            "value": claim.get("value"),
            "reported_value": claim.get("value"),
            "reported_unit": claim.get("unit"),
            "value_si": claim.get("value_SI"),
            "si_unit": claim.get("unit_SI"),
            "valid_range": claim.get("valid_range"),
        }
    if not context:
        context = {
            "material_id": legacy_applies.get("material_id"),
            "phase_id": legacy_applies.get("phase_id"),
            "process_state_id": legacy_applies.get("sample_id"),
            "condition_id": legacy_applies.get("condition_id"),
            "notes": legacy_applies.get("notes"),
            "mechanism_scope": {
                "level": legacy_applies.get("scope"),
                "mechanism_type": legacy_applies.get("mechanism"),
                "family_id": legacy_applies.get("family_id"),
                "family_name": legacy_applies.get("family_name"),
                "system_ids": _safe_list(legacy_applies.get("system_ids")),
            },
        }
        mechanism_scope = _safe_dict(context.get("mechanism_scope"))
    if evidence and "text" not in evidence and "evidence_text" in evidence:
        evidence = {
            "text": evidence.get("evidence_text"),
            "table_evidence": _safe_dict(evidence.get("table_evidence")) or None,
            "notes": evidence.get("notes"),
        }
    legacy_item = {
        "claim_id": claim.get("claim_id"),
        "domain": parameter.get("domain"),
        "canonical_name": parameter.get("canonical_name"),
        "symbol": parameter.get("symbol"),
        "description": parameter.get("description"),
        "value": assertion.get("reported_value", assertion.get("value")),
        "unit": assertion.get("reported_unit", _safe_dict(parameter.get("units")).get("reported_unit")),
        "value_SI": assertion.get("value_si"),
        "unit_SI": assertion.get("si_unit", _safe_dict(parameter.get("units")).get("si_unit")),
        "applies_to": {
            "scope": mechanism_scope.get("level"),
            "material_id": context.get("material_id"),
            "sample_id": context.get("process_state_id"),
            "bundle_id": None,
            "condition_id": context.get("condition_id"),
            "phase_id": context.get("phase_id"),
            "mechanism": mechanism_scope.get("mechanism_type"),
            "family_id": mechanism_scope.get("family_id"),
            "family_name": mechanism_scope.get("family_name"),
            "system_ids": _safe_list(mechanism_scope.get("system_ids")),
            "system_count": len(_safe_list(mechanism_scope.get("system_ids"))),
            "notes": _first_non_empty(context.get("notes"), mechanism_scope.get("notes")),
        },
        "temperature_dependent": claim.get("temperature_dependent"),
        "strain_rate_dependent": claim.get("strain_rate_dependent"),
        "valid_range": assertion.get("valid_range"),
        "source": {
            "origin_type": provenance.get("origin_type"),
            "reference_ids": _safe_list(provenance.get("reference_ids")),
            "adopted_from_reference_ids": _safe_list(provenance.get("adopted_from_reference_ids")),
            "calibration_based_on_reference_ids": _safe_list(provenance.get("calibration_based_on_reference_ids")),
            "calibration_in_this_study": provenance.get("calibration_in_this_study"),
            "calibration_method": provenance.get("calibration_method"),
            "notes": provenance.get("notes"),
        },
        "evidence": {
            "evidence_text": evidence.get("text"),
            "table_evidence": _safe_dict(evidence.get("table_evidence")) or None,
            "notes": evidence.get("notes"),
        },
        "notes": claim.get("notes"),
    }
    return legacy_item


def _feature_matches(feature: Dict[str, Any], *, material_id: Any = None, phase_id: Any = None, process_state_id: Any = None) -> bool:
    applies_to = _safe_dict(feature.get("applies_to"))
    if material_id and applies_to.get("material_id") not in (None, material_id):
        return False
    if phase_id and applies_to.get("phase_id") not in (None, phase_id):
        return False
    if process_state_id and applies_to.get("process_state_id") not in (None, process_state_id):
        return False
    return True


def _first_feature(
    features: List[Dict[str, Any]],
    feature_type: str,
    *,
    material_id: Any = None,
    phase_id: Any = None,
    process_state_id: Any = None,
) -> Dict[str, Any]:
    for feature in features:
        if not isinstance(feature, dict):
            continue
        feature_name = str(feature.get("feature_name") or feature.get("type") or "").strip().lower()
        if feature_name != feature_type:
            continue
        if _feature_matches(feature, material_id=material_id, phase_id=phase_id, process_state_id=process_state_id):
            return feature
    return {}


def _inject_legacy_compat_views_from_v3(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    document = _safe_dict(payload.get("document"))
    materials = [m for m in _safe_list(payload.get("materials")) if isinstance(m, dict)]
    process_states = [s for s in _safe_list(payload.get("process_states")) if isinstance(s, dict)]
    legacy_samples = [s for s in _safe_list(payload.get("samples")) if isinstance(s, dict)]
    compat_process_states = process_states or legacy_samples
    conditions = [c for c in _safe_list(payload.get("conditions")) if isinstance(c, dict)]
    models = [m for m in _safe_list(payload.get("models")) if isinstance(m, dict)]
    deformation_systems = [d for d in _safe_list(payload.get("deformation_systems")) if isinstance(d, dict)]
    raw_mechanisms = payload.get("mechanisms")
    mechanisms = _safe_dict(raw_mechanisms)
    if not mechanisms and isinstance(raw_mechanisms, list):
        mechanism_rows = [m for m in raw_mechanisms if isinstance(m, dict)]
        mechanisms = {
            "slip_families": [m for m in mechanism_rows if str(m.get("mechanism_type") or "").strip().lower() == "slip" and str(m.get("level") or "").strip().lower() == "family"],
            "twinning_families": [m for m in mechanism_rows if str(m.get("mechanism_type") or "").strip().lower() == "twinning" and str(m.get("level") or "").strip().lower() == "family"],
            "cleavage_families": [m for m in mechanism_rows if str(m.get("mechanism_type") or "").strip().lower() == "cleavage" and str(m.get("level") or "").strip().lower() == "family"],
            "damage_mechanisms": [m for m in mechanism_rows if str(m.get("mechanism_type") or "").strip().lower() == "damage"],
            "transformation_mechanisms": [m for m in mechanism_rows if str(m.get("mechanism_type") or "").strip().lower() == "transformation"],
            "other_mechanisms": [
                m for m in mechanism_rows
                if str(m.get("mechanism_type") or "").strip().lower() not in {"slip", "twinning", "cleavage", "damage", "transformation"}
            ],
            "notes": None,
        }
    microstructure_features = [f for f in _safe_list(payload.get("microstructure_features")) if isinstance(f, dict)]
    claims = [c for c in _safe_list(payload.get("parameter_claims")) if isinstance(c, dict)]

    primary_material = materials[0] if materials else {}
    primary_phases = [p for p in _safe_list(primary_material.get("phases")) if isinstance(p, dict)]
    primary_phase = primary_phases[0] if primary_phases else {}
    primary_model = models[0] if models else {}
    primary_condition = conditions[0] if conditions else {}
    primary_process_state = compat_process_states[0] if compat_process_states else {}
    primary_grain_feature = _first_feature(
        microstructure_features,
        "grain_size",
        material_id=primary_material.get("material_id"),
        phase_id=primary_phase.get("phase_id"),
        process_state_id=primary_process_state.get("process_state_id"),
    )
    primary_texture_feature = _first_feature(
        microstructure_features,
        "texture",
        material_id=primary_material.get("material_id"),
        phase_id=primary_phase.get("phase_id"),
        process_state_id=primary_process_state.get("process_state_id"),
    )
    primary_grain_structure_feature = _first_feature(
        microstructure_features,
        "grain_structure",
        material_id=primary_material.get("material_id"),
        phase_id=primary_phase.get("phase_id"),
        process_state_id=primary_process_state.get("process_state_id"),
    )
    primary_dislocation_feature = _first_feature(
        microstructure_features,
        "dislocation_density",
        material_id=primary_material.get("material_id"),
        phase_id=primary_phase.get("phase_id"),
        process_state_id=primary_process_state.get("process_state_id"),
    )
    primary_precip_feature = _first_feature(
        microstructure_features,
        "precipitates",
        material_id=primary_material.get("material_id"),
        phase_id=primary_phase.get("phase_id"),
        process_state_id=primary_process_state.get("process_state_id"),
    )

    payload["record_id"] = payload.get("record_id")
    payload["source_document"] = {
        "title": _first_non_empty(document.get("title"), _safe_dict(payload.get("source_document")).get("title")),
        "authors": _safe_list(document.get("authors")) or _safe_list(_safe_dict(payload.get("source_document")).get("authors")),
        "year": _first_non_empty(document.get("year"), _safe_dict(payload.get("source_document")).get("year")),
        "journal_or_venue": _first_non_empty(document.get("journal"), _safe_dict(payload.get("source_document")).get("journal_or_venue")),
        "doi": _first_non_empty(document.get("doi"), _safe_dict(payload.get("source_document")).get("doi")),
    }
    payload["material"] = {
        "name": primary_material.get("name"),
        "chemical_formula": primary_material.get("formula"),
        "phase": ("multi" if len(primary_phases) > 1 else "single") if primary_phases else None,
        "phases": [
            {
                "phase_id": phase.get("phase_id"),
                "phase_name": phase.get("name"),
                "role": phase.get("role"),
                "crystal_structure": _safe_dict(phase.get("crystal_structure")) or None,
                "volume_fraction": {
                    "value_SI": _safe_dict(phase.get("volume_fraction")).get("value"),
                    "unit_SI": _safe_dict(phase.get("volume_fraction")).get("unit"),
                    "reported_value": _safe_dict(phase.get("volume_fraction")).get("reported_value"),
                    "reported_unit": _safe_dict(phase.get("volume_fraction")).get("reported_unit"),
                    "notes": _safe_dict(phase.get("volume_fraction")).get("notes"),
                },
                "notes": phase.get("notes"),
            }
            for phase in primary_phases
        ],
        "notes": primary_material.get("notes"),
    }
    payload["paper_profile"] = {
        "studied_materials": [
            {
                "material_id": material.get("material_id"),
                "name": material.get("name"),
                "chemical_formula": material.get("formula"),
                "material_class": material.get("material_class"),
                "phase_mode": material.get("phase_mode") or (("multi_phase" if len(_safe_list(material.get("phases"))) > 1 else "single_phase") if _safe_list(material.get("phases")) else None),
                "crystal_aggregate": material.get("crystal_aggregate"),
                "composition": {
                    "basis": _safe_dict(material.get("composition")).get("basis"),
                    "rows": [
                        {
                            "component": component.get("component"),
                            "value": component.get("value"),
                            "unit": component.get("unit"),
                            "notes": component.get("notes"),
                        }
                        for component in _safe_list(_safe_dict(material.get("composition")).get("components"))
                        if isinstance(component, dict)
                    ],
                    "notes": _safe_dict(material.get("composition")).get("notes"),
                },
                "phase_ids": [
                    phase.get("phase_id")
                    for phase in _safe_list(material.get("phases"))
                    if isinstance(phase, dict) and phase.get("phase_id")
                ],
                "notes": material.get("notes"),
            }
            for material in materials
        ],
        "sample_profiles": [
            {
                "sample_id": _first_non_empty(process_state.get("process_state_id"), process_state.get("sample_id")),
                "material_id": process_state.get("material_id"),
                "condition_id": _safe_list(process_state.get("condition_ids"))[0] if _safe_list(process_state.get("condition_ids")) else None,
                "label": process_state.get("label"),
                "processing_state": _first_non_empty(process_state.get("label"), process_state.get("processing_state")),
                "temperature": None,
                "grain_size": {
                    "value": _first_feature(
                        microstructure_features,
                        "grain_size",
                        material_id=process_state.get("material_id"),
                        process_state_id=_first_non_empty(process_state.get("process_state_id"), process_state.get("sample_id")),
                    ).get("value"),
                    "unit": _first_feature(
                        microstructure_features,
                        "grain_size",
                        material_id=process_state.get("material_id"),
                        process_state_id=_first_non_empty(process_state.get("process_state_id"), process_state.get("sample_id")),
                    ).get("unit"),
                    "notes": _first_feature(
                        microstructure_features,
                        "grain_size",
                        material_id=process_state.get("material_id"),
                        process_state_id=_first_non_empty(process_state.get("process_state_id"), process_state.get("sample_id")),
                    ).get("notes"),
                } if _first_feature(
                    microstructure_features,
                    "grain_size",
                    material_id=process_state.get("material_id"),
                    process_state_id=_first_non_empty(process_state.get("process_state_id"), process_state.get("sample_id")),
                ) else None,
                "phase_fractions": [
                    {
                        "phase_id": _safe_dict(feature.get("applies_to")).get("phase_id"),
                        "value": feature.get("value"),
                        "unit": feature.get("unit"),
                        "notes": feature.get("notes"),
                    }
                    for feature in microstructure_features
                    if isinstance(feature, dict)
                    and str(feature.get("type") or "").strip().lower() == "phase_fraction"
                    and _feature_matches(
                        feature,
                        material_id=process_state.get("material_id"),
                        process_state_id=_first_non_empty(process_state.get("process_state_id"), process_state.get("sample_id")),
                    )
                ],
                "selected_grains": [
                    {
                        "grain_id": feature.get("feature_id"),
                        "label": feature.get("description"),
                        "phase_id": _safe_dict(feature.get("applies_to")).get("phase_id"),
                        "grain_size": {"value": feature.get("value"), "unit": feature.get("unit"), "notes": feature.get("notes")} if feature.get("value") not in (None, "") else None,
                        "orientation_notes": feature.get("description"),
                        "notes": feature.get("notes"),
                    }
                    for feature in microstructure_features
                    if isinstance(feature, dict)
                    and str(feature.get("type") or "").strip().lower() == "selected_grain"
                    and _feature_matches(
                        feature,
                        material_id=process_state.get("material_id"),
                        process_state_id=_first_non_empty(process_state.get("process_state_id"), process_state.get("sample_id")),
                    )
                ],
                "texture_or_orientation": _first_feature(
                    microstructure_features,
                    "texture",
                    material_id=process_state.get("material_id"),
                    process_state_id=_first_non_empty(process_state.get("process_state_id"), process_state.get("sample_id")),
                ).get("description"),
                "notes": process_state.get("notes"),
            }
            for process_state in compat_process_states
        ],
        "notes": payload.get("global_notes"),
    }
    payload["microstructure"] = {
        "grain_structure": primary_grain_structure_feature.get("value"),
        "grain_size": {
            "value": primary_grain_feature.get("value"),
            "unit": primary_grain_feature.get("unit"),
            "notes": primary_grain_feature.get("notes"),
        } if primary_grain_feature else None,
        "orientation_texture": {
            "description": primary_texture_feature.get("description"),
            "texture_type": primary_texture_feature.get("method"),
            "texture_data_available": None,
            "data_location": None,
            "notes": primary_texture_feature.get("notes"),
        },
        "initial_defect_state": {
            "dislocation_density": {
                "value": primary_dislocation_feature.get("value"),
                "unit": primary_dislocation_feature.get("unit"),
                "notes": primary_dislocation_feature.get("notes"),
            } if primary_dislocation_feature else None,
            "precipitate_state": primary_precip_feature.get("description") or primary_precip_feature.get("value"),
            "solute_state": None,
            "prestrain": None,
            "notes": _first_non_empty(primary_dislocation_feature.get("notes"), primary_precip_feature.get("notes")),
        },
        "notes": payload.get("global_notes"),
    }
    payload["constitutive_model"] = {
        "class": _first_non_empty(primary_model.get("class"), primary_model.get("model_type")),
        "framework": _first_non_empty(primary_model.get("framework"), _safe_dict(primary_model.get("implementation")).get("platform"), _safe_dict(primary_model.get("implementation")).get("software")),
        "implementation": _safe_dict(primary_model.get("implementation")) or None,
        "kinematics": _first_non_empty(primary_model.get("kinematics"), _safe_dict(primary_model.get("constitutive_description")).get("kinematics")),
        "rate_dependence": _first_non_empty(primary_model.get("rate_dependence"), _safe_dict(_safe_dict(primary_model.get("constitutive_description")).get("flow_kinetics")).get("rate_dependence")),
        "single_or_poly": primary_model.get("single_or_poly"),
        "homogenization_assumption": primary_model.get("homogenization_assumption"),
        "notes": primary_model.get("notes"),
    }
    payload["parameters"] = {
        "registry": [_project_v3_claim_to_legacy_registry_item(claim) for claim in claims],
        "notes": None,
    }
    payload["parameter_bundles"] = _safe_list(payload.get("parameter_bundles"))
    payload["deformation_conditions"] = primary_condition or {}
    payload["condition_profiles"] = conditions
    if deformation_systems and not mechanisms:
        payload["deformation_mechanisms"] = {
            "slip_families": [
                {
                    "family_id": system.get("system_id"),
                    "family_name": system.get("family_name"),
                    "plane_direction": " // ".join(
                        part for part in (
                            str(system.get("plane") or "").strip(),
                            str(system.get("direction") or "").strip(),
                        )
                        if part
                    ) or None,
                    "num_systems": system.get("number_of_systems"),
                    "systems": [],
                    "active": None,
                    "notes": system.get("notes"),
                }
                for system in deformation_systems
                if str(system.get("system_type") or "").strip().lower() == "slip"
            ],
            "twinning_families": [
                {
                    "family_id": system.get("system_id"),
                    "family_name": system.get("family_name"),
                    "plane_direction": " // ".join(
                        part for part in (
                            str(system.get("plane") or "").strip(),
                            str(system.get("direction") or "").strip(),
                        )
                        if part
                    ) or None,
                    "num_systems": system.get("number_of_systems"),
                    "systems": [],
                    "active": None,
                    "reorientation_rule": None,
                    "notes": system.get("notes"),
                }
                for system in deformation_systems
                if str(system.get("system_type") or "").strip().lower() == "twin"
            ],
            "cleavage_families": [],
            "damage_mechanisms": [],
            "transformation_mechanisms": [
                system for system in deformation_systems
                if str(system.get("system_type") or "").strip().lower() == "transformation"
            ],
            "other_mechanisms": [
                system for system in deformation_systems
                if str(system.get("system_type") or "").strip().lower() not in {"slip", "twin", "transformation"}
            ],
            "notes": None,
        }
    else:
        payload["deformation_mechanisms"] = {
            "slip_families": [
                {
                    "family_id": family.get("family_id"),
                    "family_name": family.get("name"),
                    "plane_direction": family.get("plane_direction"),
                    "num_systems": family.get("num_systems"),
                    "systems": _safe_list(family.get("systems")),
                    "active": family.get("active"),
                    "notes": family.get("notes"),
                }
                for family in _safe_list(mechanisms.get("slip_families"))
                if isinstance(family, dict)
            ],
            "twinning_families": [
                {
                    "family_id": family.get("family_id"),
                    "family_name": family.get("name"),
                    "plane_direction": family.get("plane_direction"),
                    "num_systems": family.get("num_systems"),
                    "systems": _safe_list(family.get("systems")),
                    "active": family.get("active"),
                    "reorientation_rule": family.get("reorientation_rule"),
                    "notes": family.get("notes"),
                }
                for family in _safe_list(mechanisms.get("twinning_families"))
                if isinstance(family, dict)
            ],
            "cleavage_families": [
                {
                    "family_id": family.get("family_id"),
                    "family_name": family.get("name"),
                    "plane_direction": family.get("plane_direction"),
                    "num_systems": family.get("num_systems"),
                    "systems": _safe_list(family.get("systems")),
                    "active": family.get("active"),
                    "notes": family.get("notes"),
                }
                for family in _safe_list(mechanisms.get("cleavage_families"))
                if isinstance(family, dict)
            ],
            "damage_mechanisms": _safe_list(mechanisms.get("damage_mechanisms")),
            "transformation_mechanisms": _safe_list(mechanisms.get("transformation_mechanisms")),
            "other_mechanisms": _safe_list(mechanisms.get("other_mechanisms")),
            "notes": mechanisms.get("notes"),
        }
    return payload


def _build_extract_prompt(context: str) -> str:
    # Avoid str.format() here because the template contains literal braces such as crystallographic {111}.
    prompt = EXTRACT_USER_PROMPT_TEMPLATE.replace("__SCHEMA_JSON__", EXTRACT_SCHEMA_JSON_TEMPLATE)
    return prompt.replace("{context}", context)


SOURCE_ENRICH_SYSTEM_PROMPT = """
You refine provenance fields for previously extracted CP parameters.
Return JSON only.
"""


SOURCE_ENRICH_USER_PROMPT_TEMPLATE = """
1 Task description
Refine and complete parameter source/provenance fields using the excerpt.

2 Task requirements
- Only fill source fields for existing parameters by index.
- Do not change numeric values or symbols.
- Output JSON:
{{
  "elastic_sources": [
    {{
      "index": "number",
      "source": {{
        "origin_type": "original / adopted / calibrated / adopted_then_calibrated / null",
        "reference_ids": ["string"],
        "adopted_from_reference_ids": ["string"],
        "calibration_based_on_reference_ids": ["string"],
        "calibration_in_this_study": "yes / no / null",
        "calibration_method": "string or null",
        "notes": "string or null"
      }}
    }}
  ],
  "plastic_sources": [
    {{
      "index": "number",
      "source": {{
        "origin_type": "original / adopted / calibrated / adopted_then_calibrated / null",
        "reference_ids": ["string"],
        "adopted_from_reference_ids": ["string"],
        "calibration_based_on_reference_ids": ["string"],
        "calibration_in_this_study": "yes / no / null",
        "calibration_method": "string or null",
        "notes": "string or null"
      }}
    }}
  ]
}}

3 Processing suggestions
- If text says both adopted and calibrated, use adopted_then_calibrated.
- Copy citation labels (numbers) into the reference id arrays.
- Do not put this_study/present_study/current_study into reference id arrays.
- Do not rewrite or replace the existing `evidence` object. This step is only for provenance/source refinement.
- If a parameter has an explicit value but no explicit unit, keep the value and leave unit-related fields null; do not delete or downgrade the parameter solely because the unit is missing.
- If a parameter value is explicitly `0`, preserve it as a real value.
- If one table row applies to multiple families/scopes, preserve that shared semantics in existing `evidence` and `applies_to`; do not blur it in provenance notes.
- Preserve material/sample/condition identity already present in the extracted JSON; provenance refinement should not blur which material, sample, or bundle a parameter belongs to.

4 Few-shot examples
If text says adopted from [60,61] then fitted to curves, then include both adopted and calibration refs.

Excerpt:
----------------
{context}
----------------

Current extracted JSON:
----------------
{extracted_json}
----------------

Return JSON only.
"""


def _merge_source_payload(existing: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    existing = existing if isinstance(existing, dict) else {}
    update = update if isinstance(update, dict) else {}
    out = dict(existing)

    for key, value in update.items():
        if value in (None, "", [], {}):
            continue
        if key == "table_evidence":
            current_te = out.get("table_evidence") if isinstance(out.get("table_evidence"), dict) else {}
            new_te = value if isinstance(value, dict) else {}
            merged_te = dict(current_te)
            for te_key, te_value in new_te.items():
                if te_value in (None, "", [], {}):
                    continue
                if merged_te.get(te_key) in (None, "", [], {}):
                    merged_te[te_key] = te_value
            if merged_te:
                out["table_evidence"] = merged_te
            continue
        if out.get(key) in (None, "", [], {}):
            out[key] = value
    return out


def _build_source_enrich_prompt(context: str, extracted: Dict[str, Any]) -> str:
    prompt = SOURCE_ENRICH_USER_PROMPT_TEMPLATE.replace("{context}", context)
    return prompt.replace("{extracted_json}", json.dumps(extracted, ensure_ascii=False)[:20000])


def _merge_source_enrichment(extracted: Dict[str, Any], enrich: Dict[str, Any]) -> Dict[str, Any]:
    registry = extracted.get("parameters", {}).get("registry", [])
    claim_mode = False
    has_legacy_registry = isinstance(extracted.get("parameters"), dict) and isinstance(registry, list) and bool(registry)
    if not has_legacy_registry:
        registry = extracted.get("parameter_claims", [])
        claim_mode = isinstance(registry, list)
    if not isinstance(registry, list):
        return extracted

    elastic_idx = []
    plastic_idx = []
    for i, it in enumerate(registry):
        if not isinstance(it, dict):
            continue
        domain = str(it.get("domain") or _safe_dict(_safe_dict(it.get("parameter")).get("domain")) or "").strip().lower()
        cname = str(it.get("canonical_name") or _safe_dict(it.get("parameter")).get("canonical_name") or "").strip().lower()
        if domain == "elastic" or cname in _ELASTIC_CANONICALS:
            elastic_idx.append(i)
        else:
            plastic_idx.append(i)

    for item in enrich.get("elastic_sources", []) or []:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(elastic_idx):
            ridx = elastic_idx[idx]
            if isinstance(item.get("source"), dict):
                target_key = "provenance" if claim_mode else "source"
                registry[ridx][target_key] = _merge_source_payload(registry[ridx].get(target_key), item["source"])

    for item in enrich.get("plastic_sources", []) or []:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(plastic_idx):
            ridx = plastic_idx[idx]
            if isinstance(item.get("source"), dict):
                target_key = "provenance" if claim_mode else "source"
                registry[ridx][target_key] = _merge_source_payload(registry[ridx].get(target_key), item["source"])

    return extracted


_ELASTIC_CANONICALS = {"c11", "c12", "c13", "c33", "c44", "c55", "c66", "e", "nu", "g", "k"}


def _strip_internal_keys(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            out[k] = _strip_internal_keys(v)
        return out
    if isinstance(node, list):
        return [_strip_internal_keys(x) for x in node]
    return node


def _drop_legacy_parameter_blocks(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict):
        payload.pop("elastic_parameters", None)
        payload.pop("plastic_parameters", None)
    return payload


def _normalize_equation_reference_value(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    return f"({int(nums[-1])})"


def _normalize_equation_reference_list(values: Any) -> List[str]:
    out: List[str] = []
    for value in values if isinstance(values, list) else []:
        normalized = _normalize_equation_reference_value(value)
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _normalize_equation_references_in_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    models = payload.get("models")
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, dict):
                continue
            model["equation_ids"] = _normalize_equation_reference_list(model.get("equation_ids"))
            branches = model.get("constitutive_branches")
            if isinstance(branches, list):
                for branch in branches:
                    if not isinstance(branch, dict):
                        continue
                    branch["governing_equation_ids"] = _normalize_equation_reference_list(
                        branch.get("governing_equation_ids")
                    )

    claims = payload.get("parameter_claims")
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim["governing_equation_ids"] = _normalize_equation_reference_list(
                claim.get("governing_equation_ids")
            )
            provenance = claim.get("provenance")
    return payload


def build_context(selected_sections, selected_tables, selected_equations, max_context_chars: int) -> Tuple[str, Dict[str, Any]]:
    parts = []
    total = 0
    context_meta: Dict[str, Any] = {
        "max_context_chars": max_context_chars,
        "included_tables": [],
        "included_equations": [],
        "included_sections": [],
        "truncated_tables": [],
        "truncated_equations": [],
        "truncated_sections": [],
        "omitted_tables": [],
        "omitted_equations": [],
        "omitted_sections": [],
    }

    equation_budget_total = int(max_context_chars * 0.2) if selected_equations else 0
    remaining_after_equations = max_context_chars - equation_budget_total
    table_budget_total = remaining_after_equations if not selected_sections else int(remaining_after_equations * 0.8)
    section_budget_total = remaining_after_equations - table_budget_total
    per_equation_budget = equation_budget_total // max(1, len(selected_equations)) if selected_equations else 0
    per_table_budget = table_budget_total // max(1, len(selected_tables)) if selected_tables else 0
    per_section_budget = section_budget_total // max(1, len(selected_sections)) if selected_sections else 0

    for idx, eq in enumerate(selected_equations):
        remaining_budget = max_context_chars - total
        if remaining_budget <= 0:
            context_meta["omitted_equations"].extend(
                [row["name"] for row in selected_equations[idx:] if isinstance(row, dict) and row.get("name")]
            )
            break
        local_budget = max(400, min(remaining_budget, per_equation_budget or remaining_budget))
        content = trim_text(str(eq.get("extract_text") or eq.get("text") or ""), local_budget)
        if len(str(eq.get("extract_text") or eq.get("text") or "")) > local_budget:
            context_meta["truncated_equations"].append(eq["name"])
        chunk = f"\n\n=== EQUATION: {eq['name']} ===\n{content}"
        parts.append(chunk)
        total += len(chunk)
        context_meta["included_equations"].append(eq["name"])

    for idx, t in enumerate(selected_tables):
        remaining_budget = max_context_chars - total
        if remaining_budget <= 0:
            context_meta["omitted_tables"].extend(
                [tbl["name"] for tbl in selected_tables[idx:] if isinstance(tbl, dict) and tbl.get("name")]
            )
            break
        full_table_text = str(t.get("extract_text") or t.get("json_summary") or t.get("text") or "").strip()
        semantic_hint = _table_semantic_hint(t)
        full_table_text = f"{semantic_hint}\n{full_table_text}" if full_table_text else semantic_hint
        local_budget = max(1200, min(remaining_budget, per_table_budget or remaining_budget))
        content = trim_text(full_table_text, local_budget)
        if len(full_table_text) > local_budget:
            context_meta["truncated_tables"].append(t["name"])
        chunk = f"\n\n=== TABLE: {t['name']} ===\n{content}"
        parts.append(chunk)
        total += len(chunk)
        context_meta["included_tables"].append(t["name"])
        if total > max_context_chars:
            overflow = [tbl["name"] for tbl in selected_tables[idx + 1:] if isinstance(tbl, dict) and tbl.get("name")]
            context_meta["omitted_tables"].extend(overflow)
            context_meta["final_context_chars"] = len("\n".join(parts))
            return "\n".join(parts), context_meta

    for idx, s in enumerate(selected_sections):
        remaining_budget = max_context_chars - total
        if remaining_budget <= 0:
            context_meta["omitted_sections"].extend(
                [sec["name"] for sec in selected_sections[idx:] if isinstance(sec, dict) and sec.get("name")]
            )
            break
        local_budget = max(800, min(remaining_budget, per_section_budget or remaining_budget))
        content = trim_text(s["text"], local_budget)
        if len(str(s.get("text") or "")) > local_budget:
            context_meta["truncated_sections"].append(s["name"])
        chunk = f"\n\n=== SECTION: {s['name']} ===\n{content}"
        parts.append(chunk)
        total += len(chunk)
        context_meta["included_sections"].append(s["name"])
        if total > max_context_chars:
            overflow = [sec["name"] for sec in selected_sections[idx + 1:] if isinstance(sec, dict) and sec.get("name")]
            context_meta["omitted_sections"].extend(overflow)
            break

    context_meta["final_context_chars"] = len("\n".join(parts).strip())
    return "\n".join(parts).strip(), context_meta

def _build_extract_messages(prompt: str, selected_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    for table in selected_tables:
        if str(table.get("table_kind") or "") != "image_backed":
            continue
        image_path = str(table.get("image_local_path") or "").strip()
        if not image_path or not os.path.exists(image_path):
            continue
        table_json = table.get("table_json") or {}
        caption = str(table_json.get("caption") or "").strip()
        label = str(table_json.get("table_label") or table.get("name") or "").strip()
        table_hint = f"Image-backed table {label}."
        if caption:
            table_hint += f" Caption: {caption}"
        content.append({"type": "text", "text": sanitize_text_for_openai(table_hint)})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})

    return [
        {"role": "system", "content": sanitize_text_for_openai(EXTRACT_SYSTEM_PROMPT)},
        {"role": "user", "content": content},
    ]


def llm_extract(context: str, selected_tables: List[Dict[str, Any]], model: str, max_retries: int = 2) -> Tuple[Dict[str, Any], Any, float]:
    prompt = _build_extract_prompt(context)
    attempts = max(1, max_retries + 1)
    start_all = time.perf_counter()
    last_errors: List[str] = []
    last_usage = None

    for attempt in range(1, attempts + 1):
        resp = _chat_completion_with_retry(
            model=model,
            messages=_build_extract_messages(prompt, selected_tables),
            max_retries=4,
        )

        last_usage = resp.usage
        raw_payload = json.loads(resp.choices[0].message.content)
        payload = _coerce_to_schema_shape(EXTRACT_SCHEMA_SKELETON, raw_payload)
        errors = _validate_extracted_payload(payload)
        if not errors:
            elapsed = time.perf_counter() - start_all
            return payload, resp.usage, elapsed

        last_errors = errors
        if attempt < attempts:
            prompt = (
                _build_extract_prompt(context)
                + "\n\nValidation errors from your previous output:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\nPlease regenerate and return valid JSON only."
            )

    elapsed = time.perf_counter() - start_all
    raise RuntimeError(f"Extraction JSON validation failed after {attempts} attempts: {last_errors}")

def run_llm_on_paper_dir(
    paper_dir: str,
    model_select: str,
    model_extract: str,
    max_snippet_chars: int,
    max_context_chars: int,
    max_extract_retries: int = 2,
    enable_source_enrichment: bool = True,
    direct_image_table_input: bool = True,
    image_download_api_key: str | None = None,
    image_download_inst_token: str | None = None,
):
    sections_dir = os.path.join(paper_dir, "sections")
    tables_dir = os.path.join(paper_dir, "tables")
    equations_dir = os.path.join(paper_dir, "equations")

    sections = load_md_files(sections_dir) if os.path.exists(sections_dir) else []
    tables = load_table_files(tables_dir) if os.path.exists(tables_dir) else []
    equations = load_equation_files(equations_dir) if os.path.exists(equations_dir) else []
    if direct_image_table_input and tables:
        tables = ensure_image_backed_table_images(
            paper_dir,
            tables,
            api_key=image_download_api_key,
            inst_token=image_download_inst_token,
        )

    selection, sel_usage, sel_time = llm_select_files(
        sections, tables, equations,
        model=model_select,
        max_snippet_chars=max_snippet_chars
    )

    selected_section_names = set(selection.get("selected_sections", []))
    selected_table_ids = set()
    selected_equation_ids = set()
    for name in (selection.get("selected_tables", []) or []):
        raw = str(name).strip()
        if not raw:
            continue
        if raw.endswith(".md"):
            raw = raw[:-3]
        if raw.endswith(".json"):
            raw = raw[:-5]
        selected_table_ids.add(raw)
    for name in (selection.get("selected_equations", []) or []):
        raw = str(name).strip()
        if raw:
            selected_equation_ids.add(raw)

    selected_sections = [s for s in sections if s["name"] in selected_section_names]
    selected_tables = [t for t in tables if t.get("selection_id") in selected_table_ids or t["name"] in selected_table_ids]
    selected_equations = [e for e in equations if e.get("selection_id") in selected_equation_ids or e["name"] in selected_equation_ids]
    used_fallback_sections = False
    used_fallback_tables = False
    used_fallback_equations = False

    # Fallback for robustness when selection stage returns empty.
    if not selected_sections and sections:
        selected_sections = sections[:2]
        used_fallback_sections = True
    if not selected_tables and tables:
        selected_tables = _fallback_select_tables(tables)
        used_fallback_tables = True
    if not selected_equations and equations:
        selected_equations = _fallback_select_equations(equations)
        used_fallback_equations = True

    if equations:
        augmented_equations = _augment_selected_equations(
            selected_equations,
            equations,
            selected_sections,
            limit=8,
        )
        if len(augmented_equations) > len(selected_equations):
            selected_equations = augmented_equations
            if not selection.get("selected_equations"):
                used_fallback_equations = True

    if selected_tables and tables:
        has_material_profile_table = any(_is_material_profile_table(t) for t in selected_tables)
        if not has_material_profile_table:
            profile_candidates = [t for t in tables if _is_material_profile_table(t)]
            if profile_candidates:
                profile_candidates = sorted(
                    profile_candidates,
                    key=lambda t: (_fallback_table_score(t), -int(t.get("length") or 0)),
                    reverse=True,
                )
                best = profile_candidates[0]
                if all(best.get("name") != t.get("name") for t in selected_tables):
                    selected_tables.append(best)

    if sections and not _has_selected_microstructure_section(selected_sections):
        micro_candidates = sorted(
            [s for s in sections if _microstructure_section_score(s) > 0],
            key=lambda s: (_microstructure_section_score(s), -int(s.get("length") or 0)),
            reverse=True,
        )
        for candidate in micro_candidates[:2]:
            if all(candidate.get("name") != s.get("name") for s in selected_sections):
                selected_sections.append(candidate)

    with open(os.path.join(paper_dir, "llm_selected_files.json"), "w", encoding="utf-8") as f:
        selection_out = dict(selection)
        selection_out["selected_sections"] = [s["name"] for s in selected_sections]
        selection_out["selected_tables"] = [t.get("selection_id") or Path(t["name"]).stem for t in selected_tables]
        selection_out["selected_equations"] = [e.get("selection_id") or e["name"] for e in selected_equations]
        selection_out["resolved_section_files"] = [s["name"] for s in selected_sections]
        selection_out["resolved_selected_table_files"] = [t["name"] for t in selected_tables]
        selection_out["resolved_selected_equation_ids"] = [e.get("selection_id") or e["name"] for e in selected_equations]
        selection_out["used_fallback_selection"] = used_fallback_sections or used_fallback_tables or used_fallback_equations
        selection_out["fallback"] = {
            "sections": used_fallback_sections,
            "tables": used_fallback_tables,
            "equations": used_fallback_equations,
        }
        json.dump(selection_out, f, ensure_ascii=False, indent=2)

    extract_tables = selected_tables if direct_image_table_input else [
        t for t in selected_tables if str(t.get("table_kind") or "") != "image_backed"
    ]
    context, context_meta = build_context(selected_sections, extract_tables, selected_equations, max_context_chars=max_context_chars)
    skipped_no_explicit_parameters = _should_skip_extraction_no_explicit_parameters(selected_sections, selected_tables)
    if skipped_no_explicit_parameters:
        extracted = _empty_extraction_payload(
            "Skipped LLM extraction because the selected context did not contain explicit parameter values."
        )
        ext_usage = None
        ext_time = 0.0
    else:
        extracted, ext_usage, ext_time = llm_extract(
            context,
            extract_tables,
            model=model_extract,
            max_retries=max_extract_retries,
        )
    enrich_usage = None
    enrich_time = None
    source_enrichment_applied = False
    source_enrichment_error = None
    if enable_source_enrichment and not skipped_no_explicit_parameters:
        try:
            enrich_start = time.perf_counter()
            enrich_resp = _chat_completion_with_retry(
                model=model_extract,
                messages=[
                    {"role": "system", "content": SOURCE_ENRICH_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_source_enrich_prompt(context, extracted)},
                ],
                max_retries=3,
            )
            enrich_time = time.perf_counter() - enrich_start
            enrich_usage = enrich_resp.usage
            enrich_payload = json.loads(enrich_resp.choices[0].message.content)
            extracted = _merge_source_enrichment(extracted, enrich_payload)
            source_enrichment_applied = True
        except Exception as exc:
            # Keep pipeline robust: provenance enrichment is optional.
            source_enrichment_error = str(exc)

    extracted = _normalize_equation_references_in_payload(extracted)
    extracted = _strip_internal_keys(extracted)
    with open(os.path.join(paper_dir, "materials_extracted.extractor_raw.json"), "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)
    print(
        "Extraction complete with "
        f"{(getattr(ext_usage, 'total_tokens', 0) if ext_usage else 0)+sel_usage.total_tokens} total tokens, "
        f"in {sel_time+ext_time:.2f} seconds."
    )
    return {
        "selection": selection,
        "extracted": extracted,
        "metrics": {
            "select": {
                "input_tokens": sel_usage.prompt_tokens,
                "output_tokens": sel_usage.completion_tokens,
                "total_tokens": sel_usage.total_tokens,
                "time_seconds": sel_time,
            },
            "extract": {
                "input_tokens": getattr(ext_usage, "prompt_tokens", 0) if ext_usage else 0,
                "output_tokens": getattr(ext_usage, "completion_tokens", 0) if ext_usage else 0,
                "total_tokens": getattr(ext_usage, "total_tokens", 0) if ext_usage else 0,
                "time_seconds": ext_time,
                "skipped_no_explicit_parameters": skipped_no_explicit_parameters,
            },
            "source_enrichment": {
                "enabled": bool(enable_source_enrichment),
                "applied": source_enrichment_applied,
                "error": source_enrichment_error,
                "input_tokens": getattr(enrich_usage, "prompt_tokens", 0) if enrich_usage else 0,
                "output_tokens": getattr(enrich_usage, "completion_tokens", 0) if enrich_usage else 0,
                "total_tokens": getattr(enrich_usage, "total_tokens", 0) if enrich_usage else 0,
                "time_seconds": enrich_time,
            },
            "selection_resolution": {
                "used_fallback_sections": used_fallback_sections,
                "used_fallback_tables": used_fallback_tables,
                "resolved_section_files": [s["name"] for s in selected_sections],
                "resolved_table_files": [t["name"] for t in selected_tables],
                "resolved_equation_ids": [e.get("selection_id") or e["name"] for e in selected_equations],
            },
            "context": context_meta,
        }
    }
