from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


_COLUMN_CANDIDATES: Dict[str, Sequence[str]] = {
    "doi": ("doi",),
    "claim_id": ("claim_id",),
    "claim_class": ("claim_class",),
    "material_id": ("material_id",),
    "material_name": ("material_name",),
    "process_state_id": ("process_state_id", "sample_id"),
    "process_state_name": ("process_state_name", "sample_label"),
    "condition_id": ("condition_id",),
    "condition_label": ("condition_label",),
    "canonical_name": ("canonical_name",),
    "symbol": ("symbol",),
    "domain": ("domain",),
    "constituent_id": ("constituent_id", "phase_id"),
    "constituent_name": ("constituent_name", "phase_name"),
    "mechanism": ("mechanism",),
    "family_id": ("family_id",),
    "family_name": ("family_name",),
    "model_id": ("model_id",),
    "branch_id": ("branch_id",),
    "value_text": ("value_text",),
    "unit": ("unit",),
    "origin_type": ("origin_type",),
    "evidence_file": ("evidence_file",),
    "evidence_kind": ("evidence_kind",),
    "evidence_snippet": ("evidence_snippet",),
    "retrieval_text": ("retrieval_text",),
    "metadata": ("metadata",),
    "content_hash": ("content_hash",),
    "embedding_model": ("embedding_model",),
    "system_ids": ("system_ids",),
}

_TEXT_CANONICAL_FIELDS: Tuple[str, ...] = (
    "doi",
    "claim_id",
    "claim_class",
    "material_id",
    "material_name",
    "process_state_id",
    "process_state_name",
    "condition_id",
    "condition_label",
    "canonical_name",
    "symbol",
    "domain",
    "constituent_id",
    "constituent_name",
    "mechanism",
    "family_id",
    "family_name",
    "model_id",
    "branch_id",
    "value_text",
    "unit",
    "origin_type",
    "evidence_file",
    "evidence_kind",
    "evidence_snippet",
    "retrieval_text",
)


def get_table_columns(conn: Any, table_name: str) -> List[str]:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
        ORDER BY ordinal_position;
        """,
        (table_name,),
    ).fetchall()
    return [str(row["column_name"]) for row in rows if row.get("column_name")]


def resolve_row_value(row: Dict[str, Any], canonical_field: str) -> Any:
    for candidate in _COLUMN_CANDIDATES.get(canonical_field, (canonical_field,)):
        if candidate in row and row.get(candidate) is not None:
            return row.get(candidate)
    return None


def build_parameter_vector_payload(
    row: Dict[str, Any],
    available_columns: Iterable[str],
) -> List[Tuple[str, Any]]:
    payload: List[Tuple[str, Any]] = []
    for raw_column in available_columns:
        column = str(raw_column)
        value = resolve_row_value(row, column)
        if column == "metadata" and value is None:
            value = {}
        payload.append((column, value))
    return payload


def build_select_projection(
    available_columns: Iterable[str],
    table_alias: str = "",
    snippet_fields: Iterable[str] | None = None,
    snippet_limit: int | None = None,
) -> str:
    available = {str(col) for col in available_columns}
    prefix = f"{table_alias}." if table_alias else ""
    snippet_targets = set(snippet_fields or [])
    parts: List[str] = []
    for field in _TEXT_CANONICAL_FIELDS:
        source_col = next((c for c in _COLUMN_CANDIDATES.get(field, (field,)) if c in available), None)
        if source_col:
            expr = f"{prefix}{source_col}"
            if field in snippet_targets and snippet_limit is not None:
                expr = f"left({expr}, {int(snippet_limit)})"
            parts.append(f"{expr} AS {field}")
        else:
            parts.append(f"NULL::text AS {field}")
    if "system_ids" in available:
        parts.append(f"{prefix}system_ids AS system_ids")
    else:
        parts.append("NULL::jsonb AS system_ids")
    return ",\n          ".join(parts)
