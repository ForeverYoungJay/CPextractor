from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def apply_schema(conn, schema_path: str | Path | None = None) -> None:
    path = Path(schema_path) if schema_path else Path(__file__).with_name("schema.sql")
    conn.execute(path.read_text(encoding="utf-8"))


def sync_graph_to_postgres(conn, graph: dict[str, Any], *, clear_existing: bool = False) -> None:
    """Upsert KG nodes and edges into Postgres tables created by kg/schema.sql."""
    if clear_existing:
        conn.execute("DELETE FROM kg_edges;")
        conn.execute("DELETE FROM kg_nodes;")

    for node in graph.get("nodes", []):
        conn.execute(
            """
            INSERT INTO kg_nodes (node_id, node_type, label, properties)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (node_id) DO UPDATE
            SET node_type = EXCLUDED.node_type,
                label = EXCLUDED.label,
                properties = EXCLUDED.properties;
            """,
            (
                node["node_id"],
                node["node_type"],
                node.get("label"),
                json.dumps(node.get("properties") or {}, ensure_ascii=False),
            ),
        )

    for edge in graph.get("edges", []):
        conn.execute(
            """
            INSERT INTO kg_edges (
              edge_id, source_id, target_id, edge_type, doi, claim_id,
              confidence_score, evidence_ids, properties
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (edge_id) DO UPDATE
            SET source_id = EXCLUDED.source_id,
                target_id = EXCLUDED.target_id,
                edge_type = EXCLUDED.edge_type,
                doi = EXCLUDED.doi,
                claim_id = EXCLUDED.claim_id,
                confidence_score = EXCLUDED.confidence_score,
                evidence_ids = EXCLUDED.evidence_ids,
                properties = EXCLUDED.properties;
            """,
            (
                edge["edge_id"],
                edge["source_id"],
                edge["target_id"],
                edge["edge_type"],
                edge.get("doi"),
                edge.get("claim_id"),
                edge.get("confidence_score"),
                json.dumps(edge.get("evidence_ids") or [], ensure_ascii=False),
                json.dumps(edge.get("properties") or {}, ensure_ascii=False),
            ),
        )
