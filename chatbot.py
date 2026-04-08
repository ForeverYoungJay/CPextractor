import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg
import yaml
from psycopg.rows import dict_row
from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a CP (crystal plasticity) scientific assistant. "
    "Answer only with information supported by provided evidence. "
    "If evidence is insufficient, say uncertainty clearly. "
    "Always return valid JSON."
)


def to_pgvector(v: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def connect_db(db_cfg: Dict[str, Any]) -> psycopg.Connection:
    return psycopg.connect(
        host=db_cfg.get("host", "localhost"),
        port=int(db_cfg.get("port", 55432)),
        dbname=db_cfg.get("name", "cpdb"),
        user=db_cfg.get("user", "cpuser"),
        password=db_cfg.get("password", "cppassword"),
        row_factory=dict_row,
    )


def retrieve_chunks(conn: psycopg.Connection, vec: str, k: int) -> List[Dict[str, Any]]:
    return conn.execute(
        """
        SELECT doi, source_type, source_name, left(text, 1000) AS snippet,
               1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (vec, vec, k),
    ).fetchall()


def retrieve_parameter_vectors(conn: psycopg.Connection, vec: str, k: int) -> List[Dict[str, Any]]:
    return conn.execute(
        """
        SELECT
          doi,
          claim_id,
          material_id,
          material_name,
          sample_id,
          sample_label,
          condition_id,
          condition_label,
          canonical_name,
          symbol,
          domain,
          phase_id,
          phase_name,
          mechanism,
          family_id,
          family_name,
          model_id,
          value_text,
          unit,
          origin_type,
          evidence_file,
          evidence_kind,
          left(evidence_snippet, 1000) AS evidence_snippet,
          left(retrieval_text, 1000) AS retrieval_text,
          1 - (embedding <=> %s::vector) AS score
        FROM parameter_vectors
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (vec, vec, k),
    ).fetchall()


def retrieve_table_row_vectors(conn: psycopg.Connection, vec: str, k: int) -> List[Dict[str, Any]]:
    return conn.execute(
        """
        SELECT doi, table_file, row_key, left(row_text, 1000) AS row_text,
               1 - (embedding <=> %s::vector) AS score
        FROM table_row_vectors
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (vec, vec, k),
    ).fetchall()


def _safe_like_query(query: str) -> str:
    # Normalize whitespace and trim very long input to keep SQL fast/stable.
    q = re.sub(r"\s+", " ", query).strip()
    return q[:300]


def retrieve_structured(conn: psycopg.Connection, query: str, k: int) -> List[Dict[str, Any]]:
    q = _safe_like_query(query)
    like = f"%{q}%"
    return conn.execute(
        """
        SELECT
          e.doi,
          COALESCE(
            (
              SELECT m.item->>'material_id'
              FROM jsonb_array_elements(COALESCE(e.extracted_json->'materials', '[]'::jsonb)) AS m(item)
              WHERE m.item->>'material_id' = COALESCE(c.item->'applies_to'->>'material_id', '')
              LIMIT 1
            ),
            c.item->'applies_to'->>'material_id'
          ) AS material_id,
          COALESCE(
            (
              SELECT m.item->>'name'
              FROM jsonb_array_elements(COALESCE(e.extracted_json->'materials', '[]'::jsonb)) AS m(item)
              WHERE m.item->>'material_id' = COALESCE(c.item->'applies_to'->>'material_id', '')
              LIMIT 1
            ),
            (
              SELECT m.item->>'name'
              FROM jsonb_array_elements(COALESCE(e.extracted_json->'materials', '[]'::jsonb)) AS m(item)
              LIMIT 1
            ),
            e.extracted_json->'material'->>'name'
          ) AS material,
          COALESCE(
            (
              SELECT s.item->>'sample_id'
              FROM jsonb_array_elements(COALESCE(e.extracted_json->'samples', '[]'::jsonb)) AS s(item)
              WHERE s.item->>'sample_id' = COALESCE(c.item->'applies_to'->>'sample_id', '')
              LIMIT 1
            ),
            c.item->'applies_to'->>'sample_id'
          ) AS sample_id,
          COALESCE(
            (
              SELECT s.item->>'label'
              FROM jsonb_array_elements(COALESCE(e.extracted_json->'samples', '[]'::jsonb)) AS s(item)
              WHERE s.item->>'sample_id' = COALESCE(c.item->'applies_to'->>'sample_id', '')
              LIMIT 1
            ),
            ''
          ) AS sample_label,
          COALESCE(
            (
              SELECT d.item->>'condition_id'
              FROM jsonb_array_elements(COALESCE(e.extracted_json->'conditions', '[]'::jsonb)) AS d(item)
              WHERE d.item->>'condition_id' = COALESCE(c.item->'applies_to'->>'condition_id', '')
              LIMIT 1
            ),
            c.item->'applies_to'->>'condition_id'
          ) AS condition_id,
          COALESCE(
            (
              SELECT d.item->>'label'
              FROM jsonb_array_elements(COALESCE(e.extracted_json->'conditions', '[]'::jsonb)) AS d(item)
              WHERE d.item->>'condition_id' = COALESCE(c.item->'applies_to'->>'condition_id', '')
              LIMIT 1
            ),
            ''
          ) AS condition_label,
          COALESCE(
            (
              SELECT m.item->>'framework'
              FROM jsonb_array_elements(COALESCE(e.extracted_json->'models', '[]'::jsonb)) AS m(item)
              WHERE m.item->>'model_id' = COALESCE(c.item->>'model_id', '')
              LIMIT 1
            ),
            (
              SELECT m.item->>'framework'
              FROM jsonb_array_elements(COALESCE(e.extracted_json->'models', '[]'::jsonb)) AS m(item)
              LIMIT 1
            ),
            e.extracted_json->'constitutive_model'->>'framework'
          ) AS framework,
          c.item->>'claim_id' AS claim_id,
          c.item->>'canonical_name' AS canonical_name,
          c.item->>'symbol' AS symbol,
          c.item->>'value' AS value,
          c.item->>'unit' AS unit,
          c.item->'applies_to'->>'scope' AS scope,
          c.item->'applies_to'->>'phase_id' AS phase_id,
          COALESCE(
            (
              SELECT p.item->>'name'
              FROM jsonb_array_elements(COALESCE(e.extracted_json->'materials', '[]'::jsonb)) AS m(item)
              CROSS JOIN LATERAL jsonb_array_elements(COALESCE(m.item->'phases', '[]'::jsonb)) AS p(item)
              WHERE p.item->>'phase_id' = COALESCE(c.item->'applies_to'->>'phase_id', '')
              LIMIT 1
            ),
            ''
          ) AS phase_name,
          c.item->'applies_to'->>'family_id' AS family_id,
          COALESCE(c.item->'provenance'->>'origin_type', c.item->'source'->>'origin_type') AS origin_type,
          COALESCE(
            c.item->'evidence'->>'evidence_text',
            c.item->'evidence'->'table_evidence'->>'excerpt',
            c.item->'evidence'->'table_evidence'->>'value'
          ) AS evidence_text
        FROM extractions e
        LEFT JOIN LATERAL jsonb_array_elements(
          COALESCE(e.extracted_json->'parameter_claims', '[]'::jsonb)
        ) AS c(item) ON TRUE
        WHERE
          (e.extracted_json::text ILIKE %s)
          OR (COALESCE(c.item->>'canonical_name', '') ILIKE %s)
          OR (COALESCE(c.item->>'symbol', '') ILIKE %s)
          OR (COALESCE(c.item->>'description', '') ILIKE %s)
        LIMIT %s;
        """,
        (like, like, like, like, k),
    ).fetchall()


def build_user_prompt(
    query: str,
    chunk_hits: List[Dict[str, Any]],
    parameter_hits: List[Dict[str, Any]],
    table_row_hits: List[Dict[str, Any]],
    structured_hits: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    evidence: Dict[str, Any] = {}
    lines: List[str] = [f"Question:\n{query}\n", "Evidence:"]

    for i, h in enumerate(chunk_hits, start=1):
        eid = f"C{i}"
        evidence[eid] = {"type": "chunk", **h}
        lines.append(
            f"[{eid}] doi={h.get('doi')} source={h.get('source_type')}:{h.get('source_name')} score={h.get('score')}\n"
            f"{h.get('snippet')}\n"
        )

    for i, h in enumerate(parameter_hits, start=1):
        eid = f"P{i}"
        evidence[eid] = {"type": "parameter_vector", **h}
        lines.append(
            f"[{eid}] doi={h.get('doi')} claim_id={h.get('claim_id')} "
            f"material={h.get('material_name') or h.get('material_id')} "
            f"sample={h.get('sample_label') or h.get('sample_id')} "
            f"condition={h.get('condition_label') or h.get('condition_id')} "
            f"param={h.get('canonical_name')}/{h.get('symbol')} value={h.get('value_text')} {h.get('unit')} "
            f"phase={h.get('phase_name') or h.get('phase_id')} family={h.get('family_name') or h.get('family_id')} "
            f"origin={h.get('origin_type')} score={h.get('score')}\n"
            f"evidence={h.get('evidence_kind')}:{h.get('evidence_file')} :: {h.get('evidence_snippet')}\n"
        )

    for i, h in enumerate(table_row_hits, start=1):
        eid = f"T{i}"
        evidence[eid] = {"type": "table_row_vector", **h}
        lines.append(
            f"[{eid}] doi={h.get('doi')} table={h.get('table_file')} row={h.get('row_key')} score={h.get('score')}\n"
            f"{h.get('row_text')}\n"
        )

    for i, h in enumerate(structured_hits, start=1):
        eid = f"S{i}"
        evidence[eid] = {"type": "structured", **h}
        lines.append(
            f"[{eid}] doi={h.get('doi')} material={h.get('material')} framework={h.get('framework')} "
            f"material_id={h.get('material_id')} sample={h.get('sample_label') or h.get('sample_id')} "
            f"condition={h.get('condition_label') or h.get('condition_id')} "
            f"claim_id={h.get('claim_id')} param={h.get('canonical_name')}/{h.get('symbol')} value={h.get('value')} {h.get('unit')} "
            f"scope={h.get('scope')} phase={h.get('phase_name') or h.get('phase_id')} family={h.get('family_id')} "
            f"origin={h.get('origin_type')}\n"
            f"evidence_text={h.get('evidence_text')}\n"
        )

    lines.append(
        "Return JSON only with keys: "
        "answer, confidence(high/medium/low), evidence_ids(array), "
        "claims(array of {claim,evidence_ids,doi}), gaps(array)."
    )
    return "\n".join(lines), evidence


def call_llm(
    client: OpenAI,
    llm_model: str,
    query: str,
    chunk_hits: List[Dict[str, Any]],
    parameter_hits: List[Dict[str, Any]],
    table_row_hits: List[Dict[str, Any]],
    structured_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt, evidence_map = build_user_prompt(query, chunk_hits, parameter_hits, table_row_hits, structured_hits)
    resp = client.chat.completions.create(
        model=llm_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    content = resp.choices[0].message.content or "{}"
    parsed = json.loads(content)
    parsed["query"] = query
    parsed["retrieval"] = {
        "chunk_hits": chunk_hits,
        "parameter_hits": parameter_hits,
        "table_row_hits": table_row_hits,
        "structured_hits": structured_hits,
    }
    parsed["evidence_map"] = evidence_map
    return parsed


def retrieve_only_payload(
    query: str,
    chunk_hits: List[Dict[str, Any]],
    parameter_hits: List[Dict[str, Any]],
    table_row_hits: List[Dict[str, Any]],
    structured_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "query": query,
        "answer": "retrieve_only mode: skipped LLM synthesis",
        "confidence": "low",
        "evidence_ids": [],
        "claims": [],
        "gaps": ["LLM synthesis disabled"],
        "retrieval": {
            "chunk_hits": chunk_hits,
            "parameter_hits": parameter_hits,
            "table_row_hits": table_row_hits,
            "structured_hits": structured_hits,
        },
    }


def ensure_parent_dir(path: str) -> None:
    p = Path(path)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)


def run_one(
    conn: psycopg.Connection,
    client: Optional[OpenAI],
    query: str,
    embedding_model: str,
    llm_model: str,
    k_chunks: int,
    k_params: int,
    k_struct: int,
    retrieve_only: bool,
) -> Dict[str, Any]:
    if not retrieve_only and client is None:
        raise RuntimeError("OPENAI_API_KEY is required unless --retrieve-only is used.")

    if client is None:
        # Fallback retrieval if no embedding client in retrieve-only mode.
        chunk_hits: List[Dict[str, Any]] = conn.execute(
            """
            SELECT doi, source_type, source_name, left(text, 1000) AS snippet, NULL::float AS score
            FROM chunks
            WHERE text ILIKE %s
            LIMIT %s;
            """,
            (f"%{_safe_like_query(query)}%", k_chunks),
        ).fetchall()
        parameter_hits: List[Dict[str, Any]] = conn.execute(
            """
            SELECT
              doi, claim_id, canonical_name, symbol, domain, phase_id, mechanism, family_id, family_name,
              value_text, unit, origin_type, evidence_file, evidence_kind,
              left(evidence_snippet, 1000) AS evidence_snippet,
              left(retrieval_text, 1000) AS retrieval_text,
              NULL::float AS score
            FROM parameter_vectors
            WHERE retrieval_text ILIKE %s
            LIMIT %s;
            """,
            (f"%{_safe_like_query(query)}%", k_params),
        ).fetchall()
        table_row_hits: List[Dict[str, Any]] = conn.execute(
            """
            SELECT doi, table_file, row_key, left(row_text, 1000) AS row_text, NULL::float AS score
            FROM table_row_vectors
            WHERE row_text ILIKE %s
            LIMIT %s;
            """,
            (f"%{_safe_like_query(query)}%", k_params),
        ).fetchall()
    else:
        emb = client.embeddings.create(model=embedding_model, input=[query]).data[0].embedding
        vec = to_pgvector(emb)
        chunk_hits = retrieve_chunks(conn, vec, k_chunks)
        parameter_hits = retrieve_parameter_vectors(conn, vec, k_params)
        table_row_hits = retrieve_table_row_vectors(conn, vec, k_params)

    structured_hits = retrieve_structured(conn, query, k_struct)

    if retrieve_only:
        return retrieve_only_payload(query, chunk_hits, parameter_hits, table_row_hits, structured_hits)
    return call_llm(client, llm_model, query, chunk_hits, parameter_hits, table_row_hits, structured_hits)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CP chatbot: vector + structured retrieval + evidence-grounded answer")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--query", default="", help="Single-shot question; if empty enters interactive mode")
    ap.add_argument("--embedding-model", default="")
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--k-chunks", type=int, default=8)
    ap.add_argument("--k-params", type=int, default=12)
    ap.add_argument("--k-struct", type=int, default=8)
    ap.add_argument("--retrieve-only", action="store_true")
    ap.add_argument("--output", default="", help="Optional JSON output path (single-shot mode)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    db_cfg = cfg.get("db", {})
    rag_cfg = cfg.get("rag", {})
    llm_cfg = cfg.get("llm", {})

    embedding_model = args.embedding_model or rag_cfg.get("embedding_model", "text-embedding-3-small")
    llm_model = args.llm_model or llm_cfg.get("model_extract", "gpt-4.1-mini")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    client: Optional[OpenAI] = None
    if not args.retrieve_only:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)
    elif api_key:
        client = OpenAI(api_key=api_key)

    conn = connect_db(db_cfg)

    try:
        if args.query:
            result = run_one(
                conn=conn,
                client=client,
                query=args.query,
                embedding_model=embedding_model,
                llm_model=llm_model,
                k_chunks=args.k_chunks,
                k_params=args.k_params,
                k_struct=args.k_struct,
                retrieve_only=args.retrieve_only,
            )
            text = json.dumps(result, ensure_ascii=False, indent=2)
            if args.output:
                ensure_parent_dir(args.output)
                Path(args.output).write_text(text, encoding="utf-8")
                print(f"Saved -> {args.output}")
            else:
                print(text)
            return

        print("CP chatbot interactive mode. Type ':quit' to exit.")
        while True:
            query = input("\nYou> ").strip()
            if not query:
                continue
            if query.lower() in {":quit", ":q", "quit", "exit"}:
                break

            try:
                result = run_one(
                    conn=conn,
                    client=client,
                    query=query,
                    embedding_model=embedding_model,
                    llm_model=llm_model,
                    k_chunks=args.k_chunks,
                    k_params=args.k_params,
                    k_struct=args.k_struct,
                    retrieve_only=args.retrieve_only,
                )
            except Exception as e:
                print(f"Error: {e}")
                continue

            answer = result.get("answer")
            confidence = result.get("confidence")
            evidence_ids = result.get("evidence_ids", [])

            print(f"\nBot> {answer}")
            print(f"Confidence: {confidence}")
            print(f"Evidence IDs: {evidence_ids}")

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
