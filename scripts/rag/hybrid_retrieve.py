import argparse
import json
import os

import psycopg
from psycopg.rows import dict_row
from openai import OpenAI


def to_pgvector(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid retrieval over chunks + extracted parameters")
    ap.add_argument("--query", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=55432)
    ap.add_argument("--dbname", default="cpdb")
    ap.add_argument("--user", default="cpuser")
    ap.add_argument("--password", default="cppassword")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    emb = client.embeddings.create(model=args.embedding_model, input=[args.query]).data[0].embedding
    vec = to_pgvector(emb)

    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        row_factory=dict_row,
    )

    chunk_rows = conn.execute(
        """
        SELECT doi, source_type, source_name, left(text, 280) AS snippet,
               1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (vec, vec, args.k),
    ).fetchall()

    param_rows = conn.execute(
        """
        SELECT doi,
               extracted_json->'material'->>'name' AS material,
               extracted_json->'constitutive_model'->>'framework' AS framework,
               extracted_json->'plastic_parameters'->>'flow_rule' AS flow_rule
        FROM extractions
        WHERE extracted_json::text ILIKE %s
        LIMIT %s;
        """,
        (f"%{args.query}%", args.k),
    ).fetchall()

    out = {
        "query": args.query,
        "chunk_hits": chunk_rows,
        "parameter_hits": param_rows,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    conn.close()


if __name__ == "__main__":
    main()
