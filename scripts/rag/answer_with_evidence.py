import argparse
import json
import os
from typing import Dict, List

import psycopg
from psycopg.rows import dict_row
from openai import OpenAI


def to_pgvector(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def retrieve(conn, vec: str, k: int) -> List[Dict]:
    return conn.execute(
        """
        SELECT doi, source_type, source_name, left(text, 900) AS snippet,
               1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (vec, vec, k),
    ).fetchall()


def build_prompt(query: str, hits: List[Dict]) -> str:
    parts = [f"Question:\n{query}\n", "Evidence chunks:"]
    for i, h in enumerate(hits, start=1):
        parts.append(
            f"[E{i}] doi={h.get('doi')} source={h.get('source_type')}:{h.get('source_name')} score={h.get('score')}\n"
            f"{h.get('snippet')}\n"
        )
    parts.append(
        "Output JSON only with keys: answer, confidence(high/medium/low), evidence_ids(list like ['E1','E2']), "
        "claims(list of {claim,evidence_ids,doi}). "
        "If evidence is insufficient, answer must say uncertain."
    )
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG answer with explicit evidence IDs.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=55432)
    ap.add_argument("--dbname", default="cpdb")
    ap.add_argument("--user", default="cpuser")
    ap.add_argument("--password", default="cppassword")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--llm-model", default="gpt-4.1-mini")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--output", default="")
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
    hits = retrieve(conn, vec, args.k)
    conn.close()

    prompt = build_prompt(args.query, hits)
    resp = client.chat.completions.create(
        model=args.llm_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a scientific QA assistant. Never answer without citing provided evidence IDs."},
            {"role": "user", "content": prompt},
        ],
    )
    out = json.loads(resp.choices[0].message.content)
    out["retrieval"] = hits
    out["query"] = args.query

    txt = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"Saved -> {args.output}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
