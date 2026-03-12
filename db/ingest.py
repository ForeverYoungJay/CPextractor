# db/ingest.py
import os, json, glob, time
from typing import List, Dict, Any, Tuple
from openai import OpenAI

def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def list_md(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.md")))

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    step = max(1, chunk_chars - overlap)
    out = []
    for i in range(0, len(text), step):
        ch = text[i:i+chunk_chars].strip()
        if ch:
            out.append(ch)
    return out

def upsert_paper(conn, doi: str, title: str | None = None, year: int | None = None, journal: str | None = None):
    conn.execute(
        """
        INSERT INTO papers (doi, title, year, journal)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (doi) DO UPDATE
        SET title = COALESCE(EXCLUDED.title, papers.title),
            year = COALESCE(EXCLUDED.year, papers.year),
            journal = COALESCE(EXCLUDED.journal, papers.journal);
        """,
        (doi, title, year, journal),
    )

def upsert_extraction(conn, doi: str, extracted_json: Dict[str, Any], model_select: str, model_extract: str):
    conn.execute(
        """
        INSERT INTO extractions (doi, extracted_json, model_select, model_extract)
        VALUES (%s, %s::jsonb, %s, %s)
        ON CONFLICT (doi) DO UPDATE
        SET extracted_json = EXCLUDED.extracted_json,
            model_select = EXCLUDED.model_select,
            model_extract = EXCLUDED.model_extract;
        """,
        (doi, json.dumps(extracted_json, ensure_ascii=False), model_select, model_extract),
    )

def insert_chunk_rows(conn, doi: str, source_type: str, source_name: str, chunks: List[str], metadata: Dict[str, Any]) -> List[int]:
    ids = []
    for idx, ch in enumerate(chunks):
        row = conn.execute(
            """
            INSERT INTO chunks (doi, source_type, source_name, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::jsonb, NULL)
            RETURNING chunk_id;
            """,
            (doi, source_type, source_name, ch,
             json.dumps({**metadata, "chunk_index": idx}, ensure_ascii=False)),
        ).fetchone()
        ids.append(int(row["chunk_id"]))
    return ids

def embed_texts(client: OpenAI, model: str, texts: List[str], max_retries: int = 3) -> List[List[float]]:
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Embedding request failed after retries: {last_exc}")

def update_embeddings(conn, pairs: List[Tuple[int, List[float]]]):
    for chunk_id, emb in pairs:
        vec = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
        conn.execute("UPDATE chunks SET embedding = %s WHERE chunk_id = %s;", (vec, chunk_id))

def ingest_paper_dir_to_db(
    conn,
    openai_client: OpenAI,
    doi: str,
    paper_dir: str,
    extracted_json: Dict[str, Any],
    model_select: str,
    model_extract: str,
    embedding_model: str,
    embedding_dim: int,
    chunk_chars: int,
    chunk_overlap: int,
    batch_size: int,
    embedding_max_retries: int = 3,
):
    # 1) papers + extractions
    title = None
    # optional: if your extraction includes a title field, set it here
    upsert_paper(conn, doi=doi, title=title)
    upsert_extraction(conn, doi=doi, extracted_json=extracted_json, model_select=model_select, model_extract=model_extract)

    # 2) chunks from sections + tables
    conn.execute("DELETE FROM chunks WHERE doi = %s;", (doi,))
    to_embed: List[Tuple[int, str]] = []

    sec_dir = os.path.join(paper_dir, "sections")
    if os.path.isdir(sec_dir):
        for p in list_md(sec_dir):
            text = read_text(p)
            chunks = chunk_text(text, chunk_chars, chunk_overlap)
            ids = insert_chunk_rows(
                conn, doi, "section", os.path.basename(p),
                chunks, {"path": os.path.relpath(p, start=paper_dir)}
            )
            to_embed.extend(list(zip(ids, chunks)))

    tab_dir = os.path.join(paper_dir, "tables")
    if os.path.isdir(tab_dir):
        for p in list_md(tab_dir):
            text = read_text(p)
            chunks = chunk_text(text, chunk_chars, chunk_overlap)
            ids = insert_chunk_rows(
                conn, doi, "table", os.path.basename(p),
                chunks, {"path": os.path.relpath(p, start=paper_dir)}
            )
            to_embed.extend(list(zip(ids, chunks)))

    # 3) embeddings (batch)
    for i in range(0, len(to_embed), batch_size):
        sub = to_embed[i:i+batch_size]
        texts = [t for _, t in sub]
        embs = embed_texts(openai_client, embedding_model, texts, max_retries=embedding_max_retries)

        if embs and len(embs[0]) != embedding_dim:
            raise RuntimeError(f"Embedding dim mismatch: got {len(embs[0])}, expected {embedding_dim}")

        update_embeddings(conn, [(cid, emb) for (cid, _), emb in zip(sub, embs)])
        conn.commit()
        time.sleep(0.05)


def insert_pipeline_run(
    conn,
    doi: str,
    model_select: str,
    model_extract: str,
    metrics: dict,
    prompt_version: str | None = None,
    schema_version: str | None = None,
    extractor_version: str | None = None,
):
    params = (
        doi,
        model_select,
        model_extract,
        prompt_version,
        schema_version,
        extractor_version,
        metrics["select"]["input_tokens"],
        metrics["select"]["output_tokens"],
        metrics["select"]["total_tokens"],
        metrics["extract"]["input_tokens"],
        metrics["extract"]["output_tokens"],
        metrics["extract"]["total_tokens"],
        metrics["select"]["time_seconds"],
        metrics["extract"]["time_seconds"],
        metrics["select"]["time_seconds"] + metrics["extract"]["time_seconds"],
    )
    try:
        conn.execute(
            """
            INSERT INTO pipeline_runs (
                doi,
                model_select,
                model_extract,
                prompt_version,
                schema_version,
                extractor_version,
                llm_select_input_tokens,
                llm_select_output_tokens,
                llm_select_total_tokens,
                llm_extract_input_tokens,
                llm_extract_output_tokens,
                llm_extract_total_tokens,
                time_select_seconds,
                time_extract_seconds,
                time_total_seconds
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """,
            params,
        )
    except Exception:
        # Backward compatibility for existing DBs not yet migrated with lineage columns.
        conn.execute(
            """
            INSERT INTO pipeline_runs (
                doi,
                model_select,
                model_extract,
                llm_select_input_tokens,
                llm_select_output_tokens,
                llm_select_total_tokens,
                llm_extract_input_tokens,
                llm_extract_output_tokens,
                llm_extract_total_tokens,
                time_select_seconds,
                time_extract_seconds,
                time_total_seconds
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """,
            (
                doi,
                model_select,
                model_extract,
                metrics["select"]["input_tokens"],
                metrics["select"]["output_tokens"],
                metrics["select"]["total_tokens"],
                metrics["extract"]["input_tokens"],
                metrics["extract"]["output_tokens"],
                metrics["extract"]["total_tokens"],
                metrics["select"]["time_seconds"],
                metrics["extract"]["time_seconds"],
                metrics["select"]["time_seconds"] + metrics["extract"]["time_seconds"],
            ),
        )
