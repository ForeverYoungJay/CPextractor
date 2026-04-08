# db/ingest.py
import os, json, glob, time, hashlib
from typing import List, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = Any

from db.vector_builders import build_parameter_vector_rows
from llm.openai_sanitize import sanitize_text_for_openai, validate_openai_json_payload

def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return sanitize_text_for_openai(f.read())

def list_md(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.md")))


def list_table_json(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "table_*.json")))

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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(sanitize_text_for_openai(text).encode("utf-8")).hexdigest()


def _extract_vector_dim(type_name: str | None) -> int | None:
    raw = str(type_name or "").strip().lower()
    if not raw.startswith("vector(") or not raw.endswith(")"):
        return None
    try:
        return int(raw[len("vector("):-1])
    except Exception:
        return None


def ensure_embedding_schema(conn, expected_dim: int) -> None:
    for table in ("chunks", "parameter_vectors", "table_row_vectors"):
        row = conn.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS type_name
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = %s
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            LIMIT 1;
            """,
            (table,),
        ).fetchone()
        actual_dim = _extract_vector_dim(row["type_name"] if row else None)
        if actual_dim is None:
            raise RuntimeError(f"Could not determine embedding column dimension for table {table}")
        if actual_dim != expected_dim:
            raise RuntimeError(
                f"Embedding schema mismatch for {table}. embedding column is vector({actual_dim}), "
                f"but config expects dimension {expected_dim}."
            )

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

def _upsert_chunk_row(
    conn,
    doi: str,
    source_type: str,
    source_name: str,
    text: str,
    metadata: Dict[str, Any],
    embedding_model: str,
) -> Tuple[int, bool]:
    content_hash = _sha256_text(text)
    row = conn.execute(
        """
        SELECT chunk_id, embedding, embedding_model
        FROM chunks
        WHERE doi = %s AND source_type = %s AND source_name = %s AND content_hash = %s
        LIMIT 1;
        """,
        (doi, source_type, source_name, content_hash),
    ).fetchone()
    if row:
        chunk_id = int(row["chunk_id"])
        needs_embedding = row["embedding"] is None or str(row.get("embedding_model") or "") != embedding_model
        conn.execute(
            """
            UPDATE chunks
            SET text = %s,
                metadata = %s::jsonb,
                embedding_model = %s,
                embedding = CASE
                    WHEN embedding_model = %s THEN embedding
                    ELSE NULL
                END
            WHERE chunk_id = %s;
            """,
            (text, json.dumps(metadata, ensure_ascii=False), embedding_model, embedding_model, chunk_id),
        )
        return chunk_id, needs_embedding

    row = conn.execute(
        """
        INSERT INTO chunks (doi, source_type, source_name, text, metadata, content_hash, embedding_model, embedding)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, NULL)
        RETURNING chunk_id;
        """,
        (doi, source_type, source_name, text, json.dumps(metadata, ensure_ascii=False), content_hash, embedding_model),
    ).fetchone()
    return int(row["chunk_id"]), True


def sync_chunk_rows(
    conn,
    doi: str,
    source_type: str,
    source_name: str,
    chunks: List[str],
    metadata: Dict[str, Any],
    embedding_model: str,
) -> List[Tuple[int, str]]:
    to_embed: List[Tuple[int, str]] = []
    keep_ids: List[int] = []
    for idx, ch in enumerate(chunks):
        row_meta = {**metadata, "chunk_index": idx}
        chunk_id, needs_embedding = _upsert_chunk_row(
            conn=conn,
            doi=doi,
            source_type=source_type,
            source_name=source_name,
            text=ch,
            metadata=row_meta,
            embedding_model=embedding_model,
        )
        keep_ids.append(chunk_id)
        if needs_embedding:
            to_embed.append((chunk_id, ch))

    if keep_ids:
        conn.execute(
            """
            DELETE FROM chunks
            WHERE doi = %s
              AND source_type = %s
              AND source_name = %s
              AND NOT (chunk_id = ANY(%s));
            """,
            (doi, source_type, source_name, keep_ids),
        )
    else:
        conn.execute(
            "DELETE FROM chunks WHERE doi = %s AND source_type = %s AND source_name = %s;",
            (doi, source_type, source_name),
        )
    return to_embed

def embed_texts(client: OpenAI, model: str, texts: List[str], max_retries: int = 3) -> List[List[float]]:
    delay = 1.0
    last_exc: Exception | None = None
    request_payload = validate_openai_json_payload({
        "model": model,
        "input": [sanitize_text_for_openai(t) for t in texts],
    })
    for attempt in range(max_retries + 1):
        try:
            resp = client.embeddings.create(**request_payload)
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


def _table_rows_from_json_file(path: str) -> List[Tuple[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []

    table_label = str(payload.get("table_label") or os.path.basename(path)).strip()
    caption = str(payload.get("caption") or "").strip()
    out: List[Tuple[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            continue
        cells = [sanitize_text_for_openai(str(cell).strip()) for cell in row if str(cell).strip()]
        if not cells:
            continue
        text = " | ".join(cells)
        if caption:
            text = f"{table_label} | {caption} | {text}"
        else:
            text = f"{table_label} | {text}"
        out.append((f"row_{idx:04d}", text))
    return out


def _upsert_table_row_vector(
    conn,
    doi: str,
    table_file: str,
    row_key: str,
    row_text: str,
    metadata: Dict[str, Any],
    embedding_model: str,
) -> Tuple[int, bool]:
    content_hash = _sha256_text(row_text)
    existing = conn.execute(
        """
        SELECT row_vector_id, embedding, embedding_model, content_hash
        FROM table_row_vectors
        WHERE doi = %s AND table_file = %s AND row_key = %s
        LIMIT 1;
        """,
        (doi, table_file, row_key),
    ).fetchone()
    if existing:
        row_vector_id = int(existing["row_vector_id"])
        needs_embedding = (
            existing["embedding"] is None
            or str(existing.get("embedding_model") or "") != embedding_model
            or str(existing.get("content_hash") or "") != content_hash
        )
        conn.execute(
            """
            UPDATE table_row_vectors
            SET row_text = %s,
                metadata = %s::jsonb,
                content_hash = %s,
                embedding_model = %s,
                embedding = CASE
                    WHEN content_hash = %s AND embedding_model = %s THEN embedding
                    ELSE NULL
                END
            WHERE row_vector_id = %s;
            """,
            (
                row_text,
                json.dumps(metadata, ensure_ascii=False),
                content_hash,
                embedding_model,
                content_hash,
                embedding_model,
                row_vector_id,
            ),
        )
        return row_vector_id, needs_embedding

    inserted = conn.execute(
        """
        INSERT INTO table_row_vectors (doi, table_file, row_key, row_text, metadata, content_hash, embedding_model, embedding)
        VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,NULL)
        RETURNING row_vector_id;
        """,
        (
            doi,
            table_file,
            row_key,
            row_text,
            json.dumps(metadata, ensure_ascii=False),
            content_hash,
            embedding_model,
        ),
    ).fetchone()
    return int(inserted["row_vector_id"]), True


def sync_table_row_vectors(conn, doi: str, table_file: str, rows: List[Tuple[str, str]], embedding_model: str) -> List[Tuple[int, str]]:
    to_embed: List[Tuple[int, str]] = []
    keep_ids: List[int] = []
    for row_key, row_text in rows:
        row_vector_id, needs_embedding = _upsert_table_row_vector(
            conn=conn,
            doi=doi,
            table_file=table_file,
            row_key=row_key,
            row_text=row_text,
            metadata={"table_file": table_file, "row_key": row_key},
            embedding_model=embedding_model,
        )
        keep_ids.append(row_vector_id)
        if needs_embedding:
            to_embed.append((row_vector_id, row_text))

    if keep_ids:
        conn.execute(
            """
            DELETE FROM table_row_vectors
            WHERE doi = %s AND table_file = %s AND NOT (row_vector_id = ANY(%s));
            """,
            (doi, table_file, keep_ids),
        )
    else:
        conn.execute(
            "DELETE FROM table_row_vectors WHERE doi = %s AND table_file = %s;",
            (doi, table_file),
        )
    return to_embed


def update_table_row_embeddings(conn, pairs: List[Tuple[int, List[float]]]):
    for row_vector_id, emb in pairs:
        vec = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
        conn.execute(
            "UPDATE table_row_vectors SET embedding = %s WHERE row_vector_id = %s;",
            (vec, row_vector_id),
        )


def _upsert_parameter_vector_row(
    conn,
    row: Dict[str, Any],
    embedding_model: str,
) -> Tuple[int, bool]:
    retrieval_text = str(row.get("retrieval_text") or "")
    content_hash = _sha256_text(retrieval_text)
    existing = conn.execute(
        """
        SELECT vector_id, embedding, embedding_model, content_hash
        FROM parameter_vectors
        WHERE doi = %s AND claim_id = %s
        LIMIT 1;
        """,
        (row["doi"], row["claim_id"]),
    ).fetchone()
    if existing:
        vector_id = int(existing["vector_id"])
        needs_embedding = (
            existing["embedding"] is None
            or str(existing.get("embedding_model") or "") != embedding_model
            or str(existing.get("content_hash") or "") != content_hash
        )
        conn.execute(
            """
            UPDATE parameter_vectors
            SET material_id = %s,
                material_name = %s,
                sample_id = %s,
                sample_label = %s,
                condition_id = %s,
                condition_label = %s,
                canonical_name = %s,
                symbol = %s,
                domain = %s,
                phase_id = %s,
                phase_name = %s,
                mechanism = %s,
                family_id = %s,
                family_name = %s,
                model_id = %s,
                value_text = %s,
                unit = %s,
                origin_type = %s,
                evidence_file = %s,
                evidence_kind = %s,
                evidence_snippet = %s,
                retrieval_text = %s,
                metadata = %s::jsonb,
                content_hash = %s,
                embedding_model = %s,
                embedding = CASE
                    WHEN content_hash = %s AND embedding_model = %s THEN embedding
                    ELSE NULL
                END
            WHERE vector_id = %s;
            """,
            (
                row.get("material_id"),
                row.get("material_name"),
                row.get("sample_id"),
                row.get("sample_label"),
                row.get("condition_id"),
                row.get("condition_label"),
                row.get("canonical_name"),
                row.get("symbol"),
                row.get("domain"),
                row.get("phase_id"),
                row.get("phase_name"),
                row.get("mechanism"),
                row.get("family_id"),
                row.get("family_name"),
                row.get("model_id"),
                row.get("value_text"),
                row.get("unit"),
                row.get("origin_type"),
                row.get("evidence_file"),
                row.get("evidence_kind"),
                row.get("evidence_snippet"),
                retrieval_text,
                json.dumps(row.get("metadata") or {}, ensure_ascii=False),
                content_hash,
                embedding_model,
                content_hash,
                embedding_model,
                vector_id,
            ),
        )
        return vector_id, needs_embedding

    inserted = conn.execute(
        """
        INSERT INTO parameter_vectors (
            doi, claim_id, material_id, material_name, sample_id, sample_label, condition_id, condition_label,
            canonical_name, symbol, domain, phase_id, phase_name, mechanism, family_id, family_name, model_id,
            value_text, unit, origin_type, evidence_file, evidence_kind, evidence_snippet,
            retrieval_text, metadata, content_hash, embedding_model, embedding
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,NULL)
        RETURNING vector_id;
        """,
        (
            row["doi"],
            row["claim_id"],
            row.get("material_id"),
            row.get("material_name"),
            row.get("sample_id"),
            row.get("sample_label"),
            row.get("condition_id"),
            row.get("condition_label"),
            row.get("canonical_name"),
            row.get("symbol"),
            row.get("domain"),
            row.get("phase_id"),
            row.get("phase_name"),
            row.get("mechanism"),
            row.get("family_id"),
            row.get("family_name"),
            row.get("model_id"),
            row.get("value_text"),
            row.get("unit"),
            row.get("origin_type"),
            row.get("evidence_file"),
            row.get("evidence_kind"),
            row.get("evidence_snippet"),
            retrieval_text,
            json.dumps(row.get("metadata") or {}, ensure_ascii=False),
            content_hash,
            embedding_model,
        ),
    ).fetchone()
    return int(inserted["vector_id"]), True


def sync_parameter_vectors(conn, doi: str, rows: List[Dict[str, Any]], embedding_model: str) -> List[Tuple[int, str]]:
    to_embed: List[Tuple[int, str]] = []
    keep_ids: List[int] = []
    for row in rows:
        vector_id, needs_embedding = _upsert_parameter_vector_row(conn, row, embedding_model)
        keep_ids.append(vector_id)
        if needs_embedding:
            to_embed.append((vector_id, str(row.get("retrieval_text") or "")))

    if keep_ids:
        conn.execute(
            "DELETE FROM parameter_vectors WHERE doi = %s AND NOT (vector_id = ANY(%s));",
            (doi, keep_ids),
        )
    else:
        conn.execute("DELETE FROM parameter_vectors WHERE doi = %s;", (doi,))
    return to_embed


def update_parameter_embeddings(conn, pairs: List[Tuple[int, List[float]]]):
    for vector_id, emb in pairs:
        vec = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
        conn.execute("UPDATE parameter_vectors SET embedding = %s WHERE vector_id = %s;", (vec, vector_id))

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
    ensure_embedding_schema(conn, embedding_dim)

    # 1) papers + extractions
    source_doc = extracted_json.get("source_document", {}) if isinstance(extracted_json.get("source_document"), dict) else {}
    upsert_paper(
        conn,
        doi=doi,
        title=source_doc.get("title"),
        year=source_doc.get("year"),
        journal=source_doc.get("journal_or_venue"),
    )
    upsert_extraction(conn, doi=doi, extracted_json=extracted_json, model_select=model_select, model_extract=model_extract)

    # 2) chunks from sections + tables, reusing existing embeddings when content hash is unchanged
    to_embed: List[Tuple[int, str]] = []
    table_rows_to_embed: List[Tuple[int, str]] = []

    sec_dir = os.path.join(paper_dir, "sections")
    if os.path.isdir(sec_dir):
        for p in list_md(sec_dir):
            text = read_text(p)
            chunks = chunk_text(text, chunk_chars, chunk_overlap)
            pending = sync_chunk_rows(
                conn=conn,
                doi=doi,
                source_type="section",
                source_name=os.path.basename(p),
                chunks=chunks,
                metadata={"path": os.path.relpath(p, start=paper_dir)},
                embedding_model=embedding_model,
            )
            to_embed.extend(pending)

    tab_dir = os.path.join(paper_dir, "tables")
    if os.path.isdir(tab_dir):
        for p in list_md(tab_dir):
            text = read_text(p)
            chunks = chunk_text(text, chunk_chars, chunk_overlap)
            pending = sync_chunk_rows(
                conn=conn,
                doi=doi,
                source_type="table",
                source_name=os.path.basename(p),
                chunks=chunks,
                metadata={"path": os.path.relpath(p, start=paper_dir)},
                embedding_model=embedding_model,
            )
            to_embed.extend(pending)
        for p in list_table_json(tab_dir):
            pending_rows = sync_table_row_vectors(
                conn=conn,
                doi=doi,
                table_file=os.path.basename(p),
                rows=_table_rows_from_json_file(p),
                embedding_model=embedding_model,
            )
            table_rows_to_embed.extend(pending_rows)

    # 3) parameter-level retrieval vectors
    parameter_rows = build_parameter_vector_rows(doi=doi, extracted_json=extracted_json)
    parameter_to_embed = sync_parameter_vectors(conn, doi=doi, rows=parameter_rows, embedding_model=embedding_model)

    # 4) embeddings (batch) for chunks
    for i in range(0, len(to_embed), batch_size):
        sub = to_embed[i:i+batch_size]
        texts = [t for _, t in sub]
        embs = embed_texts(openai_client, embedding_model, texts, max_retries=embedding_max_retries)

        if embs and len(embs[0]) != embedding_dim:
            raise RuntimeError(f"Embedding dim mismatch: got {len(embs[0])}, expected {embedding_dim}")

        update_embeddings(conn, [(cid, emb) for (cid, _), emb in zip(sub, embs)])
        conn.commit()
        time.sleep(0.05)

    # 5) embeddings (batch) for parameter vectors
    for i in range(0, len(parameter_to_embed), batch_size):
        sub = parameter_to_embed[i:i+batch_size]
        texts = [t for _, t in sub]
        embs = embed_texts(openai_client, embedding_model, texts, max_retries=embedding_max_retries)

        if embs and len(embs[0]) != embedding_dim:
            raise RuntimeError(f"Embedding dim mismatch: got {len(embs[0])}, expected {embedding_dim}")

        update_parameter_embeddings(conn, [(vid, emb) for (vid, _), emb in zip(sub, embs)])
        conn.commit()
        time.sleep(0.05)

    # 6) embeddings (batch) for table-row vectors
    for i in range(0, len(table_rows_to_embed), batch_size):
        sub = table_rows_to_embed[i:i+batch_size]
        texts = [t for _, t in sub]
        embs = embed_texts(openai_client, embedding_model, texts, max_retries=embedding_max_retries)

        if embs and len(embs[0]) != embedding_dim:
            raise RuntimeError(f"Embedding dim mismatch: got {len(embs[0])}, expected {embedding_dim}")

        update_table_row_embeddings(conn, [(row_id, emb) for (row_id, _), emb in zip(sub, embs)])
        conn.commit()
        time.sleep(0.05)


def insert_pipeline_run(
    conn,
    doi: str,
    model_select: str,
    model_extract: str,
    metrics: dict,
    evaluator_metrics: dict | None = None,
    prompt_version: str | None = None,
    schema_version: str | None = None,
    extractor_version: str | None = None,
):
    evaluator_metrics = evaluator_metrics or {}
    eval_input = evaluator_metrics.get("input_tokens")
    eval_output = evaluator_metrics.get("output_tokens")
    eval_total = evaluator_metrics.get("total_tokens")
    eval_time = evaluator_metrics.get("time_seconds")
    total_time = metrics["select"]["time_seconds"] + metrics["extract"]["time_seconds"] + float(eval_time or 0.0)
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
        eval_input,
        eval_output,
        eval_total,
        metrics["select"]["time_seconds"],
        metrics["extract"]["time_seconds"],
        eval_time,
        total_time,
    )
    conn.execute("SAVEPOINT sp_pipeline_run;")
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
                llm_evaluate_input_tokens,
                llm_evaluate_output_tokens,
                llm_evaluate_total_tokens,
                time_select_seconds,
                time_extract_seconds,
                time_evaluate_seconds,
                time_total_seconds
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """,
            params,
        )
        conn.execute("RELEASE SAVEPOINT sp_pipeline_run;")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT sp_pipeline_run;")
        try:
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
                    llm_evaluate_input_tokens,
                    llm_evaluate_output_tokens,
                    llm_evaluate_total_tokens,
                    time_select_seconds,
                    time_extract_seconds,
                    time_evaluate_seconds,
                    time_total_seconds
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
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
                    eval_input,
                    eval_output,
                    eval_total,
                    metrics["select"]["time_seconds"],
                    metrics["extract"]["time_seconds"],
                    eval_time,
                    total_time,
                ),
            )
            conn.execute("RELEASE SAVEPOINT sp_pipeline_run;")
        except Exception:
            conn.execute("ROLLBACK TO SAVEPOINT sp_pipeline_run;")
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
            conn.execute("RELEASE SAVEPOINT sp_pipeline_run;")
