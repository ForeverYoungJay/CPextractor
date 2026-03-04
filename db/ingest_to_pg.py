import os
import json
import glob
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple

import yaml
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI


# ----------------------------
# Config + Utilities
# ----------------------------

@dataclass
class DBConfig:
    host: str
    port: int
    name: str
    user: str
    password: str

@dataclass
class RagConfig:
    embedding_model: str
    embedding_dim: int
    chunk_chars: int
    chunk_overlap: int
    batch_size: int

def load_config(path="config.yaml") -> Tuple[DBConfig, RagConfig, str]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    db = cfg["db"]
    rag = cfg["rag"]
    fulltext_root = cfg["paths"]["fulltext"]

    return (
        DBConfig(db["host"], int(db["port"]), db["name"], db["user"], db["password"]),
        RagConfig(rag["embedding_model"], int(rag["embedding_dim"]), int(rag["chunk_chars"]),
                  int(rag["chunk_overlap"]), int(rag["batch_size"])),
        fulltext_root,
    )

def connect_db(db: DBConfig) -> psycopg.Connection:
    conn = psycopg.connect(
        host=db.host,
        port=db.port,
        dbname=db.name,
        user=db.user,
        password=db.password,
        row_factory=dict_row,
    )
    conn.execute("SET statement_timeout TO '5min';")
    return conn

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_md_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.md")))

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    step = max(1, chunk_chars - overlap)
    while i < n:
        chunk = text[i:i+chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


# ----------------------------
# DB operations
# ----------------------------

def upsert_paper(conn: psycopg.Connection, doi: str, title: str = None, year: int = None, journal: str = None):
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

def upsert_extraction(conn: psycopg.Connection, doi: str, extracted_json: Dict[str, Any],
                     model_select: str = None, model_extract: str = None):
    conn.execute(
        """
        INSERT INTO extractions (doi, extracted_json, model_select, model_extract)
        VALUES (%s, %s::jsonb, %s, %s)
        ON CONFLICT (doi) DO UPDATE
        SET extracted_json = EXCLUDED.extracted_json,
            model_select = COALESCE(EXCLUDED.model_select, extractions.model_select),
            model_extract = COALESCE(EXCLUDED.model_extract, extractions.model_extract);
        """,
        (doi, json.dumps(extracted_json, ensure_ascii=False), model_select, model_extract),
    )

def insert_chunks(conn: psycopg.Connection, doi: str, source_type: str, source_name: str,
                  chunks: List[str], metadata: Dict[str, Any]) -> List[int]:
    """
    Insert chunks without embeddings first. Returns chunk_ids.
    """
    chunk_ids = []
    for idx, ch in enumerate(chunks):
        row = conn.execute(
            """
            INSERT INTO chunks (doi, source_type, source_name, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::jsonb, NULL)
            RETURNING chunk_id;
            """,
            (doi, source_type, source_name, ch, json.dumps({**metadata, "chunk_index": idx}, ensure_ascii=False)),
        ).fetchone()
        chunk_ids.append(int(row["chunk_id"]))
    return chunk_ids

def update_embeddings(conn: psycopg.Connection, pairs: List[Tuple[int, List[float]]]):
    """
    pairs: [(chunk_id, embedding_list), ...]
    """
    # pgvector accepts array-ish string format: '[0.1, 0.2, ...]'
    for chunk_id, emb in pairs:
        vec = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
        conn.execute(
            "UPDATE chunks SET embedding = %s WHERE chunk_id = %s;",
            (vec, chunk_id),
        )


# ----------------------------
# Embeddings
# ----------------------------

def embed_texts(client: OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    # keep order
    return [d.embedding for d in resp.data]


# ----------------------------
# Ingest one DOI directory
# ----------------------------

def infer_doi_from_dirname(dirname: str) -> str:
    # Your dirs are safe_id(doi): e.g., 10.1016_j.jmrt.2025.11.003
    # Best: store DOI in a file at generation time.
    # For now, we keep dir name as id and store it in papers.doi as-is if DOI unknown.
    return dirname.replace("_", "/") if dirname.startswith("10.") else dirname

def ingest_one_paper(conn: psycopg.Connection, client: OpenAI, rag: RagConfig,
                     paper_dir: str, doi: str):
    # 1) extracted JSON (optional)
    extracted_path = os.path.join(paper_dir, "materials_extracted.json")
    extracted_json = None
    if os.path.exists(extracted_path):
        extracted_json = read_json(extracted_path)

    # 2) minimal metadata
    title = None
    year = None
    journal = None
    if extracted_json:
        # best-effort; safe if missing
        title = extracted_json.get("title") or None
        # year/journal often not in your extraction schema; keep placeholders

    upsert_paper(conn, doi=doi, title=title, year=year, journal=journal)

    if extracted_json:
        upsert_extraction(conn, doi=doi, extracted_json=extracted_json)

    # 3) insert chunk rows from sections + tables
    all_pairs_for_embedding: List[Tuple[int, str]] = []  # (chunk_id, text)

    # sections
    sec_dir = os.path.join(paper_dir, "sections")
    if os.path.isdir(sec_dir):
        for p in list_md_files(sec_dir):
            text = read_text(p)
            chunks = chunk_text(text, rag.chunk_chars, rag.chunk_overlap)
            ids = insert_chunks(
                conn, doi=doi, source_type="section",
                source_name=os.path.basename(p),
                chunks=chunks,
                metadata={"path": os.path.relpath(p, start=paper_dir)},
            )
            for cid, ch in zip(ids, chunks):
                all_pairs_for_embedding.append((cid, ch))

    # tables
    tab_dir = os.path.join(paper_dir, "tables")
    if os.path.isdir(tab_dir):
        for p in list_md_files(tab_dir):
            text = read_text(p)
            chunks = chunk_text(text, rag.chunk_chars, rag.chunk_overlap)
            ids = insert_chunks(
                conn, doi=doi, source_type="table",
                source_name=os.path.basename(p),
                chunks=chunks,
                metadata={"path": os.path.relpath(p, start=paper_dir)},
            )
            for cid, ch in zip(ids, chunks):
                all_pairs_for_embedding.append((cid, ch))

    # 4) embeddings in batches
    if not all_pairs_for_embedding:
        return

    # Avoid re-embedding if some are already embedded (optional optimization)
    # For now, embed everything inserted in this run.
    batch = rag.batch_size
    for i in range(0, len(all_pairs_for_embedding), batch):
        sub = all_pairs_for_embedding[i:i+batch]
        texts = [t for _, t in sub]
        embs = embed_texts(client, rag.embedding_model, texts)

        # dimension check
        if embs and len(embs[0]) != rag.embedding_dim:
            raise RuntimeError(
                f"Embedding dim mismatch: got {len(embs[0])}, expected {rag.embedding_dim}. "
                f"Check schema VECTOR({rag.embedding_dim}) and config rag.embedding_dim."
            )

        update_embeddings(conn, [(cid, emb) for (cid, _), emb in zip(sub, embs)])
        conn.commit()
        time.sleep(0.05)


# ----------------------------
# Main
# ----------------------------

def main():
    db_cfg, rag_cfg, fulltext_root = load_config()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    conn = connect_db(db_cfg)

    doi_dirs = [d for d in sorted(os.listdir(fulltext_root))
                if os.path.isdir(os.path.join(fulltext_root, d))]

    print(f"Found {len(doi_dirs)} paper directories under {fulltext_root}")

    for d in doi_dirs:
        paper_dir = os.path.join(fulltext_root, d)
        doi = infer_doi_from_dirname(d)
        try:
            print(f"➡️ Ingesting: {doi}  ({d})")
            ingest_one_paper(conn, client, rag_cfg, paper_dir, doi)
            conn.commit()
            print("✅ OK")
        except Exception as e:
            conn.rollback()
            print(f"❌ Failed {doi}: {e}")

    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
