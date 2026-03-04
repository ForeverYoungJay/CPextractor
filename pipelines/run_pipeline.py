# pipelines/run_pipeline.py
import os
import yaml
from openai import OpenAI
import sys
import json
import time
from datetime import datetime, timezone

sys.path.append(".")

from db.pg import connect_pg
from db.ingest import ingest_paper_dir_to_db
from db.ingest import upsert_paper,insert_pipeline_run
from db.ingest_ref import ingest_references
from scopus.scopus_search import scopus_search
from elsevier.fulltext_parser import save_paper_as_markdown_and_tables, safe_id  # import safe_id from parser
from llm.extractor import run_llm_on_paper_dir
from postprocess.reference_resolver import resolve_references, load_references
from postprocess.unit_normalizer import normalize_extracted_units

def main():

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # env keys
    elsevier_key = os.environ.get("ELSEVIER_API_KEY") or cfg["elsevier"].get("api_key")
    inst_token = cfg["elsevier"].get("inst_token")
    crossref_mailto = cfg["elsevier"].get("crossref_mailto")
    resolve_missing_reference_doi = bool(cfg["elsevier"].get("resolve_missing_reference_doi", True))
    fulltext_http_max_retries = int(cfg["elsevier"].get("http_max_retries", 3))

    if not elsevier_key:
        raise RuntimeError("Elsevier API key not found in config.yaml")

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # db
    db = cfg["db"]
    conn = connect_pg(db["host"], int(db["port"]), db["name"], db["user"], db["password"])

    # configs
    qcfg = cfg["search"]
    llm_cfg = cfg["llm"]
    rag = cfg["rag"]
    paths = cfg["paths"]
    os.makedirs(paths["fulltext"], exist_ok=True)
    status_path = os.path.join(paths["fulltext"], "pipeline_status.jsonl")

    # 1) search scopus
    dois = scopus_search(
        api_key=elsevier_key,
        query=qcfg["query"],
        count=qcfg["count"],
        max_pages=qcfg["max_pages"],
        outdir=paths["scopus_export"],
        year_from=qcfg.get("year_from"),
        year_to=qcfg.get("year_to"),
        require_doi=bool(qcfg.get("require_doi", True)),
        allowed_doctypes=qcfg.get("allowed_doctypes", []),
        rank_keywords=qcfg.get("rank_keywords", []),
        max_retries=int(qcfg.get("http_max_retries", 3)),
    )


    print(f"Found {len(dois)} DOIs")
    dois_limit = int(qcfg.get("dois_limit", 0) or 0)
    if dois_limit > 0:
        dois = dois[:dois_limit]
        print(f"Using first {len(dois)} DOIs due to search.dois_limit={dois_limit}")
    
    # 2) for each doi: parse fulltext -> LLM extract -> ingest to DB
    for doi in dois:
        t0 = time.perf_counter()
        try:
            # parse
            save_paper_as_markdown_and_tables(
                doi=doi,
                api_key=elsevier_key,
                inst_token=inst_token,
                outdir=paths["fulltext"],
                crossref_mailto=crossref_mailto,
                resolve_missing_reference_doi=resolve_missing_reference_doi,
                http_max_retries=fulltext_http_max_retries,
            )

            paper_dir = os.path.join(paths["fulltext"], doi.replace("/", "_").replace(":", "_"))

            # llm extraction (writes json files too, but returns extracted json)
            llm_result = run_llm_on_paper_dir(
                paper_dir=paper_dir,
                model_select=llm_cfg["model_select"],
                model_extract=llm_cfg["model_extract"],
                max_snippet_chars=int(llm_cfg["max_snippet_chars"]),
                max_context_chars=int(llm_cfg["max_context_chars"]),
                max_extract_retries=int(llm_cfg.get("max_extract_retries", 2)),
            )
            selection = llm_result["selection"]
            extracted = llm_result["extracted"]
            metrics = llm_result["metrics"]
            reports = {}

            # Load references.json produced by fulltext_parser
            ref_path = os.path.join(paper_dir, "references.json")
            if os.path.exists(ref_path):
                reference_map = load_references(ref_path)
                extracted, reports["reference_resolution"] = resolve_references(extracted, reference_map)

            extracted, reports["unit_normalization"] = normalize_extracted_units(extracted)

            out_path = os.path.join(paper_dir, "materials_extracted.json")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(extracted, f, ensure_ascii=False, indent=2)

            with open(os.path.join(paper_dir, "postprocess_report.json"), "w", encoding="utf-8") as f:
                json.dump(reports, f, ensure_ascii=False, indent=2)

            # ensure DOI exists in papers BEFORE pipeline_runs insert
            upsert_paper(conn, doi=doi, title=None, year=None, journal=None)

            insert_pipeline_run(
                conn=conn,
                doi=doi,
                model_select=llm_cfg["model_select"],
                model_extract=llm_cfg["model_extract"],
                metrics=metrics,
            )

            ingest_references(conn, doi, extracted)

            conn.commit()
            
            # ingest into DB (structured + chunks + embeddings)
            ingest_paper_dir_to_db(
                conn=conn,
                openai_client=openai_client,
                doi=doi,
                paper_dir=paper_dir,
                extracted_json=extracted,
                model_select=llm_cfg["model_select"],
                model_extract=llm_cfg["model_extract"],
                embedding_model=rag["embedding_model"],
                embedding_dim=int(rag["embedding_dim"]),
                chunk_chars=int(rag["chunk_chars"]),
                chunk_overlap=int(rag["chunk_overlap"]),
                batch_size=int(rag["batch_size"]),
                embedding_max_retries=int(rag.get("embedding_max_retries", 3)),
            )

            elapsed = time.perf_counter() - t0
            with open(status_path, "a", encoding="utf-8") as sf:
                sf.write(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "doi": doi,
                    "status": "success",
                    "seconds": elapsed,
                }, ensure_ascii=False) + "\n")
            print(f"✅ Done: {doi}")

        except Exception as e:
            conn.rollback()
            elapsed = time.perf_counter() - t0
            with open(status_path, "a", encoding="utf-8") as sf:
                sf.write(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "doi": doi,
                    "status": "failed",
                    "seconds": elapsed,
                    "error": str(e),
                }, ensure_ascii=False) + "\n")
            print(f"❌ Failed {doi}: {e}")

    conn.close()

if __name__ == "__main__":
    main()
