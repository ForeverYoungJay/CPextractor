# pipelines/run_pipeline.py
import os
import yaml
from openai import OpenAI
import sys
import json
import time
import re
from datetime import datetime, timezone
from pathlib import Path

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


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)


def infer_doi_from_paper_dir(paper_dir: str) -> str | None:
    xml_path = os.path.join(paper_dir, "paper.xml")
    if os.path.exists(xml_path):
        try:
            text = Path(xml_path).read_text(encoding="utf-8", errors="ignore")
            m = DOI_PATTERN.search(text)
            if m:
                return m.group(0)
        except Exception:
            pass
    return None


def discover_local_fulltext_dois(fulltext_root: str) -> list[str]:
    root = Path(fulltext_root)
    dois = []
    seen = set()

    if not root.exists():
        return []

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not ((d / "paper.xml").exists() or (d / "sections").exists()):
            continue

        doi = infer_doi_from_paper_dir(str(d))
        if not doi:
            continue

        low = doi.lower()
        if low in seen:
            continue
        seen.add(low)
        dois.append(doi)

    return dois


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
    pipeline_cfg = cfg.get("pipeline", {})
    os.makedirs(paths["fulltext"], exist_ok=True)
    status_path = os.path.join(paths["fulltext"], "pipeline_status.jsonl")

    skip_fulltext_download = bool(pipeline_cfg.get("skip_fulltext_download", False))
    prefer_local_fulltext = bool(pipeline_cfg.get("prefer_local_fulltext", True))
    local_dois_mode = bool(pipeline_cfg.get("use_local_fulltext_dois", skip_fulltext_download))
    configured_dois = pipeline_cfg.get("dois", []) or []

    # 1) collect target DOIs
    if configured_dois:
        dois = configured_dois
        print(f"Using {len(dois)} DOIs from pipeline.dois")
    elif local_dois_mode:
        dois = discover_local_fulltext_dois(paths["fulltext"])
        print(f"Using {len(dois)} DOIs discovered from local fulltext directory")
    else:
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
            paper_dir = os.path.join(paths["fulltext"], safe_id(doi))

            # parse/download
            local_exists = os.path.isdir(paper_dir)
            if skip_fulltext_download:
                if not os.path.isdir(paper_dir):
                    raise RuntimeError(f"Local paper directory not found for DOI: {doi} ({paper_dir})")
                print(f"↪ Skip download, use local fulltext: {paper_dir}")
            elif prefer_local_fulltext and local_exists:
                print(f"↪ Local fulltext exists, skip download: {paper_dir}")
            else:
                save_paper_as_markdown_and_tables(
                    doi=doi,
                    api_key=elsevier_key,
                    inst_token=inst_token,
                    outdir=paths["fulltext"],
                    crossref_mailto=crossref_mailto,
                    resolve_missing_reference_doi=resolve_missing_reference_doi,
                    http_max_retries=fulltext_http_max_retries,
                )

            # llm extraction (writes json files too, but returns extracted json)
            llm_result = run_llm_on_paper_dir(
                paper_dir=paper_dir,
                model_select=llm_cfg["model_select"],
                model_extract=llm_cfg["model_extract"],
                max_snippet_chars=int(llm_cfg["max_snippet_chars"]),
                max_context_chars=int(llm_cfg["max_context_chars"]),
                max_extract_retries=int(llm_cfg.get("max_extract_retries", 2)),
                enable_source_enrichment=bool(llm_cfg.get("enable_source_enrichment", True)),
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
                prompt_version=str(llm_cfg.get("prompt_version", "v2.0.0")),
                schema_version=str(llm_cfg.get("schema_version", "2.0.0")),
                extractor_version=str(llm_cfg.get("extractor_version", "extractor_v2")),
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
