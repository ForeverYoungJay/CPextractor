# pipelines/run_pipeline.py
import os
import yaml

from CPextractor.scopus.scopus_search import scopus_search
from CPextractor.elsevier.fulltext_parser import save_paper_as_markdown_and_tables
from CPextractor.llm.extractor import run_llm_on_paper_dir

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- config ---
    elsevier_key = cfg["elsevier"]["api_key"]
    inst_token = cfg["elsevier"].get("inst_token")

    query = cfg["search"]["query"]
    count = cfg["search"]["count"]
    max_pages = cfg["search"]["max_pages"]

    scopus_out = cfg["paths"]["scopus_export"]
    fulltext_out = cfg["paths"]["fulltext"]

    model_select = cfg["llm"]["model_select"]
    model_extract = cfg["llm"]["model_extract"]
    max_snippet_chars = cfg["llm"]["max_snippet_chars"]
    max_context_chars = cfg["llm"]["max_context_chars"]

    # --- 1) search DOIs ---
    dois = scopus_search(
        api_key=elsevier_key,
        query=query,
        count=count,
        max_pages=max_pages,
        outdir=scopus_out,
    )
    print(f"Found {len(dois)} DOIs")

    # --- 2) fulltext parse + 3) llm extraction ---
    for doi in dois:
        try:
            save_paper_as_markdown_and_tables(
                doi=doi,
                api_key=elsevier_key,
                inst_token=inst_token,
                outdir=fulltext_out,
            )

            paper_dir = os.path.join(fulltext_out, doi.replace("/", "_").replace(":", "_"))
            if not os.path.isdir(paper_dir):
                # fallback to safe_id logic used by parser (if different)
                # easiest: scan for directory that contains the doi-safe string
                candidates = [d for d in os.listdir(fulltext_out) if doi.replace("/", "_") in d]
                if candidates:
                    paper_dir = os.path.join(fulltext_out, candidates[0])

            run_llm_on_paper_dir(
                paper_dir=paper_dir,
                model_select=model_select,
                model_extract=model_extract,
                max_snippet_chars=max_snippet_chars,
                max_context_chars=max_context_chars,
            )

            print(f"✅ Completed: {doi}")

        except Exception as e:
            print(f"❌ Failed {doi}: {e}")

if __name__ == "__main__":
    main()
