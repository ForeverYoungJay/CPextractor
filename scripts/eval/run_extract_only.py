import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from llm.extractor import run_llm_on_paper_dir


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value or "")


def _resolve_doi(cfg: dict, doi_override: str) -> str:
    if doi_override.strip():
        return doi_override.strip()

    dois = ((cfg.get("pipeline") or {}).get("dois") or [])
    if not dois:
        raise RuntimeError("No DOI provided. Set pipeline.dois in config.yaml or pass --doi.")
    return str(dois[0]).strip()


def _resolve_paper_dir(cfg: dict, doi: str, root_override: str) -> Path:
    paths = cfg.get("paths") or {}
    pipeline_cfg = cfg.get("pipeline") or {}
    candidate_roots = []

    if root_override.strip():
        candidate_roots.append(Path(root_override))
    else:
        fulltext_root = paths.get("fulltext")
        local_root = pipeline_cfg.get("local_fulltext_input_root")
        if fulltext_root:
            candidate_roots.append(Path(str(fulltext_root)))
        if local_root:
            local_path = Path(str(local_root))
            if local_path not in candidate_roots:
                candidate_roots.append(local_path)

    paper_id = _safe_id(doi)
    for root in candidate_roots:
        paper_dir = root / paper_id
        if paper_dir.exists():
            return paper_dir

    searched = ", ".join(str(root / paper_id) for root in candidate_roots) or "<none>"
    raise RuntimeError(f"Paper directory not found for DOI {doi}. Searched: {searched}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run extractor only for one paper and stop before postprocess.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--doi", default="")
    ap.add_argument("--root", default="")
    ap.add_argument("--disable-source-enrichment", action="store_true")
    ap.add_argument("--disable-direct-image-table-input", action="store_true")
    ap.add_argument("--print-json", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    llm_cfg = cfg.get("llm") or {}

    doi = _resolve_doi(cfg, args.doi)
    paper_dir = _resolve_paper_dir(cfg, doi, args.root)

    result = run_llm_on_paper_dir(
        paper_dir=str(paper_dir),
        model_select=llm_cfg["model_select"],
        model_extract=llm_cfg["model_extract"],
        max_snippet_chars=int(llm_cfg["max_snippet_chars"]),
        max_context_chars=int(llm_cfg["max_context_chars"]),
        max_extract_retries=int(llm_cfg.get("max_extract_retries", 2)),
        enable_source_enrichment=not args.disable_source_enrichment and bool(
            llm_cfg.get("enable_source_enrichment", True)
        ),
        direct_image_table_input=not args.disable_direct_image_table_input and bool(
            llm_cfg.get("direct_image_table_input", True)
        ),
    )

    metrics = result.get("metrics") or {}
    print(f"DOI: {doi}")
    print(f"Paper dir: {paper_dir}")
    print(f"Selected file record: {paper_dir / 'llm_selected_files.json'}")
    print(f"Extraction output: {paper_dir / 'materials_extracted.extractor_raw.json'}")
    print("Extractor metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.print_json:
        print(json.dumps(result.get("extracted") or {}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
