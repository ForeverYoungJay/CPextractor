import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))


def _iter_paper_dirs(root: Path):
    if not root.exists():
        raise RuntimeError(f"Root does not exist: {root}")

    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "paper.xml").exists() or (path / "sections").exists():
            yield path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run extractor only for local paper folders in batch, skipping completed papers by default."
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--root", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--disable-source-enrichment", action="store_true")
    ap.add_argument("--disable-direct-image-table-input", action="store_true")
    args = ap.parse_args()

    from llm.extractor import run_llm_on_paper_dir

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    llm_cfg = cfg.get("llm") or {}
    root = Path(args.root or cfg.get("paths", {}).get("fulltext", "data/fulltext"))

    paper_dirs = list(_iter_paper_dirs(root))
    if args.limit > 0:
        paper_dirs = paper_dirs[: args.limit]

    processed = 0
    skipped = 0
    failed = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for paper_dir in paper_dirs:
        raw_path = paper_dir / "materials_extracted.extractor_raw.json"
        if raw_path.exists() and not args.force:
            skipped += 1
            print(f"Skip existing extractor output: {paper_dir.name}")
            continue

        try:
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
            raw_path.write_text(
                json.dumps(result.get("extracted") or {}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            metrics = result.get("metrics") or {}
            total_input_tokens += int(metrics.get("input_tokens") or 0)
            total_output_tokens += int(metrics.get("output_tokens") or 0)
            processed += 1
            print(
                f"Extracted: {paper_dir.name} "
                f"(input_tokens={metrics.get('input_tokens', 0)}, output_tokens={metrics.get('output_tokens', 0)})"
            )
        except Exception as exc:
            failed += 1
            print(f"Failed: {paper_dir.name} -> {exc}")

    summary = {
        "root": str(root),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "papers_considered": len(paper_dirs),
    }
    print("Batch extraction summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
