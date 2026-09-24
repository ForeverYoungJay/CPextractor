"""Read-only, paired descriptive analysis of archived pipeline runs.

This is observational evidence: run dates differ and source input hashes were
not stored. It cannot establish model accuracy or causal ablation effects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import psycopg
import yaml

MODEL_A = "gpt-4.1-mini"
MODEL_B = "gpt-5.1"
QUERY = """
SELECT doi, model_extract, run_id, llm_select_total_tokens,
       llm_extract_total_tokens, time_total_seconds, created_at
FROM pipeline_runs
WHERE schema_version = '2.1.1'
  AND prompt_version = 'v2.1.1'
  AND extractor_version = 'extractor_v2'
  AND model_select = 'gpt-4.1-mini'
  AND model_extract IN ('gpt-4.1-mini', 'gpt-5.1')
ORDER BY doi, model_extract, created_at DESC, run_id DESC
"""


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def analyze(records):
    by_doi = defaultdict(dict)
    for doi, model, run_id, select_tokens, extract_tokens, seconds, created_at in records:
        # The SQL sorts newest first. Retain one run per DOI and model.
        by_doi[doi].setdefault(model, {
            "run_id": run_id, "select_tokens": select_tokens,
            "extract_tokens": extract_tokens, "seconds": seconds,
            "created_at": created_at.isoformat(),
        })
    rows = []
    for doi, variants in sorted(by_doi.items()):
        if MODEL_A not in variants or MODEL_B not in variants:
            continue
        a, b = variants[MODEL_A], variants[MODEL_B]
        at = a["select_tokens"] + a["extract_tokens"] if a["select_tokens"] is not None and a["extract_tokens"] is not None else None
        bt = b["select_tokens"] + b["extract_tokens"] if b["select_tokens"] is not None and b["extract_tokens"] is not None else None
        rows.append({
            "doi": doi, "mini_run_id": a["run_id"], "large_run_id": b["run_id"],
            "mini_created_at": a["created_at"], "large_created_at": b["created_at"],
            "mini_seconds": a["seconds"], "large_seconds": b["seconds"],
            "seconds_delta_large_minus_mini": b["seconds"] - a["seconds"] if a["seconds"] is not None and b["seconds"] is not None else None,
            "mini_tokens": at, "large_tokens": bt,
            "tokens_delta_large_minus_mini": bt - at if at is not None and bt is not None else None,
        })
    both_seconds = [r for r in rows if r["seconds_delta_large_minus_mini"] is not None]
    both_tokens = [r for r in rows if r["tokens_delta_large_minus_mini"] is not None]
    metrics = {
        "analysis_type": "retrospective_observational_pairing",
        "publication_ready": False,
        "paper_count": len(rows),
        "model_a": MODEL_A, "model_b": MODEL_B,
        "selection_policy": "Latest recorded run for each DOI and model after fixed selector/prompt/schema/extractor filters",
        "matched_seconds": len(both_seconds), "matched_tokens": len(both_tokens),
        "median_seconds": {MODEL_A: median(r["mini_seconds"] for r in both_seconds), MODEL_B: median(r["large_seconds"] for r in both_seconds)},
        "median_tokens": {MODEL_A: median(r["mini_tokens"] for r in both_tokens), MODEL_B: median(r["large_tokens"] for r in both_tokens)},
        "median_paired_seconds_delta_large_minus_mini": median(r["seconds_delta_large_minus_mini"] for r in both_seconds),
        "median_paired_tokens_delta_large_minus_mini": median(r["tokens_delta_large_minus_mini"] for r in both_tokens),
        "large_faster_papers": sum(r["seconds_delta_large_minus_mini"] < 0 for r in both_seconds),
        "large_more_tokens_papers": sum(r["tokens_delta_large_minus_mini"] > 0 for r in both_tokens),
        "run_ids_sha256": hashlib.sha256(json.dumps([(r["doi"], r["mini_run_id"], r["large_run_id"]) for r in rows], separators=(",", ":")).encode()).hexdigest(),
        "limitations": [
            "Runs occurred in different calendar windows, so infrastructure and inputs may differ.",
            "No source-input hashes or per-run extraction gold are stored in pipeline_runs.",
            "The database contains only current extractions, not immutable extraction outputs for each historical run.",
            "Token counts are quantities, not USD costs; model prices are not assumed.",
        ],
    }
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    conf = yaml.safe_load(Path(args.config).read_text())["db"]
    with psycopg.connect(host=conf["host"], port=conf["port"], dbname=conf["name"],
                         user=conf["user"], password=conf["password"], connect_timeout=5) as db:
        with db.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(QUERY)
            records = cursor.fetchall()
    metrics, rows = analyze(records)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "historical_model_comparison.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    with (outdir / "paired_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["doi"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"paper_count": metrics["paper_count"],
                      "median_seconds": metrics["median_seconds"],
                      "median_tokens": metrics["median_tokens"],
                      "publication_ready": False}))


if __name__ == "__main__":
    main()
