# Evaluation Scripts

All scripts are standalone CLI tools. Inputs are `.json`/`.jsonl` unless noted.

## 1) Normalize labels and predictions

```bash
python3 scripts/eval/normalize_gold.py --input gold.jsonl --output gold_norm.json
python3 scripts/eval/normalize_pred.py --input-root data/fulltext --output pred_norm.json
```

## 0) Export claim-level annotation draft

Use model output as a pre-annotation draft, then let humans correct it.

```bash
python3 scripts/eval/export_annotation_draft.py \
  --input-root data/fulltext \
  --source materials_extracted.json \
  --output data/annotations/annotation_draft.jsonl
```

Convert draft jsonl into an editable CSV sheet:

```bash
python3 scripts/eval/export_annotation_sheet.py \
  --input data/annotations/annotation_draft.jsonl \
  --output data/annotations/annotation_draft.csv \
  --profile compact
```

`--profile compact` keeps the sheet simple:
- `material_name`
- `canonical_name`
- `symbol`
- `value`
- `unit`
- `evidence.file`
- `evidence.row_name`
- `evidence.column_name`
- `evidence.value_text`
- `annotation.status`
- `annotation.notes`

Use `--profile full` only when you want to annotate grounding/provenance details too.

After editing in Excel/Numbers, convert it back:

```bash
python3 scripts/eval/import_annotation_sheet.py \
  --input data/annotations/annotation_draft.csv \
  --output data/annotations/gold_claims.jsonl
```

Export one file per paper for paper-by-paper annotation:

```bash
python3 scripts/eval/export_annotation_packets.py \
  --input-root data/fulltext \
  --source materials_extracted.json \
  --output-root data/annotations/packets \
  --csv-profile compact
```

Prepare a difficulty-aware pilot packet set instead of exporting every paper:

```bash
python3 scripts/eval/prepare_annotation_pilot.py \
  --input-root data/fulltext \
  --output-root data/annotations/pilot_packets \
  --n 50 \
  --strategy journal_balanced \
  --source materials_extracted.json \
  --csv-profile compact
```

This creates a claim-level pilot set that is still diverse by journal, but prioritizes harder papers:
- image-backed tables
- flagged / review-required claims
- missing-unit cases
- grouped-row-like claims
- higher claim-count papers

This creates:

- `data/annotations/packets/manifest.csv`
- one folder per DOI, each with:
  - `claims.csv` (compact by default: parameter, value, unit, table file/row/column context, minimal annotation columns; no DOI/claim ID columns)
  - `claims.jsonl`
  - `packet_meta.json`
  - `README.md`

Recommended packet workflow:

1. Open one paper folder and edit `claims.csv`.
2. Only change:
   - `material_name`
   - `canonical_name`
   - `symbol`
   - `value`
   - `unit`
   - `annotation.status`
   - `annotation.notes`
3. Keep `claims.jsonl` unchanged; it stores the full original context.
4. After editing all desired packets, build a combined gold file:

```bash
python3 scripts/eval/build_gold_from_packets.py \
  --packets-root data/annotations/packets \
  --output data/annotations/gold_claims.jsonl \
  --write-per-packet-jsonl
```

Run the claim-level annotation benchmarks with one command:

```bash
python3 scripts/eval/run_annotation_benchmarks.py \
  --gold data/annotations/gold_claims.jsonl \
  --pred-root data/fulltext \
  --pred-source materials_extracted.json \
  --outdir results/eval_annotation
```

This writes:
- `results/eval_annotation/metrics/benchmark_claims.json`
- `results/eval_annotation/metrics/benchmark_gate.json`
- `results/eval_annotation/metrics/benchmark_slices.json`
- `results/eval_annotation/metrics/annotation_benchmark_summary.json`
- `results/eval_annotation/tables/table_bundle_completeness.csv`
- `results/eval_annotation/tables/table_gate_by_paper.csv`
- `results/eval_annotation/tables/table_slice_results.csv`

## 0b) Benchmarks after manual annotation

Claim-level benchmark:

```bash
python3 scripts/eval/benchmark_claims.py \
  --gold data/annotations/gold_claims.jsonl \
  --pred-root data/fulltext \
  --pred-source materials_extracted.json \
  --output results/eval/metrics/benchmark_claims.json \
  --by-paper-csv results/eval/tables/table_bundle_completeness.csv
```

Gate / ingest benchmark:

```bash
python3 scripts/eval/benchmark_gate.py \
  --gold data/annotations/gold_claims.jsonl \
  --pred-root data/fulltext \
  --output results/eval/metrics/benchmark_gate.json \
  --by-paper-csv results/eval/tables/table_gate_by_paper.csv
```

Difficulty-slice benchmark:

```bash
python3 scripts/eval/benchmark_slices.py \
  --gold data/annotations/gold_claims.jsonl \
  --pred-root data/fulltext \
  --pred-source materials_extracted.json \
  --output results/eval/metrics/benchmark_slices.json \
  --output-csv results/eval/tables/table_slice_results.csv
```

These scripts are the most useful after your 50-paper annotation pass. They summarize:
- claim precision / recall / F1
- value / unit / canonical-name accuracy
- grounding accuracy when gold grounding is annotated
- bundle completeness
- gate rate / ingest rate / gate-reason distribution
- slice-wise results for image-backed tables, grouped rows, missing-unit cases, etc.

Reference schema and examples:

- `docs/annotation_schema.md`
- `docs/gold_claims.example.jsonl`
- `docs/gold_bundles.example.jsonl`

## 2) Main extraction metrics

```bash
python3 scripts/eval/field_metrics.py --gold gold_norm.json --pred pred_norm.json --output metrics_field.json
python3 scripts/eval/numeric_metrics.py --gold gold_norm.json --pred pred_norm.json --output metrics_numeric.json
python3 scripts/eval/unit_metrics.py --gold gold_norm.json --pred pred_norm.json --output metrics_unit.json
python3 scripts/eval/citation_metrics.py --gold gold_norm.json --pred pred_norm.json --output metrics_citation.json
```

## 3) Retrieval metrics

`qrels.jsonl` format:
`{"query_id":"q1","doc_id":"chunk_12","relevant":1}`

`runs.jsonl` format:
`{"query_id":"q1","ranked_doc_ids":["chunk_3","chunk_12"]}`

```bash
python3 scripts/eval/retrieval_metrics.py --qrels qrels.jsonl --runs runs.jsonl --ks 5,10 --output metrics_retrieval.json
```

## 4) Cost/latency report

Use CSV exported from `pipeline_runs`.

```bash
python3 scripts/eval/cost_latency_report.py \
  --input-csv pipeline_runs.csv \
  --usd-per-1k-input 0.001 \
  --usd-per-1k-output 0.004 \
  --output metrics_cost.json
```

## 5) Error buckets

```bash
python3 scripts/eval/error_bucket.py --gold gold_norm.json --pred pred_norm.json --output metrics_errors.json
```

## 6) Quality gate (paper-grade pass/fail)

```bash
python3 scripts/eval/quality_gate.py \
  --metrics-dir results/eval/metrics \
  --postprocess-report data/fulltext/<doi>/postprocess_report.json \
  --output results/eval/metrics/quality_gate.json
```

## 7) Judge benchmark against reviewed queue

```bash
python3 scripts/eval/judge_benchmark.py \
  --review-csv output/review/review_queue.csv \
  --pred-root data/fulltext \
  --output results/eval/metrics/metrics_judge.json
```

Outputs include:
- `precision_error_detection`
- `recall_error_detection`
- `f1_error_detection`
- `brier_score`
- `ece`
- `cohen_kappa`
- `wrong_at_high_confidence`
- `risk_coverage_curve`
- `by_judge.evidence_judge`
- `by_judge.normalization_judge`
- `by_judge.consistency_judge`
- `by_judge.meta_judge_doc_level`

## 8) Utility under confidence filtering

```bash
python3 scripts/eval/utility_by_confidence.py \
  --pred-root data/fulltext \
  --qrels qrels.jsonl \
  --runs runs.jsonl \
  --qa-jsonl qa_results.jsonl \
  --output results/eval/metrics/metrics_utility_by_confidence.json
```

## One-click full run (9 scripts + paper CSV tables)

```bash
python3 scripts/eval/run_all.py \
  --gold gold.jsonl \
  --pred-root data/fulltext \
  --qrels qrels.jsonl \
  --runs runs.jsonl \
  --pipeline-csv pipeline_runs.csv \
  --review-csv output/review/review_queue.csv \
  --outdir results/eval \
  --method-name CPextractor \
  --ks 5,10 \
  --usd-per-1k-input 0.001 \
  --usd-per-1k-output 0.004
```

Generated files:
- `results/eval/metrics/*.json`
- `results/eval/tables/table_main_results.csv`
- `results/eval/tables/table_extraction_correctness.csv`
- `results/eval/tables/table_retrieval_results.csv`
- `results/eval/tables/table_error_buckets.csv`
- `results/eval/tables/table_field_by_prefix.csv`
- `results/eval/tables/table_quality_gate.csv`
- `results/eval/tables/table_judge_results.csv`
- `results/eval/tables/table_judge_correctness.csv`
- `results/eval/tables/table_utility_by_confidence.csv`

Paper-facing interpretation:
- `table_extraction_correctness.csv`: extraction correctness
- `table_judge_correctness.csv`: judge correctness
- `table_utility_by_confidence.csv`: database utility
