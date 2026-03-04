# Evaluation Scripts

All scripts are standalone CLI tools. Inputs are `.json`/`.jsonl` unless noted.

## 1) Normalize labels and predictions

```bash
python3 scripts/eval/normalize_gold.py --input gold.jsonl --output gold_norm.json
python3 scripts/eval/normalize_pred.py --input-root data/fulltext --output pred_norm.json
```

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
