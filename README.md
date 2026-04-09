# CPextractor

CPextractor is a literature-to-database pipeline for crystal plasticity parameter curation.

It turns Elsevier/Scopus papers into:
- structured CP records in `parameters.registry`
- parameter-level claims with provenance and calibrated confidence
- evidence-grounded review artifacts
- PostgreSQL records for retrieval, RAG, and analytics

## What The Code Does

The main entry point is [pipelines/run_pipeline.py](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/pipelines/run_pipeline.py).

The runtime flow is:
1. collect DOI targets from `pipeline.dois`, local fulltext folders, or Scopus
2. download and parse Elsevier XML into `paper.xml`, `sections/*.md`, `tables/*.md`, and `references.json`
3. run a two-stage LLM extractor:
   - file selection
   - schema-constrained extraction
4. normalize parameters, units, provenance, and document metadata
5. resolve condition binding for material, phase, family, system, and loading context
6. ground evidence back to file spans and table cells
7. run multi-agent LLM evaluation:
   - evidence judge
   - normalization judge
   - consistency judge
   - meta judge
8. fuse rule score and judge score into final confidence and quality tier
9. build `parameter_claims`
10. gate low-quality papers from formal DB ingest
11. ingest structured records, chunks, embeddings, references, and evaluation artifacts

## Core Data Model

The extractor still uses `parameters.registry` as an intermediate extraction target, but the finalized stored document is now a hierarchical v3 schema centered on:

- `document`
- `materials[] -> phases[]`
- `process_states[]`
- `conditions[]`
- `models[]`
- `mechanisms`
- `microstructure_features[]`
- `parameter_claims[]`

Each registry item still represents one extracted parameter record with:
- normalized parameter identity
- reported and SI-normalized value
- scope and mechanism binding
- provenance source
- confidence

On top of that, the pipeline now also builds:
- `materials[]` with nested `phases[]`
- `process_states[]` and `conditions[]`
- `models[]` and `mechanisms`
- `evidence_objects`: reusable evidence spans and table-cell references
- `parameter_claims`: the smallest trusted unit for review, auditing, and downstream use

`parameter_claims` are designed for publication-grade curation. A claim keeps:
- `canonical_name`
- normalized value
- `applies_to`
- binding context
- material / process-state / condition / mechanism scope
- source provenance
- direct evidence locator plus grounded evidence ids
- confidence
- audit verdict
- uncertainty typing

## Quality Control Stack

The QA stack is layered rather than a single LLM verdict.

### Layer 1: Rule Validation
- schema completeness
- unit sanity
- scope consistency
- provenance conflict detection
- basic physics-aware checks

### Layer 2: Evidence Grounding
- file-level matching
- char span and line span
- table-cell fallback grounding
- reusable `evidence_objects`

### Layer 3: Multi-Agent Judge
- evidence support
- normalization correctness
- document consistency
- document-level meta verdict

### Layer 4: Confidence Fusion And Tiering
- `document_confidence_score`
- `document_confidence`
- `quality_tier = gold / silver / candidate`

## Database Outputs

Main tables are defined in [schema.sql](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/schema.sql).

Structured content:
- `papers`
- `extractions`
- `chunks`
- `parameter_vectors`
- `references`
- `paper_references`
- `parameter_references`

Evaluation content:
- `pipeline_runs`
- `evaluation_runs`
- `parameter_audits`
- `evaluation_paper_summary`

## Retrieval And Chatbot

The chatbot in [chatbot.py](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/chatbot.py) uses hybrid retrieval:
- vector retrieval over `chunks`
- vector retrieval over `parameter_vectors`
- structured retrieval over finalized `extractions.extracted_json -> parameter_claims[]` plus hierarchical `materials[] / samples[] / conditions[]`
- LLM synthesis constrained to provided evidence

This supports:
- CP chatbot
- evidence-grounded QA
- structured parameter lookup

The ingest layer now reuses embeddings when source text has not changed:
- `chunks.content_hash + embedding_model`
- `parameter_vectors.content_hash + embedding_model`

So if `sections/*.md`, `tables/*.md`, or parameter claims are unchanged for the same DOI, re-ingest can skip repeated embedding cost.

## Evaluation Structure

The evaluation pipeline is now explicitly split into three paper-facing targets:

### A. Extraction Correctness
- field precision / recall / F1
- numeric accuracy proxy
- unit normalization accuracy
- provenance completeness proxy

### B. Judge Correctness
- evidence judge correctness
- normalization judge correctness
- consistency judge correctness
- meta-judge reliability
- Brier / ECE / risk-coverage / wrong@high-confidence

### C. Database Utility
- structured retrieval hit rate
- RAG answer grounding rate
- analyst query success rate
- downstream analytics consistency

## Quick Start

```bash
docker compose up -d
docker exec -i cp_pgvector psql -U cpuser -d cpdb < schema.sql

export ELSEVIER_API_KEY="..."
export OPENAI_API_KEY="..."

python3 pipelines/run_pipeline.py
```

For already downloaded local papers:

```bash
python3 scripts/eval/audit_extractions.py --config config.yaml
```

## Paper-Oriented Outputs

One-click evaluation:

```bash
python3 scripts/eval/run_all.py \
  --gold gold.jsonl \
  --pred-root data/fulltext \
  --qrels qrels.jsonl \
  --runs runs.jsonl \
  --pipeline-csv pipeline_runs.csv \
  --review-csv output/review/review_queue.csv \
  --outdir results/eval \
  --method-name CPextractor
```

Case-study analytics:

```bash
python3 scripts/analytics/materials_insight.py --root data/fulltext --outdir output/analytics
```

## Key Files

- [pipelines/run_pipeline.py](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/pipelines/run_pipeline.py)
- [llm/extractor.py](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/llm/extractor.py)
- [llm/evaluator.py](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/llm/evaluator.py)
- [postprocess/evidence_grounding.py](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/postprocess/evidence_grounding.py)
- [postprocess/condition_binding.py](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/postprocess/condition_binding.py)
- [postprocess/claim_builder.py](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/postprocess/claim_builder.py)
- [postprocess/confidence_fusion.py](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/postprocess/confidence_fusion.py)
- [chatbot.py](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/chatbot.py)
