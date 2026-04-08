# Methods Workflow

This note is the paper-facing summary of the implemented system.

## Problem Definition

The system targets automatic extraction of crystal plasticity parameters from literature and converts them into a searchable, evidence-grounded database suitable for:
- parameter retrieval
- RAG-based question answering
- materials analytics

## Pipeline

### 1. Literature Acquisition

Candidate DOIs are gathered from:
- Scopus search
- manually configured DOI lists
- local fulltext folders already present in `data/fulltext`

### 2. Full-Text Parsing

Elsevier XML is parsed into local evidence assets:
- `paper.xml`
- `sections/*.md`
- `tables/*.md`
- `references.json`

### 3. Two-Stage LLM Extraction

The extraction model operates in two stages:
- file selection to choose the minimum relevant sections and tables
- schema-constrained extraction into the CP JSON schema

### 4. Deterministic Postprocessing

The extracted JSON is refined by:
- reference resolution
- parameter normalization
- unit normalization
- provenance normalization
- document metadata backfill
- condition binding resolution

### 5. Evidence Grounding

Each extracted parameter is grounded back to local evidence.

The system records:
- matched file
- char span
- line span
- matched snippet
- table coordinates when relevant

These are promoted into reusable `evidence_objects`.

### 6. Multi-Agent Evaluation

Evaluation is split into:
- evidence judge
- normalization judge
- consistency judge
- meta judge

The evaluator emits parameter-level audits and a document-level verdict.

### 7. Confidence Fusion And Tiering

Rule-based quality signals and judge outputs are fused into:
- parameter confidence
- document confidence
- quality tier: `gold`, `silver`, or `candidate`

### 8. Claim Construction

Each normalized parameter record is converted into a `parameter_claim`.

A claim is the minimal trusted unit used for:
- review queue generation
- calibration analysis
- provenance-aware downstream use

### 9. Database Ingest And Gating

Low-quality records may be blocked from formal structured ingestion based on:
- evaluator verdict
- document confidence threshold

Audit artifacts are still preserved locally and in evaluation tables.

## Evaluation Design

The system is evaluated at three levels.

### A. Extraction Correctness
- field precision / recall / F1
- numeric accuracy
- unit normalization accuracy
- provenance completeness proxy

### B. Judge Correctness
- agreement between reviewed queue adjudication and each judge
- document-level meta-judge reliability
- calibration metrics: Brier, ECE, risk-coverage

### C. Database Utility
- retrieval hit rate
- QA grounding rate
- analyst query success rate
- downstream analytics consistency

## Why This Matters

The system is not only an extractor.

It is a full evidence-grounded curation workflow that separates:
- extraction
- normalization
- provenance grounding
- uncertainty typing
- quality-tiered database construction

That separation is what makes the project suitable for both a database paper and a scientific information extraction paper.
