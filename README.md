# CPextractor

CPextractor curates crystal-plasticity literature into parameter claims with
material/model context, provenance and evidence pointers, then supports review,
PostgreSQL/pgvector retrieval, a Streamlit chatbot and a CP knowledge graph.

## Current implementation

The active extractor schema is **6.0.0**. The main pipeline is
`pipelines/run_pipeline.py`. It selects relevant local sections/tables/equations,
extracts hierarchical `parameter_claims`, links evidence and equations, evaluates
claims, fuses confidence and applies the database gate.

The default configuration uses `gpt-4.1-mini` for selection, `gpt-5.1` for
extraction and `gpt-4.1` in **single_judge** mode for evaluation. Committee review
is available but not the default. The extractor-internal double pass is off.
`pipeline.skip_quality_checks` defaults to true. New extractor-first schemas skip
legacy parameter/provenance/table/condition rewriting while retaining unit,
material, model and evidence processing. `parameters.registry` is a compatibility
view, not the primary v6 extraction target.

The acquisition path implemented by the main pipeline uses Elsevier XML and
Scopus/local DOI discovery. A general PDF fallback is not established by this
release. Quality tiers and confidence are operational signals, not validated
probabilities or expert acceptance labels.

## Install and run

The core environment was checked with Python 3.13. Offline evaluation needs only
Python's standard library. Pipeline/annotation dependencies are pinned separately
from the optional UI:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Edit the ignored `config.yaml` to select local input/output roots and DOIs. Provider
credentials are read from `OPENAI_API_KEY` and `ELSEVIER_API_KEY`. The committed
example contains no provider credentials. The local database example is for
local development.

```bash
docker compose up -d
docker exec -i cp_pgvector psql -U cpuser -d cpdb < schema.sql
.venv/bin/python pipelines/run_pipeline.py --config config.yaml
```

For the UI, install `requirements-ui.txt` and run `streamlit run chatbot_ui.py`.
Live pipeline runs require provider access and a running database; offline
benchmarking and release checks do not call either service.

## Data and applications

The schema links `document`, `materials -> phases`, `process_states`, `conditions`,
`models`, mechanisms, microstructure features, claims and evidence objects.
A claim records its reported value/unit, parameter identity, applicability,
provenance roles and governing equations.

`schema.sql` defines paper/extraction storage, text/parameter/table-row vectors,
reference and lineage relations, and pipeline/evaluation audits. Embedding reuse
uses content hashes and model identifiers. `chatbot.py` combines structured,
vector, table-row and equation retrieval. `kg/builder.py` exports a property graph:

```bash
.venv/bin/python -m kg.builder --root data/fulltext --output-json output/kg/cp_kg.json --output-graphml output/kg/cp_kg.graphml --output-cypher output/kg/cp_kg.cypher
.venv/bin/python -m kg.ingest_to_pg --config config.yaml --root data/fulltext
```

## Benchmark and review workflow

The evaluation contract is [docs/benchmark_protocol.md](docs/benchmark_protocol.md).
It fixes the paper universe, DOI normalization, one-to-one semantic matching,
field denominators, error taxonomy and calibration/test separation. Generated
claim IDs do not determine cross-run identity. Unreviewed papers are not negative
gold, and draft annotations start as `pending`.

```bash
python3 scripts/eval/prepare_reliable_pilot.py --input-root data/fulltext --output-root output/pilot50 --n 50
python3 scripts/eval/prepare_reliable_pilot.py --collect output/pilot50 --output-jsonl output/pilot50/reviewed.jsonl
python3 scripts/eval/evaluate_release.py --gold output/pilot50/reviewed.jsonl --pred-root data/fulltext --manifest output/pilot50/manifest.json --split development --outdir output/eval/development
```

Review the original source, correct records, add omitted claims and record who
reviewed which fields. AI-assisted field review remains diagnostic. After genuine
exhaustive expert adjudication, add `--strict` for publication evaluation.
Unknown labels and missing confidence stay unknown.

```bash
python3 scripts/eval/evaluate_release.py --gold output/pilot50/reviewed.jsonl --pred-root data/fulltext --manifest output/pilot50/manifest.json --split calibration --strict --outdir output/eval/calibration
python3 scripts/eval/calibrate_gate.py --evaluation-dir output/eval/calibration --outdir output/eval/calibration_policy
python3 scripts/eval/evaluate_release.py --gold output/pilot50/reviewed.jsonl --pred-root data/fulltext --manifest output/pilot50/manifest.json --split test --strict --outdir output/eval/test
python3 scripts/eval/calibrate_gate.py --evaluation-dir output/eval/test --policy output/eval/calibration_policy/frozen_policy.json --outdir output/eval/test_confidence
```

Explicit paper gate labels are required for false-admission/false-block rates.
Field-only review cannot certify full simulation usability. Historical metrics in
`results/eval_annotation` used an earlier matching/counting policy and are not
release-validation results.

## Ablations and release checks

Plan isolated baseline/committee/rules/double-pass/model comparisons without API
calls, then execute after the evaluation protocol and credentials are ready:

```bash
.venv/bin/python scripts/eval/run_ablations.py --config config.example.yaml --manifest output/pilot50/manifest.json --outdir output/ablations --split development
```

`--execute` performs model calls; `--diagnostic` explicitly permits pre-validation
experiments. Jobs are isolated by variant/repeat/DOI and never ingest the formal
DB. `summarize_ablations.py` retains failed and unrun jobs, and leaves unavailable
costs null rather than assigning a misleading price to combined model usage.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/release/check_release.py --manifest output/pilot50/manifest.json --outdir output/release --run-tests --archive
```

The source archive excludes corpus files and personal configuration. A passing
software test run does not imply an expert-validated scientific release.
See [docs/release_workflow.md](docs/release_workflow.md) for deliverables and
remaining scientific acceptance criteria.
