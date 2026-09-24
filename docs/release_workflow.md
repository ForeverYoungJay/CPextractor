# Reproducible evaluation and release workflow

## Delivered software

| Stage | Entry point / artifact | Completion condition |
|---|---|---|
| Benchmark repair | `evaluate_release.py`, `benchmark_protocol.py`, `diagnose_benchmark.py` | Fixed universe, adapters, one-to-one matching, auditable counts, no unresolved assignments |
| Review set | `prepare_reliable_pilot.py` | Original-source review, added omissions, reviewer identity, exhaustive adjudication |
| Error taxonomy | `audit/errors.json`, `audit/fields.json` | Field denominators and source-backed module attribution |
| Confidence/gate | `calibrate_gate.py` | Explicit usability/gate labels; calibrated thresholds evaluated on disjoint test papers |
| Ablation | `run_ablations.py`, `summarize_ablations.py` | Fixed inputs; complete attempted runs including failures; usage and latency |
| Release | `scripts/release/check_release.py` | Tests, source hashes, verified evaluation evidence and explicit license |

Read [benchmark_protocol.md](benchmark_protocol.md) before promoting any output
to a manuscript performance claim. `config.example.yaml` is the shareable default;
`config.yaml` is ignored and local. The source snapshot is a candidate and contains
no literature corpus or provider credentials.

## Current local implementation pass (2026-09-07)

The prepared workspace is `output/benchmark_release/pilot50`. It contains 50
papers, 1,042 extracted claims, a fixed 20/15/15 development/calibration/test split,
and source review packets. Six papers have zero extracted claims; they remain in
the universe and do not automatically become zero-claim gold.

The first review pass covered 20 development papers, including one zero-output
paper. It inspected reported values and units for the 520 existing claims, left
one unresolved nonnumeric claim pending, corrected 10 source-table column
misassignments and added 54 omitted records. These are **AI-assisted,
field-limited review records**, not exhaustive expert gold or independent test
results. Material/model/condition applicability and paper-level gate truth still
require full review. The remaining 30 packets retain pending labels.

Concrete source findings:

- `10.1016/j.commatsci.2025.114470`, Table 1: five interaction coefficients omitted
  after the self-interaction row.
- `10.1016/j.commatsci.2025.114088`, Tables 7 and 9: 32 T&E calibration parameters
  omitted while GA rows were retained. Alternative reported methods are separate
  parameter sets under the pilot's broad reported-parameter scope.
- `10.1016/j.actamat.2025.121314`, Table 2: 17 omissions, 10 column/value errors,
  and 50 table claims whose evidence IDs do not resolve. The local XML's CALS
  spans and row inheritance were checked before recording corrections.

Historical zero-score results do not include input hashes. The available older
usable-parameter draft has 1,015 rows across 44 DOIs, all default-labeled correct
without named reviewers; it does not reproduce the historical 1,709-row gold
count. `legacy_diagnosis.json` records this provenance gap. No exact root cause of
that historical run is asserted beyond the verified code and input-contract bugs.

## Re-run the local diagnostic

```bash
python3 scripts/eval/prepare_reliable_pilot.py --collect output/benchmark_release/pilot50 --output-jsonl output/benchmark_release/pilot50/ai_reviewed_claims.jsonl
python3 scripts/eval/evaluate_release.py --gold output/benchmark_release/pilot50/ai_reviewed_claims.jsonl --pred-root data/fulltext_20260514_run50 --manifest output/benchmark_release/pilot50/manifest.json --split development --outdir output/benchmark_release/ai_development
python3 scripts/eval/calibrate_gate.py --evaluation-dir output/benchmark_release/ai_development --outdir output/benchmark_release/field_confidence --diagnostic --describe-only
```

The descriptive confidence curve measures correctness only of reviewed fields;
it does not select a threshold or certify simulation usability. The strict
benchmark command is expected to refuse this AI-reviewed workspace.

## Continue annotation

For each paper, edit `packets/<paper>/claims.jsonl` and add omissions in
`missing_claims.jsonl`. Do not alter source predictions. Set `reviewed_fields`
explicitly for field-limited checks. Keep source hashes, notes and reviewer type.
The manifest is authoritative for split, review status and gate labels; the
packet's `paper.json` and progress CSV are convenience copies.

Before setting `exhaustive: true`, inspect the complete relevant paper, not only
the extracted evidence. Decide whether numerical settings, loading parameters,
comparison models and non-CP ML hyperparameters belong to the target universe.
An expert should adjudicate ambiguity and establish `gate_should_block` according
to a prespecified simulation-reuse requirement. Do not infer gate truth from a
single mismatched parameter or a paper's model confidence.

## Run ablations

```bash
.venv/bin/python scripts/eval/run_ablations.py --config config.example.yaml --manifest output/benchmark_release/pilot50/manifest.json --outdir output/benchmark_release/ablations --split development
.venv/bin/python scripts/eval/summarize_ablations.py --plan output/benchmark_release/ablations/plan.json --gold output/benchmark_release/pilot50/ai_reviewed_claims.jsonl --outdir output/benchmark_release/ablation_comparison --diagnostic
```

Planning does not call providers. Add `--execute --diagnostic` to the first
command for explicitly exploratory model runs when `OPENAI_API_KEY` is available.
The implementation pass did not have that environment variable; no live ablation
results or costs are claimed. All 100 planned jobs are visible as unrun until
executed. Cloud placeholders that cannot be read within the source-freeze timeout
block that paper; hydrate the local assets and create a new plan rather than
silently omitting evidence.

Final ablations should use locked gold, sufficient repetitions and the same
paper universe. Different model variants have different reported usage; combined
selector/extractor usage must not be priced as though it came from one model.

## Verify and package

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/release/check_release.py --manifest output/benchmark_release/pilot50/manifest.json --outdir output/benchmark_release/release --run-tests --archive
```

The checker writes test logs, dependency versions, source hashes, current Git
identity/dirty state and a deterministic source ZIP. `--require-ready` exits
nonzero when scientific release prerequisites remain. `--validation-root` can
point to verified test, confidence and ablation artifacts, using the documented
directory layout. A software source snapshot is not a published release and does
not assign a software license on the owner's behalf.

No pipeline predictions, database rows, remote repository, public deployment or
Git tag are changed by the diagnostic/release commands above.
