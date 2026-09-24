# Offline experiments from archived results

The OpenAI API returned `credit_balance_exhausted` for the September 2026 live
smoke run. Existing files and the local PostgreSQL database still support
descriptive, read-only experiments. They do not replace the planned controlled
ablation or expert gold evaluation.

## What was available

The local database contained 289 papers, 198 current extraction records, 808
pipeline run records, and 516 evaluator run records when inspected. These table
counts refer to different grains and must not be added or presented as numbers
of independent experiments. The local `data/fulltext_20260514_run50` collection
has 50 archived extraction outputs. The 20-paper AI-reviewed development
diagnostic is recorded in `output/benchmark_release/ai_development`; its labels
cover selected fields only.

## Historical model comparison

Run:

```bash
python scripts/eval/analyze_historical_runs.py \
  --config config.yaml \
  --outdir output/benchmark_release/historical_offline
```

The script issues a read-only database query and pairs the latest recorded run
per DOI and extraction model. It requires the same recorded selector
(`gpt-4.1-mini`), prompt (`v2.1.1`), schema (`2.1.1`), and extractor
(`extractor_v2`). The resulting 287 papers have both `gpt-4.1-mini` and
`gpt-5.1` extraction runs. The observed medians were:

| Extraction model | Median recorded pipeline seconds | Median select + extract tokens |
|---|---:|---:|
| `gpt-4.1-mini` | 105.87 | 19,087 |
| `gpt-5.1` | 155.57 | 17,279 |

The mini runs occurred between 2026-03-12 and 2026-03-24; the `gpt-5.1` runs
between 2026-03-24 and 2026-04-17. The database did not store hashes of the
actual selected source snippets or immutable per-run extraction outputs. Even
with the matching recorded versions and DOI, changes in inputs, runtime,
retries, or infrastructure may explain differences. These numbers measure
historical throughput and token use only; they say nothing about model accuracy
or causal effects. Token totals are not USD costs.

The output contains `paired_runs.csv` with DOI and run IDs, plus
`historical_model_comparison.json` with selection rules, aggregate values, and
a hash of selected run IDs. Keep those outputs local unless the underlying
paper and database release scope is separately decided.

## Existing claim and gate diagnostics

The previously prepared 20-paper development annotation is explicitly marked
as AI-assisted and field-limited. Its 54 identified omissions, 10 corrected
values, and 50 evidence-file mismatches can guide error repair. Precision,
recall, gate false-admission, and gate false-block rates remain uncertified
because exhaustive expert labels and gate truth do not exist yet. See
[`benchmark_protocol.md`](benchmark_protocol.md) for the criteria that must be
met before reporting scientific performance.

All 50 archived postprocess reports were also available locally. Their stored
gate decisions admitted 43 papers and blocked 7: six because extraction was
empty, one because review escalation was required. The stored median document
confidence was 100. These are historical gate *outputs*, not correctness
labels; no false-admission or false-block rate can be calculated from them.
The local audit is `output/benchmark_release/historical_offline/pilot_gate_profile.json`.

The next controlled experiment still requires one frozen paper source per run,
the same paper universe across variants, model access with usable API credits,
and independent review labels for quality comparisons.
