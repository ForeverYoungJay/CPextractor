# CPextractor evaluation protocol 1.0

The executable contract is `scripts/eval/benchmark_protocol.py` (`claim-benchmark-1.0`).
The release entry point is `scripts/eval/evaluate_release.py`; the legacy
`run_annotation_benchmarks.py` command delegates to it. Existing historical
metrics are retained as historical artifacts and must not be cited as validation
of this protocol or the current schema.

## 1. Evaluation universe

Freeze a paper manifest before evaluating. Each paper has a canonical DOI, split
(`development`, `calibration`, `test`), source directory, prediction hash, review
status, reviewer identity, completeness declaration, and an optional explicit
paper gate label. DOI prefixes and case are normalized; missing/invalid DOIs and
duplicate manifest DOIs are errors. Claim IDs must never substitute for a DOI.

Strict evaluation includes every manifest paper in the chosen split, including
extraction failures. A failed extraction contributes false negatives for its
gold claims. An exhaustively reviewed zero-claim paper requires an explicit
`gold_claim_count: 0`; its predictions are false positives. Unannotated papers
outside the universe are excluded and counted, never treated as negative gold.

Without `--strict`, output is **diagnostic_only**. Entirely pending papers and
predictions unambiguously matched to pending claims are excluded. Partial
annotations do not establish exhaustive precision/recall. No diagnostic run may
be promoted to a publication benchmark by relabeling a summary file.

## 2. Claim identity and assignment

A claim identifies a reported parameter in a paper and its applicability context.
Generated `claim_id` values are tracking handles, not cross-run matching keys.
Flat exports, nested usable-parameter annotations, and v6 parameter/assertion
objects are converted to a common row representation.

Candidate matches require the same DOI and at least one of:

- the same canonical parameter name;
- the same reported symbol (case-sensitive: `m` and `M` differ);
- the same explicit table file, row and column locator.

Named material, phase, process state, condition, temperature, strain rate, model,
mechanism, family and systems disambiguate repeated parameters. Generated object
IDs do not participate. A matching table row plus parameter identity can
disambiguate grain-size or other row variants when no column locator exists.
Complete cell anchors can preserve a match when the predicted binding is wrong,
allowing that binding to be evaluated as a field error.

Assignment repeatedly accepts mutually unique highest-scoring pairs. It is
one-to-one and never collapses records into a dictionary. Equally good remaining
assignments are written to `diagnostics.ambiguities`; strict publication output
is refused until those cases are resolved. Counts from ambiguous diagnostic runs
are conservative, not best-case matching estimates. The weights and exact
matching rules are in `candidate_score` and are versioned with this protocol.

Values, units, confidence, verdicts and correctness labels never choose a match.
Neither fuzzy semantic equivalence nor arbitrary ontology mappings are inferred.
If both identity and evidence are corrupted, a claim may remain unmatched; this
limitation must accompany detection results.

## 3. Field scoring and error taxonomy

Score fields only after matching. Unknown labels are excluded from denominators;
explicit `annotation.reviewed_fields` restricts scoring to the fields actually
checked. Explicitly reviewed absence can be scored (e.g. no reported unit).
Numeric tolerance is `rtol=1e-4`, `atol=1e-9`. Explicit scientific notation,
fractions, zero and arrays are supported. Ambiguous flattened exponents are not
guessed. Unit prefixes retain case (`mPa` is not `MPa`). Reported-unit fidelity
is separate from physical dimensional correctness or SI conversion accuracy.

| Module | Typical error |
|---|---|
| detection | Omitted, spurious, duplicate or misidentified parameter |
| value_unit | Wrong number, range, exponent or reported unit |
| material_scope | Wrong material, phase or process state |
| condition | Wrong temperature, rate or loading condition |
| model_mechanism | Wrong model, branch, deformation family or system |
| source_attribution | Wrong origin, adopted reference or calibration reference |
| grounding | Missing/wrong evidence file, row or column |

The audit outputs contain match reasons, original/corrected values, evidence and
field denominators. A copied model field is not an expert label. A source's
physically suspicious printed value/unit should be preserved and flagged,
not silently corrected by the annotator.

Difficulty slices must select gold and predictions independently; otherwise
unmatched predictions disappear and precision becomes artificially optimistic.
The legacy slice CLI implements that rule. Missing-unit and other error-defined
slices can move records between slices; report their construction explicitly.

## 4. Expert and AI review

`prepare_reliable_pilot.py` freezes up to 50 papers using seeded journal round
robin, including zero-output cases. At 50 papers the split is 20/15/15. It writes
one packet per paper with editable claims, an empty omitted-claims file, and
source snapshots/hashes. Source snippets generated by the extraction model are
not independent evidence; inspect the source file or original XML.

Review all relevant text, tables, captions and equations. Add omissions manually.
Separate fitted values, bounds, comparison methods and different conditions.
Do not label a comparative parameter set as duplicate merely because its symbol
matches the primary model. Decide target scope (constitutive parameters versus
coupled-model, numerical, loading or ML parameters) before claiming completeness.

AI-assisted reviews carry `reviewer_type: ai_assistant` and may establish a
field-limited diagnostic set. They **do not satisfy expert gold**. Strict mode
requires paper-level `annotation_status: adjudicated`, `exhaustive: true`, a named
human expert and adjudicator, plus human-expert reviewer identity on every row.
Unresolved statuses prevent strict evaluation. Keep original prediction IDs and
source hashes; never manufacture reviewer signatures.

## 5. Confidence and gate calibration

Claim confidence is on a 0–100 scale; missing/nonfinite/out-of-range scores are
unknown, not zero. A claim contributes a correctness label only when its review
explicitly declares `usability_review_complete: true` and the required core
fields are reviewed. Field-limited checks cannot certify complete usability.

Paper gate truth is the explicit `gate_should_block` boolean in the manifest.
It is not inferred from a missing label, a single parameter warning, or an LLM
verdict. The gate report separates missing decisions from unlabeled papers.

The calibration CLI writes precision–coverage curves and a frozen policy:

- false admission rate = admitted bad papers / all bad papers;
- false block rate = blocked good papers / all good papers;
- admitted precision = admitted good papers / all admitted papers;
- coverage = admitted papers / evaluated papers.

Choose maximum coverage meeting target empirical precision and minimum support
on **calibration** papers only. Evaluate that exact policy on disjoint **test**
papers. The CLI rejects split leakage and diagnostic-to-validated promotion.
No feasible threshold produces `null`, not an invented threshold. Wilson
intervals are descriptive claim-level intervals; paper clustering limits their
interpretation. This is threshold selection, not proof that scores are calibrated
probabilities. A final study should add paper-bootstrap uncertainty and compare
raw versus independently calibrated probabilities where sufficient labels exist.

## 6. Ablation and release

`run_ablations.py` plans baseline, committee, rules-on, double-pass and alternate
extractor runs. Each variant changes one baseline setting; feedback examples are
disabled to prevent annotation leakage. All outputs are isolated by variant,
repeat and DOI; no formal DB ingest occurs. Source hashes, config hashes,
failures, reported token usage and latency are retained. Provider-billed retries
without returned usage may not appear in token totals; costs must be reconciled
with actual provider usage before publication.

The default action only creates a plan. `--execute` performs live model calls;
`--diagnostic` permits runs before expert benchmark lock but keeps them outside
publication validation. A runtime API key and installed dependencies are required.
Do not compare variants on different paper subsets or remove failed papers.

Use `scripts/release/check_release.py` to record environment, tests, source
hashes and remaining scientific blockers. A local software snapshot can be
reproducible while the scientific release remains not ready. Publishing, public
data licensing and human review are separate completion conditions.
