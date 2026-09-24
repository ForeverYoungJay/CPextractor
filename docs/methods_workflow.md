# Implemented methods workflow

The authoritative evaluation specification is [benchmark_protocol.md](benchmark_protocol.md).
The runtime configuration example is `config.example.yaml`.

1. Discover DOIs from Scopus, explicit lists or local fulltext directories.
2. Parse Elsevier XML into sections, tables, equations and references.
3. Select relevant evidence and extract hierarchical v6.0.0 parameter claims.
4. Apply extractor-first structure handling, units, material/model normalization,
   equation binding and evidence linking. Legacy rewrite stages are skipped for
   newer payloads.
5. Run a configurable LLM evaluator; the current default is single_judge, with
   committee mode available as an experimental variant.
6. Optionally run deterministic quality checks (off by default), fuse confidence,
   finalize claims and gate structured DB ingestion. Audit artifacts remain local.
7. Store structured records, vectors, reference relations and evaluation records;
   support hybrid retrieval, the chatbot and graph projections.

## Validation status and reporting

No general PDF fallback or universal successful grounding is asserted. Evidence
references may be dangling and must be checked against actual source objects.
Internal confidence and quality tiers are operational, not expert-calibrated truth.

The paper-level evaluation universe is fixed before scoring. Matching is
one-to-one, independent of values/units and generated claim IDs. Pending papers
are not negatives. Field errors are scored only against actually reviewed labels.
Explicit gate labels support false-admission/false-block rates; calibration/test
DOIs must be disjoint. AI review remains a diagnostic tier.

Historical corpus counts do not establish extraction accuracy. Final manuscript
performance claims require an exhaustive expert gold set, a frozen test run,
confidence/gate validation, complete ablation records and a reproducible source
snapshot with versioned inputs.
