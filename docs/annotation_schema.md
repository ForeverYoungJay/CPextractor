# Annotation Schema

This project supports claim-level manual annotation using model output as a draft and human correction as gold.

## Goals

The annotation format is designed to support:

- `precision`
- `recall`
- `f1`
- `grounding_accuracy`
- `bundle_completeness`
- `unit_accuracy`
- `mapping_accuracy`
- `binding_accuracy`
- `provenance_accuracy`

Annotators should correct claim drafts rather than annotate papers from scratch whenever possible.

## Files

Recommended files:

- `annotation_draft.jsonl`
  Model-generated draft for human correction.
- `gold_claims.jsonl`
  Human-corrected gold claim file.
- `gold_bundles.jsonl`
  Optional per-paper bundle completeness reference.

## Claim-Level Schema

One JSON object per line:

```json
{
  "doi": "10.1016/j.actamat.2024.120250",
  "paper_dir": "data/fulltext/10.1016_j.actamat.2024.120250",
  "claim_id": "claim_0016",
  "record_index": 15,
  "canonical_name": "hardening_coefficient_h",
  "symbol": "h",
  "domain": "plastic",
  "value": 3555.0,
  "unit": "MPa",
  "value_SI": 3555000000.0,
  "unit_SI": "Pa",
  "scope": {
    "scope": "global",
    "phase_id": null,
    "mechanism": "all_slip",
    "family_id": null,
    "family_name": null,
    "system_ids": []
  },
  "provenance": {
    "origin_type": "calibrated",
    "provenance_id": "prov_0001",
    "reference_ids": [],
    "adopted_from_reference_ids": [],
    "calibration_based_on_reference_ids": [],
    "calibration_in_this_study": true,
    "calibration_method": null
  },
  "evidence": {
    "kind": "table",
    "file": "table_002.json",
    "page": null,
    "row_name": "h",
    "column_name": "Hardening coefficients",
    "value_text": "3555 MPa",
    "snippet": "Row 12: h, h D | Hardening coefficients | 3555 MPa, 245"
  },
  "prediction_context": {
    "source_file": "materials_extracted.json",
    "grounding_status": "table_cell_match",
    "llm_verdict": "warning",
    "review_required": true,
    "policy_adjustments": [
      "suppress_false_unit_conversion"
    ]
  },
  "annotation": {
    "status": "correct",
    "error_tags": [],
    "notes": ""
  }
}
```

## Annotation Status

Use exactly one primary status:

- `correct`
- `wrong_value`
- `wrong_unit`
- `wrong_mapping`
- `wrong_binding`
- `wrong_provenance`
- `insufficient_evidence`
- `spurious_claim`
- `missing_from_prediction`

`missing_from_prediction` is used for claims added manually that were absent from model output.

## Error Tags

Use `annotation.error_tags` for optional finer labels:

- `numeric_tolerance`
- `si_only_issue`
- `row_split_error`
- `grouped_row_error`
- `shared_scope_not_expanded`
- `image_table_error`
- `missing_zero_value`
- `missing_unit_allowed`
- `wrong_table`
- `wrong_section`

## Bundle Schema

Optional per-paper bundle file for completeness metrics:

```json
{
  "doi": "10.1016/j.actamat.2024.120250",
  "bundle_id": "main_cp_parameter_set",
  "required_claim_ids": [
    "claim_0001",
    "claim_0002",
    "claim_0016",
    "claim_0017"
  ],
  "required_groups": [
    "elastic_constants",
    "rate_parameters",
    "hardening_parameters"
  ],
  "notes": ""
}
```

## Recommended Workflow

1. Export draft claims from predictions.
2. For the default lightweight workflow, edit only:
   - `canonical_name` or `symbol` when the parameter label is wrong
   - `value`
   - `unit`
   - `annotation.status`
   - `annotation.notes`
3. Keep the CSV compact during first-pass review; use the full JSONL only when you need to inspect grounding or provenance.
4. Add missing claims manually with `status = missing_from_prediction`.
5. Save as `gold_claims.jsonl`.
6. Normalize and benchmark against predictions.

## Matching Guidance

Recommended claim match key for benchmark:

- `doi`
- `canonical_name`
- coarse binding fields:
  - `scope.scope`
  - `scope.phase_id`
  - `scope.mechanism`
  - `scope.family_id` or `scope.family_name`

Then compare:

- value with tolerance
- unit
- provenance
- grounding

## Practical Advice

- Do not require annotators to read the full paper first.
- Start from the draft and only open the source table/section when needed.
- Prioritize high-value cases:
  - image-backed tables
  - grouped rows
  - missing-unit cases
  - warnings/fails
  - disagreement cases
