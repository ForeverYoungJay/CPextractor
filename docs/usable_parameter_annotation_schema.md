# Usable Parameter Annotation Schema

This schema is for **parameter-centric manual annotation** when the goal is not only to judge whether an extracted claim is "correct", but whether the parameter record is **usable by another researcher** with the minimum necessary scientific context.

The annotation target is a **usable parameter record** rather than a raw JSON subtree.

## Design Goal

A record should contain enough information for another person to decide:

1. what material or constituent the parameter belongs to
2. what CP model family / constitutive setting it belongs to
3. what the parameter itself is
4. what scope the parameter acts on
5. what evidence supports it
6. how the value was obtained or sourced

This is intentionally narrower than full-document annotation and broader than value-only claim review.

## Core Blocks

Each record is a JSON object with these top-level blocks:

- `material_object`
- `cp_model`
- `parameter_body`
- `parameter_scope`
- `evidence`
- `provenance`
- `annotation`

Support fields such as `doi`, `paper_dir`, `claim_id`, and `record_index` may also be included for tracking and packet assembly, but they are not part of the scientific minimum-use definition.

## Record Example

```json
{
  "doi": "10.1016/j.actamat.2024.120250",
  "paper_dir": "data/fulltext/10.1016_j.actamat.2024.120250",
  "claim_id": "claim_c11_316h",
  "record_index": 0,
  "material_object": {
    "material_id": "mat_316h",
    "material_name": "316H austenitic stainless steel",
    "material_class": "steel",
    "phase_mode": "single_phase",
    "constituent_id": "const_316h_austenite_fcc",
    "constituent_name": "austenite",
    "constituent_type": "phase",
    "crystal_structure": "FCC",
    "process_state_id": "ps_316h_as_received_solution_annealed",
    "process_state_label": "as-received 316H, forged and solution annealed"
  },
  "cp_model": {
    "model_id": "model_cpfe_bristol_316h",
    "model_label": "BRISTOL crystal plasticity solver for 316H | power_law | Voce-type with saturation and recovery including creep term",
    "model_name": "BRISTOL crystal plasticity solver for 316H",
    "model_type": "crystal_plasticity",
    "model_role": "primary_simulation",
    "kinematics": "finite_strain",
    "flow_rule_form": "power_law",
    "rate_dependence": "rate_dependent",
    "hardening_law": "Voce-type with saturation and recovery including creep term",
    "solver_scale": "polycrystal",
    "discretization": "fem",
    "software": "Abaqus",
    "code_name": "BRISTOL",
    "branch_ids": [],
    "branch_labels": []
  },
  "parameter_body": {
    "canonical_name": "c11",
    "symbol": "c_{11}",
    "parameter_family": "elastic_constants",
    "raw_name": "c 11",
    "domain": "elastic",
    "value": 183.9,
    "unit": "GPa"
  },
  "parameter_scope": {
    "scope_level": "constituent",
    "scope_target": "constituent:austenite",
    "mechanism": "other",
    "family_id": null,
    "family_name": null,
    "system_ids": [],
    "system_names": [],
    "condition_id": "cond_316h_823k_550c_cpfe",
    "condition_label": "CPFE calibration and simulation at 550°C",
    "temperature_text": "823 K",
    "strain_rate_text": null,
    "notes": "From parameter table captioned 'Calibrated model parameters used for CPFE simulations'."
  },
  "evidence": {
    "evidence_id": "ev_table_cpfe_c11",
    "evidence_type": "table_row",
    "table_id": "Table 2",
    "snippet": "c11 = 183.9 GPa",
    "section_heading": "Methods",
    "page": null
  },
  "provenance": {
    "origin_type": "calibrated",
    "reference_ids": [],
    "adopted_from_reference_ids": [],
    "calibration_based_on_reference_ids": ["5"],
    "calibration_method": "manual_fitting",
    "target_type": "stress_strain_curve",
    "target_description": "Macroscopic stress-strain data at 550°C for as-received 316H steel.",
    "observation_scope": "macroscopic",
    "notes": "Same calibration campaign as other CPFE parameters."
  },
  "annotation": {
    "status": "correct"
  }
}
```

## Minimal Scientific Requirement

For a record to be considered scientifically usable, the annotation should be able to answer all of these:

- **Material object:** who this parameter belongs to
- **CP model:** which constitutive model family this parameter belongs to
- **Parameter body:** what the parameter is and what numeric value/unit is claimed
- **Scope:** whether it is global, constituent-level, family-level, or system-level
- **Evidence:** what source sentence / table cell supports the record
- **Provenance:** whether it was calibrated, adopted, assumed, original, or otherwise obtained

## Field Guidance

### 1. `material_object`

Minimum fields:

- `material_name`
- `constituent_name` when applicable
- `process_state_label` when applicable

Recommended additions:

- `material_class`
- `phase_mode`
- `crystal_structure`

### 2. `cp_model`

Minimum fields:

- `model_type`
- `model_name` or model summary

Recommended additions:

- `model_label`
- `kinematics`
- `flow_rule_form`
- `rate_dependence`
- `hardening_law`
- `solver_scale`
- `discretization`

### 3. `parameter_body`

Required:

- `canonical_name`
- `symbol`
- `value`
- `unit`

Recommended additions:

- `parameter_family`
- `raw_name`
- `domain`

### 4. `parameter_scope`

Required:

- `scope_level`
- `scope_target`

Recommended additions:

- `mechanism`
- `family_name`
- `system_names`
- `condition_label`
- `temperature_text`
- `strain_rate_text`

When the source contains deformation-system metadata, `scope_target` can be made more specific by resolving `family_id` into family details such as family name, plane/direction, and number of systems. For example:

- `family:prismatic ({10-10}<11-20>, n=3)`

### 5. `evidence`

Required:

- `evidence_type`
- `snippet`

Recommended additions:

- `table_id`
- `section_heading`

### 6. `provenance`

Required:

- `origin_type`

Recommended additions:

- `reference_ids`
- `adopted_from_reference_ids`
- `calibration_based_on_reference_ids`
- `calibration_method`
- `target_type`

## Annotation Fields

The `annotation` block is intentionally minimal.

### `annotation.status`

Recommended values:

- `correct`
- `incorrect`

Draft exports should initialize this field as `correct`. During manual review, change it to `incorrect` only when the record should not be accepted as-is.

## Packet Editing Recommendation

For spreadsheet-style editing, the recommended core columns are:

- `material_object.material_name`
- `material_object.constituent_name`
- `cp_model.model_type`
- `parameter_body.canonical_name`
- `parameter_body.symbol`
- `parameter_body.domain`
- `parameter_body.value`
- `parameter_body.unit`
- `parameter_scope.scope_level`
- `parameter_scope.scope_target`
- `parameter_scope.condition_label`
- `evidence.evidence_type`
- `evidence.snippet`
- `provenance.origin_type`
- `annotation.status`

## Scoring Direction

This schema is suitable for a future benchmark that asks:

1. is the parameter record present?
2. is it sufficiently specified to reuse?
3. which minimum-use blocks are correct or missing?

That benchmark should score by block rather than by raw JSON path.
