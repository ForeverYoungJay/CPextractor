# Extractor Schema Design

This note explains the intended role of the extractor schema in CPextractor.

The extractor should answer five linked questions for each paper:

1. What material is studied?
2. Under what processing state or loading condition?
3. Which model/framework is used?
4. Which phase / mechanism / family / system does a parameter act on?
5. Where is the supporting evidence?

## Design Principles

- `parameter_claims[]` remains the atomic review and database unit.
- `materials[]`, `process_states[]`, `conditions[]`, `models[]`, `mechanisms`, and `microstructure_features[]` are support graphs for binding and interpretation.
- Keep nested `materials[] -> phases[]` because the existing postprocess and compatibility projection already assume phase ownership under a material.
- Keep `process_states[]` separate from `conditions[]`.
  Processing state means material history or specimen state.
  Condition means deformation, testing, calibration, or service-like loading context.
- Keep evidence lightweight at extraction time, then let `postprocess/evidence_grounding.py` turn it into reusable `evidence_objects`.

## Why This Fits The Project

The project is not only extracting material descriptions.
Its real downstream product is a trustworthy parameter claim with:

- parameter identity
- numeric value
- applicability binding
- provenance
- evidence

So the schema must make those bindings explicit rather than hide them in free text.

## Recommended Object Roles

### `materials[]`

Use for stable material identity:

- alloy or material name
- formula
- material class
- aliases used in the paper
- composition
- phase catalog

Do not overload `materials[]` with every heat treatment or test case.

### `process_states[]`

Use for material-state variants:

- as-built
- annealed
- aged
- solution-treated
- different reduction ratios
- different grain-size bins caused by processing

This is where sample-to-sample material history belongs.

### `conditions[]`

Use for loading or testing context:

- temperature
- strain rate
- fatigue mode
- indentation settings
- loading path
- calibration versus validation condition

This keeps "under what condition" separate from "what material state".

### `models[]`

Use for:

- CPFE / FFT / VPSC / DAMASK / UMAT
- constitutive law choices
- rate dependence
- hardening law
- implementation platform

A parameter claim should point to `model_id` when the model is explicit.

### `microstructure_features[]`

Use for contextual structure facts that are not parameters:

- grain size
- phase fraction
- texture
- precipitates
- porosity
- selected grains

Each feature should say:

- stable family bucket
- paper-facing feature name
- value type
- applicability target
- direct evidence

### `parameter_claims[]`

This is the main curated unit.
Each claim should carry:

- parameter identity
- assertion/value
- context
- provenance
- evidence

The important binding pattern is:

- `context.material_id`
- `context.phase_id`
- `context.process_state_id`
- `context.condition_id`
- `context.mechanism_scope.level`
- `context.mechanism_scope.mechanism_type`
- `context.mechanism_scope.family_id`
- `context.mechanism_scope.system_ids`

## Evidence Strategy

At extraction time, store direct human-readable evidence:

- `evidence.text`
- `evidence.source_type`
- `evidence.source_id`
- `evidence.source_file`
- `evidence.section_heading`
- `evidence.table_evidence`

Later, grounding can resolve these into reusable normalized evidence objects.

This two-stage approach is better than forcing the extractor to predict final span ids.

## Schema Changes Added In v3.2

- `materials[].aliases`
- `materials[].source_label`
- `materials[].phases[].source_label`
- `materials[].phases[].volume_fraction.basis`
- `process_states[].state_type`
- `process_states[].state_variables[]`
- `conditions[].condition_role`
- `models[].applies_to`
- `microstructure_features[].feature_family`
- `microstructure_features[].feature_name`
- `microstructure_features[].value_type`
- `microstructure_features[].reported_value`
- `microstructure_features[].evidence`
- `parameter_claims[].context.target_description`
- `parameter_claims[].evidence.source_type`
- `parameter_claims[].evidence.source_id`
- `parameter_claims[].evidence.source_file`
- `parameter_claims[].evidence.section_heading`

## What We Deliberately Did Not Do

- We did not move `phases[]` to top level.
  The project already treats phase as material-owned, and that relationship is useful.
- We did not collapse `process_states[]` and `conditions[]`.
  They answer different scientific questions.
- We did not make grounding ids mandatory at extraction time.
  Those are better produced by postprocess.
