# Schema History

This document preserves the schema history of CPextractor as found from git history and the current workspace on 2026-04-10.

## Scope

There are two schema layers in this repository:

- extractor schema: the JSON schema used by `llm/extractor.py` during LLM extraction
- final hierarchy schema: the normalized document shape produced by `postprocess/final_hierarchy.py`

These two layers do not currently share the same version number, so they are listed separately below.

## Extractor Schema Versions

| Version | Date | Source | Evidence |
| --- | --- | --- | --- |
| `1.1.0` | 2026-03-04 | git commits `3b3c01f`, `abdb505` | `git show abdb505:llm/extractor.py` contains `"schema_version": "1.1.0"` |
| `2.0.0` | 2026-03-12 | git commit `d5346b5` | `git show d5346b5:llm/extractor.py` contains `"schema_version": "2.0.0"` |
| `2.1.1` | 2026-03-12 | git commit `09f65de` | `git show 09f65de:llm/extractor.py` contains `"schema_version": "2.1.1"` |
| `3.0.0` | 2026-04-08 | git commit `9ac9896` | `git show 9ac9896:llm/extractor.py` contains `"schema_version": "3.0.0"` |
| `3.1.0` | 2026-04-08 | git commit `ac7f652` | `git show ac7f652:llm/extractor.py` contains `"schema_version": "3.1.0"` |
| `4.2.0` | 2026-04-09 | git commit `662057d` | `git show 662057d:llm/extractor.py` contains `"schema_version": "4.2.0"` |
| `5.0.2` | 2026-04-21 workspace state | current uncommitted working tree | historical workspace version before the 5.1.0 schema expansion |
| `5.1.0` | 2026-05-12 workspace state | historical working-tree state before extractor-first tightening | previous schema line with document/output/evaluation blocks still inside extractor |
| `5.1.1` | 2026-05-13 workspace state | current uncommitted working tree | [`llm/extractor.py`](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/llm/extractor.py#L1221) contains `"schema_version": "5.1.1"` |

## Extractor Schema Timeline

1. `1.1.0`
   Early schema used before the March 2026 schema-v2 refactor.

2. `2.0.0`
   Introduced with the seven-layer pipeline upgrade.

3. `2.1.1`
   Refined the v2 structure into a registry-oriented extraction shape.

4. `3.0.0`
   Introduced a v3 extractor document with `document`, `materials`, and richer hierarchy fields.

5. `3.1.0`
   Restored the extractor schema after subsequent edits.

6. `4.2.0`
   Refined the extractor again and removed legacy output views.

7. `5.1.0`
   Added top-level `deformation_systems`, `simulation_geometries`, `orientation_inputs`, `numerical_methods`, `simulation_outputs`, and `model_evaluations`.
   Added `parameter_claims[].claim_class`.
   Removed `assertion.qualifier`.
   Clarified that `reference_ids` is the full claim-level reference union, while `adopted_from_reference_ids` and `calibration_based_on_reference_ids` are role-specific subsets that may overlap and may also repeat IDs already present in `reference_ids`.

8. `5.1.1`
   Tightened the extractor into a more extractor-first schema.
   Removed `document`, `simulation_outputs`, and `model_evaluations` from the extractor schema.
   Added direct cross-link IDs across materials, process states, conditions, geometries, orientation inputs, numerical methods, families, branches, and systems.
   Restored explicit family/system binding fields on `parameter_claims[].applies_to`.
   Normalized mixed value fields into `*_value_text` plus structured numeric/range/comparator fields where high ambiguity previously caused drift.

## Final Hierarchy Schema

The finalized stored document is still treated as hierarchical `v3`, even though the extractor schema has already advanced to the `4.x` line.

- README states that "the finalized stored document is now a hierarchical v3 schema":
  [`README.md`](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/README.md#L36)
- The final hierarchy writer still overwrites output to `3.0.0`:
  [`postprocess/final_hierarchy.py`](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/postprocess/final_hierarchy.py#L463)
- The related test also asserts `3.0.0`:
  [`tests/test_final_hierarchy.py`](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/tests/test_final_hierarchy.py#L80)

## Current Version Split

As of 2026-05-13:

- extractor schema: `5.1.1`
- configured pipeline schema version in config: `5.1.1`
- final hierarchy schema written by postprocess: `5.1.0`

The repository still preserves historical schema lines, but the active extractor now operates on the `5.1.1` line while the downstream finalized hierarchy remains on the `5.1.0` family.
