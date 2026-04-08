import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

from postprocess.param_iter import iter_parameter_items_with_index


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _iter_paper_dirs(root: Path) -> Iterable[Path]:
    for paper_dir in sorted(root.iterdir()):
        if not paper_dir.is_dir():
            continue
        if (paper_dir / "materials_extracted.json").exists():
            yield paper_dir


def _audit_map(llm_evaluation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _safe_list(llm_evaluation.get("parameter_audits")):
        if not isinstance(row, dict):
            continue
        location = str(row.get("location") or "").strip()
        if location:
            out[location] = row
    return out


def _provenance_map(extracted: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _safe_list(extracted.get("provenance_records")):
        if not isinstance(row, dict):
            continue
        pid = str(row.get("provenance_id") or "").strip()
        if pid:
            out[pid] = row
    return out


def _claim_rows_for_paper(paper_dir: Path, source_name: str) -> List[Dict[str, Any]]:
    src_path = paper_dir / source_name
    extracted = _load_json(src_path)
    if not extracted:
        return []

    doi = (
        extracted.get("record_id")
        or _safe_dict(extracted.get("source_document")).get("doi")
        or paper_dir.name.replace("_", "/")
    )
    material = _safe_dict(extracted.get("material"))
    document = _safe_dict(extracted.get("document"))
    materials = [m for m in _safe_list(extracted.get("materials")) if isinstance(m, dict)]
    primary_material = _safe_dict(materials[0]) if materials else {}
    material_name = primary_material.get("name") or material.get("name")
    material_phase_mode = _safe_dict(extracted.get("study")).get("study_type") or material.get("phase")
    llm_eval = _load_json(paper_dir / "llm_evaluation.json")
    audit_map = _audit_map(llm_eval)

    rows: List[Dict[str, Any]] = []
    for idx, _, item in iter_parameter_items_with_index(extracted):
        location = str(item.get("claim_id") or f"parameters.registry[{idx}]")
        location = f"claim:{location}" if not location.startswith("parameters.registry[") and not location.startswith("claim:") else location
        audit = audit_map.get(location) or audit_map.get(f"parameters.registry[{idx}]") or {}
        applies_to = _safe_dict(item.get("applies_to"))
        source = _safe_dict(item.get("source"))
        provenance = _safe_dict(item.get("provenance")) or _safe_dict(source)
        claim_evidence = _safe_dict(item.get("evidence"))
        table_evidence = _safe_dict(claim_evidence.get("table_evidence"))

        rows.append({
            "doi": document.get("doi") or doi,
            "paper_dir": str(paper_dir),
            "claim_id": item.get("claim_id") or f"claim_{idx + 1:04d}",
            "record_index": idx,
            "material_name": material_name,
            "material_phase_mode": material_phase_mode,
            "canonical_name": item.get("canonical_name"),
            "symbol": item.get("symbol"),
            "domain": item.get("domain"),
            "value": item.get("value"),
            "unit": item.get("unit"),
            "value_SI": item.get("value_SI"),
            "unit_SI": item.get("unit_SI"),
            "scope": {
                "scope": applies_to.get("scope"),
                "phase_id": applies_to.get("phase_id"),
                "mechanism": applies_to.get("mechanism"),
                "family_id": applies_to.get("family_id"),
                "family_name": applies_to.get("family_name"),
                "system_ids": _safe_list(applies_to.get("system_ids")),
            },
            "provenance": {
                "origin_type": provenance.get("origin_type"),
                "provenance_id": source.get("provenance_id"),
                "reference_ids": _safe_list(provenance.get("references")),
                "adopted_from_reference_ids": _safe_list(provenance.get("adopted_from_references")),
                "calibration_based_on_reference_ids": _safe_list(provenance.get("calibration_based_on_references")),
                "calibration_in_this_study": provenance.get("calibration_in_this_study"),
                "calibration_method": provenance.get("calibration_method"),
            },
            "evidence": {
                "kind": "table" if table_evidence else None,
                "file": claim_evidence.get("file"),
                "page": None,
                "row_name": table_evidence.get("row_name"),
                "column_name": table_evidence.get("column_name"),
                "value_text": table_evidence.get("value"),
                "snippet": claim_evidence.get("evidence_text") or table_evidence.get("excerpt"),
            },
            "prediction_context": {
                "source_file": source_name,
                "grounding_status": _safe_dict(item.get("quality_assessment")).get("grounding_status"),
                "llm_verdict": audit.get("verdict"),
                "review_required": audit.get("review_required"),
                "policy_adjustments": _safe_list(audit.get("policy_adjustments")),
            },
            "annotation": {
                "status": "correct",
                "error_tags": [],
                "notes": "",
            },
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Export claim-level annotation draft jsonl from pipeline outputs.")
    ap.add_argument("--input-root", default="data/fulltext")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--source",
        default="materials_extracted.json",
        choices=["materials_extracted.json", "materials_extracted.extractor_raw.json", "materials_extracted.pre_evaluator.json"],
        help="Prediction artifact to export as annotation draft source.",
    )
    args = ap.parse_args()

    root = Path(args.input_root)
    rows: List[Dict[str, Any]] = []
    for paper_dir in _iter_paper_dirs(root):
        rows.extend(_claim_rows_for_paper(paper_dir, args.source))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Exported {len(rows)} draft claims -> {out_path}")


if __name__ == "__main__":
    main()
