import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from postprocess.record_links import inflate_registry_from_claims
from postprocess.claim_builder import build_parameter_claims
from postprocess.provenance_normalizer import normalize_provenance
from postprocess.parameter_table_resolver import resolve_parameter_tables
from postprocess.condition_binding import resolve_condition_bindings
from postprocess.evidence_grounding import verify_evidence_grounding
from postprocess.quality_checks import run_quality_checks
from postprocess.compact_export import write_compact_summary


def _load_fix_input(doc_path: Path) -> tuple[dict, str]:
    raw_path = doc_path.parent / "materials_extracted.extractor_raw.json"
    if raw_path.exists():
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}, raw_path.name

    payload = json.loads(doc_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}, doc_path.name
    payload = inflate_registry_from_claims(payload)
    return payload, doc_path.name


def _claim_loc_map(extracted: dict) -> dict[str, str]:
    out = {}
    registry = ((extracted.get("parameters") or {}).get("registry") or [])
    for idx, item in enumerate(registry):
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("claim_id") or "").strip()
        if claim_id:
            out[f"parameters.registry[{idx}]"] = f"claim:{claim_id}"
    return out


def _rewrite_eval_locations(evaluation: dict, loc_map: dict[str, str]) -> dict:
    if not isinstance(evaluation, dict):
        return evaluation
    for audit in evaluation.get("parameter_audits") or []:
        if isinstance(audit, dict):
            loc = str(audit.get("location") or "")
            if loc in loc_map:
                audit["location"] = loc_map[loc]
            committee = audit.get("committee") if isinstance(audit.get("committee"), dict) else {}
            for rec in committee.values():
                if isinstance(rec, dict):
                    rloc = str(rec.get("location") or "")
                    if rloc in loc_map:
                        rec["location"] = loc_map[rloc]
    for issue in evaluation.get("critical_issues") or []:
        if isinstance(issue, dict):
            loc = str(issue.get("location") or "")
            if loc in loc_map:
                issue["location"] = loc_map[loc]
    return evaluation


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply deterministic issue-driven fixes to existing extracted JSON files.")
    ap.add_argument("--paper-dir", default="")
    ap.add_argument("--root", default="data/fulltext")
    args = ap.parse_args()

    if args.paper_dir:
        paper_dirs = [Path(args.paper_dir)]
    else:
        paper_dirs = sorted(Path(args.root).glob("*"))

    for paper_dir in paper_dirs:
        doc_path = paper_dir / "materials_extracted.json"
        if not doc_path.exists():
            continue
        try:
            extracted, source_name = _load_fix_input(doc_path)
        except Exception:
            continue
        if not extracted:
            continue
        extracted, prov_report = normalize_provenance(extracted)
        extracted, table_report = resolve_parameter_tables(extracted, str(paper_dir))
        extracted, binding_report = resolve_condition_bindings(extracted)
        extracted, evidence_report = verify_evidence_grounding(extracted, str(paper_dir))
        extracted, quality_report = run_quality_checks(extracted)

        reports = {
            "provenance_normalization": prov_report,
            "parameter_table_resolution": table_report,
            "condition_binding": binding_report,
            "evidence_grounding": evidence_report,
            "quality_checks": quality_report,
            "autofix_input": {
                "source_file": source_name,
                "prefer_extractor_raw": source_name == "materials_extracted.extractor_raw.json",
            },
        }
        llm_eval_path = paper_dir / "llm_evaluation.json"
        llm_evaluation = {}
        if llm_eval_path.exists():
            try:
                llm_evaluation = json.loads(llm_eval_path.read_text(encoding="utf-8"))
            except Exception:
                llm_evaluation = {}
        loc_map = _claim_loc_map(extracted)
        llm_evaluation = _rewrite_eval_locations(llm_evaluation, loc_map)
        extracted, _ = build_parameter_claims(
            extracted,
            evaluation_report=llm_evaluation,
            confidence_report={},
        )

        doc_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
        (paper_dir / "postprocess_report.autofix.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if llm_evaluation:
            llm_eval_path.write_text(json.dumps(llm_evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        write_compact_summary(paper_dir, extracted, postprocess_report=reports, llm_evaluation=llm_evaluation)
        print(f"Auto-fixed -> {paper_dir}")


if __name__ == "__main__":
    main()
