import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from llm.evaluator import run_llm_evaluation
from postprocess.record_links import inflate_registry_from_claims
from postprocess.workflow import (
    run_structure_normalization,
    run_linking,
    run_deterministic_validation,
    run_finalization,
)
from postprocess.compact_export import write_compact_summary


def _load_audit_input(paper_dir: Path) -> tuple[dict, str]:
    raw_path = paper_dir / "materials_extracted.extractor_raw.json"
    final_path = paper_dir / "materials_extracted.json"

    if raw_path.exists():
        extracted = json.loads(raw_path.read_text(encoding="utf-8"))
        return extracted if isinstance(extracted, dict) else {}, raw_path.name

    extracted = json.loads(final_path.read_text(encoding="utf-8"))
    if not isinstance(extracted, dict):
        return {}, final_path.name
    extracted = inflate_registry_from_claims(extracted)
    return extracted, final_path.name


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit existing extracted papers without rerunning extraction.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--root", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    llm_cfg = cfg.get("llm", {})
    root = Path(args.root or cfg.get("paths", {}).get("fulltext", "data/fulltext"))

    files = sorted(root.glob("*/materials_extracted.json"))
    if args.limit > 0:
        files = files[:args.limit]

    for path in files:
        paper_dir = path.parent
        backup_path = paper_dir / "materials_extracted.pre_audit_backup.json"
        if not backup_path.exists():
            backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        extracted, source_name = _load_audit_input(paper_dir)
        if not extracted:
            print(f"Skip {paper_dir.name}: unable to load audit input")
            continue

        reports = {}
        reports["audit_input"] = {
            "source_file": source_name,
            "prefer_extractor_raw": source_name == "materials_extracted.extractor_raw.json",
        }
        extracted, structure_reports = run_structure_normalization(
            extracted,
            paper_dir=str(paper_dir),
        )
        reports.update(structure_reports)
        extracted, linking_reports = run_linking(
            extracted,
            paper_dir=str(paper_dir),
        )
        reports.update(linking_reports)
        extracted, validation_reports = run_deterministic_validation(extracted)
        reports.update(validation_reports)

        if bool(llm_cfg.get("enable_evaluator", False)):
            (paper_dir / "materials_extracted.pre_evaluator.json").write_text(
                json.dumps(extracted, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            evaluation, eval_metrics = run_llm_evaluation(
                paper_dir=str(paper_dir),
                extracted_json=extracted,
                model_evaluate=llm_cfg.get("model_evaluate", llm_cfg.get("model_extract", "gpt-4.1-mini")),
                max_context_chars=int(llm_cfg.get("max_evaluate_context_chars", 18000)),
                max_retries=int(llm_cfg.get("max_evaluate_retries", 2)),
                parameter_limit=int(llm_cfg.get("evaluate_parameter_limit", 40)),
                field_batch_size=int(llm_cfg.get("evaluate_parameter_batch_size", 12)),
                per_evidence_chars=int(llm_cfg.get("evaluate_evidence_chars", 800)),
                quality_report=reports.get("quality_checks"),
                evidence_report=reports.get("evidence_grounding"),
                feedback_artifact_path=llm_cfg.get("evaluation_feedback_json"),
            )
            reports["llm_evaluation"] = evaluation
            reports["llm_evaluation_metrics"] = eval_metrics
            (paper_dir / "llm_evaluation.json").write_text(
                json.dumps(evaluation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        extracted, final_reports = run_finalization(
            extracted,
            evaluation_report=reports.get("llm_evaluation"),
            quality_report=reports.get("quality_checks"),
        )
        reports.update(final_reports)

        path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
        (paper_dir / "postprocess_report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_compact_summary(
            paper_dir,
            extracted,
            postprocess_report=reports,
            llm_evaluation=reports.get("llm_evaluation"),
        )
        print(f"Audited -> {paper_dir.name}")


if __name__ == "__main__":
    main()
