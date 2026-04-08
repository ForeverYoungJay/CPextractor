import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _approved_paper_dirs(root: Path) -> list[Path]:
    out = []
    for paper_dir in sorted(root.iterdir()):
        if not paper_dir.is_dir():
            continue
        decision = _load_json(paper_dir / "review_decision.json")
        if bool(decision.get("approved_for_ingest")):
            out.append(paper_dir)
    return out


def _resolve_paper_doi(extracted: dict, paper_dir_name: str) -> str:
    source_document = extracted.get("source_document", {}) if isinstance(extracted.get("source_document"), dict) else {}
    return (
        source_document.get("doi")
        or extracted.get("record_id")
        or paper_dir_name.replace("_", "/")
    )


def main() -> None:
    import yaml
    from openai import OpenAI
    from db.pg import connect_pg
    from pipelines.decision_layer import apply_decision_layer, build_ingest_gate_report

    ap = argparse.ArgumentParser(description="Ingest manually approved papers without rerunning extraction.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--root", default="")
    ap.add_argument("--doi", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = cfg.get("paths", {})
    llm_cfg = cfg.get("llm", {})
    rag_cfg = cfg.get("rag", {})
    pipeline_cfg = cfg.get("pipeline", {})
    db = cfg["db"]

    root = Path(args.root or paths.get("fulltext", "data/fulltext"))
    if args.doi:
        target_dir = root / args.doi.replace("/", "_")
        paper_dirs = [target_dir] if target_dir.exists() else []
    else:
        paper_dirs = _approved_paper_dirs(root)

    if not paper_dirs:
        print("No approved papers found.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    conn = connect_pg(db["host"], int(db["port"]), db["name"], db["user"], db["password"])
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    gate_on_evaluation = bool(pipeline_cfg.get("gate_on_evaluation", True))
    gate_block_verdicts = {
        str(v).strip().lower()
        for v in (pipeline_cfg.get("db_ingest_block_verdicts", ["fail"]) or ["fail"])
    }
    gate_min_confidence = float(pipeline_cfg.get("db_ingest_min_document_confidence_score", 65.0))
    gate_block_review_escalation_on_pass = bool(pipeline_cfg.get("db_ingest_block_review_escalation_on_pass", False))
    gate_review_escalation_pass_min_confidence = float(
        pipeline_cfg.get("db_ingest_review_escalation_pass_min_confidence_score", 85.0)
    )
    gate_review_escalation_pass_max_review_required = int(
        pipeline_cfg.get("db_ingest_review_escalation_pass_max_review_required_parameters", 2)
    )

    for paper_dir in paper_dirs:
        extracted = _load_json(paper_dir / "materials_extracted.json")
        reports = _load_json(paper_dir / "postprocess_report.json")
        if not extracted or not reports:
            print(f"Skip {paper_dir.name}: missing materials_extracted.json or postprocess_report.json")
            continue

        doi = _resolve_paper_doi(extracted, paper_dir.name)

        reports["ingest_gate"] = build_ingest_gate_report(
            evaluation_report=reports.get("llm_evaluation"),
            confidence_report=reports.get("confidence_fusion"),
            extracted_json=extracted,
            enabled=gate_on_evaluation,
            blocked_verdicts=gate_block_verdicts,
            min_document_confidence_score=gate_min_confidence,
            paper_dir=str(paper_dir),
            block_review_escalation_on_pass=gate_block_review_escalation_on_pass,
            review_escalation_pass_min_confidence_score=gate_review_escalation_pass_min_confidence,
            review_escalation_pass_max_review_required_parameters=gate_review_escalation_pass_max_review_required,
        )
        (paper_dir / "postprocess_report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        blocked = apply_decision_layer(
            conn=conn,
            openai_client=openai_client,
            doi=doi,
            paper_dir=str(paper_dir),
            extracted_json=extracted,
            reports=reports,
            llm_cfg=llm_cfg,
            rag_cfg=rag_cfg,
        )
        conn.commit()
        if blocked:
            print(f"Still gated -> {doi}")
        else:
            print(f"Ingested approved paper -> {doi}")

    conn.close()


if __name__ == "__main__":
    main()
