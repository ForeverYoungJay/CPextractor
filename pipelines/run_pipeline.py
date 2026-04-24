# pipelines/run_pipeline.py
import os
import yaml
from openai import OpenAI
import sys
import json
import time
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup

sys.path.append(".")

from db.pg import connect_pg
from scopus.scopus_search import scopus_search
from elsevier.fulltext_parser import (
    save_paper_as_markdown_and_tables,
    safe_id,
    extract_references_from_xml,
)
from llm.extractor import run_llm_on_paper_dir
from llm.evaluator import run_llm_evaluation
from postprocess.reference_resolver import load_references
from postprocess.compact_export import write_compact_summary
from postprocess.workflow import (
    run_structure_normalization,
    run_evidence_linking,
    run_deterministic_validation,
    run_finalization,
)
from pipelines.decision_layer import build_ingest_gate_report, apply_decision_layer


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)


def infer_doi_from_paper_dir(paper_dir: str) -> str | None:
    xml_path = os.path.join(paper_dir, "paper.xml")
    if os.path.exists(xml_path):
        try:
            text = Path(xml_path).read_text(encoding="utf-8", errors="ignore")
            m = DOI_PATTERN.search(text)
            if m:
                return m.group(0)
        except Exception:
            pass
    return None


def discover_local_fulltext_dois(fulltext_root: str) -> list[str]:
    root = Path(fulltext_root)
    dois = []
    seen = set()

    if not root.exists():
        return []

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not ((d / "paper.xml").exists() or (d / "sections").exists()):
            continue

        doi = infer_doi_from_paper_dir(str(d))
        if not doi:
            continue

        low = doi.lower()
        if low in seen:
            continue
        seen.add(low)
        dois.append(doi)

    return dois


def write_json_snapshot(path: str | Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sanitize_snapshot_for_disk(payload: dict) -> dict:
    def _walk(node, path=()):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if isinstance(key, str) and key.startswith("_"):
                    continue
                if not path and key == "references":
                    continue
                if path and path[-1] == "table_evidence" and key in {"row_index", "column_index"}:
                    continue
                if len(path) >= 3 and path[-3:] == ("parameters", "registry", "[]") and key in {"value_SI", "unit_SI"}:
                    continue
                if key in {"evidence_ids", "evidence_location", "provenance_id", "binding_id"}:
                    continue
                if key in {"evidence_objects", "binding_contexts", "provenance_records"}:
                    continue
                if key in {"confidence", "quality_assessment"}:
                    continue
                out[key] = _walk(value, path + (key,))
            return out
        if isinstance(node, list):
            return [_walk(item, path + ("[]",)) for item in node]
        return node

    if not isinstance(payload, dict):
        return payload
    return _walk(payload)


def _parameter_count_for_pipeline(payload: dict) -> int:
    if not isinstance(payload, dict):
        return 0
    registry = ((payload.get("parameters") or {}).get("registry") or [])
    if isinstance(registry, list) and registry:
        return len(registry)
    claims = payload.get("parameter_claims") or []
    if isinstance(claims, list):
        return len([claim for claim in claims if isinstance(claim, dict)])
    return 0


def _sync_local_paper_source(source_paper_dir: str, output_paper_dir: str) -> None:
    source = Path(source_paper_dir)
    target = Path(output_paper_dir)
    target.mkdir(parents=True, exist_ok=True)

    for name in ("paper.xml", "paper.md", "references.json"):
        src = source / name
        if src.exists():
            shutil.copy2(src, target / name)

    for name in ("sections", "tables", "equations"):
        src = source / name
        dst = target / name
        if src.exists() and src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _ensure_references_json_from_xml(
    paper_dir: str,
    *,
    crossref_mailto: str | None = None,
    resolve_missing_reference_doi: bool = True,
) -> str | None:
    ref_path = os.path.join(paper_dir, "references.json")
    if os.path.exists(ref_path):
        return ref_path

    xml_path = os.path.join(paper_dir, "paper.xml")
    if not os.path.exists(xml_path):
        return None

    try:
        xml_text = Path(xml_path).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(xml_text, "xml")
        references = extract_references_from_xml(
            soup,
            crossref_mailto=crossref_mailto,
            resolve_missing_reference_doi=resolve_missing_reference_doi,
        )
        if not references:
            return None
        Path(ref_path).write_text(
            json.dumps(references, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"↪ Built missing references.json from local paper.xml: {paper_dir}")
        return ref_path
    except Exception as exc:
        print(f"↪ Failed to rebuild references.json from local paper.xml: {paper_dir} ({exc})")
        return None


def main():

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # configs
    qcfg = cfg["search"]
    llm_cfg = cfg["llm"]
    rag = cfg["rag"]
    paths = cfg["paths"]
    pipeline_cfg = cfg.get("pipeline", {})
    output_fulltext_root = str(paths["fulltext"])
    input_fulltext_root = str(pipeline_cfg.get("local_fulltext_input_root") or output_fulltext_root)
    os.makedirs(output_fulltext_root, exist_ok=True)
    status_path = os.path.join(output_fulltext_root, "pipeline_status.jsonl")

    # env keys
    elsevier_key = os.environ.get("ELSEVIER_API_KEY") or cfg["elsevier"].get("api_key")
    inst_token = cfg["elsevier"].get("inst_token")
    crossref_mailto = cfg["elsevier"].get("crossref_mailto")
    resolve_missing_reference_doi = bool(cfg["elsevier"].get("resolve_missing_reference_doi", True))
    fulltext_http_max_retries = int(cfg["elsevier"].get("http_max_retries", 3))

    skip_fulltext_download = bool(pipeline_cfg.get("skip_fulltext_download", False))
    prefer_local_fulltext = bool(pipeline_cfg.get("prefer_local_fulltext", True))
    local_dois_mode = bool(pipeline_cfg.get("use_local_fulltext_dois", skip_fulltext_download))
    configured_dois = pipeline_cfg.get("dois", []) or []
    gate_on_evaluation = bool(pipeline_cfg.get("gate_on_evaluation", True))
    gate_block_verdicts = {
        str(v).strip().lower()
        for v in (pipeline_cfg.get("db_ingest_block_verdicts", ["rejected"]) or ["rejected"])
    }
    gate_min_confidence = float(pipeline_cfg.get("db_ingest_min_document_confidence_score", 65.0))
    gate_block_review_escalation_on_accepted = bool(
        pipeline_cfg.get(
            "db_ingest_block_review_escalation_on_accepted",
            pipeline_cfg.get("db_ingest_block_review_escalation_on_pass", False),
        )
    )
    gate_review_escalation_accepted_min_confidence = float(
        pipeline_cfg.get(
            "db_ingest_review_escalation_accepted_min_confidence_score",
            pipeline_cfg.get("db_ingest_review_escalation_pass_min_confidence_score", 85.0),
        )
    )
    gate_review_escalation_accepted_max_review_required = int(
        pipeline_cfg.get(
            "db_ingest_review_escalation_accepted_max_review_required_parameters",
            pipeline_cfg.get("db_ingest_review_escalation_pass_max_review_required_parameters", 2),
        )
    )
    skip_quality_checks = bool(pipeline_cfg.get("skip_quality_checks", True))
    image_table_cfg = pipeline_cfg.get("image_backed_tables", {}) or {}
    image_table_keywords = image_table_cfg.get("relevant_keywords") or []
    direct_image_table_input = bool(llm_cfg.get("direct_image_table_input", True))

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # db
    db = cfg["db"]
    conn = connect_pg(db["host"], int(db["port"]), db["name"], db["user"], db["password"])

    # 1) collect target DOIs
    if configured_dois:
        dois = configured_dois
        print(f"Using {len(dois)} DOIs from pipeline.dois")
    elif local_dois_mode:
        dois = discover_local_fulltext_dois(input_fulltext_root)
        print(f"Using {len(dois)} DOIs discovered from local fulltext directory")
    else:
        if not elsevier_key:
            raise RuntimeError("Elsevier API key is required for Scopus search/fulltext discovery")
        dois = scopus_search(
            api_key=elsevier_key,
            query=qcfg["query"],
            count=qcfg["count"],
            max_pages=qcfg["max_pages"],
            outdir=paths["scopus_export"],
            year_from=qcfg.get("year_from"),
            year_to=qcfg.get("year_to"),
            require_doi=bool(qcfg.get("require_doi", True)),
            allowed_doctypes=qcfg.get("allowed_doctypes", []),
            rank_keywords=qcfg.get("rank_keywords", []),
            max_retries=int(qcfg.get("http_max_retries", 3)),
        )


    print(f"Found {len(dois)} DOIs")
    dois_limit = int(qcfg.get("dois_limit", 0) or 0)
    if dois_limit > 0:
        dois = dois[:dois_limit]
        print(f"Using first {len(dois)} DOIs due to search.dois_limit={dois_limit}")

    # 2) for each doi: parse fulltext -> LLM extract -> ingest to DB
    for doi in dois:
        t0 = time.perf_counter()
        try:
            source_paper_dir = os.path.join(input_fulltext_root, safe_id(doi))
            paper_dir = os.path.join(output_fulltext_root, safe_id(doi))
            stage_timings = {}

            # parse/download
            download_start = time.perf_counter()
            source_local_exists = os.path.isdir(source_paper_dir)
            output_local_exists = os.path.isdir(paper_dir)
            if skip_fulltext_download:
                if not source_local_exists:
                    raise RuntimeError(f"Local paper directory not found for DOI: {doi} ({source_paper_dir})")
                if source_paper_dir != paper_dir:
                    _sync_local_paper_source(source_paper_dir, paper_dir)
                    print(f"↪ Skip download, copy local fulltext: {source_paper_dir} -> {paper_dir}")
                else:
                    print(f"↪ Skip download, use local fulltext: {paper_dir}")
            elif prefer_local_fulltext and source_local_exists:
                if source_paper_dir != paper_dir:
                    _sync_local_paper_source(source_paper_dir, paper_dir)
                    print(f"↪ Local fulltext exists, copy to output root: {source_paper_dir} -> {paper_dir}")
                else:
                    print(f"↪ Local fulltext exists, skip download: {paper_dir}")
            else:
                if not elsevier_key:
                    raise RuntimeError(
                        f"Elsevier API key is required to download missing fulltext for DOI: {doi}"
                    )
                save_paper_as_markdown_and_tables(
                    doi=doi,
                    api_key=elsevier_key,
                    inst_token=inst_token,
                    outdir=output_fulltext_root,
                    crossref_mailto=crossref_mailto,
                    resolve_missing_reference_doi=resolve_missing_reference_doi,
                    http_max_retries=fulltext_http_max_retries,
                    image_backed_table_keywords=image_table_keywords,
                )
            stage_timings["fulltext_prepare_seconds"] = round(time.perf_counter() - download_start, 3)

            # llm extraction (writes json files too, but returns extracted json)
            llm_result = run_llm_on_paper_dir(
                paper_dir=paper_dir,
                model_select=llm_cfg["model_select"],
                model_extract=llm_cfg["model_extract"],
                max_snippet_chars=int(llm_cfg["max_snippet_chars"]),
                max_context_chars=int(llm_cfg["max_context_chars"]),
                max_extract_retries=int(llm_cfg.get("max_extract_retries", 2)),
                enable_source_enrichment=bool(llm_cfg.get("enable_source_enrichment", True)),
                direct_image_table_input=direct_image_table_input,
                image_download_api_key=elsevier_key,
                image_download_inst_token=inst_token,
            )
            selection = llm_result["selection"]
            extracted = llm_result["extracted"]
            metrics = llm_result["metrics"]
            write_json_snapshot(
                os.path.join(paper_dir, "materials_extracted.extractor_raw.json"),
                extracted,
            )
            reports = {}
            reports["lineage"] = {
                "prompt_version": str(llm_cfg.get("prompt_version", "v2.0.0")),
                "schema_version": str(extracted.get("schema_version") or llm_cfg.get("schema_version", "5.0.2")),
                "extractor_version": str(llm_cfg.get("extractor_version", "extractor_v2")),
                "postprocess_version": "workflow_v5_0_2_finalized",
            }

            # Load references.json produced by fulltext_parser
            ref_path = _ensure_references_json_from_xml(
                paper_dir,
                crossref_mailto=crossref_mailto,
                resolve_missing_reference_doi=resolve_missing_reference_doi,
            ) or os.path.join(paper_dir, "references.json")
            reference_map = load_references(ref_path) if os.path.exists(ref_path) else None

            structure_start = time.perf_counter()
            extracted, structure_reports = run_structure_normalization(
                extracted,
                paper_dir=paper_dir,
                doi_hint=doi,
                reference_map=reference_map,
            )
            reports.update(structure_reports)
            stage_timings["structure_normalization_seconds"] = round(time.perf_counter() - structure_start, 3)

            evidence_start = time.perf_counter()
            extracted, evidence_reports = run_evidence_linking(
                extracted,
                paper_dir=paper_dir,
            )
            reports.update(evidence_reports)
            stage_timings["evidence_linking_seconds"] = round(time.perf_counter() - evidence_start, 3)

            out_path = os.path.join(paper_dir, "materials_extracted.json")

            parameter_count_pre_eval = _parameter_count_for_pipeline(extracted)
            if bool(llm_cfg.get("enable_evaluator", False)) and parameter_count_pre_eval > 0:
                eval_start = time.perf_counter()
                evaluation, eval_metrics = run_llm_evaluation(
                    paper_dir=paper_dir,
                    extracted_json=extracted,
                    model_evaluate=llm_cfg.get("model_evaluate", llm_cfg["model_extract"]),
                    max_context_chars=int(llm_cfg.get("max_evaluate_context_chars", 18000)),
                    max_retries=int(llm_cfg.get("max_evaluate_retries", 2)),
                    parameter_limit=int(llm_cfg.get("evaluate_parameter_limit", 40)),
                    field_batch_size=int(llm_cfg.get("evaluate_parameter_batch_size", 12)),
                    per_evidence_chars=int(llm_cfg.get("evaluate_evidence_chars", 800)),
                    quality_report={},
                    evidence_report=reports.get("evidence_grounding") or {},
                    feedback_artifact_path=llm_cfg.get("evaluation_feedback_json"),
                )
                reports["llm_evaluation"] = evaluation
                reports["llm_evaluation_metrics"] = eval_metrics
                with open(os.path.join(paper_dir, "llm_evaluation.json"), "w", encoding="utf-8") as f:
                    json.dump(evaluation, f, ensure_ascii=False, indent=2)
                stage_timings["llm_evaluation_seconds"] = round(time.perf_counter() - eval_start, 3)
            else:
                stage_timings["llm_evaluation_seconds"] = 0.0
                if parameter_count_pre_eval == 0:
                    reports["llm_evaluation_skipped_reason"] = "no_explicit_parameters"

            deterministic_start = time.perf_counter()
            extracted, deterministic_reports = run_deterministic_validation(
                extracted,
                skip_quality_checks=skip_quality_checks,
            )
            reports.update(deterministic_reports)
            stage_timings["deterministic_validation_seconds"] = round(
                time.perf_counter() - deterministic_start, 3
            )

            finalization_start = time.perf_counter()
            extracted, finalization_reports = run_finalization(
                extracted,
                evaluation_report=reports.get("llm_evaluation"),
                quality_report=reports.get("quality_checks"),
            )
            reports.update(finalization_reports)
            stage_timings["finalization_seconds"] = round(
                time.perf_counter() - finalization_start, 3
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(extracted, f, ensure_ascii=False, indent=2)

            reports["pipeline_metrics"] = dict(metrics)
            reports["pipeline_metrics"]["stages"] = stage_timings
            reports["ingest_gate"] = build_ingest_gate_report(
                evaluation_report=reports.get("llm_evaluation"),
                confidence_report=reports.get("confidence_fusion"),
                extracted_json=extracted,
                enabled=gate_on_evaluation,
                blocked_verdicts=gate_block_verdicts,
                min_document_confidence_score=gate_min_confidence,
                paper_dir=paper_dir,
                block_review_escalation_on_accepted=gate_block_review_escalation_on_accepted,
                review_escalation_accepted_min_confidence_score=gate_review_escalation_accepted_min_confidence,
                review_escalation_accepted_max_review_required_parameters=gate_review_escalation_accepted_max_review_required,
            )
            blocked_by_gate = bool(reports["ingest_gate"].get("blocked"))

            with open(os.path.join(paper_dir, "postprocess_report.json"), "w", encoding="utf-8") as f:
                json.dump(reports, f, ensure_ascii=False, indent=2)

            write_compact_summary(
                paper_dir,
                extracted,
                postprocess_report=reports,
                llm_evaluation=reports.get("llm_evaluation"),
            )

            ingest_start = time.perf_counter()
            blocked_by_gate = apply_decision_layer(
                conn=conn,
                openai_client=openai_client,
                doi=doi,
                paper_dir=paper_dir,
                extracted_json=extracted,
                reports=reports,
                llm_cfg=llm_cfg,
                rag_cfg=rag,
            )
            stage_timings["ingest_seconds"] = round(time.perf_counter() - ingest_start, 3)

            elapsed = time.perf_counter() - t0
            reports["pipeline_metrics"]["stages"] = stage_timings
            reports["pipeline_metrics"]["total_seconds"] = round(elapsed, 3)
            with open(os.path.join(paper_dir, "postprocess_report.json"), "w", encoding="utf-8") as f:
                json.dump(reports, f, ensure_ascii=False, indent=2)
            with open(status_path, "a", encoding="utf-8") as sf:
                sf.write(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "doi": doi,
                    "status": "gated" if blocked_by_gate else "success",
                    "seconds": elapsed,
                    "ingest_gate": reports.get("ingest_gate"),
                }, ensure_ascii=False) + "\n")
            if blocked_by_gate:
                print(f"⚠️ Gated from DB ingest: {doi}")
            else:
                print(f"✅ Done: {doi}")

        except Exception as e:
            conn.rollback()
            elapsed = time.perf_counter() - t0
            with open(status_path, "a", encoding="utf-8") as sf:
                sf.write(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "doi": doi,
                    "status": "failed",
                    "seconds": elapsed,
                    "error": str(e),
                }, ensure_ascii=False) + "\n")
            print(f"❌ Failed {doi}: {e}")

    conn.close()

if __name__ == "__main__":
    main()
