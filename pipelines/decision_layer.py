from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set

from db.ingest import ingest_paper_dir_to_db, insert_pipeline_run, upsert_paper
from db.ingest_eval import ingest_evaluation
from db.ingest_ref import ingest_references

REVIEW_DECISION_FILENAME = "review_decision.json"


def _load_review_decision(paper_dir: str | None) -> Dict[str, Any]:
    if not paper_dir:
        return {}
    path = Path(paper_dir) / REVIEW_DECISION_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_ingest_gate_report(
    *,
    evaluation_report: Dict[str, Any] | None,
    confidence_report: Dict[str, Any] | None,
    extracted_json: Dict[str, Any] | None = None,
    enabled: bool,
    blocked_verdicts: Set[str],
    min_document_confidence_score: float,
    paper_dir: str | None = None,
    block_review_escalation_on_pass: bool = False,
    review_escalation_pass_min_confidence_score: float = 85.0,
    review_escalation_pass_max_review_required_parameters: int = 2,
) -> Dict[str, Any]:
    evaluation_report = evaluation_report or {}
    confidence_report = confidence_report or {}
    extracted_json = extracted_json or {}
    review_decision = _load_review_decision(paper_dir)

    eval_verdict = str(evaluation_report.get("verdict") or "").strip().lower()
    doc_conf_score = float(confidence_report.get("document_confidence_score") or 0.0)
    review_escalation = evaluation_report.get("review_escalation") or {}
    fail_parameter_count = int(confidence_report.get("fail_parameter_count") or 0)
    review_required_parameter_count = int(confidence_report.get("review_required_parameter_count") or 0)
    review_recommended = bool(confidence_report.get("review_recommended"))
    approved_for_ingest = bool(review_decision.get("approved_for_ingest"))
    review_escalation_required = bool(review_escalation.get("required"))
    parameter_claim_count = len(extracted_json.get("parameter_claims") or [])
    if parameter_claim_count <= 0:
        parameter_claim_count = len(((extracted_json.get("parameters") or {}).get("registry") or []))
    empty_extraction = parameter_claim_count == 0

    high_confidence_pass_with_limited_review = (
        eval_verdict == "pass"
        and fail_parameter_count == 0
        and not empty_extraction
        and doc_conf_score >= review_escalation_pass_min_confidence_score
        and review_required_parameter_count <= review_escalation_pass_max_review_required_parameters
    )
    review_escalation_blocks = review_escalation_required and (
        block_review_escalation_on_pass or not high_confidence_pass_with_limited_review
    )

    blocked_by_default = enabled and (
        (eval_verdict in blocked_verdicts if eval_verdict else False)
        or doc_conf_score < min_document_confidence_score
        or fail_parameter_count > 0
        or empty_extraction
        or review_escalation_blocks
    )
    blocked = blocked_by_default and not approved_for_ingest

    return {
        "enabled": enabled,
        "blocked": blocked,
        "blocked_by_default": blocked_by_default,
        "verdict": eval_verdict or None,
        "document_confidence_score": doc_conf_score,
        "min_document_confidence_score": min_document_confidence_score,
        "blocked_verdicts": sorted(blocked_verdicts),
        "parameter_claim_count": parameter_claim_count,
        "fail_parameter_count": fail_parameter_count,
        "review_required_parameter_count": review_required_parameter_count,
        "review_escalation_required": review_escalation_required,
        "review_escalation_blocks": review_escalation_blocks,
        "review_recommended": review_recommended,
        "gate_reasons": {
            "blocked_verdict": bool(eval_verdict and eval_verdict in blocked_verdicts),
            "low_document_confidence": doc_conf_score < min_document_confidence_score,
            "has_fail_parameters": fail_parameter_count > 0,
            "empty_extraction": empty_extraction,
            "review_escalation": review_escalation_required,
        },
        "soft_review_escalation_policy": {
            "block_review_escalation_on_pass": block_review_escalation_on_pass,
            "pass_min_confidence_score": review_escalation_pass_min_confidence_score,
            "pass_max_review_required_parameters": review_escalation_pass_max_review_required_parameters,
            "high_confidence_pass_with_limited_review": high_confidence_pass_with_limited_review,
        },
        "manual_override": {
            "approved_for_ingest": approved_for_ingest,
            "review_decision_file": REVIEW_DECISION_FILENAME if review_decision else None,
            "reviewer": review_decision.get("reviewer"),
            "decision_timestamp_utc": review_decision.get("decision_timestamp_utc"),
            "notes": review_decision.get("notes"),
        },
    }


def apply_decision_layer(
    *,
    conn,
    openai_client,
    doi: str,
    paper_dir: str,
    extracted_json: Dict[str, Any],
    reports: Dict[str, Any],
    llm_cfg: Dict[str, Any],
    rag_cfg: Dict[str, Any],
) -> bool:
    gate = reports.get("ingest_gate") or {}
    blocked = bool(gate.get("blocked"))

    source_doc = extracted_json.get("source_document", {}) if isinstance(extracted_json.get("source_document"), dict) else {}
    upsert_paper(
        conn,
        doi=doi,
        title=source_doc.get("title"),
        year=source_doc.get("year"),
        journal=source_doc.get("journal_or_venue"),
    )

    insert_pipeline_run(
        conn=conn,
        doi=doi,
        model_select=llm_cfg["model_select"],
        model_extract=llm_cfg["model_extract"],
        metrics=reports.get("pipeline_metrics"),
        evaluator_metrics=reports.get("llm_evaluation_metrics"),
        prompt_version=str(llm_cfg.get("prompt_version", "v2.0.0")),
        schema_version=str(llm_cfg.get("schema_version", "2.0.0")),
        extractor_version=str(llm_cfg.get("extractor_version", "extractor_v2")),
    )

    ingest_evaluation(
        conn,
        doi=doi,
        evaluation=reports.get("llm_evaluation"),
        confidence=reports.get("confidence_fusion"),
        model_evaluate=llm_cfg.get("model_evaluate"),
        metrics=reports.get("llm_evaluation_metrics"),
    )

    if not blocked:
        ingest_references(conn, doi, extracted_json)
    else:
        conn.execute("DELETE FROM parameter_references WHERE paper_doi = %s;", (doi,))
        conn.execute("DELETE FROM paper_references WHERE paper_doi = %s;", (doi,))
        conn.execute("DELETE FROM parameter_vectors WHERE doi = %s;", (doi,))
        conn.execute("DELETE FROM table_row_vectors WHERE doi = %s;", (doi,))
        conn.execute("DELETE FROM chunks WHERE doi = %s;", (doi,))
        conn.execute("DELETE FROM extractions WHERE doi = %s;", (doi,))

    conn.commit()

    if not blocked:
        ingest_paper_dir_to_db(
            conn=conn,
            openai_client=openai_client,
            doi=doi,
            paper_dir=paper_dir,
            extracted_json=extracted_json,
            model_select=llm_cfg["model_select"],
            model_extract=llm_cfg["model_extract"],
            embedding_model=rag_cfg["embedding_model"],
            embedding_dim=int(rag_cfg["embedding_dim"]),
            chunk_chars=int(rag_cfg["chunk_chars"]),
            chunk_overlap=int(rag_cfg["chunk_overlap"]),
            batch_size=int(rag_cfg["batch_size"]),
            embedding_max_retries=int(rag_cfg.get("embedding_max_retries", 3)),
        )
    return blocked
