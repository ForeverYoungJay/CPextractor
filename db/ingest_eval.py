import json
from typing import Any, Dict


def _normalize_verdict(value: Any, default: str | None = None) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    mapping = {
        "accepted": "accepted",
        "pass": "accepted",
        "passed": "accepted",
        "flagged": "flagged",
        "warning": "flagged",
        "warn": "flagged",
        "needs_review": "flagged",
        "rejected": "rejected",
        "fail": "rejected",
        "failed": "rejected",
    }
    return mapping.get(raw, raw)


def ingest_evaluation(
    conn,
    doi: str,
    evaluation: Dict[str, Any] | None,
    confidence: Dict[str, Any] | None,
    model_evaluate: str | None = None,
    metrics: Dict[str, Any] | None = None,
) -> None:
    evaluation = evaluation or {}
    confidence = confidence or {}
    metrics = metrics or {}
    stages = metrics.get("stages") or {}
    conn.execute("SAVEPOINT sp_eval_ingest;")
    primary_exc: Exception | None = None
    try:
        conn.execute("DELETE FROM evaluation_runs WHERE doi = %s;", (doi,))
        conn.execute("DELETE FROM parameter_audits WHERE doi = %s;", (doi,))
        conn.execute(
            """
            INSERT INTO evaluation_runs (
                doi, model_evaluate, verdict,
                document_confidence, document_confidence_score, quality_tier,
                review_recommended,
                llm_evaluate_input_tokens, llm_evaluate_output_tokens, llm_evaluate_total_tokens,
                time_evaluate_seconds,
                evidence_judge_input_tokens, evidence_judge_output_tokens, evidence_judge_total_tokens, evidence_judge_time_seconds,
                normalization_judge_input_tokens, normalization_judge_output_tokens, normalization_judge_total_tokens, normalization_judge_time_seconds,
                consistency_judge_input_tokens, consistency_judge_output_tokens, consistency_judge_total_tokens, consistency_judge_time_seconds,
                meta_judge_input_tokens, meta_judge_output_tokens, meta_judge_total_tokens, meta_judge_time_seconds,
                evaluation_json, confidence_json
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb);
            """,
            (
                doi,
                model_evaluate,
                _normalize_verdict(evaluation.get("verdict")),
                confidence.get("document_confidence"),
                confidence.get("document_confidence_score"),
                confidence.get("quality_tier"),
                confidence.get("review_recommended"),
                metrics.get("input_tokens"),
                metrics.get("output_tokens"),
                metrics.get("total_tokens"),
                metrics.get("time_seconds"),
                (stages.get("evidence_judge") or {}).get("input_tokens"),
                (stages.get("evidence_judge") or {}).get("output_tokens"),
                (stages.get("evidence_judge") or {}).get("total_tokens"),
                (stages.get("evidence_judge") or {}).get("time_seconds"),
                (stages.get("normalization_judge") or {}).get("input_tokens"),
                (stages.get("normalization_judge") or {}).get("output_tokens"),
                (stages.get("normalization_judge") or {}).get("total_tokens"),
                (stages.get("normalization_judge") or {}).get("time_seconds"),
                (stages.get("consistency_judge") or {}).get("input_tokens"),
                (stages.get("consistency_judge") or {}).get("output_tokens"),
                (stages.get("consistency_judge") or {}).get("total_tokens"),
                (stages.get("consistency_judge") or {}).get("time_seconds"),
                (stages.get("meta_judge") or {}).get("input_tokens"),
                (stages.get("meta_judge") or {}).get("output_tokens"),
                (stages.get("meta_judge") or {}).get("total_tokens"),
                (stages.get("meta_judge") or {}).get("time_seconds"),
                json.dumps(evaluation, ensure_ascii=False),
                json.dumps(confidence, ensure_ascii=False),
            ),
        )

        for audit in evaluation.get("parameter_audits", []) or []:
            if not isinstance(audit, dict):
                continue
            conn.execute(
                """
                INSERT INTO parameter_audits (
                    doi, location, canonical_name, symbol, verdict,
                    supportiveness, exactness, normalization_correctness,
                    completeness, provenance_quality, confidence, error_types, uncertainty_types, audit_json
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb);
                """,
                (
                    doi,
                    audit.get("location"),
                    audit.get("canonical_name"),
                    audit.get("symbol"),
                    _normalize_verdict(audit.get("verdict")),
                    audit.get("supportiveness"),
                    audit.get("exactness"),
                    audit.get("normalization_correctness"),
                    audit.get("completeness"),
                    audit.get("provenance_quality"),
                    audit.get("confidence"),
                    "|".join(audit.get("error_types", []) or []),
                    "|".join(audit.get("uncertainty_types", []) or []),
                    json.dumps(audit, ensure_ascii=False),
                ),
            )
        conn.execute("RELEASE SAVEPOINT sp_eval_ingest;")
    except Exception as exc:
        primary_exc = exc
        conn.execute("ROLLBACK TO SAVEPOINT sp_eval_ingest;")
        try:
            conn.execute(
                """
                INSERT INTO evaluation_runs (
                    doi, model_evaluate, verdict,
                    document_confidence, document_confidence_score, quality_tier,
                    review_recommended,
                    evaluation_json, confidence_json
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb);
                """,
                (
                    doi,
                    model_evaluate,
                    _normalize_verdict(evaluation.get("verdict")),
                    confidence.get("document_confidence"),
                    confidence.get("document_confidence_score"),
                    confidence.get("quality_tier"),
                    confidence.get("review_recommended"),
                    json.dumps(evaluation, ensure_ascii=False),
                    json.dumps(confidence, ensure_ascii=False),
                ),
            )
            conn.execute("RELEASE SAVEPOINT sp_eval_ingest;")
        except Exception as fallback_exc:
            conn.execute("ROLLBACK TO SAVEPOINT sp_eval_ingest;")
            conn.execute("RELEASE SAVEPOINT sp_eval_ingest;")
            raise RuntimeError(
                f"Failed to ingest evaluation for DOI {doi}: primary={primary_exc}; fallback={fallback_exc}"
            ) from fallback_exc
