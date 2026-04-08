import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_avg(values: list[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def _qa_metrics(rows: list[dict], kept_dois: set[str]) -> dict:
    if not rows:
        return {}
    total = len(rows)
    total_correct = 0
    kept_total = 0
    kept_correct = 0
    for row in rows:
        is_correct = row.get("is_correct")
        if is_correct is None:
            is_correct = row.get("correct")
        is_correct = bool(is_correct)
        supporting = {str(d).lower() for d in (row.get("supporting_dois") or [])}
        kept = bool(supporting & kept_dois) if supporting else False
        total_correct += int(is_correct)
        if kept:
            kept_total += 1
            kept_correct += int(is_correct)
    return {
        "qa_accuracy_all": (total_correct / total if total else None),
        "qa_covered_count": kept_total,
        "qa_accuracy_kept_covered": (kept_correct / kept_total if kept_total else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate downstream utility under confidence filtering.")
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--qrels", default="")
    ap.add_argument("--runs", default="")
    ap.add_argument("--qa-jsonl", default="")
    ap.add_argument("--min-doc-score", type=float, default=65.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    docs = []
    for report_path in sorted(pred_root.glob("*/postprocess_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cf = report.get("confidence_fusion", {}) if isinstance(report, dict) else {}
        qc = report.get("quality_checks", {}) if isinstance(report, dict) else {}
        eg = report.get("evidence_grounding", {}) if isinstance(report, dict) else {}
        evaluation = report.get("llm_evaluation", {}) if isinstance(report, dict) else {}
        docs.append({
            "doi": report_path.parent.name.replace("_", "/"),
            "document_confidence": cf.get("document_confidence"),
            "document_confidence_score": float(cf.get("document_confidence_score") or 0.0),
            "quality_tier": cf.get("quality_tier"),
            "rule_score": float(cf.get("rule_score") or qc.get("rule_score") or 0.0),
            "llm_score": float(cf.get("llm_score") or evaluation.get("overall_score") or 0.0),
            "evidence_grounding_score": float(eg.get("evidence_grounding_score") or 0.0),
            "review_recommended": bool(cf.get("review_recommended")),
            "issue_count": int(qc.get("issue_count") or 0),
        })

    kept = [d for d in docs if d["document_confidence_score"] >= args.min_doc_score]
    kept_dois = {d["doi"].lower() for d in kept}
    out = {
        "documents_total": len(docs),
        "documents_kept": len(kept),
        "document_keep_ratio": (len(kept) / len(docs) if docs else 0.0),
        "min_doc_score": args.min_doc_score,
        "avg_document_confidence_score_all": _safe_avg([d["document_confidence_score"] for d in docs]),
        "avg_document_confidence_score_kept": _safe_avg([d["document_confidence_score"] for d in kept]),
        "avg_rule_score_all": _safe_avg([d["rule_score"] for d in docs]),
        "avg_rule_score_kept": _safe_avg([d["rule_score"] for d in kept]),
        "avg_llm_score_all": _safe_avg([d["llm_score"] for d in docs]),
        "avg_llm_score_kept": _safe_avg([d["llm_score"] for d in kept]),
        "avg_evidence_grounding_score_all": _safe_avg([d["evidence_grounding_score"] for d in docs]),
        "avg_evidence_grounding_score_kept": _safe_avg([d["evidence_grounding_score"] for d in kept]),
        "avg_issue_count_all": _safe_avg([d["issue_count"] for d in docs]),
        "avg_issue_count_kept": _safe_avg([d["issue_count"] for d in kept]),
        "review_rate_all": (sum(1 for d in docs if d["review_recommended"]) / len(docs) if docs else 0.0),
        "review_rate_kept": (sum(1 for d in kept if d["review_recommended"]) / len(kept) if kept else 0.0),
        "gold_docs": sum(1 for d in docs if d["quality_tier"] == "gold"),
        "silver_docs": sum(1 for d in docs if d["quality_tier"] == "silver"),
        "candidate_docs": sum(1 for d in docs if d["quality_tier"] == "candidate"),
    }

    if args.qrels and args.runs:
        qrels = _load_jsonl(Path(args.qrels))
        runs = _load_jsonl(Path(args.runs))
        total_relevant = 0
        kept_relevant = 0
        for row in qrels:
            doi = str(row.get("doc_id") or row.get("doi") or "").lower()
            rel = int(row.get("relevance", 1))
            if rel > 0:
                total_relevant += 1
                if doi in kept_dois:
                    kept_relevant += 1
        total_run_hits = 0
        kept_run_hits = 0
        for row in runs:
            doi = str(row.get("doc_id") or row.get("doi") or "").lower()
            total_run_hits += 1
            if doi in kept_dois:
                kept_run_hits += 1
        out["retrieval_relevant_coverage"] = (kept_relevant / total_relevant if total_relevant else 0.0)
        out["retrieval_run_hit_coverage"] = (kept_run_hits / total_run_hits if total_run_hits else 0.0)
        out["structured_retrieval_hit_rate"] = out["retrieval_run_hit_coverage"]

    if args.qa_jsonl:
        qa_rows = _load_jsonl(Path(args.qa_jsonl))
        out.update(_qa_metrics(qa_rows, kept_dois))
        out["rag_answer_grounding_rate"] = out.get("qa_accuracy_kept_covered")

    out["analyst_query_success_rate"] = 1.0 - out["review_rate_kept"] if kept else 0.0
    out["downstream_analytics_consistency"] = out["avg_issue_count_kept"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved utility-by-confidence -> {out_path}")


if __name__ == "__main__":
    main()
