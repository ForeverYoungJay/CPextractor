import argparse
import csv
import json
import math
from pathlib import Path


def _load_report(root: Path, doi: str) -> dict:
    path = root / doi.replace("/", "_") / "postprocess_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_map(report: dict) -> dict:
    rows = (((report.get("llm_evaluation") or {}).get("parameter_audits")) or [])
    out = {}
    for row in rows:
        if isinstance(row, dict) and row.get("location"):
            out[row["location"]] = row
    return out


def _safe_div(a: float, b: float) -> float | None:
    return (a / b) if b else None


def _judge_metrics(rows: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for row in rows:
        human_error = bool(row["human_error"])
        judge_error = bool(row["judge_error"])
        if human_error and judge_error:
            tp += 1
        elif (not human_error) and judge_error:
            fp += 1
        elif human_error and (not judge_error):
            fn += 1
        else:
            tn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None and (precision + recall) else None
    return {
        "samples": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _safe_div(tp + tn, len(rows)),
    }


def _cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    n = tp + fp + fn + tn
    if n == 0:
        return None
    p0 = (tp + tn) / n
    p_yes_h = (tp + fn) / n
    p_yes_j = (tp + fp) / n
    p_no_h = (fp + tn) / n
    p_no_j = (fn + tn) / n
    pe = (p_yes_h * p_yes_j) + (p_no_h * p_no_j)
    if pe == 1:
        return None
    return (p0 - pe) / (1 - pe)


def _ece(rows: list[dict], bins: int = 10) -> tuple[float | None, list[dict]]:
    if not rows:
        return None, []
    bucket_rows = []
    ece = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        bucket = [r for r in rows if lo <= r["p_correct"] < hi or (i == bins - 1 and r["p_correct"] == 1.0)]
        if not bucket:
            continue
        conf = sum(r["p_correct"] for r in bucket) / len(bucket)
        acc = sum(r["y_correct"] for r in bucket) / len(bucket)
        gap = abs(acc - conf)
        weight = len(bucket) / len(rows)
        ece += weight * gap
        bucket_rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "count": len(bucket),
            "avg_confidence": conf,
            "avg_accuracy": acc,
            "gap": gap,
        })
    return ece, bucket_rows


def _risk_coverage(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: r["p_correct"], reverse=True)
    out = []
    kept = 0
    correct = 0
    for idx, row in enumerate(ordered, start=1):
        kept += 1
        correct += int(row["y_correct"])
        coverage = kept / len(ordered)
        accuracy = correct / kept
        risk = 1.0 - accuracy
        out.append({
            "coverage": coverage,
            "accuracy": accuracy,
            "risk": risk,
            "threshold_confidence": row["p_correct"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark LLM judge against reviewed queue adjudications.")
    ap.add_argument("--review-csv", required=True)
    ap.add_argument("--pred-root", default="data/fulltext")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.pred_root)
    rows = list(csv.DictReader(Path(args.review_csv).open("r", encoding="utf-8")))

    tp = fp = fn = tn = 0
    brier_sum = 0.0
    judged_rows = []
    by_error_type = {}
    committee_rows = {
        "evidence_judge": [],
        "normalization_judge": [],
        "consistency_judge": [],
    }
    doi_human_error = {}
    doi_meta_verdict = {}

    cached_reports = {}
    cached_full_reports = {}
    for row in rows:
        adjudication = (row.get("adjudication") or "").strip().lower()
        if adjudication not in {"correct", "accepted", "incorrect", "rejected", "fixed", "corrected"}:
            continue
        doi = row.get("doi") or ""
        location = row.get("location") or ""
        if doi not in cached_reports:
            cached_full_reports[doi] = _load_report(root, doi)
            cached_reports[doi] = _audit_map(cached_full_reports[doi])
        audit = cached_reports[doi].get(location, {})
        report = cached_full_reports[doi]
        llm_eval = report.get("llm_evaluation", {}) if isinstance(report, dict) else {}
        if doi not in doi_meta_verdict:
            doi_meta_verdict[doi] = str(llm_eval.get("verdict") or "").strip().lower()

        human_error = adjudication in {"incorrect", "rejected", "fixed", "corrected"}
        judge_error = (audit.get("verdict") or "").strip().lower() in {"warning", "fail"}
        doi_human_error[doi] = doi_human_error.get(doi, False) or human_error

        if human_error and judge_error:
            tp += 1
        elif (not human_error) and judge_error:
            fp += 1
        elif human_error and (not judge_error):
            fn += 1
        else:
            tn += 1

        score = audit.get("score")
        try:
            p_correct = max(0.0, min(1.0, float(score) / 100.0))
        except Exception:
            p_correct = 0.5
        y_correct = 0.0 if human_error else 1.0
        brier_sum += (p_correct - y_correct) ** 2
        judged_rows.append({
            "doi": doi,
            "location": location,
            "p_correct": p_correct,
            "y_correct": y_correct,
            "judge_error": judge_error,
            "human_error": human_error,
        })

        et = (row.get("human_error_type") or "unspecified").strip().lower()
        bucket = by_error_type.setdefault(et, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        if human_error and judge_error:
            bucket["tp"] += 1
        elif (not human_error) and judge_error:
            bucket["fp"] += 1
        elif human_error and (not judge_error):
            bucket["fn"] += 1
        else:
            bucket["tn"] += 1

        committee = audit.get("committee", {}) if isinstance(audit.get("committee"), dict) else {}
        evidence_related = et in {"missing_evidence", "weak_grounding", "cross_reference_unresolved", "unsupported_claim", "table_parse_uncertain"}
        normalization_related = et in {"wrong_unit_conversion", "wrong_parameter_mapping", "symbol_mapping_conflict", "normalization_ambiguous"}
        consistency_related = et in {"condition_binding_error", "condition_binding_ambiguous", "cross_material_mixup", "model_variant_confusion", "physics_inconsistency"}

        evidence_vote = str((committee.get("evidence_judge") or {}).get("verdict") or "").strip().lower() in {"warning", "fail"}
        normalization_vote = str((committee.get("normalization_judge") or {}).get("verdict") or "").strip().lower() in {"warning", "fail"}
        consistency_vote = str((committee.get("consistency_judge") or {}).get("verdict") or "").strip().lower() in {"warning", "fail"}

        committee_rows["evidence_judge"].append({"human_error": human_error and evidence_related, "judge_error": evidence_vote})
        committee_rows["normalization_judge"].append({"human_error": human_error and normalization_related, "judge_error": normalization_vote})
        committee_rows["consistency_judge"].append({"human_error": human_error and consistency_related, "judge_error": consistency_vote})

    n = len(judged_rows)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None and (precision + recall) else None
    accuracy = _safe_div(tp + tn, n)
    brier = _safe_div(brier_sum, n)
    kappa = _cohen_kappa(tp, fp, fn, tn)
    ece, reliability_bins = _ece(judged_rows)
    risk_coverage = _risk_coverage(judged_rows)

    high_conf_rows = [r for r in judged_rows if r["p_correct"] >= 0.9]
    wrong_high_conf = _safe_div(sum(1 for r in high_conf_rows if r["y_correct"] == 0.0), len(high_conf_rows))
    meta_rows = []
    for doi, human_error in doi_human_error.items():
        meta_rows.append({
            "human_error": human_error,
            "judge_error": doi_meta_verdict.get(doi, "") in {"warning", "fail"},
        })

    out = {
        "samples": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_error_detection": precision,
        "recall_error_detection": recall,
        "f1_error_detection": f1,
        "accuracy": accuracy,
        "brier_score": brier,
        "ece": ece,
        "cohen_kappa": kappa,
        "wrong_at_high_confidence": wrong_high_conf,
        "high_confidence_samples": len(high_conf_rows),
        "reliability_bins": reliability_bins,
        "risk_coverage_curve": risk_coverage,
        "by_human_error_type": by_error_type,
        "by_judge": {
            "evidence_judge": _judge_metrics(committee_rows["evidence_judge"]),
            "normalization_judge": _judge_metrics(committee_rows["normalization_judge"]),
            "consistency_judge": _judge_metrics(committee_rows["consistency_judge"]),
            "meta_judge_doc_level": _judge_metrics(meta_rows),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved judge benchmark -> {out_path}")


if __name__ == "__main__":
    main()
