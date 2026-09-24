"""Select thresholds on calibration papers, then evaluate a frozen policy on test."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.benchmark_protocol import fingerprint, PROTOCOL_VERSION
from scripts.eval.common import save_json, save_csv
from scripts.eval.evaluate_release import classification


def score(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and 0 <= value <= 100 else None
    except (ValueError, TypeError):
        return None


def wilson(correct, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = correct / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0, center-half), min(1, center+half)]


def precision_coverage(rows, thresholds=None):
    labeled = [r for r in rows if isinstance(r.get("correct"), bool)]
    scored = [{**r, "score": score(r.get("score"))} for r in labeled if score(r.get("score")) is not None]
    thresholds = thresholds if thresholds is not None else sorted({0.0, 101.0, *(r["score"] for r in scored)})
    curve = []
    for threshold in thresholds:
        kept = [r for r in scored if r["score"] >= threshold]
        n, correct = len(kept), sum(r["correct"] for r in kept)
        curve.append({"threshold": threshold, "kept": n, "correct": correct,
                      "precision": correct/n if n else None,
                      "coverage": n/len(labeled) if labeled else None,
                      "precision_wilson95": wilson(correct, n)})
    return {"labeled": len(labeled), "unlabeled": len(rows)-len(labeled),
            "missing_score": len(labeled)-len(scored), "curve": curve,
            "interval_note": "Wilson intervals treat claims as independent; papers may induce correlation."}


def choose_threshold(curve, target, min_kept):
    eligible = [r for r in curve if r["kept"] >= min_kept and r["precision"] is not None
                and r["precision"] >= target]
    return max(eligible, key=lambda r: (r["coverage"], -r["threshold"]))["threshold"] if eligible else None


def calibrate(evaluation_dir, output, *, policy_path=None, target=0.95, min_kept=10, diagnostic=False):
    if not 0 < target <= 1 or min_kept < 1:
        raise ValueError("target must be in (0,1] and min_kept must be positive")
    root = Path(evaluation_dir)
    summary_path = root / "metrics/benchmark_claims.json"
    summary = json.loads(summary_path.read_text())
    if not summary.get("publication_ready") and not diagnostic:
        raise ValueError("Unvalidated/AI annotations: pass --diagnostic to produce non-publication analysis")
    split = summary["universe"].get("split")
    expected = "test" if policy_path else "calibration"
    if split != expected:
        raise ValueError(f"This operation requires split={expected}; got {split}")
    claims = json.loads((root / "audit/claim_labels.json").read_text())
    gates = json.loads((root / "audit/gate_labels.json").read_text())
    gate_scored = [{**r, "score": score(r.get("score"))} for r in gates if score(r.get("score")) is not None]
    claim_curve = precision_coverage(claims)
    gate_as_claims = [{"score": r["score"], "correct": not r["should_block"]} for r in gate_scored]
    gate_curve = precision_coverage(gate_as_claims)
    if policy_path:
        policy = json.loads(Path(policy_path).read_text())
        if policy.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Policy protocol mismatch")
        if policy.get("calibration_dois") is None:
            raise ValueError("Policy lacks calibration DOI provenance")
        if set(policy["calibration_dois"]) & set(summary["universe"]["dois"]):
            raise ValueError("Calibration/test paper leakage detected")
        if not diagnostic and not policy.get("publication_ready"):
            raise ValueError("Diagnostic policy cannot be promoted to validated test results")
    else:
        policy = {"protocol_version": PROTOCOL_VERSION, "publication_ready": summary["publication_ready"] and not diagnostic,
                  "calibration_dois": summary["universe"]["dois"], "calibration_summary_sha256": fingerprint(summary_path),
                  "target_precision": target, "min_kept": min_kept,
                  "claim_threshold": choose_threshold(claim_curve["curve"], target, min_kept),
                  "document_threshold": choose_threshold(gate_curve["curve"], target, min_kept),
                  "selection": "maximum coverage satisfying empirical precision and minimum support",
                  "limitation": "No success guarantee; test evaluation is required."}
    ct, dt = policy["claim_threshold"], policy["document_threshold"]
    result = {"split": split, "publication_ready": summary["publication_ready"] and policy["publication_ready"] and not diagnostic,
              "policy": policy, "claim_precision_coverage": claim_curve,
              "gate_threshold_curve": [{"threshold": r["threshold"], **classification(gate_scored, r["threshold"])}
                                       for r in gate_curve["curve"]],
              "claim_at_frozen_threshold": precision_coverage(claims, [ct]) if ct is not None else None,
              "gate_at_frozen_threshold": classification(gate_scored, dt) if dt is not None else None,
              "actual_gate": classification(gates), "gate_missing_scores": len(gates)-len(gate_scored),
              "status": "evaluated" if ct is not None and dt is not None else "insufficient_labels_or_no_feasible_threshold"}
    out = Path(output)
    save_json(out / "confidence_gate_report.json", result)
    if not policy_path:
        save_json(out / "frozen_policy.json", policy)
    save_csv(out / "precision_coverage.csv", claim_curve["curve"])
    save_csv(out / "gate_thresholds.csv", result["gate_threshold_curve"])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluation-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--policy", help="Frozen calibration policy; requires test split")
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--min-kept", type=int, default=10)
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--describe-only", action="store_true", help="Describe field-limited confidence; do not select thresholds")
    args = ap.parse_args()
    if args.describe_only:
        if not args.diagnostic:
            ap.error("Field-limited descriptive analysis requires --diagnostic")
        root = Path(args.evaluation_dir)
        rows = json.loads((root / "audit/claim_labels.json").read_text())
        limited = [{**r, "correct": r.get("reviewed_fields_correct")} for r in rows]
        result = precision_coverage(limited)
        result.update(publication_ready=False, threshold_selected=False,
                      target="correctness only of explicitly reviewed fields; not full claim usability")
        save_json(Path(args.outdir) / "field_confidence_diagnostic.json", result)
        save_csv(Path(args.outdir) / "field_precision_coverage.csv", result["curve"])
        print("Saved descriptive field-confidence diagnostic; no threshold selected")
        return
    result = calibrate(args.evaluation_dir, args.outdir, policy_path=args.policy,
                       target=args.target_precision, min_kept=args.min_kept, diagnostic=args.diagnostic)
    print(result["status"])


if __name__ == "__main__":
    main()
