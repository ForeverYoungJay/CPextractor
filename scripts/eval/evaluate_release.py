"""One evaluation universe for detection, fields, taxonomy, slices and gates."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.benchmark_protocol import (
    POSITIVE, PROTOCOL_VERSION, fingerprint, prepare_universe, read_manifest,
    match_rows, normalize_row, text, symbol_text, value_equal, doi,
)
from scripts.eval.claim_eval_common import load_gold_claim_rows, load_pred_claim_rows, load_gate_rows
from scripts.eval.common import save_csv, save_json

FIELDS = {
    "detection": ["canonical_name", "symbol"],
    "value_unit": ["value", "unit"],
    "material_scope": ["context.material", "context.phase", "context.process_state"],
    "condition": ["context.condition", "context.temperature", "context.strain_rate"],
    "model_mechanism": ["context.model", "context.mechanism", "context.family", "context.systems"],
    "source_attribution": ["provenance.origin_type", "provenance.source_scope", "provenance.reference_ids",
                           "provenance.adopted_from_reference_ids", "provenance.calibration_based_on_reference_ids"],
    "grounding": ["evidence.kind", "evidence.file", "evidence.row_name", "evidence.column_name"],
}


def get(row, key):
    for part in key.split("."):
        row = row.get(part) if isinstance(row, dict) else None
    return row


def field_equal(key, gold, pred):
    if key == "value":
        return value_equal(gold, pred)
    if key.endswith("reference_ids") or key == "context.systems":
        return sorted(map(text, gold or [])) == sorted(map(text, pred or []))
    if key == "evidence.file":
        return text(gold).replace("\\", "/").split("/")[-1] == text(pred).replace("\\", "/").split("/")[-1]
    if key == "symbol":
        return symbol_text(gold) == symbol_text(pred)
    if key == "unit":
        aliases = {"1/s": "s^-1", "s−1": "s^-1", "s-1": "s^-1", "s⁻¹": "s^-1"}
        return aliases.get(symbol_text(gold), symbol_text(gold)) == aliases.get(symbol_text(pred), symbol_text(pred))
    return text(gold) == text(pred)


def compare_fields(gold, pred):
    rows = []
    explicitly_reviewed = set((gold.get("annotation") or {}).get("reviewed_fields") or [])
    for module, fields in FIELDS.items():
        for key in fields:
            if explicitly_reviewed and key not in explicitly_reviewed:
                continue
            g, p = get(gold, key), get(pred, key)
            # Missing labels are unknown, not automatically correct. Explicitly
            # reviewed absence can be scored (e.g. a genuinely dimensionless unit).
            if g in (None, "", [], {}) and key not in explicitly_reviewed:
                continue
            rows.append({"module": module, "field": key, "gold": g, "prediction": p,
                         "correct": field_equal(key, g, p)})
    return rows


def classification(rows, threshold=None):
    tp = fp = fn = tn = 0
    for row in rows:
        blocked = row["blocked"] if threshold is None else row["score"] < threshold
        bad = row["should_block"]
        tp += int(blocked and bad)
        fp += int(blocked and not bad)
        fn += int(not blocked and bad)
        tn += int(not blocked and not bad)
    return {"paper_count": len(rows), "tp_block_bad": tp, "fp_block_good": fp,
            "fn_admit_bad": fn, "tn_admit_good": tn,
            "false_admission_rate": fn / (tp + fn) if tp + fn else None,
            "false_block_rate": fp / (fp + tn) if fp + tn else None,
            "admitted_precision": tn / (tn + fn) if tn + fn else None,
            "coverage": (tn + fn) / len(rows) if rows else None}


def difficulty_slices(gold, pred):
    materials = defaultdict(set)
    for r in gold + pred:
        if r.get("context", {}).get("material"):
            materials[r["doi"]].add(text(r["context"]["material"]))
    def tags(r):
        result = {"all"}
        kind = text(r.get("evidence", {}).get("kind"))
        if kind.startswith("table"):
            result.add("table_backed")
        elif kind:
            result.add("text_or_other_evidence")
        if not r.get("unit"):
            result.add("missing_reported_unit")
        if len(materials[r["doi"]]) > 1:
            result.add("multi_material_paper")
        return result
    names = sorted(set().union(*(tags(r) for r in gold + pred))) if gold or pred else []
    rows = []
    for name in names:
        gs, ps = [r for r in gold if name in tags(r)], [r for r in pred if name in tags(r)]
        result = match_rows(gs, ps)
        rows.append({"slice": name, "gold_rows": len(gs), "pred_rows": len(ps),
                     **result["prf"], **{k: result[k] for k in ("tp", "fp", "fn")},
                     "ambiguities": len(result["ambiguities"])})
    return rows


def build_report(gold, pred, universe, manifest=None, gate_rows=()):
    matched = match_rows(gold, pred)
    detail, errors, claim_labels = [], [], []
    for g, p in matched["matched_positive"]:
        comparisons = compare_fields(g, p)
        ref = {"doi": g["doi"], "gold_claim_id": g.get("claim_id"), "pred_claim_id": p.get("claim_id")}
        detail.extend({**ref, **r} for r in comparisons)
        errors.extend({**ref, **r, "error_type": "field_mismatch", "evidence": g.get("evidence")}
                      for r in comparisons if not r["correct"])
        required = {"canonical_name", "value", "unit"}
        reviewed = {r["field"] for r in comparisons}
        claim_labels.append({**ref, "score": p.get("confidence_score"),
                             "reviewed_fields_correct": all(r["correct"] for r in comparisons) if comparisons else None,
                             "reviewed_field_count": len(comparisons),
                             "correct": all(r["correct"] for r in comparisons) if required <= reviewed and (g.get("annotation") or {}).get("usability_review_complete") is True else None})
    for kind, rows in (("missing_claim", matched["missing_gold_positive"]),
                       ("unmatched_prediction", matched["unmatched_pred"]),
                       ("spurious_prediction", [p for _, p in matched["matched_spurious"]])):
        for row in rows:
            errors.append({"doi": row["doi"], "claim_id": row.get("claim_id"), "module": "detection",
                           "error_type": kind, "evidence": row.get("evidence"), "record": row})
    if universe["publication_ready"] and not matched["ambiguities"]:
        claim_labels.extend({"doi": p["doi"], "pred_claim_id": p.get("claim_id"),
                             "score": p.get("confidence_score"), "correct": False}
                            for p in matched["unmatched_pred"] + [p for _, p in matched["matched_spurious"]])
    fields = {}
    for key in [k for values in FIELDS.values() for k in values]:
        vals = [r["correct"] for r in detail if r["field"] == key]
        fields[key] = {"annotated": len(vals), "correct": sum(vals),
                       "accuracy": sum(vals) / len(vals) if vals else None}
    by_paper = []
    for paper_doi in universe["active_dois"]:
        total = sum(r["doi"] == paper_doi and r["annotation"]["status"] in POSITIVE for r in gold)
        found = sum(g["doi"] == paper_doi for g, _ in matched["matched_positive"])
        by_paper.append({"doi": paper_doi, "gold_claims": total, "matched_claims": found,
                         "bundle_completeness": found / total if total else None})
    actual_gates = {doi(r["doi"]): r for r in gate_rows}
    evaluated_gates, unlabeled, missing = [], [], []
    for paper in (manifest or {}).get("papers", []):
        if paper["doi"] not in universe["active_dois"]:
            continue
        label = paper.get("gate_should_block")
        if not isinstance(label, bool):
            unlabeled.append(paper["doi"])
            continue
        gate = actual_gates.get(paper["doi"], {})
        if not isinstance(gate.get("blocked"), bool):
            missing.append(paper["doi"])
            continue
        evaluated_gates.append({"doi": paper["doi"], "split": paper["split"],
                                "should_block": label, "blocked": gate["blocked"],
                                "score": gate.get("document_confidence_score")})
    complete = [r["bundle_completeness"] for r in by_paper if r["bundle_completeness"] is not None]
    aliases = {"canonical_name_accuracy": "canonical_name", "value_accuracy": "value", "unit_accuracy": "unit"}
    summary = {
        "protocol_version": PROTOCOL_VERSION, "universe": universe,
        "publication_ready": universe["publication_ready"] and not matched["ambiguities"],
        "gold_rows": len(gold), "gold_positive_rows": sum(r["annotation"]["status"] in POSITIVE for r in gold),
        "pred_rows": len(pred),
        "claim_detection": {**matched["prf"], **{k: matched[k] for k in ("tp", "fp", "fn")}},
        "field_accuracy": {k: fields[v]["accuracy"] for k, v in aliases.items()},
        "fields": fields,
        "bundle": {"paper_count": len(by_paper), "macro_bundle_completeness": sum(complete) / len(complete) if complete else None},
        "diagnostics": {"ambiguities": matched["ambiguities"],
                        "duplicate_gold_identity_groups": matched["duplicate_gold_identity_groups"],
                        "duplicate_pred_identity_groups": matched["duplicate_pred_identity_groups"]},
        "error_taxonomy": dict(Counter(r["module"] for r in errors)),
        "slices": difficulty_slices(gold, pred),
        "gate": {**classification(evaluated_gates), "unlabeled_papers": unlabeled, "missing_decisions": missing,
                 "label_policy": "Explicit paper labels only; claim error statuses are not gate gold."},
    }
    return summary, {"matches": matched["match_log"], "errors": errors, "fields": detail, "slices": summary["slices"],
                     "by_paper": by_paper, "claim_labels": claim_labels, "gate_labels": evaluated_gates}


def run(gold_path, pred_root, outdir, *, manifest_path=None, split=None, strict=False, source="materials_extracted.json"):
    manifest = read_manifest(manifest_path) if manifest_path else None
    gold, pred, universe = prepare_universe(load_gold_claim_rows(gold_path), load_pred_claim_rows(pred_root, source),
                                            manifest, split, strict)
    summary, artifacts = build_report(gold, pred, universe, manifest, load_gate_rows(pred_root))
    summary["inputs"] = {"gold_sha256": fingerprint(gold_path), "prediction_source": source,
                         "manifest_sha256": fingerprint(manifest_path) if manifest_path else None,
                         "predictions": {str(p.relative_to(pred_root)): fingerprint(p)
                                         for p in sorted(Path(pred_root).glob(f"*/{source}"))}}
    out = Path(outdir)
    save_json(out / "metrics/benchmark_claims.json", summary)
    save_json(out / "metrics/annotation_benchmark_summary.json", summary)
    save_json(out / "metrics/benchmark_gate.json", summary["gate"])
    save_json(out / "metrics/benchmark_slices.json", {"slices": summary["slices"], "universe": universe})
    for key, rows in artifacts.items():
        save_json(out / f"audit/{key}.json", rows)
        save_csv(out / f"tables/{key}.csv", rows)
    if strict and not summary["publication_ready"]:
        raise ValueError("Ambiguous claim assignments; inspect audit/matches.json and resolve before release")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--split", choices=["development", "calibration", "test"])
    ap.add_argument("--strict", action="store_true", help="Require exhaustive human-expert adjudication")
    ap.add_argument("--pred-source", default="materials_extracted.json")
    args = ap.parse_args()
    summary = run(args.gold, args.pred_root, args.outdir, manifest_path=args.manifest, split=args.split,
                  strict=args.strict, source=args.pred_source)
    print(json.dumps({"publication_ready": summary["publication_ready"], "universe": summary["universe"],
                      "claim_detection": summary["claim_detection"]}, indent=2))


if __name__ == "__main__":
    main()
