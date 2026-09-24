"""Explain identity, schema and universe mismatches before scoring a benchmark."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.benchmark_protocol import normalize_row, valid_doi, fingerprint
from scripts.eval.claim_eval_common import load_pred_claim_rows
from scripts.eval.common import load_jsonl, save_json


def diagnose(gold_path, pred_root, source="materials_extracted.json", historic_path=None):
    raw = load_jsonl(gold_path)
    gold = [normalize_row(r) for r in raw]
    pred = load_pred_claim_rows(pred_root, source)
    old_gold_keys = [(str(r.get("doi") or r.get("record_id") or "").strip(), str(r.get("claim_id") or "")) for r in raw]
    old_pred_keys = [(str(r.get("doi") or r.get("record_id") or "").strip(), str(r.get("claim_id") or "")) for r in pred]
    gdois = {r["doi"] for r in gold if valid_doi(r["doi"])}
    pdois = {normalize_row(r)["doi"] for r in pred}
    historic = json.loads(Path(historic_path).read_text()) if historic_path else {}
    return {"gold_sha256": fingerprint(gold_path), "gold_rows": len(gold), "prediction_rows": len(pred),
            "gold_valid_dois": len(gdois), "prediction_dois": len(pdois),
            "common_dois": sorted(gdois & pdois), "gold_only_dois": sorted(gdois-pdois),
            "prediction_only_paper_count": len(pdois-gdois),
            "invalid_gold_doi_rows": sum(not valid_doi(r["doi"]) for r in gold),
            "legacy_duplicate_key_rows_lost": len(old_gold_keys)-len(set(old_gold_keys)),
            "legacy_exact_id_intersection": len(set(old_gold_keys) & set(old_pred_keys)),
            "nested_parameter_body_rows": sum(isinstance(r.get("parameter_body"), dict) for r in raw),
            "gold_status_counts": dict(Counter(r["annotation"]["status"] for r in gold)),
            "rows_without_reviewer_identity": sum(not (r.get("annotation") or {}).get("reviewer") for r in raw),
            "historical_counts_match_selected_inputs": historic.get("gold_rows") == len(gold) and historic.get("pred_rows") == len(pred),
            "historical_input_hashes_present": bool(historic.get("inputs")),
            "conclusion": "A historical result without input hashes cannot be exactly reproduced from counts alone. The selected inputs diagnose migration hazards, not independent gold quality."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--pred-source", default="materials_extracted.json")
    ap.add_argument("--historic-metrics")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = diagnose(args.gold, args.pred_root, args.pred_source, args.historic_metrics)
    save_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"common_dois", "gold_only_dois"}}, indent=2))


if __name__ == "__main__":
    main()
