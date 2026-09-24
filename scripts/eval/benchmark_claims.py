import argparse
from collections import Counter
from pathlib import Path

from common import save_csv, save_json
from claim_eval_common import (
    bundle_metrics,
    field_accuracy,
    gold_positive,
    load_gold_claim_rows,
    load_pred_claim_rows,
    match_gold_pred,
)


def main() -> None:
    from scripts.eval.benchmark_protocol import prepare_universe, read_manifest
    from scripts.eval.evaluate_release import build_report
    ap = argparse.ArgumentParser(description="Versioned claim benchmark; diagnostic without --strict.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--pred-source", default="materials_extracted.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--by-paper-csv", default="")
    ap.add_argument("--manifest")
    ap.add_argument("--split", choices=["development", "calibration", "test"])
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    manifest = read_manifest(args.manifest) if args.manifest else None
    gold, pred, universe = prepare_universe(load_gold_claim_rows(args.gold), load_pred_claim_rows(args.pred_root, args.pred_source), manifest, args.split, args.strict)
    summary, artifacts = build_report(gold, pred, universe, manifest)
    save_json(args.output, summary)
    save_json(str(args.output) + ".matches.json", artifacts["matches"])
    if args.by_paper_csv:
        save_csv(args.by_paper_csv, artifacts["by_paper"])
    if args.strict and not summary["publication_ready"]:
        raise ValueError("Unresolved matching ambiguities; inspect diagnostics")
    print(f"Saved claim benchmark -> {args.output}")


if __name__ == "__main__":
    main()
