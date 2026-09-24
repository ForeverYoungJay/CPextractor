import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.common import save_json, save_csv
from scripts.eval.benchmark_protocol import prepare_universe, read_manifest
from scripts.eval.claim_eval_common import load_gold_claim_rows, load_pred_claim_rows, load_gate_rows
from scripts.eval.evaluate_release import build_report


def main():
    ap = argparse.ArgumentParser(description="Evaluate gates on explicitly labeled papers only.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--by-paper-csv", default="")
    ap.add_argument("--manifest", required=True, help="Must contain explicit gate_should_block labels")
    ap.add_argument("--split", choices=["development", "calibration", "test"])
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    manifest = read_manifest(args.manifest)
    gold, pred, universe = prepare_universe(load_gold_claim_rows(args.gold), load_pred_claim_rows(args.pred_root), manifest, args.split, args.strict)
    summary, artifacts = build_report(gold, pred, universe, manifest, load_gate_rows(args.pred_root))
    save_json(args.output, {**summary["gate"], "universe": universe})
    if args.by_paper_csv:
        save_csv(args.by_paper_csv, artifacts["gate_labels"])


if __name__ == "__main__":
    main()
