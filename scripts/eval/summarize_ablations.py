"""Compare ablations on the same manifest, retaining failures and unrun jobs."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.common import save_json, save_csv
from scripts.eval.evaluate_release import run


def summarize(plan_path, gold, outdir, diagnostic=False, pricing=None):
    plan = json.loads(Path(plan_path).read_text())
    output = Path(outdir)
    manifest_path = output / "evaluation_manifest.json"
    save_json(manifest_path, plan["manifest"])
    groups = defaultdict(list)
    for job in plan["jobs"]:
        groups[job["variant"], job["repeat"]].append(job)
    rows = []
    for (variant, repeat), jobs in sorted(groups.items()):
        records = []
        for job in jobs:
            status_path = Path(job["output_dir"]) / "experiment_status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "not_run"}
            records.append(status)
        row = {"variant": variant, "repeat": repeat, "papers": len(jobs),
               "successful_papers": sum(r["status"] == "success" for r in records),
               "failed_papers": sum(r["status"] == "failed" for r in records),
               "not_run_papers": sum(r["status"] not in {"success", "failed"} for r in records),
               "seconds": sum(r.get("seconds", 0) for r in records),
               "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": None,
               "publication_ready": False}
        priced_cost, unpriced = 0.0, False
        for status, job in zip(records, jobs):
            for stage, model_key in (("extraction", "model_extract"), ("evaluation", "model_evaluate")):
                metrics = status.get("metrics", {}).get(stage, {})
                input_tokens, output_tokens = metrics.get("input_tokens", 0), metrics.get("output_tokens", 0)
                row["input_tokens"] += input_tokens
                row["output_tokens"] += output_tokens
                # Extraction combines selector and extractor usage in legacy metrics;
                # assigning all tokens to one model would fabricate an exact cost.
                if input_tokens or output_tokens:
                    rates = (pricing or {}).get(job["config"]["llm"][model_key])
                    if stage == "extraction" or rates is None:
                        unpriced = True
                    else:
                        priced_cost += (input_tokens*rates["input_per_million"] + output_tokens*rates["output_per_million"])/1e6
        if (not unpriced and row["successful_papers"] == len(jobs)
                and row["input_tokens"] + row["output_tokens"] > 0):
            row["estimated_cost_usd"] = priced_cost
        if row["not_run_papers"] == 0:
            pred_root = Path(jobs[0]["output_dir"]).parent
            result = run(gold, pred_root, output / variant / f"repeat_{repeat}", manifest_path=manifest_path,
                         split=plan["split"], strict=not diagnostic)
            row.update(result["claim_detection"])
            row["publication_ready"] = result["publication_ready"]
        else:
            row["status"] = "not_run_or_incomplete"
        rows.append(row)
    save_json(output / "ablation_comparison.json", {"rows": rows, "publication_ready": bool(rows) and all(r["publication_ready"] for r in rows),
              "cost_note": "Combined selector/extractor usage cannot be exactly priced; unavailable cost remains null."})
    save_csv(output / "ablation_comparison.csv", rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--pricing", help="Optional model rates JSON; no assumed pricing")
    args = ap.parse_args()
    summarize(args.plan, args.gold, args.outdir, args.diagnostic,
              json.loads(Path(args.pricing).read_text()) if args.pricing else None)


if __name__ == "__main__":
    main()
