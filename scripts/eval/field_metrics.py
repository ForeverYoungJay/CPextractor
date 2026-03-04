import argparse
from collections import defaultdict
from typing import Dict

from common import flatten_dict, index_by_record_id, load_json, load_jsonl, norm_text, prf, save_json


def load_any(path: str):
    return load_jsonl(path) if path.endswith(".jsonl") else load_json(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute field-level precision/recall/F1.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gold_rows = load_any(args.gold)
    pred_rows = load_any(args.pred)
    if isinstance(gold_rows, dict):
        gold_rows = [gold_rows]
    if isinstance(pred_rows, dict):
        pred_rows = [pred_rows]

    gidx = index_by_record_id(gold_rows)
    pidx = index_by_record_id(pred_rows)

    tp = fp = fn = 0
    by_prefix = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    keys = sorted(set(gidx) | set(pidx))
    for rid in keys:
        gflat = flatten_dict(gidx.get(rid, {}))
        pflat = flatten_dict(pidx.get(rid, {}))
        all_fields = set(gflat) | set(pflat)
        for f in all_fields:
            g = gflat.get(f, None)
            p = pflat.get(f, None)
            prefix = f.split(".")[0] if f else "root"
            if p is None and g is not None:
                fn += 1
                by_prefix[prefix]["fn"] += 1
            elif p is not None and g is None:
                fp += 1
                by_prefix[prefix]["fp"] += 1
            elif norm_text(p) == norm_text(g):
                tp += 1
                by_prefix[prefix]["tp"] += 1
            else:
                fp += 1
                fn += 1
                by_prefix[prefix]["fp"] += 1
                by_prefix[prefix]["fn"] += 1

    out: Dict[str, object] = {
        "micro": {**prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn},
        "by_prefix": {},
    }

    for k, v in sorted(by_prefix.items()):
        out["by_prefix"][k] = {**v, **prf(v["tp"], v["fp"], v["fn"])}

    save_json(args.output, out)
    print(f"Saved field metrics -> {args.output}")


if __name__ == "__main__":
    main()
