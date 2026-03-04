import argparse
from collections import defaultdict
from typing import Dict, Tuple

from common import build_param_key, extract_param_rows, load_json, load_jsonl, prf, save_json


def load_any(path: str):
    return load_jsonl(path) if path.endswith('.jsonl') else load_json(path)


def main() -> None:
    ap = argparse.ArgumentParser(description='Compute numeric error (MAE/MAPE) on matched parameters.')
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    gold_docs = load_any(args.gold)
    pred_docs = load_any(args.pred)
    if isinstance(gold_docs, dict):
        gold_docs = [gold_docs]
    if isinstance(pred_docs, dict):
        pred_docs = [pred_docs]

    g = {}
    for d in gold_docs:
        for r in extract_param_rows(d):
            g[build_param_key(r)] = r

    p = {}
    for d in pred_docs:
        for r in extract_param_rows(d):
            p[build_param_key(r)] = r

    abs_err = []
    ape = []
    by_block = defaultdict(list)
    missing = 0

    for k, gv in g.items():
        pv = p.get(k)
        if not pv:
            missing += 1
            continue
        try:
            a = float(gv.get('value'))
            b = float(pv.get('value'))
        except Exception:
            continue
        e = abs(a - b)
        abs_err.append(e)
        if a != 0:
            ape.append(e / abs(a))
        by_block[gv.get('block', 'unknown')].append(e)

    out: Dict[str, object] = {
        'matched': len(abs_err),
        'missing_from_pred': missing,
        'mae': (sum(abs_err) / len(abs_err)) if abs_err else None,
        'mape': (sum(ape) / len(ape)) if ape else None,
        'by_block': {
            k: {
                'count': len(v),
                'mae': sum(v) / len(v) if v else None,
            }
            for k, v in sorted(by_block.items())
        },
    }

    save_json(args.output, out)
    print(f'Saved numeric metrics -> {args.output}')


if __name__ == '__main__':
    main()
