import argparse
from collections import defaultdict

from common import build_param_key, extract_param_rows, load_json, load_jsonl, norm_text, save_json


def load_any(path: str):
    return load_jsonl(path) if path.endswith('.jsonl') else load_json(path)


def main() -> None:
    ap = argparse.ArgumentParser(description='Compute unit match accuracy for parameters.')
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
    p = {}
    for d in gold_docs:
        for r in extract_param_rows(d):
            g[build_param_key(r)] = r
    for d in pred_docs:
        for r in extract_param_rows(d):
            p[build_param_key(r)] = r

    total = hit = 0
    by_block = defaultdict(lambda: {'total': 0, 'hit': 0})

    for k, gv in g.items():
        pv = p.get(k)
        if not pv:
            continue
        total += 1
        blk = gv.get('block', 'unknown')
        by_block[blk]['total'] += 1
        if norm_text(gv.get('unit')) == norm_text(pv.get('unit')):
            hit += 1
            by_block[blk]['hit'] += 1

    out = {
        'matched': total,
        'unit_accuracy': (hit / total) if total else None,
        'by_block': {
            b: {
                'matched': v['total'],
                'unit_accuracy': (v['hit'] / v['total']) if v['total'] else None,
            }
            for b, v in sorted(by_block.items())
        },
    }

    save_json(args.output, out)
    print(f'Saved unit metrics -> {args.output}')


if __name__ == '__main__':
    main()
