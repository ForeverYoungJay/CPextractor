import argparse
from collections import defaultdict

from common import build_param_key, extract_param_rows, load_json, load_jsonl, save_json


def load_any(path: str):
    return load_jsonl(path) if path.endswith('.jsonl') else load_json(path)


def citation_dois(row):
    src = row.get('source', {}) if isinstance(row.get('source', {}), dict) else {}
    dois = set()
    for c in src.get('citations', []) or []:
        d = c.get('doi')
        if d:
            dois.add(str(d).lower())
    return dois


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate citation DOI linking quality.')
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

    tp = fp = fn = 0
    matched_params = 0

    for k, gv in g.items():
        pv = p.get(k)
        if not pv:
            continue
        matched_params += 1
        gd = citation_dois(gv)
        pd = citation_dois(pv)
        tp += len(gd & pd)
        fp += len(pd - gd)
        fn += len(gd - pd)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None

    out = {
        'matched_parameters': matched_params,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    save_json(args.output, out)
    print(f'Saved citation metrics -> {args.output}')


if __name__ == '__main__':
    main()
