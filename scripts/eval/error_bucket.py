import argparse
from collections import Counter

from common import build_param_key, extract_param_rows, load_json, load_jsonl, norm_text, save_json


def load_any(path: str):
    return load_jsonl(path) if path.endswith('.jsonl') else load_json(path)


def citation_count(row):
    src = row.get('source', {}) if isinstance(row.get('source', {}), dict) else {}
    return len(src.get('citations', []) or [])


def main() -> None:
    ap = argparse.ArgumentParser(description='Bucket extraction errors by type.')
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

    c = Counter()

    for k, gv in g.items():
        pv = p.get(k)
        if not pv:
            c['missing_parameter'] += 1
            continue

        if norm_text(gv.get('unit')) != norm_text(pv.get('unit')):
            c['unit_mismatch'] += 1

        try:
            gval = float(gv.get('value'))
            pval = float(pv.get('value'))
            if abs(gval - pval) > max(1e-6, 0.05 * abs(gval)):
                c['value_mismatch_gt5pct'] += 1
        except Exception:
            c['non_numeric_value'] += 1

        if citation_count(gv) > 0 and citation_count(pv) == 0:
            c['missing_citation_link'] += 1

    for k in p:
        if k not in g:
            c['spurious_parameter'] += 1

    total = sum(c.values())
    out = {
        'total_errors': total,
        'buckets': [
            {'error_type': k, 'count': v, 'ratio': (v / total if total else 0.0)}
            for k, v in c.most_common()
        ],
    }
    save_json(args.output, out)
    print(f'Saved error buckets -> {args.output}')


if __name__ == '__main__':
    main()
