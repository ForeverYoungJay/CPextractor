import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import save_json


def load_qrels(path: str):
    # JSONL rows: {"query_id":..., "doc_id":..., "relevant": 0/1}
    rel = defaultdict(set)
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r.get('relevant', 1)) > 0:
                rel[str(r['query_id'])].add(str(r['doc_id']))
    return rel


def load_runs(path: str):
    # JSONL rows: {"query_id":..., "ranked_doc_ids": [...]}
    runs = {}
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            runs[str(r['query_id'])] = [str(x) for x in r.get('ranked_doc_ids', [])]
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description='Compute Recall@k and MRR for retrieval.')
    ap.add_argument('--qrels', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--ks', default='5,10')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    qrels = load_qrels(args.qrels)
    runs = load_runs(args.runs)

    qids = sorted(set(qrels) & set(runs))
    if not qids:
        save_json(args.output, {'queries': 0})
        print(f'Saved retrieval metrics -> {args.output}')
        return

    recall_hits = {k: 0 for k in ks}
    rr_sum = 0.0

    for qid in qids:
        rel = qrels[qid]
        ranked = runs[qid]
        for k in ks:
            topk = ranked[:k]
            if rel.intersection(topk):
                recall_hits[k] += 1

        rr = 0.0
        for i, doc_id in enumerate(ranked, start=1):
            if doc_id in rel:
                rr = 1.0 / i
                break
        rr_sum += rr

    out = {
        'queries': len(qids),
        'mrr': rr_sum / len(qids),
        **{f'recall@{k}': recall_hits[k] / len(qids) for k in ks},
    }
    save_json(args.output, out)
    print(f'Saved retrieval metrics -> {args.output}')


if __name__ == '__main__':
    main()
