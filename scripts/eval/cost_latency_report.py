import argparse
import csv
from pathlib import Path

from common import save_json


def to_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def to_int(x):
    try:
        return int(x)
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description='Aggregate token/cost/latency from pipeline run CSV export.')
    ap.add_argument('--input-csv', required=True, help='CSV with pipeline_runs-like fields')
    ap.add_argument('--usd-per-1k-input', type=float, default=0.0)
    ap.add_argument('--usd-per-1k-output', type=float, default=0.0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    rows = []
    with Path(args.input_csv).open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    in_tok = sum(to_int(r.get('llm_select_input_tokens')) + to_int(r.get('llm_extract_input_tokens')) for r in rows)
    out_tok = sum(to_int(r.get('llm_select_output_tokens')) + to_int(r.get('llm_extract_output_tokens')) for r in rows)
    total_time = sum(to_float(r.get('time_total_seconds')) for r in rows)

    cost = (in_tok / 1000.0) * args.usd_per_1k_input + (out_tok / 1000.0) * args.usd_per_1k_output

    out = {
        'papers': n,
        'input_tokens_total': in_tok,
        'output_tokens_total': out_tok,
        'time_seconds_total': total_time,
        'tokens_per_paper': (in_tok + out_tok) / n if n else None,
        'time_seconds_per_paper': total_time / n if n else None,
        'cost_total_usd': cost,
        'cost_per_paper_usd': cost / n if n else None,
    }
    save_json(args.output, out)
    print(f'Saved cost/latency report -> {args.output}')


if __name__ == '__main__':
    main()
