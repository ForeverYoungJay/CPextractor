import argparse
import json
from pathlib import Path


def _load(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Quality gate for extraction/retrieval pipeline.")
    ap.add_argument("--metrics-dir", required=True, help="Directory containing metrics_*.json")
    ap.add_argument("--postprocess-report", default="", help="Optional postprocess_report.json")
    ap.add_argument("--min-field-f1", type=float, default=0.65)
    ap.add_argument("--min-citation-f1", type=float, default=0.55)
    ap.add_argument("--min-recall-at-10", type=float, default=0.60)
    ap.add_argument("--max-missing-evidence-ratio", type=float, default=0.35)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    md = Path(args.metrics_dir)
    field = _load(str(md / "metrics_field.json"))
    citation = _load(str(md / "metrics_citation.json"))
    retrieval = _load(str(md / "metrics_retrieval.json"))
    post = _load(args.postprocess_report) if args.postprocess_report else {}
    qc = post.get("quality_checks", {}) if isinstance(post, dict) else {}

    field_f1 = float((field.get("micro", {}) or {}).get("f1") or 0.0)
    citation_f1 = float(citation.get("f1") or 0.0)
    recall10 = float(retrieval.get("recall@10") or retrieval.get("recall@5") or 0.0)

    total = int(qc.get("total_parameters") or 0)
    missing_evidence = int(qc.get("missing_evidence") or 0)
    missing_ratio = (missing_evidence / total) if total > 0 else 0.0

    checks = [
        ("field_f1", field_f1, ">=", args.min_field_f1),
        ("citation_f1", citation_f1, ">=", args.min_citation_f1),
        ("retrieval_recall@10", recall10, ">=", args.min_recall_at_10),
        ("missing_evidence_ratio", missing_ratio, "<=", args.max_missing_evidence_ratio),
    ]

    failed = []
    for name, v, op, th in checks:
        ok = v >= th if op == ">=" else v <= th
        if not ok:
            failed.append({"name": name, "value": v, "threshold": th, "op": op})

    out = {
        "pass": len(failed) == 0,
        "checks": [
            {"name": n, "value": v, "op": op, "threshold": th, "pass": not any(f["name"] == n for f in failed)}
            for n, v, op, th in checks
        ],
        "failed": failed,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved quality gate -> {out_path} pass={out['pass']}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
