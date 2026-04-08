import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from postprocess.param_iter import iter_parameter_items
from postprocess.record_links import resolve_primary_crystal_structure, resolve_provenance_record


def iter_docs(root: Path):
    for p in sorted(root.glob("*/materials_extracted.json")):
        try:
            yield json.loads(p.read_text(encoding="utf-8")), p
        except Exception:
            continue


def source_bucket(doc: dict, src: dict) -> str:
    prov = resolve_provenance_record(doc, src if isinstance(src, dict) else {})
    t = (prov.get("origin_type") or (src.get("type") if isinstance(src, dict) else None) or "unknown").strip()
    return t or "unknown"


def _safe_year(doc: dict) -> int | None:
    source_doc = (doc.get("source_document", {}) or {})
    y = source_doc.get("year")
    try:
        return int(y) if y is not None else None
    except Exception:
        return None


def _safe_write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Publication-style analytics for the CP extracted dataset")
    ap.add_argument("--root", default="data/fulltext")
    ap.add_argument("--output", default="")
    ap.add_argument("--outdir", default="output/analytics")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-quality-tier", default="", help="Optional filter: gold / silver / candidate")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    output = Path(args.output) if args.output else outdir / "materials_insight.json"
    material_counter = Counter()
    framework_counter = Counter()
    source_counter = Counter()
    symbol_counter = Counter()
    canonical_counter = Counter()
    year_counter = Counter()
    journal_counter = Counter()
    lattice_counter = Counter()
    quality_counter = Counter()
    framework_year_counter = defaultdict(Counter)
    canonical_by_framework = defaultdict(Counter)
    canonical_by_lattice = defaultdict(Counter)
    docs_total = 0

    for doc, _ in iter_docs(root):
        tier = str(doc.get("quality_tier") or "").strip().lower()
        if args.min_quality_tier and tier != args.min_quality_tier.strip().lower():
            continue
        docs_total += 1

        m = (doc.get("material", {}) or {}).get("name")
        if m:
            material_counter[str(m)] += 1

        framework = (doc.get("constitutive_model", {}) or {}).get("framework")
        if framework:
            framework_counter[str(framework)] += 1
        year = _safe_year(doc)
        if year is not None:
            year_counter[str(year)] += 1
            if framework:
                framework_year_counter[str(year)][str(framework)] += 1
        journal = (doc.get("source_document", {}) or {}).get("journal_or_venue")
        if journal:
            journal_counter[str(journal)] += 1
        lattice = (resolve_primary_crystal_structure(doc).get("lattice_type"))
        if lattice:
            lattice_counter[str(lattice)] += 1
        if tier:
            quality_counter[tier] += 1

        for _, p in iter_parameter_items(doc):
            if p.get("symbol"):
                symbol_counter[str(p.get("symbol"))] += 1
            canonical = p.get("canonical_name")
            if canonical:
                canonical_counter[str(canonical)] += 1
                if framework:
                    canonical_by_framework[str(framework)][str(canonical)] += 1
                if lattice:
                    canonical_by_lattice[str(lattice)][str(canonical)] += 1
            source_counter[source_bucket(doc, p.get("source", {}))] += 1

    out = {
        "documents_total": docs_total,
        "quality_tier_distribution": dict(quality_counter),
        "top_materials": material_counter.most_common(args.top_k),
        "top_frameworks": framework_counter.most_common(args.top_k),
        "top_parameter_symbols": symbol_counter.most_common(max(30, args.top_k)),
        "top_canonical_parameters": canonical_counter.most_common(max(30, args.top_k)),
        "top_journals": journal_counter.most_common(args.top_k),
        "top_lattice_types": lattice_counter.most_common(args.top_k),
        "year_distribution": year_counter,
        "source_type_distribution": source_counter,
        "framework_year_trend": {year: dict(counter) for year, counter in framework_year_counter.items()},
        "top_parameters_by_framework": {
            framework: counter.most_common(10)
            for framework, counter in canonical_by_framework.items()
        },
        "top_parameters_by_lattice": {
            lattice: counter.most_common(10)
            for lattice, counter in canonical_by_lattice.items()
        },
    }

    outdir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    _safe_write_csv(
        outdir / "case_top_materials.csv",
        [{"material": k, "paper_count": v} for k, v in material_counter.most_common(args.top_k)],
    )
    _safe_write_csv(
        outdir / "case_top_frameworks.csv",
        [{"framework": k, "paper_count": v} for k, v in framework_counter.most_common(args.top_k)],
    )
    _safe_write_csv(
        outdir / "case_top_canonical_parameters.csv",
        [{"canonical_name": k, "count": v} for k, v in canonical_counter.most_common(max(30, args.top_k))],
    )
    _safe_write_csv(
        outdir / "case_quality_tiers.csv",
        [{"quality_tier": k, "count": v} for k, v in quality_counter.items()],
    )
    _safe_write_csv(
        outdir / "case_year_framework_trend.csv",
        [
            {"year": year, "framework": framework, "count": count}
            for year, counter in sorted(framework_year_counter.items())
            for framework, count in counter.items()
        ],
    )
    _safe_write_csv(
        outdir / "case_parameter_by_framework.csv",
        [
            {"framework": framework, "canonical_name": canonical_name, "count": count}
            for framework, counter in canonical_by_framework.items()
            for canonical_name, count in counter.most_common(10)
        ],
    )
    _safe_write_csv(
        outdir / "case_parameter_by_lattice.csv",
        [
            {"lattice_type": lattice, "canonical_name": canonical_name, "count": count}
            for lattice, counter in canonical_by_lattice.items()
            for canonical_name, count in counter.most_common(10)
        ],
    )

    print(f"Saved analytics json -> {output}")
    print(f"Saved analytics tables -> {outdir}")


if __name__ == "__main__":
    main()
