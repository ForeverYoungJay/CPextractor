import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from postprocess.param_iter import iter_parameter_items


def iter_docs(root: Path):
    for p in sorted(root.glob("*/materials_extracted.json")):
        try:
            yield json.loads(p.read_text(encoding="utf-8")), p
        except Exception:
            continue


def source_bucket(src: dict) -> str:
    t = (src.get("origin_type") or src.get("type") or "unknown").strip() if isinstance(src, dict) else "unknown"
    return t or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple analytics for CP extracted dataset")
    ap.add_argument("--root", default="data/fulltext")
    ap.add_argument("--output", default="output/analytics/materials_insight.json")
    args = ap.parse_args()

    root = Path(args.root)
    material_counter = Counter()
    framework_counter = Counter()
    source_counter = Counter()
    symbol_counter = Counter()

    for doc, _ in iter_docs(root):
        m = (doc.get("material", {}) or {}).get("name")
        if m:
            material_counter[str(m)] += 1

        framework = (doc.get("constitutive_model", {}) or {}).get("framework")
        if framework:
            framework_counter[str(framework)] += 1

        for _, p in iter_parameter_items(doc):
            if p.get("symbol"):
                symbol_counter[str(p.get("symbol"))] += 1
            source_counter[source_bucket(p.get("source", {}))] += 1

    out = {
        "top_materials": material_counter.most_common(20),
        "top_frameworks": framework_counter.most_common(20),
        "top_parameter_symbols": symbol_counter.most_common(30),
        "source_type_distribution": source_counter,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved analytics -> {out_path}")


if __name__ == "__main__":
    main()
