import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import load_json, save_json


def _to_number(v: Any):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return None
    return None


def _factor_to_si(unit: str) -> float | None:
    u = (unit or "").strip().lower()
    table = {"pa": 1.0, "kpa": 1e3, "mpa": 1e6, "gpa": 1e9}
    return table.get(u)


def normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    for block, key in (("elastic_parameters", "constants"), ("plastic_parameters", "parameters")):
        items: List[Dict[str, Any]] = doc.get(block, {}).get(key, [])
        for p in items:
            v = _to_number(p.get("value"))
            u = p.get("unit")
            fac = _factor_to_si(u)
            if v is not None and fac is not None:
                p["value"] = v
                p["value_SI"] = v * fac
                p["unit_SI"] = "Pa"
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize prediction JSONs under fulltext folders.")
    ap.add_argument("--input-root", required=True, help="Folder containing DOI folders")
    ap.add_argument("--filename", default="materials_extracted.json")
    ap.add_argument("--output", required=True, help="Aggregated normalized output JSON array")
    args = ap.parse_args()

    docs = []
    for p in sorted(Path(args.input_root).glob(f"*/{args.filename}")):
        doc = load_json(p)
        doc = normalize_doc(doc)
        doi = p.parent.name.replace("_", "/")
        doc.setdefault("record_id", doi)
        docs.append(doc)

    save_json(args.output, docs)
    print(f"Saved normalized predictions: {len(docs)} docs -> {args.output}")


if __name__ == "__main__":
    main()
