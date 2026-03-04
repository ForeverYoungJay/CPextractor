import argparse
from typing import Any, Dict, List

from common import load_json, load_jsonl, save_json


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
    table = {
        "pa": 1.0,
        "kpa": 1e3,
        "mpa": 1e6,
        "gpa": 1e9,
        "1/s": 1.0,
        "s^-1": 1.0,
        "": 1.0,
    }
    return table.get(u)


def normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    for block, key in (("elastic_parameters", "constants"), ("plastic_parameters", "parameters")):
        items: List[Dict[str, Any]] = doc.get(block, {}).get(key, [])
        for p in items:
            value = _to_number(p.get("value"))
            unit = p.get("unit")
            fac = _factor_to_si(unit)
            if value is not None and fac is not None:
                p["value"] = value
                p["value_SI"] = value * fac
                p["unit_SI"] = "Pa" if (unit or "").lower() in {"pa", "kpa", "mpa", "gpa"} else unit
            symbol = p.get("symbol") or p.get("canonical_name")
            if symbol is not None:
                p["symbol"] = str(symbol).strip()
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize gold labels (units, numeric values, symbols).")
    ap.add_argument("--input", required=True, help="Input .json or .jsonl")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.input.endswith(".jsonl"):
        rows = load_jsonl(args.input)
        out = [normalize_doc(r) for r in rows]
    else:
        obj = load_json(args.input)
        out = [normalize_doc(r) for r in obj] if isinstance(obj, list) else normalize_doc(obj)

    save_json(args.output, out)
    print(f"Saved normalized gold to {args.output}")


if __name__ == "__main__":
    main()
