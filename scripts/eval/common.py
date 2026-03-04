import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return

    keys = sorted({k for r in rows for k in r.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def norm_text(v: Any) -> str:
    if v is None:
        return ""
    return " ".join(str(v).strip().lower().split())


def almost_equal(a: Any, b: Any, atol: float = 1e-9, rtol: float = 1e-6) -> bool:
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return False
    return abs(af - bf) <= (atol + rtol * abs(bf))


def flatten_dict(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_dict(v, p))
        return out
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.update(flatten_dict(v, p))
        return out
    out[prefix] = obj
    return out


def index_by_record_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = r.get("record_id") or r.get("doi") or r.get("source_document", {}).get("doi")
        if rid:
            out[str(rid)] = r
    return out


def extract_param_rows(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rid = doc.get("record_id") or doc.get("doi") or doc.get("source_document", {}).get("doi")

    elastic = doc.get("elastic_parameters", {}).get("constants", [])
    for p in elastic:
        rows.append(
            {
                "record_id": rid,
                "block": "elastic",
                "symbol": norm_text(p.get("symbol") or p.get("canonical_name")),
                "value": p.get("value"),
                "unit": norm_text(p.get("unit")),
                "source": p.get("source", {}),
            }
        )

    plastic = doc.get("plastic_parameters", {}).get("parameters", [])
    for p in plastic:
        rows.append(
            {
                "record_id": rid,
                "block": "plastic",
                "symbol": norm_text(p.get("symbol") or p.get("canonical_name")),
                "value": p.get("value"),
                "unit": norm_text(p.get("unit")),
                "source": p.get("source", {}),
            }
        )

    return rows


def build_param_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("record_id") or ""),
        str(row.get("block") or ""),
        str(row.get("symbol") or ""),
    )


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def parse_args(desc: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=desc)
