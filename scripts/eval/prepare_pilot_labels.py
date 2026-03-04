import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def folder_to_doi(folder_name: str) -> str:
    return folder_name.replace("_", "/")


def journal_key_from_doi(doi: str) -> str:
    m = re.search(r"/j\.([^.]+)", doi)
    if m:
        return m.group(1)
    return "other"


def read_title_from_paper_md(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    if first.startswith("#"):
        return first.lstrip("#").strip()
    return first


def collect_candidates(input_root: Path) -> List[Dict]:
    out = []
    for d in sorted(input_root.iterdir()):
        if not d.is_dir():
            continue
        pred = d / "materials_extracted.json"
        if not pred.exists():
            continue

        doi = folder_to_doi(d.name)
        out.append(
            {
                "paper_dir": str(d),
                "folder": d.name,
                "doi": doi,
                "journal_key": journal_key_from_doi(doi),
                "title": read_title_from_paper_md(d / "paper.md"),
                "pred_path": str(pred),
            }
        )
    return out


def stratified_round_robin_sample(rows: List[Dict], n: int, seed: int) -> List[Dict]:
    rnd = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["journal_key"]].append(r)

    for g in groups.values():
        rnd.shuffle(g)

    keys = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)
    picked = []

    while len(picked) < n:
        added = False
        for k in keys:
            if groups[k]:
                picked.append(groups[k].pop())
                added = True
                if len(picked) >= n:
                    break
        if not added:
            break

    return picked


def build_seed_record(meta: Dict) -> Dict:
    with Path(meta["pred_path"]).open("r", encoding="utf-8") as f:
        pred = json.load(f)

    pred["record_id"] = pred.get("record_id") or meta["doi"]
    pred["source_document"] = pred.get("source_document", {})
    pred["source_document"]["doi"] = pred["source_document"].get("doi") or meta["doi"]
    pred["source_document"]["title"] = pred["source_document"].get("title") or meta["title"]

    pred["annotation_metadata"] = {
        "stage": "pilot_50",
        "status": "seeded",
        "seed_source": "materials_extracted.json",
        "paper_dir": meta["paper_dir"],
        "journal_key": meta["journal_key"],
    }
    return pred


def write_manifest(path: Path, rows: List[Dict]) -> None:
    fields = [
        "index",
        "doi",
        "journal_key",
        "title",
        "paper_dir",
        "annotator_a",
        "annotator_b",
        "status_a",
        "status_b",
        "adjudication_status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            w.writerow(
                {
                    "index": i,
                    "doi": r["doi"],
                    "journal_key": r["journal_key"],
                    "title": r["title"],
                    "paper_dir": r["paper_dir"],
                    "annotator_a": "",
                    "annotator_b": "",
                    "status_a": "todo",
                    "status_b": "todo",
                    "adjudication_status": "todo",
                    "notes": "",
                }
            )


def write_seed_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: List[Dict], n_total: int) -> None:
    by_journal = defaultdict(int)
    for r in rows:
        by_journal[r["journal_key"]] += 1

    lines = [
        "# Pilot 50 Trial Annotation Set",
        "",
        f"- total_candidates: {n_total}",
        f"- sampled: {len(rows)}",
        "",
        "## Journal distribution",
        "",
    ]

    for k, v in sorted(by_journal.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "## Files",
        "",
        "- pilot50_manifest.csv: assignment/progress tracking",
        "- gold_trial_50_seed.jsonl: seeded labels for manual correction",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare 50-paper pilot annotation package.")
    ap.add_argument("--input-root", default="data/fulltext")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data/annotations/pilot50")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    outdir = Path(args.outdir)

    rows = collect_candidates(input_root)
    if not rows:
        raise RuntimeError("No candidates found (missing materials_extracted.json)")

    n = min(args.n, len(rows))
    picked = stratified_round_robin_sample(rows, n=n, seed=args.seed)
    picked = sorted(picked, key=lambda x: x["doi"])

    seed_records = [build_seed_record(r) for r in picked]

    write_manifest(outdir / "pilot50_manifest.csv", picked)
    write_seed_jsonl(outdir / "gold_trial_50_seed.jsonl", seed_records)
    write_summary(outdir / "README.md", picked, n_total=len(rows))

    print(f"Prepared pilot set: {len(picked)} papers")
    print(f"Output dir: {outdir}")


if __name__ == "__main__":
    main()
