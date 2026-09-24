"""Freeze a small paper universe and source-backed, explicitly pending review packets."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.benchmark_protocol import PROTOCOL_VERSION, normalize_row, valid_doi, fingerprint
from scripts.eval.export_annotation_draft import _claim_rows_for_paper
from scripts.eval.common import save_json, save_csv


def source_path(paper_dir, source_file):
    if not source_file:
        return None
    root = Path(paper_dir).resolve()
    for relative in (source_file, f"tables/{source_file}", f"sections/{source_file}", f"equations/{source_file}"):
        candidate = (root / relative).resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return candidate
    return None


def source_packet(row, paper_dir):
    path = source_path(paper_dir, row["evidence"].get("file"))
    result = {"doi": row["doi"], "claim_id": row.get("claim_id"),
              "canonical_name": row.get("canonical_name"), "symbol": row.get("symbol"),
              "value": row.get("value"), "unit": row.get("unit"),
              "context": row.get("context"), "provenance": row.get("provenance"),
              "claimed_locator": row.get("evidence"), "source_available": path is not None}
    if path:
        raw = path.read_text(encoding="utf-8")
        result.update(source_file=str(path), source_sha256=fingerprint(path))
        if path.suffix == ".json":
            data = json.loads(raw)
            result["source_content"] = data
        else:
            result["source_content"] = raw
    return result


def prepare(root, output, n=50, seed=20260907):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite a review workspace: {output}")
    candidates = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        guessed = directory.name.replace("_", "/", 1).lower()
        if not valid_doi(guessed):
            continue
        artifact = directory / "materials_extracted.json"
        data = json.loads(artifact.read_text()) if artifact.exists() else {}
        paper_doi = (data.get("document") or {}).get("doi") or guessed
        candidates.append({"doi": paper_doi.lower(), "paper_dir": str(directory),
                           "title": (data.get("document") or {}).get("title"),
                           "prediction_sha256": fingerprint(artifact) if artifact.exists() else None,
                           "prediction_available": artifact.exists(),
                           "claim_count": len(data.get("parameter_claims") or []),
                           "material_count": len(data.get("materials") or []),
                           "condition_count": len(data.get("conditions") or [])})
        print(f"Indexed {paper_doi}", flush=True)
    # Round robin by DOI journal segment, seeded within each group. Include
    # failed/zero-claim papers so selection is not conditional on model success.
    groups = defaultdict(list)
    rng = random.Random(seed)
    for row in candidates:
        parts = row["doi"].split(".")
        journal = parts[2] if len(parts) > 3 else "other"
        groups[journal].append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < min(n, len(candidates)):
        for key in sorted(groups):
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == min(n, len(candidates)):
                    break
    rng.shuffle(selected)
    total = len(selected)
    dev = min(20, max(1, int(total * .4)))
    cal = (total-dev)//2
    all_rows, sources = [], []
    for i, paper in enumerate(selected):
        paper.update(split="development" if i < dev else "calibration" if i < dev+cal else "test",
                     annotation_status="pending", exhaustive=False, reviewer_type=None,
                     reviewer=None, adjudicator=None, gate_should_block=None,
                     gold_claim_count=None)
        directory = Path(paper["paper_dir"])
        packet = output / "packets" / directory.name
        rows = [normalize_row(r) for r in _claim_rows_for_paper(directory, "materials_extracted.json")]
        for row in rows:
            row["annotation"] = {"status": "pending", "reviewer_type": None, "reviewer": None,
                                 "reviewed_fields": [], "error_tags": [], "notes": ""}
        packet.mkdir(parents=True, exist_ok=True)
        (packet / "claims.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
        save_json(packet / "paper.json", paper)
        evidence = [source_packet(r, directory) for r in rows]
        save_json(packet / "source_review.json", evidence)
        (packet / "missing_claims.jsonl").write_text("")
        (packet / "README.md").write_text(
            "# Source review packet\n\nReview the original paper, not only the selected evidence.\n"
            "Edit claims.jsonl, add omitted claims to missing_claims.jsonl, and record reviewer identity.\n"
            "Generated evidence snippets are not independent support; source_review.json contains local source bytes.\n"
            "Set exhaustive=true only after checking all relevant sections, tables and equations.\n"
            "AI review must retain reviewer_type=ai_assistant. Test papers must not guide prompt/threshold tuning.\n")
        all_rows.extend(rows)
        sources.extend({k: v for k, v in r.items() if k != "source_content"} for r in evidence)
        print(f"Prepared {paper['doi']}: {len(rows)} claims ({paper['split']})", flush=True)
    manifest = {"protocol_version": PROTOCOL_VERSION, "seed": seed,
                "selection": "seeded journal round-robin, including extraction failures",
                "source_root": str(root), "papers": selected}
    save_json(output / "manifest.json", manifest)
    save_csv(output / "progress.csv", selected)
    (output / "annotation_draft.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in all_rows))
    save_json(output / "source_inventory.json", sources)
    save_json(output / "preparation_summary.json", {"papers": total, "claims": len(all_rows),
               "splits": dict(Counter(p["split"] for p in selected)),
               "missing_prediction_papers": [p["doi"] for p in selected if not p["prediction_available"]],
               "source_available_claims": sum(r["source_available"] for r in sources),
               "publication_ready": False})
    return manifest


def collect(workspace, output):
    workspace = Path(workspace)
    rows = []
    manifest = json.loads((workspace / "manifest.json").read_text())
    for paper in manifest["papers"]:
        packet = workspace / "packets" / Path(paper["paper_dir"]).name
        for name in ("claims.jsonl", "missing_claims.jsonl"):
            for line in (packet / name).read_text().splitlines():
                if line.strip():
                    row = normalize_row(json.loads(line))
                    if row["doi"] != paper["doi"]:
                        raise ValueError(f"DOI mismatch in {packet / name}")
                    rows.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root")
    ap.add_argument("--output-root")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--collect", help="Collect reviewed packets from this workspace")
    ap.add_argument("--output-jsonl")
    args = ap.parse_args()
    if args.collect:
        if not args.output_jsonl:
            ap.error("--collect requires --output-jsonl")
        collect(args.collect, args.output_jsonl)
    elif args.input_root and args.output_root and args.n > 0:
        prepare(args.input_root, args.output_root, args.n, args.seed)
    else:
        ap.error("Provide --input-root --output-root and positive --n")


if __name__ == "__main__":
    main()
