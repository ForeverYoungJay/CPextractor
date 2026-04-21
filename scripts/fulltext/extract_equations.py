import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup

from elsevier.fulltext_parser import extract_equations_from_xml, save_equations


def process_paper_dir(paper_dir: Path) -> dict:
    xml_path = paper_dir / "paper.xml"
    equations_dir = paper_dir / "equations"

    if not xml_path.exists():
        return {
            "paper_dir": str(paper_dir),
            "status": "skipped",
            "reason": "missing_paper_xml",
            "equation_count": 0,
        }

    xml_text = xml_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(xml_text, "xml")
    equations = extract_equations_from_xml(soup)
    save_equations(equations, str(equations_dir))
    return {
        "paper_dir": str(paper_dir),
        "status": "processed",
        "equation_count": len(equations),
    }


def main():
    ap = argparse.ArgumentParser(description="Extract equations from local Elsevier paper.xml files.")
    ap.add_argument("--root", default="data/fulltext", help="Root directory containing per-paper folders.")
    ap.add_argument("--summary-out", default=None, help="Optional JSON summary path.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    results = []
    total_equations = 0
    processed = 0
    skipped = 0

    for paper_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        result = process_paper_dir(paper_dir)
        results.append(result)
        if result["status"] == "processed":
            processed += 1
            total_equations += int(result["equation_count"])
        else:
            skipped += 1

    summary = {
        "root": str(root),
        "processed_papers": processed,
        "skipped_papers": skipped,
        "total_equations": total_equations,
        "results": results,
    }

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "root": str(root),
        "processed_papers": processed,
        "skipped_papers": skipped,
        "total_equations": total_equations,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
