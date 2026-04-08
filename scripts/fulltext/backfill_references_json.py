import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from postprocess.reference_xml import extract_references_from_xml_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill missing references.json from local paper.xml files.")
    ap.add_argument("--root", default="data/fulltext")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    built = 0
    skipped = 0
    failed = 0

    for paper_dir in sorted(root.iterdir()):
        if not paper_dir.is_dir():
            continue
        xml_path = paper_dir / "paper.xml"
        ref_path = paper_dir / "references.json"
        if not xml_path.exists():
            continue
        if ref_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            xml_text = xml_path.read_text(encoding="utf-8", errors="ignore")
            references = extract_references_from_xml_text(xml_text)
            if not references:
                failed += 1
                print(f"NO_REFS {paper_dir.name}")
                continue
            ref_path.write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
            built += 1
            print(f"BUILT {paper_dir.name} {len(references)}")
        except Exception as exc:
            failed += 1
            print(f"FAILED {paper_dir.name} {exc}")

    print(f"SUMMARY built={built} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
