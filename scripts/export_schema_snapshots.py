#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = "llm/extractor.py"
OUTPUT_DIR = REPO_ROOT / "docs" / "schema_snapshots"

SNAPSHOTS = [
    {
        "version": "1.1.0",
        "source_kind": "git",
        "ref": "abdb505",
        "date": "2026-03-04",
        "note": "Early prompt-embedded extraction schema.",
    },
    {
        "version": "2.0.0",
        "source_kind": "git",
        "ref": "d5346b5",
        "date": "2026-03-12",
        "note": "Seven-layer pipeline upgrade.",
    },
    {
        "version": "2.1.1",
        "source_kind": "git",
        "ref": "09f65de",
        "date": "2026-03-12",
        "note": "Registry-oriented v2.1.1 schema.",
    },
    {
        "version": "3.0.0",
        "source_kind": "git",
        "ref": "9ac9896",
        "date": "2026-04-08",
        "note": "First v3 extractor document snapshot.",
    },
    {
        "version": "3.1.0",
        "source_kind": "git",
        "ref": "ac7f652",
        "date": "2026-04-08",
        "note": "Restored v3.1 extractor schema.",
    },
    {
        "version": "4.2.0",
        "source_kind": "git",
        "ref": "662057d",
        "date": "2026-04-09",
        "note": "Refined extractor schema before current workspace changes.",
    },
    {
        "version": "4.3.0",
        "source_kind": "workspace",
        "ref": "working-tree",
        "date": "2026-04-10",
        "note": "Current uncommitted extractor schema.",
    },
]


def _load_source(snapshot: dict[str, str]) -> str:
    if snapshot["source_kind"] == "workspace":
        return (REPO_ROOT / SOURCE_FILE).read_text(encoding="utf-8")

    cmd = ["git", "show", f'{snapshot["ref"]}:{SOURCE_FILE}']
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _extract_schema_json_text(source_text: str) -> str:
    direct = re.search(
        r'EXTRACT_SCHEMA_JSON_TEMPLATE\s*=\s*r?"""(.*?)"""',
        source_text,
        re.DOTALL,
    )
    if direct:
        return direct.group(1).strip()

    prompt = re.search(
        r'EXTRACT_USER_PROMPT_TEMPLATE\s*=\s*"""(.*?)"""',
        source_text,
        re.DOTALL,
    )
    if not prompt:
        raise RuntimeError("Could not find schema template in source text")

    block = prompt.group(1)
    start_marker = "Extract the following schema from the paper excerpt:"
    start = block.find(start_marker)
    if start >= 0:
        block = block[start + len(start_marker) :]

    end_positions = []
    for marker in ("Paper excerpt:", "\n3 Processing suggestions", "\n4 Few-shot examples"):
        pos = block.find(marker)
        if pos >= 0:
            end_positions.append(pos)
    if end_positions:
        block = block[: min(end_positions)]

    block = block.strip().replace("{{", "{").replace("}}", "}")
    left = block.find("{")
    right = block.rfind("}")
    if left < 0 or right < 0 or right <= left:
        raise RuntimeError("Could not isolate JSON schema block from prompt template")
    return block[left : right + 1]


def _extract_schema_object(source_text: str) -> dict:
    json_text = _extract_schema_json_text(source_text)
    return json.loads(json_text)


def _write_snapshot(snapshot: dict[str, str]) -> str:
    source_text = _load_source(snapshot)
    schema = _extract_schema_object(source_text)
    version = str(schema.get("schema_version"))
    if version != snapshot["version"]:
        raise RuntimeError(
            f'Version mismatch for {snapshot["ref"]}: expected {snapshot["version"]}, got {version}'
        )

    filename = f"extractor_schema_v{version}.json"
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return filename


def _write_index(rows: list[tuple[dict[str, str], str]]) -> None:
    lines = [
        "# Schema Snapshots",
        "",
        "Generated from `llm/extractor.py` git history and the current workspace.",
        "",
        "| Version | Date | Source | File |",
        "| --- | --- | --- | --- |",
    ]
    for snapshot, filename in rows:
        source = snapshot["ref"]
        lines.append(
            f'| `{snapshot["version"]}` | {snapshot["date"]} | `{source}` | [{filename}](/Users/yang/Library/CloudStorage/OneDrive-国立研究開発法人物質・材料研究機構/自分/CPextractor/docs/schema_snapshots/{filename}) |'
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Versions `1.1.0` and `2.0.0` were embedded inside `EXTRACT_USER_PROMPT_TEMPLATE`, so these snapshots are reconstructed from the prompt JSON block.",
            "- Versions `2.1.1` and later are exported from `EXTRACT_SCHEMA_JSON_TEMPLATE`.",
            "- These files preserve the complete extractor schema template for each version line documented in `docs/schema_history.md`.",
        ]
    )

    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[dict[str, str], str]] = []
    for snapshot in SNAPSHOTS:
        filename = _write_snapshot(snapshot)
        rows.append((snapshot, filename))
    _write_index(rows)


if __name__ == "__main__":
    main()
