from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)


def _xml_first(text: str, tag: str) -> str | None:
    m = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    v = re.sub(r"\s+", " ", m.group(1)).strip()
    return v or None


def _xml_all(text: str, tag: str) -> list[str]:
    vals = re.findall(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.IGNORECASE | re.DOTALL)
    out = []
    seen = set()
    for v in vals:
        s = re.sub(r"\s+", " ", v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _year_from_cover_date(cover_date: str | None) -> int | None:
    if not cover_date:
        return None
    m = re.match(r"^(\d{4})", str(cover_date))
    if not m:
        return None
    return int(m.group(1))


def backfill_document_metadata(extracted_json: Dict[str, Any], paper_dir: str, doi_hint: str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report = {
        "paper_xml_found": False,
        "record_id_filled": False,
        "source_document_fields_filled": [],
    }

    sd = extracted_json.setdefault("source_document", {})
    for k, default in (
        ("title", None),
        ("authors", []),
        ("year", None),
        ("journal_or_venue", None),
        ("doi", None),
    ):
        if k not in sd:
            sd[k] = default
    # Keep schema compact.
    sd.pop("url", None)
    sd.pop("notes", None)

    xml_text = ""
    xml_path = Path(paper_dir) / "paper.xml"
    if xml_path.exists():
        report["paper_xml_found"] = True
        xml_text = xml_path.read_text(encoding="utf-8", errors="ignore")

    title = _xml_first(xml_text, "dc:title") if xml_text else None
    doi = _xml_first(xml_text, "prism:doi") if xml_text else None
    if not doi and xml_text:
        m = DOI_PATTERN.search(xml_text)
        doi = m.group(0) if m else None
    authors = _xml_all(xml_text, "dc:creator") if xml_text else []
    journal = _xml_first(xml_text, "prism:publicationName") if xml_text else None
    cover_date = _xml_first(xml_text, "prism:coverDate") if xml_text else None
    year = _year_from_cover_date(cover_date)

    if not doi:
        doi = doi_hint

    def _fill_scalar(key: str, value: Any):
        if value in (None, "", []):
            return
        if sd.get(key) in (None, "", []):
            sd[key] = value
            report["source_document_fields_filled"].append(key)

    _fill_scalar("title", title)
    if authors and (not isinstance(sd.get("authors"), list) or not sd.get("authors")):
        sd["authors"] = authors
        report["source_document_fields_filled"].append("authors")
    _fill_scalar("year", year)
    _fill_scalar("journal_or_venue", journal)
    _fill_scalar("doi", doi)

    if extracted_json.get("record_id") in (None, ""):
        if sd.get("doi"):
            extracted_json["record_id"] = str(sd["doi"]).lower()
        elif doi_hint:
            extracted_json["record_id"] = str(doi_hint).lower()
        else:
            extracted_json["record_id"] = os.path.basename(paper_dir.rstrip("/"))
        report["record_id_filled"] = True

    return extracted_json, report
