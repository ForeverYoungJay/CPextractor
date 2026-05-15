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


def _is_extractor_first_payload(extracted_json: Dict[str, Any]) -> bool:
    raw = str(extracted_json.get("schema_version") or "").strip()
    if not raw:
        return False
    try:
        major = int(raw.split(".")[0])
    except Exception:
        return False
    return major >= 4


def backfill_document_metadata(extracted_json: Dict[str, Any], paper_dir: str, doi_hint: str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report = {
        "paper_xml_found": False,
        "record_id_filled": False,
        "document_fields_filled": [],
    }

    extractor_first = _is_extractor_first_payload(extracted_json)
    document = extracted_json.setdefault("document", {})
    document_defaults = (
        ("title", None),
        ("authors", []),
        ("year", None),
        ("journal", None),
        ("doi", None),
        ("notes", None),
    )
    for k, default in document_defaults:
        if k not in document:
            document[k] = default

    source_document = None
    if not extractor_first:
        source_document = extracted_json.setdefault("source_document", {})
        source_defaults = (
            ("title", None),
            ("authors", []),
            ("year", None),
            ("journal_or_venue", None),
            ("doi", None),
        )
        for k, default in source_defaults:
            if k not in source_document:
                source_document[k] = default
        source_document.pop("url", None)
        source_document.pop("notes", None)

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

    def _fill_document_scalar(key: str, value: Any):
        if value in (None, "", []):
            return
        if document.get(key) in (None, "", []):
            document[key] = value
            report["document_fields_filled"].append(key)

    def _first_document_value(primary_key: str, legacy_key: str | None = None) -> Any:
        if document.get(primary_key) not in (None, "", []):
            return document.get(primary_key)
        if source_document and legacy_key and source_document.get(legacy_key) not in (None, "", []):
            return source_document.get(legacy_key)
        return None

    if document.get("title") in (None, "", []) and source_document and source_document.get("title") not in (None, "", []):
        document["title"] = source_document.get("title")
    if (not isinstance(document.get("authors"), list) or not document.get("authors")) and source_document:
        if isinstance(source_document.get("authors"), list) and source_document.get("authors"):
            document["authors"] = list(source_document.get("authors"))
    if document.get("year") in (None, "", []) and source_document and source_document.get("year") not in (None, "", []):
        document["year"] = source_document.get("year")
    if document.get("journal") in (None, "", []) and source_document and source_document.get("journal_or_venue") not in (None, "", []):
        document["journal"] = source_document.get("journal_or_venue")
    if document.get("doi") in (None, "", []) and source_document and source_document.get("doi") not in (None, "", []):
        document["doi"] = source_document.get("doi")

    _fill_document_scalar("title", title)
    if authors and (not isinstance(document.get("authors"), list) or not document.get("authors")):
        document["authors"] = authors
        report["document_fields_filled"].append("authors")
    _fill_document_scalar("year", year)
    _fill_document_scalar("journal", journal)
    _fill_document_scalar("doi", doi)

    if source_document is not None:
        source_document["title"] = _first_document_value("title", "title")
        source_document["authors"] = _first_document_value("authors", "authors") or []
        source_document["year"] = _first_document_value("year", "year")
        source_document["journal_or_venue"] = _first_document_value("journal", "journal_or_venue")
        source_document["doi"] = _first_document_value("doi", "doi")

    if extracted_json.get("record_id") in (None, ""):
        resolved_doi = document.get("doi") or (source_document.get("doi") if source_document else None)
        if resolved_doi:
            extracted_json["record_id"] = str(resolved_doi).lower()
        elif doi_hint:
            extracted_json["record_id"] = str(doi_hint).lower()
        else:
            extracted_json["record_id"] = os.path.basename(paper_dir.rstrip("/"))
        report["record_id_filled"] = True

    return extracted_json, report
