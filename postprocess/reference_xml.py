from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag.split(":", 1)[-1]


def _iter_descendants(root: ET.Element, names: set[str]) -> Iterable[ET.Element]:
    for elem in root.iter():
        if _local_name(elem.tag) in names:
            yield elem


def _first_descendant(root: ET.Element, names: set[str]) -> ET.Element | None:
    for elem in _iter_descendants(root, names):
        return elem
    return None


def _text_content(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    text = _normalize_text("".join(elem.itertext()))
    return text or None


def _reference_label(ref: ET.Element) -> str | None:
    label = _text_content(_first_descendant(ref, {"label"}))
    if label:
        m = re.search(r"(\d+)", label)
        if m:
            return str(int(m.group(1)))
    ref_id = str(ref.attrib.get("id") or ref.attrib.get("refid") or "").strip()
    if ref_id:
        m = re.search(r"(\d+)", ref_id)
        if m:
            return str(int(m.group(1)))
    return None


def _reference_title(ref: ET.Element) -> str | None:
    for names in ({"title"}, {"maintitle"}):
        elem = _first_descendant(ref, names)
        text = _text_content(elem)
        if text:
            return text
    return None


def _reference_doi(ref: ET.Element) -> str | None:
    elem = _first_descendant(ref, {"doi"})
    text = _text_content(elem)
    return text or None


def extract_references_from_xml_text(xml_text: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    references: List[Dict[str, Any]] = []
    for ref in _iter_descendants(root, {"bib-reference"}):
        references.append(
            {
                "label": _reference_label(ref),
                "title": _reference_title(ref),
                "doi": _reference_doi(ref),
            }
        )
    return references
