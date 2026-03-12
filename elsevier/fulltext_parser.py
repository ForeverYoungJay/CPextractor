# elsevier/fulltext_parser.py

import os
import re
import json
import time
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def safe_id(s: str) -> str:
    """Convert DOI into a filesystem-safe string."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def safe_filename(title: str) -> str:
    """Convert section title into a filesystem-safe filename."""
    if not title:
        return "Untitled"
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"[^\w\-_ .]", "_", title)
    return title[:80].strip()


def normalize_text(s: str) -> str:
    """Normalize whitespace."""
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def rows_to_markdown(rows):
    """Convert table rows into a Markdown table."""
    if not rows:
        return ""

    ncol = max(len(r) for r in rows)
    norm = [r + [""] * (ncol - len(r)) for r in rows]

    header = norm[0]
    body = norm[1:] if len(norm) > 1 else []

    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * ncol) + " |")

    for r in body:
        md.append("| " + " | ".join(r) + " |")

    return "\n".join(md)


# --------------------------------------------------
# Elsevier API
# --------------------------------------------------

BASE_URL = "https://api.elsevier.com/content/article/doi/"


def _http_get_with_retry(url, *, headers=None, params=None, timeout=30, max_retries=3):
    delay = 1.0
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            return r
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"HTTP request failed after retries: {last_exc}")


def fetch_xml_by_doi(doi, api_key, inst_token=None, max_retries=3):
    doi_safe = quote(doi, safe="")
    url = f"{BASE_URL}{doi_safe}"

    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "text/xml",
    }
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

    r = _http_get_with_retry(
        url,
        headers=headers,
        params={"view": "FULL"},
        timeout=30,
        max_retries=max_retries,
    )
    if r.ok and r.text.strip():
        return r.text
    return None


# --------------------------------------------------
# Table extraction
# --------------------------------------------------

def extract_table_rows(table_tag, refid_to_num=None):
    rows = []
    row_tags = table_tag.find_all(["ce:row", "row", "tr"])
    for row in row_tags:
        cells = []
        cell_tags = row.find_all(["ce:entry", "entry", "td", "th"])
        for c in cell_tags:
            # Work on a copy so we can safely replace tags
            c_copy = BeautifulSoup(str(c), "xml")

            if refid_to_num:
                replace_crossrefs_with_numbers(c_copy, refid_to_num)

            txt = normalize_text(c_copy.get_text(" ", strip=True))
            txt = compress_numeric_citation_groups(txt)
            cells.append(txt)

        if any(cell.strip() for cell in cells):
            rows.append(cells)
    return rows



def extract_caption(table_tag):
    cap = table_tag.find(["ce:caption", "caption", "title"])
    if cap:
        return normalize_text(cap.get_text(" ", strip=True))
    return None


def extract_tables_from_xml(soup, refid_to_num=None):
    tables = soup.find_all(["ce:table", "table-wrap", "table"])
    extracted = []
    idx = 1

    for t in tables:
        rows = extract_table_rows(t, refid_to_num=refid_to_num)
        if not rows:
            continue

        extracted.append({
            "table_index": idx,
            "caption": extract_caption(t),
            "rows": rows,
        })
        idx += 1

    return extracted



# --------------------------------------------------
# Abstract extraction
# --------------------------------------------------

def extract_abstract_from_xml(soup):
    abs_tag = soup.find(["ce:abstract", "abstract"])
    if abs_tag:
        paras = abs_tag.find_all(["ce:para", "p"])
        if paras:
            return "\n\n".join(
                normalize_text(p.get_text(" ", strip=True))
                for p in paras if p.get_text(strip=True)
            )
        return normalize_text(abs_tag.get_text(" ", strip=True))

    dc_desc = soup.find("dc:description")
    if dc_desc and dc_desc.get_text(strip=True):
        return normalize_text(dc_desc.get_text(" ", strip=True))

    return None


# --------------------------------------------------
# Cross-ref (citation) conversion
# --------------------------------------------------

def build_refid_to_number(soup):
    """
    Build mapping from bibliography entry id/refid (e.g., 'bib0001', 'bib12')
    to paper-local numeric index as a string (e.g., '1', '12').
    """
    refid_to_num = {}

    for ref in soup.find_all(["ce:bib-reference", "bib-reference"]):
        rid = ref.get("id") or ref.get("refid")
        if not rid:
            continue

        m = re.search(r"(\d+)", rid)
        if not m:
            continue

        num = str(int(m.group(1)))  # "0001" -> "1"
        refid_to_num[rid] = num

        # Also map "bib0001" -> "bib1" form (helps mixed IDs)
        prefix = re.sub(r"\d+", "", rid)
        refid_to_num[f"{prefix}{num}"] = num

    return refid_to_num


def replace_crossrefs_with_numbers(tag, refid_to_num):
    """
    In-place: replace <ce:cross-ref ... refid="bibXXXX">...</ce:cross-ref>
    with "[N]" where N comes from refid_to_num.
    """
    for cr in tag.find_all(["ce:cross-ref", "cross-ref"]):
        refid = cr.get("refid") or cr.get("rid")
        num = refid_to_num.get(refid)

        # Fallback: if we can't map it, keep visible text.
        if not num:
            cr.replace_with(cr.get_text(" ", strip=True))
            continue

        cr.replace_with(f"[{num}]")


def compress_numeric_citation_groups(text: str) -> str:
    """
    Turn patterns like:
      "([12]; [19]; [48])" -> "[12,19,48]"
      "([12], [19])" -> "[12,19]"
    """
    def repl(m):
        inside = m.group(1)
        nums = re.findall(r"\[(\d+)\]", inside)
        if not nums:
            return m.group(0)
        return "[" + ",".join(nums) + "]"

    # Parentheses groups containing ONLY bracketed numbers + separators
    text = re.sub(
        r"\(\s*((?:\[\d+\]\s*[,;]\s*)*\[\d+\])\s*\)",
        repl,
        text
    )

    # Simple repeated adjacent pairs: "[1], [2]" -> "[1,2]"
    # (Apply repeatedly to catch chains)
    while True:
        new = re.sub(
            r"\[\s*(\d+)\s*\]\s*,\s*\[\s*(\d+)\s*\]",
            r"[\1,\2]",
            text
        )
        if new == text:
            break
        text = new

    # Flatten nested citation brackets:
    #   "[ [12,19] ]" or "[[12,19]]" -> "[12,19]"
    while True:
        new = re.sub(
            r"\[\s*\[\s*((?:\d+\s*[,;]\s*)*\d+)\s*\]\s*\]",
            r"[\1]",
            text
        )
        if new == text:
            break
        text = new

    return text


# --------------------------------------------------
# Section extraction (with nested-section de-dup)
# --------------------------------------------------

def _is_inside_nested_section(tag, current_sec):
    """
    True if `tag` is inside a nested section of `current_sec`
    (i.e., there is an ancestor section between tag and current_sec).
    """
    parent = tag.parent
    while parent is not None and parent is not current_sec:
        if parent.name in ("ce:section", "sec"):
            return True
        parent = parent.parent
    return False


def section_to_markdown(sec, tables_map, refid_to_num):
    md_lines = []

    # Section title
    st = sec.find(["ce:section-title", "section-title", "title"])
    if st and st.get_text(strip=True):
        md_lines.append("## " + normalize_text(st.get_text(" ", strip=True)))
        md_lines.append("")

    # Paragraphs belonging to THIS section only (exclude nested sections)
    for p in sec.find_all(["ce:para", "para", "p"], recursive=True):
        if _is_inside_nested_section(p, sec):
            continue

        # Work on a copy of the paragraph so we can replace tags safely
        p_copy = BeautifulSoup(str(p), "xml")
        p_tag = p_copy.find(["ce:para", "para", "p"]) or p_copy

        replace_crossrefs_with_numbers(p_tag, refid_to_num)

        txt = normalize_text(p_tag.get_text(" ", strip=True))
        if txt:
            txt = compress_numeric_citation_groups(txt)
            md_lines.append(txt)
            md_lines.append("")

    # Tables inside THIS section only
    seen_table_idxs = set()
    for table in sec.find_all(["ce:table", "table-wrap", "table"], recursive=True):
        if _is_inside_nested_section(table, sec):
            continue

        rows = extract_table_rows(table, refid_to_num=refid_to_num)

        if not rows:
            continue

        cap = extract_caption(table) or ""
        key = cap + "|" + "|".join(rows[0])

        tinfo = tables_map.get(key)
        if not tinfo:
            # Fallback: write inline even if not matched to global table list
            md_lines.append("### Table")
            if cap:
                md_lines.append(f"**Caption:** {cap}")
            md_lines.append("")
            md_lines.append(rows_to_markdown(rows))
            md_lines.append("")
            continue

        idx = tinfo["table_index"]
        if idx in seen_table_idxs:
            continue
        seen_table_idxs.add(idx)

        md_lines.append(f"### Table {idx}")
        if tinfo.get("caption"):
            md_lines.append(f"**Caption:** {tinfo['caption']}")
        md_lines.append("")
        md_lines.append(rows_to_markdown(tinfo["rows"]))
        md_lines.append("")

    return "\n".join(md_lines).strip()


# --------------------------------------------------
# Reference extraction + Crossref lookup
# --------------------------------------------------

CROSSREF_URL = "https://api.crossref.org/works"


def build_bibliographic_string(journal=None, volume=None, year=None, article_number=None):
    parts = []
    if journal:
        parts.append(journal)
    if volume:
        parts.append(volume)
    if year:
        parts.append(f"({year})")
    if article_number:
        parts.append(article_number)
    return " ".join(parts)


def lookup_doi_crossref_biblio(journal=None, volume=None, year=None, article_number=None, mailto=None, max_retries=2):
    """
    Resolve DOI from bibliographic metadata using Crossref.
    Returns DOI string or None.
    """
    query = build_bibliographic_string(journal, volume, year, article_number)
    if not query.strip():
        return None

    params = {
        "query.bibliographic": query,
        "rows": 1
    }

    headers = {
        "User-Agent": (
            f"nims-demura-fulltext-parser/1.0 (mailto:{mailto})"
            if mailto else
            "nims-demura-fulltext-parser/1.0"
        )
    }

    try:
        r = _http_get_with_retry(
            CROSSREF_URL,
            params=params,
            headers=headers,
            timeout=20,
            max_retries=max_retries,
        )
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if items:
            return items[0].get("DOI")
    except Exception:
        pass

    return None


def extract_references_from_xml(soup, crossref_mailto=None, resolve_missing_reference_doi=True):
    """
    Extract reference list from Elsevier XML.
    - Preserves paper-local reference numbering
    - Uses publisher DOI if present
    - Resolves missing DOIs via Crossref (deterministic-ish based on biblio)
    - Never guesses beyond the query

    Returns list of dicts with keys:
      label, title, doi
    """
    references = []

    for ref in soup.find_all(["ce:bib-reference", "bib-reference"]):
        ref_id = ref.get("id") or ref.get("refid")

        # Reference label from id digits (e.g., bib0001 -> "1", bib12 -> "12")
        label = None
        if ref_id:
            m = re.search(r"(\d+)", ref_id)
            if m:
                label = str(int(m.group(1)))

        # Title
        title_tag = ref.find(["ce:title", "title"])
        title = normalize_text(title_tag.get_text(" ", strip=True)) if title_tag else None

        # Publisher-provided DOI
        doi_tag = ref.find("ce:doi")
        doi = doi_tag.get_text(strip=True) if doi_tag else None

        # If DOI missing → Crossref lookup
        if not doi and resolve_missing_reference_doi:
            journal = None
            volume = None
            year = None
            article_number = None

            journal_tag = ref.find("sb:maintitle")
            if journal_tag:
                journal = normalize_text(journal_tag.get_text(" ", strip=True))

            volume_tag = ref.find("sb:volume-nr")
            if volume_tag:
                volume = volume_tag.get_text(strip=True)

            year_tag = ref.find("sb:date")
            if year_tag:
                try:
                    year = int(year_tag.get_text(strip=True))
                except ValueError:
                    pass

            article_tag = ref.find("sb:article-number")
            if article_tag:
                article_number = article_tag.get_text(strip=True)

            doi = lookup_doi_crossref_biblio(
                journal=journal,
                volume=volume,
                year=year,
                article_number=article_number,
                mailto=crossref_mailto
            )

        references.append({
            "label": label,     # e.g. "60"
            "title": title,
            "doi": doi,         # may be None
        })

    return references


# --------------------------------------------------
# Main entry point
# --------------------------------------------------

def save_paper_as_markdown_and_tables(
    doi,
    api_key,
    inst_token=None,
    outdir="data/fulltext",
    crossref_mailto=None,
    resolve_missing_reference_doi=True,
    http_max_retries=3,
):
    paper_id = safe_id(doi)
    base_dir = os.path.join(outdir, paper_id)

    # --------------------------------------------------
    # Fetch XML FIRST (no folders yet)
    # --------------------------------------------------
    xml_text = fetch_xml_by_doi(doi, api_key, inst_token, max_retries=http_max_retries)
    if not xml_text:
        print(f"❌ Failed to fetch XML for DOI: {doi}")
        return

    # --------------------------------------------------
    # Create folders ONLY if fetch succeeded
    # --------------------------------------------------
    sections_dir = os.path.join(base_dir, "sections")
    tables_dir = os.path.join(base_dir, "tables")

    os.makedirs(sections_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Save raw XML
    xml_path = os.path.join(base_dir, "paper.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_text)

    soup = BeautifulSoup(xml_text, "xml")

    # --------------------------------------------------
    # Build refid -> numeric mapping for citation conversion
    # --------------------------------------------------
    refid_to_num = build_refid_to_number(soup)

    # --------------------------------------------------
    # Extract and save references
    # --------------------------------------------------
    references = extract_references_from_xml(
        soup,
        crossref_mailto=crossref_mailto,
        resolve_missing_reference_doi=resolve_missing_reference_doi,
    )
    if references:
        with open(os.path.join(base_dir, "references.json"), "w", encoding="utf-8") as f:
            json.dump(references, f, ensure_ascii=False, indent=2)

    # Paper title
    title_tag = soup.find(["dc:title", "ce:title", "article-title", "title"])
    paper_title = normalize_text(title_tag.get_text(" ", strip=True)) if title_tag else "Untitled Paper"

    # Extract tables (global)
    tables = extract_tables_from_xml(soup, refid_to_num=refid_to_num)

    tables_map = {}
    for t in tables:
        cap = t.get("caption") or ""
        key = cap + "|" + "|".join(t["rows"][0])
        tables_map[key] = t

    # Save tables
    for t in tables:
        idx = t["table_index"]

        with open(os.path.join(tables_dir, f"table_{idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False, indent=2)

        md = ""
        if t.get("caption"):
            md += f"**Caption:** {t['caption']}\n\n"
        md += rows_to_markdown(t["rows"])

        with open(os.path.join(tables_dir, f"table_{idx:03d}.md"), "w", encoding="utf-8") as f:
            f.write(md)

    # Abstract
    combined_md = [f"# {paper_title}", ""]
    abstract = extract_abstract_from_xml(soup)

    if abstract:
        abs_md = "# Abstract\n\n" + abstract
        with open(
            os.path.join(sections_dir, "000_Abstract.md"),
            "w", encoding="utf-8"
        ) as f:
            f.write(abs_md)

        combined_md.extend([abs_md, ""])


    # Sections (flat iteration, but per-section content excludes nested duplicates)
    sections = soup.find_all(["ce:section", "sec"])
    sec_idx = 1

    for sec in sections:
        sec_title_tag = sec.find(["ce:section-title", "section-title", "title"])
        sec_title = normalize_text(sec_title_tag.get_text(" ", strip=True)) if sec_title_tag else f"Section_{sec_idx}"

        md_text = section_to_markdown(sec, tables_map, refid_to_num)
        if not md_text:
            continue

        fname = f"{sec_idx:03d}_{safe_filename(sec_title)}.md"
        with open(os.path.join(sections_dir, fname), "w", encoding="utf-8") as f:
            f.write(md_text)

        combined_md.extend([md_text, ""])
        sec_idx += 1

    # Combined Markdown
    with open(os.path.join(base_dir, "paper.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(combined_md).strip())

    print(f"✅ DOI processed: {doi}")
    print(f"📂 Output directory: {base_dir}")
