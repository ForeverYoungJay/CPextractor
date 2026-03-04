# elsevier/fulltext_parser.py

import os
import re
import json
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


def fetch_xml_by_doi(doi, api_key, inst_token=None):
    doi_safe = quote(doi, safe="")
    url = f"{BASE_URL}{doi_safe}"

    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "text/xml",
    }
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

    r = requests.get(url, headers=headers, params={"view": "FULL"}, timeout=30)
    if r.ok and r.text.strip():
        return r.text
    return None


# --------------------------------------------------
# Table extraction
# --------------------------------------------------

def extract_table_rows(table_tag):
    rows = []
    row_tags = table_tag.find_all(["ce:row", "row", "tr"])
    for row in row_tags:
        cells = []
        cell_tags = row.find_all(["ce:entry", "entry", "td", "th"])
        for c in cell_tags:
            cells.append(normalize_text(c.get_text(" ", strip=True)))
        if any(cells):
            rows.append(cells)
    return rows


def extract_caption(table_tag):
    cap = table_tag.find(["ce:caption", "caption", "title"])
    if cap:
        return normalize_text(cap.get_text(" ", strip=True))
    return None


def extract_tables_from_xml(soup):
    tables = soup.find_all(["ce:table", "table-wrap", "table"])
    extracted = []
    idx = 1

    for t in tables:
        rows = extract_table_rows(t)
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
# Section extraction
# --------------------------------------------------

def section_to_markdown(sec, tables_map):
    md_lines = []

    # Section title
    st = sec.find(["ce:section-title", "title"])
    if st:
        md_lines.append("## " + normalize_text(st.get_text(" ", strip=True)))
        md_lines.append("")

    # Paragraphs
    for p in sec.find_all(["ce:para", "p"], recursive=True):
        txt = normalize_text(p.get_text(" ", strip=True))
        if txt:
            md_lines.append(txt)
            md_lines.append("")

    # Tables inside section
    for table in sec.find_all(["ce:table", "table-wrap", "table"], recursive=True):
        rows = extract_table_rows(table)
        if not rows:
            continue

        cap = extract_caption(table) or ""
        key = cap + "|" + "|".join(rows[0])

        if key in tables_map:
            tinfo = tables_map[key]
            idx = tinfo["table_index"]

            md_lines.append(f"### Table {idx}")
            if tinfo.get("caption"):
                md_lines.append(f"**Caption:** {tinfo['caption']}")
            md_lines.append("")
            md_lines.append(rows_to_markdown(tinfo["rows"]))
            md_lines.append("")

    return "\n".join(md_lines).strip()

# --------------------------------------------------
# Reference extraction
# --------------------------------------------------

import re
import requests

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


def lookup_doi_crossref_biblio(journal=None, volume=None, year=None, article_number=None):
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

    try:
        r = requests.get(CROSSREF_URL, params=params, timeout=20)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if items:
            return items[0].get("DOI")
    except Exception:
        pass

    return None




def extract_references_from_xml(soup):
    """
    Extract reference list from Elsevier XML.
    - Preserves paper-local reference numbering
    - Uses publisher DOI if present
    - Resolves missing DOIs via Crossref (deterministic)
    - Never guesses

    Returns list of dicts with keys:
      label, title, doi, doi_source, resolved
    """
    references = []

    for ref in soup.find_all(["ce:bib-reference", "bib-reference"]):
        ref_id = ref.get("id") or ref.get("refid")

        # ----------------------------
        # Reference label (e.g. [60])
        # ----------------------------
        label = None
        if ref_id:
            m = re.search(r"(\d+)", ref_id)
            if m:
                label = m.group(1)

        # ----------------------------
        # Title
        # ----------------------------
        title_tag = ref.find(["ce:title", "title"])
        title = (
            normalize_text(title_tag.get_text(" ", strip=True))
            if title_tag else None
        )

        # ----------------------------
        # DOI (publisher-provided)
        # ----------------------------
        doi_tag = ref.find("ce:doi")
        doi = doi_tag.get_text(strip=True) if doi_tag else None
        doi_source = "publisher" if doi else None
        resolved = bool(doi)

        # ----------------------------
        # If DOI missing → Crossref lookup
        # ----------------------------
        if not doi:
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
                article_number=article_number
            )

        references.append({
            "label": label,            # e.g. "60"
            "title": title,
            "doi": doi,                # may be None
        })

    return references



# --------------------------------------------------
# Main entry point
# --------------------------------------------------

def save_paper_as_markdown_and_tables(
    doi,
    api_key,
    inst_token=None,
    outdir="data/fulltext"
):
    paper_id = safe_id(doi)
    base_dir = os.path.join(outdir, paper_id)

    # --------------------------------------------------
    # Fetch XML FIRST (no folders yet)
    # --------------------------------------------------
    xml_text = fetch_xml_by_doi(doi, api_key, inst_token)
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
    # Extract and save references
    # --------------------------------------------------
    references = extract_references_from_xml(soup)

    if references:
        with open(
            os.path.join(base_dir, "references.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(references, f, ensure_ascii=False, indent=2)


    # Paper title
    title_tag = soup.find(
        ["dc:title", "ce:title", "article-title", "title"]
    )
    paper_title = (
        normalize_text(title_tag.get_text(" ", strip=True))
        if title_tag else "Untitled Paper"
    )

    # Extract tables
    tables = extract_tables_from_xml(soup)

    tables_map = {}
    for t in tables:
        cap = t.get("caption") or ""
        key = cap + "|" + "|".join(t["rows"][0])
        tables_map[key] = t

    # Save tables
    for t in tables:
        idx = t["table_index"]

        with open(
            os.path.join(tables_dir, f"table_{idx:03d}.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(t, f, ensure_ascii=False, indent=2)

        md = ""
        if t.get("caption"):
            md += f"**Caption:** {t['caption']}\n\n"
        md += rows_to_markdown(t["rows"])

        with open(
            os.path.join(tables_dir, f"table_{idx:03d}.md"),
            "w", encoding="utf-8"
        ) as f:
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

    # Sections
    sections = soup.find_all(["ce:section", "sec"])
    sec_idx = 1

    for sec in sections:
        sec_title_tag = sec.find(["ce:section-title", "title"])
        sec_title = (
            normalize_text(sec_title_tag.get_text(" ", strip=True))
            if sec_title_tag else f"Section_{sec_idx}"
        )

        md_text = section_to_markdown(sec, tables_map)
        if not md_text:
            continue

        fname = f"{sec_idx:03d}_{safe_filename(sec_title)}.md"
        with open(
            os.path.join(sections_dir, fname),
            "w", encoding="utf-8"
        ) as f:
            f.write(md_text)

        combined_md.extend([md_text, ""])
        sec_idx += 1

    # Combined Markdown
    with open(
        os.path.join(base_dir, "paper.md"),
        "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(combined_md).strip())

    print(f"✅ DOI processed: {doi}")
    print(f"📂 Output directory: {base_dir}")
