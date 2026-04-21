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


def safe_table_label(label: str, fallback_index: int) -> str:
    """Convert article table label into a filesystem-safe suffix."""
    raw = normalize_text(label or "")
    if not raw:
        return f"{fallback_index:03d}"
    raw = re.sub(r"^\s*table\s+", "", raw, flags=re.IGNORECASE)
    raw = raw.replace(".", "_")
    raw = re.sub(r"\s+", "", raw)
    raw = re.sub(r"[^A-Za-z0-9_-]+", "_", raw)
    raw = raw.strip("_")
    return raw or f"{fallback_index:03d}"


def normalize_text(s: str) -> str:
    """Normalize whitespace."""
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _tag_matches(tag, names):
    tag_name = str(getattr(tag, "name", "") or "")
    if not tag_name:
        return False
    local = tag_name.split(":")[-1]
    allowed = set()
    for name in names:
        text = str(name or "")
        if not text:
            continue
        allowed.add(text)
        allowed.add(text.split(":")[-1])
    return tag_name in allowed or local in allowed


def _nearest_ancestor(tag, names):
    parent = getattr(tag, "parent", None)
    while parent is not None:
        if _tag_matches(parent, names):
            return parent
        parent = getattr(parent, "parent", None)
    return None


def _nearest_section_title(tag):
    sec = _nearest_ancestor(tag, {"ce:section", "section", "sec"})
    if not sec:
        return None
    st = sec.find(["ce:section-title", "section-title", "title"], recursive=False) or sec.find(
        ["ce:section-title", "section-title", "title"]
    )
    if st and st.get_text(strip=True):
        return normalize_text(st.get_text(" ", strip=True))
    return None


def _paragraph_context(tag):
    para = _nearest_ancestor(tag, {"ce:para", "simple-para", "para", "p"})
    if not para:
        return None, None
    para_id = para.get("id")
    para_text = normalize_text(para.get_text(" ", strip=True))
    return para_id, para_text or None


def _equation_plain_text(tag):
    text = normalize_text(tag.get_text(" ", strip=True))
    text = text.replace("( ", "(").replace(" )", ")")
    return text


def _formula_marker_text(formula_tag) -> str:
    if formula_tag is None:
        return ""
    label_tag = formula_tag.find(["ce:label", "label"])
    label = normalize_text(label_tag.get_text(" ", strip=True)) if label_tag and label_tag.get_text(strip=True) else None
    math_tag = formula_tag.find(["mml:math", "math"])
    equation_text = _equation_plain_text(math_tag or formula_tag)
    parts = []
    if label:
        parts.append(f"Equation {label}:")
    elif equation_text:
        parts.append("Equation:")
    if equation_text:
        parts.append(equation_text)
    return " ".join(parts).strip()


def _inject_formula_markers(container):
    if container is None:
        return
    for formula_tag in list(container.find_all(["ce:formula", "formula", "disp-formula", "inline-formula"])):
        marker = _formula_marker_text(formula_tag)
        if not marker:
            continue
        formula_tag.replace_with(" " + marker + " ")


def _mathml_local_name(tag) -> str:
    return str(getattr(tag, "name", "") or "").split(":")[-1]


def _mathml_children(tag):
    return [c for c in getattr(tag, "children", []) if getattr(c, "name", None) or str(c).strip()]


def _mathml_join(parts):
    text = "".join(p for p in parts if p)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _mathml_operator(text: str) -> str:
    mapped = {
        "−": "-",
        "–": "-",
        "—": "-",
        "±": r"\pm",
        "∓": r"\mp",
        "×": r"\times",
        "·": r"\cdot",
        "⋅": r"\cdot",
        "∗": r"\ast",
        "∑": r"\sum",
        "∏": r"\prod",
        "∫": r"\int",
        "∂": r"\partial",
        "∞": r"\infty",
        "π": r"\pi",
        "∇": r"\nabla",
        "≠": r"\neq",
        "≤": r"\leq",
        "≥": r"\geq",
        "≈": r"\approx",
        "≃": r"\simeq",
        "∝": r"\propto",
        "→": r"\to",
        "⟶": r"\to",
        "↔": r"\leftrightarrow",
        "⇒": r"\Rightarrow",
        "⇔": r"\Leftrightarrow",
        "⊗": r"\otimes",
        "⨂": r"\otimes",
        "⊙": r"\odot",
        "°": r"^\circ",
    }
    text = normalize_text(text)
    return mapped.get(text, text)


def _mathml_identifier(text: str) -> str:
    text = normalize_text(text)
    greek = {
        "α": r"\alpha",
        "β": r"\beta",
        "γ": r"\gamma",
        "Γ": r"\Gamma",
        "δ": r"\delta",
        "Δ": r"\Delta",
        "ε": r"\varepsilon",
        "ϵ": r"\epsilon",
        "η": r"\eta",
        "θ": r"\theta",
        "Θ": r"\Theta",
        "κ": r"\kappa",
        "λ": r"\lambda",
        "Λ": r"\Lambda",
        "μ": r"\mu",
        "ν": r"\nu",
        "ξ": r"\xi",
        "Ξ": r"\Xi",
        "π": r"\pi",
        "ρ": r"\rho",
        "σ": r"\sigma",
        "Σ": r"\Sigma",
        "τ": r"\tau",
        "φ": r"\phi",
        "Φ": r"\Phi",
        "χ": r"\chi",
        "ψ": r"\psi",
        "Ψ": r"\Psi",
        "ω": r"\omega",
        "Ω": r"\Omega",
    }
    operators = {
        "tr": r"\operatorname{tr}",
        "sign": r"\operatorname{sign}",
        "sgn": r"\operatorname{sgn}",
        "sym": r"\operatorname{sym}",
        "skw": r"\operatorname{skw}",
        "exp": r"\exp",
        "ln": r"\ln",
        "sin": r"\sin",
        "cos": r"\cos",
        "tan": r"\tan",
        "max": r"\max",
        "min": r"\min",
        "det": r"\det",
    }
    if text in greek:
        return greek[text]
    low = text.lower()
    if low in operators:
        return operators[low]
    return text


def _cleanup_latex(text: str) -> str:
    text = text or ""
    text = text.replace(" otimes ", r" \otimes ")
    text = re.sub(r"\\overset\{˙\}\{([^}]*)\}", r"\\dot{\1}", text)
    text = re.sub(r"\\overset\{̇\}\{([^}]*)\}", r"\\dot{\1}", text)
    text = re.sub(r"\\underset\{([^}]*)\}\{\\sum\}", r"\\sum_{\1}", text)
    text = re.sub(r"(?<![A-Za-z])sgn(?=[A-Za-z\\(])", r"\\operatorname{sgn}", text)
    text = re.sub(r"(?<![A-Za-z])sign(?=[A-Za-z\\(])", r"\\operatorname{sign}", text)
    text = re.sub(r"(?<![A-Za-z\\])tr(?=[A-Za-z\\(])", r"\\operatorname{tr}", text)
    text = re.sub(r"(\\[A-Za-z]+)tr([A-Za-z\\])", r"\1 \\operatorname{tr} \2", text)
    text = re.sub(r"(\\[A-Za-z]+)(\\operatorname\{(?:sgn|sign)\})", r"\1 \2", text)
    text = re.sub(r"(\\operatorname\{(?:sgn|sign|tr)\})(\\[A-Za-z]+)", r"\1(\2)", text)
    text = re.sub(r"(\\operatorname\{(?:sgn|sign|tr)\})([A-Za-z])", r"\1(\2)", text)
    text = re.sub(r"\\otimes([A-Za-z\\])", r"\\otimes \1", text)
    text = re.sub(r"([A-Za-z\\])\\otimes", r"\1 \\otimes", text)
    text = re.sub(r"nosumon\s*([A-Za-z\\]+)", r"\\text{no sum on } \1", text)
    text = re.sub(r"([a-zA-Z0-9\}\]])\\operatorname", r"\1 \\operatorname", text)
    text = re.sub(r"(\\operatorname\{(?:sgn|sign|tr)\})([A-Za-z\\][^,\s)]*)", r"\1(\2)", text)
    text = re.sub(r"([A-Za-z])\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+\}", "}", text)
    return text


def mathml_to_latex(node) -> str:
    if node is None:
        return ""

    if getattr(node, "name", None) is None:
        return str(node).strip()

    name = _mathml_local_name(node)
    children = _mathml_children(node)

    if name in {"math", "mrow", "mstyle", "semantics"}:
        return _mathml_join([mathml_to_latex(child) for child in children])

    if name in {"mi", "mn"}:
        return _mathml_identifier(node.get_text(" ", strip=True))

    if name == "mtext":
        return _mathml_identifier(node.get_text(" ", strip=True))

    if name == "mo":
        return _mathml_operator(node.get_text(" ", strip=True))

    if name == "msub":
        base = mathml_to_latex(children[0]) if len(children) > 0 else ""
        sub = mathml_to_latex(children[1]) if len(children) > 1 else ""
        return f"{base}_{{{sub}}}"

    if name == "msup":
        base = mathml_to_latex(children[0]) if len(children) > 0 else ""
        sup = mathml_to_latex(children[1]) if len(children) > 1 else ""
        return f"{base}^{{{sup}}}"

    if name == "msubsup":
        base = mathml_to_latex(children[0]) if len(children) > 0 else ""
        sub = mathml_to_latex(children[1]) if len(children) > 1 else ""
        sup = mathml_to_latex(children[2]) if len(children) > 2 else ""
        return f"{base}_{{{sub}}}^{{{sup}}}"

    if name == "mfrac":
        num = mathml_to_latex(children[0]) if len(children) > 0 else ""
        den = mathml_to_latex(children[1]) if len(children) > 1 else ""
        return rf"\frac{{{num}}}{{{den}}}"

    if name == "msqrt":
        body = _mathml_join([mathml_to_latex(child) for child in children])
        return rf"\sqrt{{{body}}}"

    if name == "mroot":
        body = mathml_to_latex(children[0]) if len(children) > 0 else ""
        degree = mathml_to_latex(children[1]) if len(children) > 1 else ""
        return rf"\sqrt[{degree}]{{{body}}}"

    if name == "mover":
        base = mathml_to_latex(children[0]) if len(children) > 0 else ""
        over = mathml_to_latex(children[1]) if len(children) > 1 else ""
        accent_map = {
            "^": r"\hat",
            "~": r"\tilde",
            "¯": r"\bar",
            "→": r"\vec",
            "˙": r"\dot",
            "̇": r"\dot",
        }
        if over in accent_map:
            return rf"{accent_map[over]}{{{base}}}"
        return rf"\overset{{{over}}}{{{base}}}"

    if name == "munder":
        base = mathml_to_latex(children[0]) if len(children) > 0 else ""
        under = mathml_to_latex(children[1]) if len(children) > 1 else ""
        return rf"\underset{{{under}}}{{{base}}}"

    if name == "munderover":
        base = mathml_to_latex(children[0]) if len(children) > 0 else ""
        under = mathml_to_latex(children[1]) if len(children) > 1 else ""
        over = mathml_to_latex(children[2]) if len(children) > 2 else ""
        return f"{base}_{{{under}}}^{{{over}}}"

    if name == "mfenced":
        open_delim = node.get("open", "(")
        close_delim = node.get("close", ")")
        body = ", ".join(mathml_to_latex(child) for child in children)
        return f"{open_delim}{body}{close_delim}"

    if name == "mtable":
        rows = []
        for row in children:
            if _mathml_local_name(row) != "mtr":
                continue
            cols = [mathml_to_latex(cell) for cell in _mathml_children(row) if _mathml_local_name(cell) == "mtd"]
            rows.append(" & ".join(c for c in cols if c))
        if len(rows) == 1 and "&" not in rows[0]:
            return rows[0]
        return r"\begin{matrix} " + r" \\ ".join(rows) + r" \end{matrix}"

    if name in {"mtr", "mtd"}:
        return _mathml_join([mathml_to_latex(child) for child in children])

    if name == "annotation":
        return normalize_text(node.get_text(" ", strip=True))

    latex = _mathml_join([mathml_to_latex(child) for child in children]) or _mathml_identifier(node.get_text(" ", strip=True))
    return _cleanup_latex(latex)


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


def extract_table_label(table_tag):
    label = table_tag.find(["ce:label", "label"])
    if label:
        return normalize_text(label.get_text(" ", strip=True))
    return None


def build_object_ref_map(soup):
    """
    Build a map from object ref (e.g., fx1, gr1) to downloadable URLs declared
    in the <objects> block of Elsevier full-text XML.
    """
    ref_map = {}
    for obj in soup.find_all(["object", "xocs:object"]):
        ref = obj.get("ref")
        if not ref:
            continue
        url = normalize_text(obj.get_text(" ", strip=True))
        category = (obj.get("category") or "").strip().lower()
        mimetype = obj.get("mimetype")
        entry = ref_map.setdefault(ref, {"urls": {}, "mimetype": mimetype})
        if url:
            entry["urls"][category or "default"] = url
        if mimetype and not entry.get("mimetype"):
            entry["mimetype"] = mimetype
    return ref_map


def extract_table_image_info(table_tag, object_ref_map=None):
    """Return metadata for image-backed tables embedded as inline figures."""
    inline = table_tag.find(["ce:inline-figure", "inline-figure", "ce:graphic", "graphic"])
    if not inline:
        return None

    link = inline.find(["ce:link", "link"])
    graphic = inline.find(["ce:graphic", "graphic"])

    info = {
        "kind": "image_backed",
        "locator": None,
        "href": None,
        "local_path": None,
    }
    if link:
        info["locator"] = link.get("locator")
        info["href"] = link.get("xlink:href") or link.get("href")
    if graphic and not info["href"]:
        info["href"] = graphic.get("xlink:href") or graphic.get("href")
    ref = info.get("locator")
    if ref and object_ref_map and ref in object_ref_map:
        urls = object_ref_map[ref].get("urls") or {}
        info["download_url"] = (
            urls.get("high")
            or urls.get("standard")
            or urls.get("default")
            or urls.get("thumbnail")
        )
        info["download_urls"] = urls
        if object_ref_map[ref].get("mimetype"):
            info["mimetype"] = object_ref_map[ref]["mimetype"]

    if info["locator"] or info["href"]:
        return info
    return None


def extract_tables_from_xml(soup, refid_to_num=None):
    tables = soup.find_all(["ce:table", "table-wrap", "table"])
    extracted = []
    idx = 1
    object_ref_map = build_object_ref_map(soup)

    for t in tables:
        rows = extract_table_rows(t, refid_to_num=refid_to_num)
        image_info = extract_table_image_info(t, object_ref_map=object_ref_map)
        if not rows and not image_info:
            continue

        record = {
            "table_index": idx,
            "table_label": extract_table_label(t),
            "caption": extract_caption(t),
            "rows": rows,
        }
        if image_info and not rows:
            record["table_kind"] = "image_backed"
            record["image"] = image_info
        extracted.append(record)
        idx += 1

    return extracted


def download_table_image(image_info, outdir, table_base, api_key=None, inst_token=None, timeout=30):
    """
    Best-effort downloader for image-backed tables.
    This is intentionally conservative: if the asset URL pattern does not work,
    the caller still keeps a stable placeholder record.
    """
    if not image_info:
        return None

    locator = image_info.get("locator")
    href = image_info.get("href")
    candidates = []
    if image_info.get("download_url"):
        candidates.append(image_info["download_url"])
    if href and href.startswith("http"):
        candidates.append(href)
    if locator:
        candidates.append(f"https://api.elsevier.com/content/object/eid/{quote(locator, safe='')}")
    if href and not href.startswith("http"):
        candidates.append(f"https://api.elsevier.com/content/object/{href.lstrip('/')}")

    headers = {}
    if api_key:
        headers["X-ELS-APIKey"] = api_key
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token
    headers["Accept"] = "image/*"

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{table_base}.png")
    for url in candidates:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if not resp.ok or not resp.content:
                continue
            content_type = (resp.headers.get("content-type") or "").lower()
            ext = ".png"
            if "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            elif "gif" in content_type:
                ext = ".gif"
            out_path = os.path.join(outdir, f"{table_base}{ext}")
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return out_path
        except Exception:
            continue
    return None


def should_download_image_backed_table(table_record, keywords=None):
    keywords = [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]
    if not keywords:
        return False
    text = " ".join(
        [
            str(table_record.get("table_label") or ""),
            str(table_record.get("caption") or ""),
        ]
    ).lower()
    if not text:
        return False
    return any(k in text for k in keywords)


def run_rule_based_table_ocr(image_path, output_json_path):
    """
    Legacy placeholder kept for compatibility.
    Image-backed tables are no longer converted into structured table JSON here.
    The active path is to download the image and pass it directly to the
    extractor as vision input when selected.
    """
    result = {
        "engine": "legacy_placeholder",
        "status": "skipped",
        "reason": "Image-backed tables are no longer converted here; use direct image input in extractor.",
        "image_path": image_path,
        "rows": [],
        "cells": [],
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# --------------------------------------------------
# Equation extraction
# --------------------------------------------------

def extract_equations_from_xml(soup):
    """
    Extract standalone display formulas from Elsevier XML.
    """
    equations = []

    formula_tags = soup.find_all(["ce:formula", "formula", "disp-formula", "inline-formula"])
    for idx, formula_tag in enumerate(formula_tags, start=1):
        math_tag = formula_tag.find(["mml:math", "math"])
        label_tag = formula_tag.find(["ce:label", "label"])
        equations.append(
            {
                "equation_index": idx,
                "kind": "display_formula",
                "label": normalize_text(label_tag.get_text(" ", strip=True)) if label_tag and label_tag.get_text(strip=True) else None,
                "section_title": _nearest_section_title(formula_tag),
                "mathml": str(math_tag or formula_tag),
                "xml": str(formula_tag),
                "text": _equation_plain_text(math_tag or formula_tag),
                "latex": _cleanup_latex(mathml_to_latex(math_tag or formula_tag)),
            }
        )

    return equations


def save_equations(equations, outdir):
    os.makedirs(outdir, exist_ok=True)
    for stale in os.listdir(outdir):
        if re.match(r"^equation_\d+\.(xml|txt)$", stale):
            try:
                os.remove(os.path.join(outdir, stale))
            except OSError:
                pass

    index = []

    for eq in equations:
        base_name = f"equation_{int(eq['equation_index']):03d}"
        txt_path = os.path.join(outdir, f"{base_name}.txt")
        equation_id = f"eq_{int(eq['equation_index']):04d}"

        with open(txt_path, "w", encoding="utf-8") as f:
            meta = [
                f"Equation ID: {equation_id}",
            ]
            if eq.get("label"):
                meta.append(f"Label: {eq['label']}")
            if eq.get("section_title"):
                meta.append(f"Section: {eq['section_title']}")

            lines = meta + [""]
            if eq.get("latex"):
                lines.extend(
                    [
                        "LaTeX",
                        "-----",
                        eq["latex"],
                        "",
                    ]
                )
            if eq.get("text"):
                lines.extend(
                    [
                        "Plain Text",
                        "----------",
                        eq["text"],
                        "",
                    ]
                )
            f.write("\n".join(lines).rstrip() + "\n")

        record = {
            "equation_id": equation_id,
            "equation_index": eq.get("equation_index"),
            "section_title": eq.get("section_title"),
            "text": eq.get("text"),
            "latex": eq.get("latex"),
            "text_file": os.path.basename(txt_path),
        }
        index.append(record)

    with open(os.path.join(outdir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return index



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


def _iter_direct_child_sections(sec):
    for child in getattr(sec, "children", []):
        if getattr(child, "name", None) in ("ce:section", "sec"):
            yield child


def _section_title(sec):
    st = sec.find(["ce:section-title", "section-title", "title"], recursive=False) or sec.find(["ce:section-title", "section-title", "title"])
    if st and st.get_text(strip=True):
        return normalize_text(st.get_text(" ", strip=True))
    return None


def _first_direct_paragraph_preview(sec, refid_to_num, max_chars=220):
    for p in sec.find_all(["ce:para", "para", "p"], recursive=True):
        if _is_inside_nested_section(p, sec):
            continue
        p_copy = BeautifulSoup(str(p), "xml")
        p_tag = p_copy.find(["ce:para", "para", "p"]) or p_copy
        _inject_formula_markers(p_tag)
        replace_crossrefs_with_numbers(p_tag, refid_to_num)
        txt = normalize_text(p_tag.get_text(" ", strip=True))
        if txt:
            txt = compress_numeric_citation_groups(txt)
            return txt[:max_chars] + ("..." if len(txt) > max_chars else "")
    return None


def section_to_markdown(sec, tables_map, refid_to_num):
    md_lines = []

    # Section title
    st = sec.find(["ce:section-title", "section-title", "title"])
    if st and st.get_text(strip=True):
        md_lines.append("## " + normalize_text(st.get_text(" ", strip=True)))
        md_lines.append("")

    child_summaries = []
    for child_sec in _iter_direct_child_sections(sec):
        title = _section_title(child_sec)
        preview = _first_direct_paragraph_preview(child_sec, refid_to_num)
        if title:
            line = f"- {title}"
            if preview:
                line += f": {preview}"
            child_summaries.append(line)
    if child_summaries:
        md_lines.append("### Subsection overview")
        md_lines.extend(child_summaries)
        md_lines.append("")

    # Paragraphs belonging to THIS section only (exclude nested sections)
    for p in sec.find_all(["ce:para", "para", "p"], recursive=True):
        if _is_inside_nested_section(p, sec):
            continue

        # Work on a copy of the paragraph so we can replace tags safely
        p_copy = BeautifulSoup(str(p), "xml")
        p_tag = p_copy.find(["ce:para", "para", "p"]) or p_copy

        _inject_formula_markers(p_tag)
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
        image_info = extract_table_image_info(table)

        if not rows and not image_info:
            continue

        cap = extract_caption(table) or ""
        key = cap + "|" + ("|".join(rows[0]) if rows else "")

        tinfo = tables_map.get(key)
        if not tinfo:
            # Fallback: write inline even if not matched to global table list
            md_lines.append(f"### {extract_table_label(table) or 'Table'}")
            if cap:
                md_lines.append(f"**Caption:** {cap}")
            md_lines.append("")
            if rows:
                md_lines.append(rows_to_markdown(rows))
            else:
                md_lines.append("_Image-backed table detected._")
            md_lines.append("")
            continue

        table_heading = tinfo.get("table_label") or f"Table {tinfo['table_index']}"
        idx = tinfo["table_index"]
        if idx in seen_table_idxs:
            continue
        seen_table_idxs.add(idx)

        md_lines.append(f"### {table_heading}")
        if tinfo.get("caption"):
            md_lines.append(f"**Caption:** {tinfo['caption']}")
        md_lines.append("")
        if tinfo.get("rows"):
            md_lines.append(rows_to_markdown(tinfo["rows"]))
        elif tinfo.get("table_kind") == "image_backed":
            md_lines.append("_Image-backed table detected._")
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
    image_backed_table_keywords=None,
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
    equations_dir = os.path.join(base_dir, "equations")

    os.makedirs(sections_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(equations_dir, exist_ok=True)

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
    equations = extract_equations_from_xml(soup)

    tables_map = {}
    for t in tables:
        cap = t.get("caption") or ""
        first_row = "|".join(t["rows"][0]) if t.get("rows") else ""
        key = cap + "|" + first_row
        tables_map[key] = t

    # Save tables
    for t in tables:
        idx = t["table_index"]
        table_suffix = safe_table_label(t.get("table_label") or "", idx)
        table_base = f"table_{table_suffix}"
        if (
            t.get("table_kind") == "image_backed"
            and should_download_image_backed_table(t, image_backed_table_keywords)
        ):
            image_dir = os.path.join(tables_dir, "images")
            downloaded = download_table_image(
                t.get("image") or {},
                image_dir,
                table_base,
                api_key=api_key,
                inst_token=inst_token,
            )
            if downloaded:
                rel = os.path.relpath(downloaded, base_dir)
                t.setdefault("image", {})["local_path"] = rel

        with open(os.path.join(tables_dir, f"{table_base}.json"), "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False, indent=2)

    # Save equations
    save_equations(equations, equations_dir)

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
