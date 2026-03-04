import csv
import json
import re
import time
from typing import Any, Dict, List

import requests
from pathlib import Path

SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"


def _request_with_retry(url: str, headers: Dict[str, str], params: Dict[str, Any], timeout: int, max_retries: int) -> Dict[str, Any]:
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Scopus request failed after retries: {last_error}")


def _year_from_date(date_str: str) -> int | None:
    if not date_str:
        return None
    m = re.match(r"^(\d{4})", str(date_str))
    if not m:
        return None
    return int(m.group(1))


def _score_entry(entry: Dict[str, Any], keywords: List[str]) -> int:
    text = " ".join([
        str(entry.get("dc:title", "")),
        str(entry.get("dc:description", "")),
    ]).lower()
    return sum(1 for k in keywords if k and k.lower() in text)


def scopus_search(
    api_key,
    query,
    count,
    max_pages,
    outdir,
    year_from=None,
    year_to=None,
    require_doi=True,
    allowed_doctypes=None,
    rank_keywords=None,
    max_retries=3,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }

    start = 0
    all_entries = []
    raw_pages = []
    page_count = 0
    rank_keywords = rank_keywords or []
    allowed_doctypes = {d.lower() for d in (allowed_doctypes or [])}

    for _ in range(max_pages):
        params = {
            "query": query,
            "count": count,
            "start": start,
            "view": "STANDARD",
        }

        data = _request_with_retry(
            SCOPUS_URL,
            headers=headers,
            params=params,
            timeout=60,
            max_retries=max_retries,
        )
        page_count += 1

        raw_pages.append(data)
        entries = data["search-results"].get("entry", [])
        if not entries:
            break

        all_entries.extend(entries)
        start += count
        time.sleep(0.2)

    # Save raw
    (outdir / "scopus_raw_pages.json").write_text(
        json.dumps(raw_pages, indent=2), encoding="utf-8"
    )

    # Deduplicate + filter + score
    dedup: Dict[str, Dict[str, Any]] = {}
    for it in all_entries:
        doi = (it.get("prism:doi") or "").strip().lower()
        eid = (it.get("eid") or "").strip().lower()
        title = (it.get("dc:title") or "").strip().lower()
        key = doi or eid or title
        if not key:
            continue

        cur_score = _score_entry(it, rank_keywords)
        prev = dedup.get(key)
        if not prev:
            dedup[key] = {**it, "__score": cur_score}
            continue
        if cur_score > prev.get("__score", 0):
            dedup[key] = {**it, "__score": cur_score}

    filtered = []
    dropped_no_doi = dropped_doctype = dropped_year = 0
    for it in dedup.values():
        doi = (it.get("prism:doi") or "").strip()
        year = _year_from_date(it.get("prism:coverDate", ""))
        doctype = (it.get("subtype") or "").strip().lower()

        if require_doi and not doi:
            dropped_no_doi += 1
            continue

        if allowed_doctypes and doctype and doctype not in allowed_doctypes:
            dropped_doctype += 1
            continue

        if year is not None and year_from is not None and year < int(year_from):
            dropped_year += 1
            continue
        if year is not None and year_to is not None and year > int(year_to):
            dropped_year += 1
            continue

        filtered.append(it)

    filtered.sort(key=lambda x: x.get("__score", 0), reverse=True)

    fieldnames = ["title", "doi", "eid", "source", "date", "doctype", "score"]
    raw_csv_path = outdir / "scopus_results_raw.csv"
    filtered_csv_path = outdir / "scopus_results.csv"
    filtered_backup_csv_path = outdir / "scopus_results_filtered.csv"

    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in dedup.values():
            w.writerow({
                "title": it.get("dc:title", ""),
                "doi": it.get("prism:doi", ""),
                "eid": it.get("eid", ""),
                "source": it.get("prism:publicationName", ""),
                "date": it.get("prism:coverDate", ""),
                "doctype": it.get("subtype", ""),
                "score": it.get("__score", 0),
            })

    for csv_path in (filtered_csv_path, filtered_backup_csv_path):
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for it in filtered:
                w.writerow({
                    "title": it.get("dc:title", ""),
                    "doi": it.get("prism:doi", ""),
                    "eid": it.get("eid", ""),
                    "source": it.get("prism:publicationName", ""),
                    "date": it.get("prism:coverDate", ""),
                    "doctype": it.get("subtype", ""),
                    "score": it.get("__score", 0),
                })

    report = {
        "query": query,
        "pages_fetched": page_count,
        "rows_raw": len(all_entries),
        "rows_dedup": len(dedup),
        "rows_filtered": len(filtered),
        "dropped_no_doi": dropped_no_doi,
        "dropped_doctype": dropped_doctype,
        "dropped_year": dropped_year,
        "year_from": year_from,
        "year_to": year_to,
        "require_doi": require_doi,
        "allowed_doctypes": sorted(allowed_doctypes),
        "rank_keywords": rank_keywords,
    }
    (outdir / "scopus_search_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dois = []
    seen = set()
    for it in filtered:
        doi = (it.get("prism:doi") or "").strip()
        if not doi:
            continue
        low = doi.lower()
        if low in seen:
            continue
        seen.add(low)
        dois.append(doi)

    return dois
