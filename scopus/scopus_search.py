import time, json, csv, requests
from pathlib import Path

SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"

def scopus_search(api_key, query, count, max_pages, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }

    start = 0
    all_entries = []
    raw_pages = []

    for _ in range(max_pages):
        params = {
            "query": query,
            "count": count,
            "start": start,
            "view": "STANDARD",
        }

        r = requests.get(SCOPUS_URL, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

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

    # Save CSV
    csv_path = outdir / "scopus_results.csv"
    fieldnames = ["title", "doi", "eid", "source", "date"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        for it in all_entries:
            w.writerow({
                "title": it.get("dc:title", ""),
                "doi": it.get("prism:doi", ""),
                "eid": it.get("eid", ""),
                "source": it.get("prism:publicationName", ""),
                "date": it.get("prism:coverDate", "")
            })

    dois = [it.get("prism:doi") for it in all_entries if it.get("prism:doi")]
    return dois
