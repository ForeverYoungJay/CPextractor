import argparse
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable


PALETTE = {
    "ink": "#102033",
    "muted": "#5a6b7b",
    "panel": "#f7fafc",
    "panel_alt": "#eef4f8",
    "stroke": "#d9e3ec",
    "accent": "#123B63",
    "accent_2": "#2A6F97",
    "accent_3": "#5D8DA8",
    "accent_4": "#9EC1D4",
    "highlight": "#E07A5F",
    "white": "#ffffff",
}


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _norm_verdict(value: Any) -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "accepted": "accepted",
        "pass": "accepted",
        "passed": "accepted",
        "flagged": "flagged",
        "warning": "flagged",
        "warn": "flagged",
        "rejected": "rejected",
        "fail": "rejected",
        "failed": "rejected",
    }
    return mapping.get(raw, raw)


def _iter_paper_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def build_summary(root: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "papers_processed": 0,
        "papers_with_raw": 0,
        "papers_with_evaluation": 0,
        "raw_parameter_records": 0,
        "normalized_claims": 0,
        "evidence_linked_claims": 0,
        "committee_accepted_claims": 0,
        "high_confidence_accepted_claims": 0,
        "database_ready_entries": 0,
        "database_ready_papers": 0,
        "document_verdicts": {"accepted": 0, "flagged": 0, "rejected": 0, "other": 0},
        "notes": [
            "raw_parameter_records counts parameters.registry entries from materials_extracted.extractor_raw.json",
            "normalized_claims counts parameter_claims entries from materials_extracted.json",
            "database_ready_entries counts claims from papers with ingest_gate.blocked == false",
        ],
    }

    for paper_dir in _iter_paper_dirs(root):
        raw = _load_json(paper_dir / "materials_extracted.extractor_raw.json")
        final = _load_json(paper_dir / "materials_extracted.json")
        evaluation = _load_json(paper_dir / "llm_evaluation.json")
        report = _load_json(paper_dir / "postprocess_report.json")

        if raw:
            summary["papers_with_raw"] += 1
            summary["raw_parameter_records"] += len(((raw.get("parameters") or {}).get("registry") or []))

        if final:
            summary["papers_processed"] += 1
            claims = final.get("parameter_claims") or []
            summary["normalized_claims"] += len(claims)
            for claim in claims:
                source = claim.get("source") if isinstance(claim.get("source"), dict) else {}
                evidence = claim.get("evidence") if isinstance(claim.get("evidence"), dict) else {}
                has_link = bool(
                    source.get("evidence_text")
                    or source.get("table_evidence")
                    or source.get("evidence_location")
                    or evidence.get("evidence_text")
                    or evidence.get("table_evidence")
                    or (isinstance(source.get("evidence_ids"), list) and source.get("evidence_ids"))
                    or (isinstance(claim.get("evidence_ids"), list) and claim.get("evidence_ids"))
                )
                if has_link:
                    summary["evidence_linked_claims"] += 1

        if evaluation:
            summary["papers_with_evaluation"] += 1
            verdict = _norm_verdict(evaluation.get("verdict"))
            if verdict in summary["document_verdicts"]:
                summary["document_verdicts"][verdict] += 1
            else:
                summary["document_verdicts"]["other"] += 1

            for audit in evaluation.get("parameter_audits") or []:
                audit_verdict = _norm_verdict(audit.get("verdict"))
                if audit_verdict != "accepted":
                    continue
                summary["committee_accepted_claims"] += 1
                if str(audit.get("confidence") or "").strip().lower() == "high":
                    summary["high_confidence_accepted_claims"] += 1

        if final and report:
            gate = report.get("ingest_gate") or {}
            if not gate.get("blocked", True):
                summary["database_ready_papers"] += 1
                summary["database_ready_entries"] += len(final.get("parameter_claims") or [])

    normalized = max(1, summary["normalized_claims"])
    high_acc = max(1, summary["high_confidence_accepted_claims"])
    papers = max(1, summary["papers_processed"])
    summary["retention"] = {
        "accepted_from_normalized_pct": round(100.0 * summary["high_confidence_accepted_claims"] / normalized, 1),
        "db_ready_from_normalized_pct": round(100.0 * summary["database_ready_entries"] / normalized, 1),
        "db_ready_papers_pct": round(100.0 * summary["database_ready_papers"] / papers, 1),
        "db_ready_from_accepted_pct": round(100.0 * summary["database_ready_entries"] / high_acc, 1),
    }
    return summary


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        "  <defs>\n"
        "    <linearGradient id=\"accent_gradient\" x1=\"0\" x2=\"1\" y1=\"0\" y2=\"0\">\n"
        f'      <stop offset="0%" stop-color="{PALETTE["accent"]}"/>\n'
        f'      <stop offset="100%" stop-color="{PALETTE["accent_2"]}"/>\n'
        "    </linearGradient>\n"
        "  </defs>\n"
    )


def _panel(x: int, y: int, w: int, h: int, fill: str | None = None) -> str:
    return (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" ry="22" '
        f'fill="{fill or PALETTE["panel"]}" stroke="{PALETTE["stroke"]}" stroke-width="2"/>\n'
    )


def _text(x: float, y: float, value: str, *, size: int, weight: int = 500, fill: str | None = None, anchor: str = "start") -> str:
    return (
        f'  <text x="{x}" y="{y}" font-family="Aptos, Helvetica Neue, Segoe UI, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill or PALETTE["ink"]}" text-anchor="{anchor}">{escape(value)}</text>\n'
    )


def render_cards_svg(summary: Dict[str, Any]) -> str:
    width, height = 1600, 320
    cards = [
        ("Papers processed", summary["papers_processed"], "Full-text papers completed"),
        ("Extracted parameter claims", summary["normalized_claims"], "Claim-level structured outputs"),
        ("Evidence-linked claims", summary["evidence_linked_claims"], "Claims tied to text or table evidence"),
        ("High-confidence accepted claims", summary["high_confidence_accepted_claims"], "Committee accepted, confidence = high"),
        ("Final database-ready entries", summary["database_ready_entries"], "Claims from ingest-ready papers"),
    ]

    pieces = [_svg_header(width, height), f'  <rect width="{width}" height="{height}" fill="{PALETTE["white"]}"/>\n']
    pieces.append(_text(60, 48, "CPextractor Corpus Results", size=30, weight=700))
    pieces.append(_text(60, 78, "Numbers below are computed directly from data/fulltext.", size=16, fill=PALETTE["muted"]))

    card_w, gap = 284, 20
    for idx, (label, value, note) in enumerate(cards):
        x = 60 + idx * (card_w + gap)
        y = 104
        pieces.append(_panel(x, y, card_w, 170))
        pieces.append(f'  <rect x="{x}" y="{y}" width="{card_w}" height="16" rx="22" ry="22" fill="url(#accent_gradient)"/>\n')
        pieces.append(_text(x + 22, y + 48, _fmt_int(value), size=40, weight=700, fill=PALETTE["accent"]))
        pieces.append(_text(x + 22, y + 84, label, size=16, weight=700))
        pieces.append(_text(x + 22, y + 118, note, size=14, fill=PALETTE["muted"]))
    pieces.append("</svg>\n")
    return "".join(pieces)


def render_funnel_svg(summary: Dict[str, Any]) -> str:
    width, height = 1600, 920
    stages = [
        ("Papers processed", summary["papers_processed"], PALETTE["accent_4"]),
        ("Raw parameter records", summary["raw_parameter_records"], "#B7D1E0"),
        ("Normalized claim records", summary["normalized_claims"], "#8FB6CA"),
        ("Evidence-linked claims", summary["evidence_linked_claims"], PALETTE["accent_3"]),
        ("Committee-accepted claims", summary["high_confidence_accepted_claims"], PALETTE["accent_2"]),
        ("Final database-ready entries", summary["database_ready_entries"], PALETTE["highlight"]),
    ]
    max_value = max(value for _, value, _ in stages)
    center_x = 520
    top_y = 150
    bar_h = 84
    gap = 18
    max_w = 700
    min_w = 180

    pieces = [_svg_header(width, height), f'  <rect width="{width}" height="{height}" fill="{PALETTE["white"]}"/>\n']
    pieces.append(_text(70, 64, "From recall-first extraction to trusted database records", size=34, weight=700))
    pieces.append(_text(70, 96, "A funnel communicates that CPextractor does not ingest everything it extracts.", size=18, fill=PALETTE["muted"]))
    pieces.append(_panel(60, 120, 980, 730, PALETTE["panel_alt"]))

    for idx, (label, value, color) in enumerate(stages):
        y = top_y + idx * (bar_h + gap)
        width_i = min_w + (max_w - min_w) * (value / max_value)
        x = center_x - width_i / 2
        pieces.append(
            f'  <rect x="{x:.1f}" y="{y}" width="{width_i:.1f}" height="{bar_h}" rx="18" ry="18" '
            f'fill="{color}" stroke="{PALETTE["white"]}" stroke-width="2"/>\n'
        )
        pieces.append(_text(center_x, y + 34, label, size=20, weight=700, fill=PALETTE["ink"], anchor="middle"))
        pieces.append(_text(center_x, y + 60, _fmt_int(value), size=28, weight=700, fill=PALETTE["accent"], anchor="middle"))

        if idx < len(stages) - 1:
            next_y = y + bar_h
            pieces.append(
                f'  <path d="M {center_x} {next_y + 4} L {center_x} {next_y + gap - 6}" '
                f'stroke="{PALETTE["accent_2"]}" stroke-width="5" stroke-linecap="round"/>\n'
            )

    side_x = 1080
    pieces.append(_panel(side_x, 120, 460, 280))
    pieces.append(_text(side_x + 24, 162, "Why this chart works for a poster", size=22, weight=700))
    pieces.append(_text(side_x + 24, 200, "Raw extraction maximizes recall.", size=18, weight=700, fill=PALETTE["accent"]))
    pieces.append(_text(side_x + 24, 228, "Evidence grounding attaches claim-level support from text or tables.", size=16, fill=PALETTE["muted"]))
    pieces.append(_text(side_x + 24, 268, "Committee review removes low-trust claims before they reach the database.", size=16, fill=PALETTE["muted"]))
    pieces.append(_text(side_x + 24, 308, "Only accepted claims from ingest-ready papers become final records.", size=16, fill=PALETTE["muted"]))

    pieces.append(_panel(side_x, 420, 460, 220))
    pieces.append(_text(side_x + 24, 462, "Retention highlights", size=22, weight=700))
    pieces.append(_text(side_x + 24, 504, f'{summary["retention"]["accepted_from_normalized_pct"]}% of normalized claims remain high-confidence after committee review.', size=17, fill=PALETTE["muted"]))
    pieces.append(_text(side_x + 24, 544, f'{summary["retention"]["db_ready_from_normalized_pct"]}% of normalized claims become database-ready entries.', size=17, fill=PALETTE["muted"]))
    pieces.append(_text(side_x + 24, 584, f'{summary["retention"]["db_ready_papers_pct"]}% of processed papers produce ingest-ready outputs.', size=17, fill=PALETTE["muted"]))

    verdicts = summary["document_verdicts"]
    pieces.append(_panel(side_x, 660, 460, 190))
    pieces.append(_text(side_x + 24, 702, "Document-level verdicts", size=22, weight=700))
    pieces.append(_text(side_x + 24, 742, f'Accepted: {_fmt_int(verdicts["accepted"])} papers', size=18, fill=PALETTE["accent"]))
    pieces.append(_text(side_x + 24, 772, f'Flagged: {_fmt_int(verdicts["flagged"])} papers', size=18, fill=PALETTE["accent_2"]))
    pieces.append(_text(side_x + 24, 802, f'Rejected: {_fmt_int(verdicts["rejected"])} papers', size=18, fill=PALETTE["highlight"]))

    pieces.append("</svg>\n")
    return "".join(pieces)


def render_dashboard_svg(summary: Dict[str, Any]) -> str:
    width, height = 1600, 1240
    pieces = [_svg_header(width, height), f'  <rect width="{width}" height="{height}" fill="{PALETTE["white"]}"/>\n']

    def _body_lines(svg_text: str) -> list[str]:
        lines = svg_text.splitlines()
        out: list[str] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if idx == 0 or stripped == "</svg>":
                continue
            out.append(line)
        return out

    cards_svg = _body_lines(render_cards_svg(summary))
    funnel_svg = _body_lines(render_funnel_svg(summary))

    pieces.append(f'  <g transform="translate(0,0)">\n')
    for line in cards_svg:
        pieces.append(f"{line}\n")
    pieces.append("  </g>\n")

    pieces.append(f'  <g transform="translate(0,320)">\n')
    for line in funnel_svg:
        pieces.append(f"{line}\n")
    pieces.append("  </g>\n")
    pieces.append("</svg>\n")
    return "".join(pieces)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate poster/slide result figures from data/fulltext")
    ap.add_argument("--root", default="data/fulltext")
    ap.add_argument("--outdir", default="output/presentation/results")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(root)
    (outdir / "results_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "results_cards.svg").write_text(render_cards_svg(summary), encoding="utf-8")
    (outdir / "results_funnel.svg").write_text(render_funnel_svg(summary), encoding="utf-8")
    (outdir / "results_dashboard.svg").write_text(render_dashboard_svg(summary), encoding="utf-8")

    print(f"Saved summary -> {outdir / 'results_summary.json'}")
    print(f"Saved cards   -> {outdir / 'results_cards.svg'}")
    print(f"Saved funnel  -> {outdir / 'results_funnel.svg'}")
    print(f"Saved board   -> {outdir / 'results_dashboard.svg'}")


if __name__ == "__main__":
    main()
