"""Versioned, dependency-free claim evaluation contract.

IDs are audit handles, not cross-run identities. Values and units never select a
match. Ambiguous assignments remain visible instead of being resolved by order.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

PROTOCOL_VERSION = "claim-benchmark-1.0"
POSITIVE = {
    "correct", "accepted", "wrong_value", "wrong_unit", "wrong_mapping",
    "wrong_binding", "wrong_provenance", "insufficient_evidence",
    "missing_from_prediction", "wrong_material_object", "wrong_cp_model",
    "wrong_parameter_body", "wrong_scope", "wrong_evidence",
    "insufficient_context", "missing_record", "incorrect",
}
NEGATIVE = {"spurious_claim", "spurious_record", "duplicate_record"}
PENDING = {"", "pending", "todo", "unreviewed", "needs_adjudication"}


def text(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def symbol_text(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def number(value):
    """Parse explicit numbers only, never infer exponents from collapsed layout."""
    if isinstance(value, bool) or value is None:
        return None
    raw = unicodedata.normalize("NFKC", str(value)).replace("−", "-").strip()
    try:
        parsed = float(raw)
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        pass
    fraction = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)\s*", raw)
    if fraction and float(fraction[2]) != 0:
        return float(fraction[1]) / float(fraction[2])
    scientific = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*[×x·]\s*10\s*\^?\s*([+-]?\d+)", raw)
    power = re.fullmatch(r"10\s*\^\s*([+-]?\d+)", raw)
    try:
        if scientific:
            parsed = float(scientific[1]) * 10.0**int(scientific[2])
            return parsed if math.isfinite(parsed) else None
        if power:
            parsed = 10.0**int(power[1])
            return parsed if math.isfinite(parsed) else None
    except OverflowError:
        return None
    return None


def doi(value):
    value = unquote(text(value))
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)
    return value.strip()


def valid_doi(value):
    return bool(re.fullmatch(r"10\.\d{4,9}/\S+", doi(value)))


def normalize_row(row):
    """Accept flat claim exports, usable-parameter rows, and v6 claims."""
    out = copy.deepcopy(row)
    body = row.get("parameter_body") or row.get("parameter") or {}
    assertion = row.get("assertion") or {}
    for key, fallback in {
        "canonical_name": body.get("canonical_name"),
        "symbol": body.get("symbol", body.get("symbol_reported")),
        "value": body.get("value", assertion.get("reported_value")),
        "unit": body.get("unit", assertion.get("reported_unit")),
    }.items():
        if key not in out:
            out[key] = fallback
    out["doi"] = doi(row.get("doi") or (row.get("document") or {}).get("doi") or row.get("record_id"))
    material = row.get("material_object") or {}
    model = row.get("cp_model") or {}
    scope = row.get("parameter_scope") or row.get("scope") or row.get("applies_to") or {}
    out["context"] = {
        "material": row.get("material_name") or material.get("material_name"),
        "phase": material.get("constituent_name") or scope.get("phase_name"),
        "process_state": material.get("process_state_label"),
        "condition": scope.get("condition_label"),
        "temperature": scope.get("temperature_text"),
        "strain_rate": scope.get("strain_rate_text"),
        "model": model.get("model_label") or model.get("model_name"),
        "mechanism": scope.get("mechanism"),
        "family": scope.get("family_name"),
        "systems": scope.get("system_names"),
        **(row.get("context") or {}),
    }
    ev = row.get("evidence") or {}
    table = ev.get("table_evidence") or {}
    out["evidence"] = {
        **ev,
        "kind": ev.get("kind") or ev.get("source_type") or ev.get("evidence_type"),
        "file": ev.get("file") or ev.get("source_file"),
        "row_name": ev.get("row_name") or table.get("row_name"),
        "column_name": ev.get("column_name") or table.get("column_name"),
        "snippet": ev.get("snippet") or ev.get("text") or ev.get("evidence_text"),
    }
    out["annotation"] = {**(row.get("annotation") or {})}
    out["annotation"]["status"] = text(out["annotation"].get("status"))
    return out


def row_key(row):
    row = normalize_row(row)
    return (row["doi"], text(row.get("canonical_name")), text(row.get("symbol")),
            *(text(row["context"].get(k)) for k in sorted(row["context"])))


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported or missing manifest protocol_version")
    papers = manifest.get("papers")
    if not isinstance(papers, list) or not papers:
        raise ValueError("Manifest must contain papers")
    seen = set()
    for paper in papers:
        paper["doi"] = doi(paper.get("doi"))
        if not valid_doi(paper["doi"]) or paper["doi"] in seen:
            raise ValueError(f"Invalid or duplicate manifest DOI: {paper['doi']}")
        seen.add(paper["doi"])
        if paper.get("split") not in {"development", "calibration", "test"}:
            raise ValueError(f"Missing/invalid split: {paper['doi']}")
    return manifest


def prepare_universe(gold, pred, manifest=None, split=None, strict=False):
    gold = [normalize_row(r) for r in gold]
    pred = [normalize_row(r) for r in pred]
    invalid = [i for i, r in enumerate(gold) if not valid_doi(r["doi"])]
    if invalid:
        raise ValueError(f"Gold rows missing valid DOI: {invalid[:10]}; do not infer paper identity from claim IDs")
    invalid_pred = [i for i, r in enumerate(pred) if not valid_doi(r["doi"])]
    if invalid_pred:
        raise ValueError(f"Prediction rows missing valid DOI: {invalid_pred[:10]}")
    if strict and manifest is None:
        raise ValueError("Publication evaluation requires --manifest")
    if split and manifest is None:
        raise ValueError("--split requires --manifest")
    papers = [p for p in (manifest or {}).get("papers", []) if not split or p["split"] == split]
    universe = {p["doi"] for p in papers} if manifest else {r["doi"] for r in gold}
    if not universe:
        raise ValueError("Empty evaluation universe")
    scoped_gold = [r for r in gold if r["doi"] in universe]
    pending = [r for r in scoped_gold if r["annotation"]["status"] in PENDING]
    unknown = sorted({r["annotation"]["status"] for r in scoped_gold} - POSITIVE - NEGATIVE - PENDING)
    if unknown:
        raise ValueError(f"Unknown annotation statuses: {unknown}")
    if strict:
        for p in papers:
            if not (p.get("annotation_status") == "adjudicated" and p.get("exhaustive") is True
                    and p.get("reviewer_type") == "human_expert" and p.get("reviewer")
                    and p.get("adjudicator")):
                raise ValueError(f"Paper is not exhaustive expert-adjudicated gold: {p['doi']}")
            if not any(r["doi"] == p["doi"] for r in scoped_gold) and p.get("gold_claim_count") != 0:
                raise ValueError(f"No gold rows or explicit zero-claim declaration: {p['doi']}")
        if pending:
            raise ValueError(f"Gold contains {len(pending)} unresolved rows")
        if any((r["annotation"].get("reviewer_type") != "human_expert"
                or not r["annotation"].get("reviewer")) for r in scoped_gold):
            raise ValueError("Every gold row needs an explicit human_expert reviewer")
    eligible = [r for r in scoped_gold if r["annotation"]["status"] in POSITIVE | NEGATIVE]
    # Diagnostic runs never treat an entirely unreviewed paper as negative gold.
    active = universe if strict else {r["doi"] for r in eligible}
    selected_pred = [r for r in pred if r["doi"] in active]
    excluded_pending_predictions = 0
    if pending and selected_pred:
        combined = match_rows(eligible + pending, selected_pred)
        pending_pred_indices = {r["pred_index"] for r in combined["match_log"] if r["gold_index"] >= len(eligible)}
        selected_pred = [r for i, r in enumerate(selected_pred) if i not in pending_pred_indices]
        excluded_pending_predictions = len(pending_pred_indices)
    report = {
        "excluded_pending_predictions": excluded_pending_predictions,
        "protocol_version": PROTOCOL_VERSION,
        "publication_ready": strict,
        "mode": "expert_benchmark" if strict else "diagnostic_only",
        "split": split, "paper_count": len(universe), "active_paper_count": len(active),
        "dois": sorted(universe), "active_dois": sorted(active),
        "excluded_gold_rows": len(gold) - len(scoped_gold),
        "excluded_prediction_rows": len(pred) - len(selected_pred),
        "pending_gold_rows": len(pending),
        "missing_prediction_papers": sorted(active - {r["doi"] for r in pred}),
        "warnings": [] if strict else ["Partial/AI annotations are diagnostic; precision and recall are not release claims."],
    }
    return eligible, selected_pred, report


def _anchor(row):
    ev = row["evidence"]
    # Cell coordinates are identity evidence; the reported cell VALUE is excluded.
    file = text(ev.get("file")).replace("\\", "/").split("/")[-1]
    return (file, text(ev.get("row_name")), text(ev.get("column_name")))


def candidate_score(gold, pred):
    if gold["doi"] != pred["doi"]:
        return None
    ga, pa = _anchor(gold), _anchor(pred)
    anchor = all(ga) and ga == pa
    name = bool(text(gold.get("canonical_name"))) and text(gold.get("canonical_name")) == text(pred.get("canonical_name"))
    symbol = bool(symbol_text(gold.get("symbol"))) and symbol_text(gold.get("symbol")) == symbol_text(pred.get("symbol"))
    row_anchor = bool(ga[0] and ga[1]) and ga[:2] == pa[:2] and (name or symbol)
    if not (name or symbol or anchor):
        return None
    # Generated local IDs deliberately do not participate.
    score = 20 * int(name) + 10 * int(symbol) + 100 * int(anchor) + 50 * int(row_anchor)
    for key in gold["context"]:
        g, p = text(gold["context"].get(key)), text(pred["context"].get(key))
        if g and p:
            if g != p and not (anchor or row_anchor):
                return None
            score += 3 if g == p else -3
    return score


def match_rows(gold, pred):
    gold = [normalize_row(r) for r in gold]
    pred = [normalize_row(r) for r in pred]
    scores = {(i, j): score for i, g in enumerate(gold) for j, p in enumerate(pred)
              if (score := candidate_score(g, p)) is not None}
    left, right = set(range(len(gold))), set(range(len(pred)))
    matches = []
    while True:
        best_g, best_p = {}, {}
        for (i, j), score in scores.items():
            if i not in left or j not in right:
                continue
            for mapping, key, partner in ((best_g, i, j), (best_p, j, i)):
                current = mapping.get(key)
                if current is None or score > current[0]:
                    mapping[key] = (score, [partner])
                elif score == current[0]:
                    current[1].append(partner)
        pairs = [(i, js[0]) for i, (_, js) in best_g.items()
                 if len(js) == 1 and best_p[js[0]][1] == [i]]
        if not pairs:
            break
        for i, j in sorted(pairs):
            matches.append((i, j, scores[i, j]))
            left.remove(i)
            right.remove(j)
    positive, spurious, logs = [], [], []
    for i, j, score in matches:
        target = positive if gold[i]["annotation"].get("status") in POSITIVE else spurious
        target.append((gold[i], pred[j]))
        logs.append({"gold_index": i, "pred_index": j, "doi": gold[i]["doi"],
                     "gold_claim_id": gold[i].get("claim_id"), "pred_claim_id": pred[j].get("claim_id"),
                     "reason": "evidence_anchor" if all(_anchor(gold[i])) and _anchor(gold[i]) == _anchor(pred[j]) else "semantic_context",
                     "score": score})
    ambiguous = [{"gold_index": i, "candidate_pred_indices": sorted(j for j in right if (i, j) in scores)}
                 for i in sorted(left) if any((i, j) in scores for j in right)]
    missing = [gold[i] for i in sorted(left) if gold[i]["annotation"].get("status") in POSITIVE]
    unmatched = [pred[j] for j in sorted(right)]
    tp, fp, fn = len(positive), len(spurious) + len(unmatched), len(missing)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "prf": {"precision": precision, "recall": recall, "f1": f1},
            "matched_positive": positive, "matched_spurious": spurious,
            "missing_gold_positive": missing, "unmatched_pred": unmatched,
            "match_log": logs, "ambiguities": ambiguous,
            "duplicate_gold_identity_groups": sum(v > 1 for v in Counter(row_key(r) for r in gold).values()),
            "duplicate_pred_identity_groups": sum(v > 1 for v in Counter(row_key(r) for r in pred).values())}


def value_equal(a, b, rtol=1e-4, atol=1e-9):
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, list) or isinstance(b, list):
        return (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
                and all(value_equal(x, y, rtol, atol) for x, y in zip(a, b)))
    for value in (a, b):
        if isinstance(value, float) and not math.isfinite(value):
            return False
    af, bf = number(a), number(b)
    if af is not None and bf is not None:
        return math.isclose(af, bf, rel_tol=rtol, abs_tol=atol)
    if (af is None) != (bf is None):
        return False
    return symbol_text(a) == symbol_text(b)
