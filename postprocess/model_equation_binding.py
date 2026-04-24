from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _merge_unique(values: List[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _normalized_equation_row_id(row: Dict[str, Any]) -> str:
    equation_id = str(row.get("equation_id") or "").strip()
    if equation_id:
        return equation_id

    eq_index = row.get("equation_index")
    try:
        if eq_index not in (None, ""):
            return f"eq_{int(eq_index):04d}"
    except Exception:
        pass

    label = str(row.get("label") or "").strip()
    if label:
        return label

    return ""


def _equation_ref_tokens(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    lowered = text.lower()
    compact = re.sub(r"\s+", "", lowered)
    tokens = {text, lowered, compact}

    nums = re.findall(r"\d+", lowered)
    if nums:
        n = str(int(nums[-1]))
        tokens.update(
            {
                n,
                f"({n})",
                f"eq({n})",
                f"eq.{n}",
                f"eq{n}",
                f"equation({n})",
                f"equation{n}",
            }
        )
    cleaned = set()
    for tok in tokens:
        t = str(tok or "").strip().lower()
        if not t:
            continue
        cleaned.add(t)
        cleaned.add(re.sub(r"[\s.]+", "", t))
    return list(cleaned)


_SYMBOL_TRANSLATION = {
    "γ": "gamma",
    "Γ": "gamma",
    "τ": "tau",
    "Τ": "tau",
    "θ": "theta",
    "Θ": "theta",
    "χ": "chi",
    "Χ": "chi",
    "σ": "sigma",
    "Σ": "sigma",
    "α": "alpha",
    "Α": "alpha",
    "β": "beta",
    "Β": "beta",
    "ρ": "rho",
    "Ρ": "rho",
    "μ": "mu",
    "Μ": "mu",
    "ν": "nu",
    "Ν": "nu",
    "κ": "kappa",
    "Κ": "kappa",
    "̇": "dot",
    "˙": "dot",
}


def _compact_symbol_token(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return ""
    for src, dst in _SYMBOL_TRANSLATION.items():
        text = text.replace(src, dst)
    text = text.lower()
    text = re.sub(r"\b(eq|equation)\b", "", text)
    text = re.sub(r"[\s_./|()\[\]{}^*=:+-]+", "", text)
    text = text.replace(",", "")
    text = text.replace("−", "")
    text = text.replace("–", "")
    text = text.replace("·", "")
    return text


def _symbol_token_variants(value: Any) -> List[str]:
    base = _compact_symbol_token(value)
    if not base:
        return []
    variants = {base}
    if len(base) <= 10 and "0" in base:
        variants.add(base.replace("0", "o"))
    if len(base) <= 10 and "o" in base:
        variants.add(base.replace("o", "0"))
    return [tok for tok in variants if tok]


def _section_title(section_path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or section_path.stem
        return stripped[:200]
    return section_path.stem


def _inline_equation_pattern() -> re.Pattern[str]:
    return re.compile(r"\((\d{1,4})\)\s*(?=[A-Za-z0-9γΓτΤθΘχΧσΣαΑβΒρΡμΜνΝκΚXx])")


def _load_inline_equations_from_sections(paper_dir: str) -> List[Dict[str, Any]]:
    sections_dir = Path(paper_dir) / "sections"
    if not sections_dir.exists():
        return []

    dedup: Dict[str, Dict[str, Any]] = {}
    for section_path in sorted(sections_dir.glob("*.md")):
        try:
            text = section_path.read_text(encoding="utf-8")
        except Exception:
            continue
        matches = list(_inline_equation_pattern().finditer(text))
        if not matches:
            continue
        title = _section_title(section_path, text)
        for idx, match in enumerate(matches):
            label = f"({int(match.group(1))})"
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            window_end = min(end, start + 650)
            block = text[start:window_end].strip()
            if not block:
                continue
            if "=" not in block[:220]:
                continue
            block = re.sub(r"\n{3,}", "\n\n", block)
            row = {
                "equation_id": label,
                "label": label,
                "equation_index": int(match.group(1)),
                "section_title": title,
                "paragraph_text": block[:1200],
                "text": block[:1200],
                "latex": "",
                "text_file": section_path.name,
                "source_file": f"sections/{section_path.name}",
            }
            previous = dedup.get(label)
            if not previous or len(str(row.get("text") or "")) > len(str(previous.get("text") or "")):
                dedup[label] = row
    return [dedup[key] for key in sorted(dedup, key=lambda item: int(re.findall(r'\d+', item)[-1]))]


def _extract_defined_symbol_aliases(text: str) -> List[str]:
    collapsed = " ".join(unicodedata.normalize("NFKC", str(text or "")).split())
    if not collapsed:
        return []

    forbidden_words = (
        "reference",
        "strain",
        "rate",
        "sensitivity",
        "factor",
        "initial",
        "slip",
        "hardening",
        "coefficient",
        "activation",
        "energy",
        "temperature",
        "constant",
        "dynamic",
        "recovery",
        "related",
        "parameter",
        "parameters",
        "regime",
        "gas",
    )

    def _looks_like_symbol(token: str) -> bool:
        raw = str(token or "").strip()
        compact = _compact_symbol_token(raw)
        if not compact:
            return False
        if any(word in compact for word in forbidden_words):
            return False
        if len(compact) == 1:
            return raw in {"A", "Q", "R", "T", "d", "n", "h", "X"}
        if len(compact) > 24:
            return False
        return True

    aliases: set[str] = set()
    for left in re.findall(r"([A-Za-zγΓτΤθΘχΧσΣαΑβΒρΡμΜνΝκΚXx0-9˙̇,\-\s]+?)\s+(?:is|are)\b", collapsed):
        cleaned_left = re.sub(
            r"^(where|here|the|in the second term|in the first term)\s+",
            "",
            left.strip(),
            flags=re.IGNORECASE,
        )
        for part in re.split(r",| and ", cleaned_left):
            token = part.strip()
            if not token or not _looks_like_symbol(token):
                continue
            for alias in _symbol_token_variants(token):
                if alias and len(alias) <= 24:
                    aliases.add(alias)
    return sorted(aliases)


def _equation_role_tags(equation: Dict[str, Any]) -> List[str]:
    haystack = " ".join(
        str(equation.get(k) or "")
        for k in ("label", "text", "latex", "paragraph_text", "section_title")
    )
    compact = _compact_symbol_token(haystack)
    tags: set[str] = set()
    if "backstress" in compact or "xdotalpha" in compact or "xdot" in compact:
        tags.update({"backstress_evolution", "backstress"})
    if "taudotcalpha" in compact or "taucdotalpha" in compact or "tauc0" in compact or "tauco" in compact:
        tags.update({"hardening", "crss_evolution"})
    if "gammadot02" in compact or "m2" in compact or "creep" in compact:
        tags.update({"creep_flow", "creep"})
    if "gammadot01" in compact or "m1" in compact or "plasticity" in compact:
        tags.update({"plastic_flow", "plastic"})
    if "gammadot01" in compact and "gammadot02" in compact:
        tags.update({"combined_flow", "plastic_flow", "creep_flow"})
    if "cumulativeslip" in compact or re.search(r"\(6\)", str(equation.get("label") or "")):
        if "gamma=" in compact or compact.startswith("6gamma") or "eq6" in compact:
            tags.add("cumulative_slip")
    return sorted(tags)


def _equation_aliases(equation: Dict[str, Any]) -> List[str]:
    aliases = set(_extract_defined_symbol_aliases(" ".join(
        str(equation.get(k) or "")
        for k in ("text", "paragraph_text")
    )))
    compact = _compact_symbol_token(" ".join(
        str(equation.get(k) or "")
        for k in ("text", "paragraph_text", "latex", "label")
    ))
    manual_aliases = {
        "gammadot01": "gammadot01",
        "gammadot02": "gammadot02",
        "m1": "m1",
        "m2": "m2",
        "h0": "h0",
        "hd": "hd",
        "tauc0": "tauc0",
        "tauco": "tauco",
        "backstress": "backstress",
    }
    for needle, alias in manual_aliases.items():
        if needle in compact:
            aliases.update(_symbol_token_variants(alias))
    return sorted(aliases)


def _branch_aliases(branch: Dict[str, Any]) -> List[str]:
    branch_type = str(branch.get("branch_type") or "").strip().lower()
    text = " ".join(
        str(branch.get(k) or "")
        for k in ("branch_type", "name", "description", "notes")
    )
    aliases = {_compact_symbol_token(text), branch_type}
    manual: Dict[str, List[str]] = {
        "plastic_flow": ["plastic_flow", "plastic", "flow", "gammadot01", "m1"],
        "creep_flow": ["creep_flow", "creep", "gammadot02", "m2", "q"],
        "hardening": ["hardening", "crss_evolution", "tauc0", "h0", "n", "a", "d", "q"],
        "backstress_evolution": ["backstress_evolution", "backstress", "h", "hd"],
        "backstress": ["backstress", "h", "hd"],
        "combined_flow": ["combined_flow", "plastic_flow", "creep_flow", "gammadot01", "gammadot02", "m1", "m2"],
    }
    for alias in manual.get(branch_type, []):
        aliases.update(_symbol_token_variants(alias))
    return [tok for tok in aliases if tok]


def _claim_aliases(claim: Dict[str, Any]) -> List[str]:
    parameter = _safe_dict(claim.get("parameter"))
    aliases: set[str] = set()
    for value in (
        parameter.get("symbol_reported"),
        parameter.get("raw_name"),
        parameter.get("canonical_name"),
        parameter.get("description"),
    ):
        aliases.update(_symbol_token_variants(value))

    canonical = str(parameter.get("canonical_name") or "").strip().lower()
    manual: Dict[str, List[str]] = {
        "reference_shear_rate_plastic": ["gammadot01"],
        "reference_shear_rate_creep": ["gammadot02"],
        "reference_shear_rate": ["gammadot01", "gammadot02"],
        "rate_sensitivity_plastic": ["m1"],
        "rate_sensitivity_creep": ["m2"],
        "initial_hardening_modulus": ["h0"],
        "hardening_h0": ["h0"],
        "initial_crss": ["tauc0"],
        "crss_initial": ["tauc0"],
        "creep_activation_energy": ["q"],
        "hardening_coefficient_h": ["h"],
        "hardening_coefficient_hd": ["hd"],
        "dynamic_recovery_coefficient": ["hd"],
    }
    for alias in manual.get(canonical, []):
        aliases.update(_symbol_token_variants(alias))
    return sorted(tok for tok in aliases if tok)


def _infer_branch_equation_ids(
    branch: Dict[str, Any],
    equation_rows: List[Dict[str, Any]],
) -> List[str]:
    branch_tokens = set(_branch_aliases(branch))
    if not branch_tokens:
        return []
    out: List[str] = []
    for equation in equation_rows:
        eq_id = _normalized_equation_row_id(equation)
        if not eq_id:
            continue
        score = 0
        eq_tokens = set(_equation_aliases(equation))
        role_tags = set(_equation_role_tags(equation))
        score += len(branch_tokens & eq_tokens) * 3
        score += len(branch_tokens & role_tags) * 4
        if score <= 0:
            continue
        if eq_id not in out:
            out.append(eq_id)
    return out


def _infer_claim_equation_ids(
    claim: Dict[str, Any],
    equation_rows: List[Dict[str, Any]],
) -> List[str]:
    claim_tokens = set(_claim_aliases(claim))
    if not claim_tokens:
        return []
    out: List[str] = []
    for equation in equation_rows:
        eq_id = _normalized_equation_row_id(equation)
        if not eq_id:
            continue
        eq_tokens = set(_equation_aliases(equation))
        if claim_tokens & eq_tokens:
            out.append(eq_id)
    return out


def _build_equation_lookup(equation_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for row in equation_rows:
        if not isinstance(row, dict):
            continue
        eq_id = _normalized_equation_row_id(row)
        if not eq_id:
            continue
        for candidate in [eq_id, row.get("label"), row.get("equation_index")]:
            for token in _equation_ref_tokens(candidate):
                lookup.setdefault(token, eq_id)
    return lookup


def _resolve_equation_bundle(
    equation_ids: List[Any],
    *,
    equation_by_id: Dict[str, Dict[str, Any]],
    equation_lookup: Dict[str, str],
    owner_prefix: str,
    owner_id: str,
) -> Tuple[List[str], List[Dict[str, Any]], List[str], int]:
    normalized_ids: List[str] = []
    for eq_id in _safe_list(equation_ids):
        eq_key = str(eq_id or "").strip()
        if not eq_key:
            continue
        resolved = None
        if eq_key in equation_by_id:
            resolved = eq_key
        else:
            for token in _equation_ref_tokens(eq_key):
                if token in equation_lookup:
                    resolved = equation_lookup[token]
                    break
        if resolved and resolved in equation_by_id and resolved not in normalized_ids:
            normalized_ids.append(resolved)
        elif eq_key and not equation_by_id and eq_key not in normalized_ids:
            normalized_ids.append(eq_key)

    resolved_equations: List[Dict[str, Any]] = []
    for eq_id in normalized_ids:
        eq = equation_by_id.get(eq_id)
        if not eq:
            continue
        resolved_equations.append({
            "equation_id": eq_id,
            "label": eq.get("label"),
            "section_title": eq.get("section_title"),
            "text": eq.get("text"),
            "latex": eq.get("latex"),
            "text_file": eq.get("text_file"),
        })

    return normalized_ids, resolved_equations, [], len(resolved_equations)


def _load_equation_index(paper_dir: str) -> List[Dict[str, Any]]:
    path = Path(paper_dir) / "equations" / "index.json"
    if not path.exists():
        return _load_inline_equations_from_sections(paper_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _load_inline_equations_from_sections(paper_dir)
    rows = [row for row in payload if isinstance(row, dict)]
    return rows or _load_inline_equations_from_sections(paper_dir)


def _backfill_equation_ids(
    extracted_json: Dict[str, Any],
    *,
    equation_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    extracted = dict(extracted_json)
    models = [m for m in _safe_list(extracted.get("models")) if isinstance(m, dict)]
    claims = [c for c in _safe_list(extracted.get("parameter_claims")) if isinstance(c, dict)]
    if not equation_rows:
        return extracted, {"branches_backfilled": 0, "claims_backfilled": 0}

    branch_map: Dict[str, List[str]] = {}
    out_models: List[Dict[str, Any]] = []
    branches_backfilled = 0
    for idx, model in enumerate(models, start=1):
        row = dict(model)
        model_id = str(row.get("model_id") or "").strip() or f"model_{idx:03d}"
        branches = [b for b in _safe_list(row.get("constitutive_branches")) if isinstance(b, dict)]
        out_branches: List[Dict[str, Any]] = []
        for branch_idx, branch in enumerate(branches, start=1):
            branch_row = dict(branch)
            branch_id = str(branch_row.get("branch_id") or "").strip() or f"{model_id}_branch_{branch_idx:02d}"
            existing = _merge_unique(_safe_list(branch_row.get("governing_equation_ids")))
            inferred = _infer_branch_equation_ids(branch_row, equation_rows) if not existing else []
            merged = _merge_unique(existing + inferred)
            if not existing and merged:
                branches_backfilled += 1
            branch_row["branch_id"] = branch_id
            branch_row["governing_equation_ids"] = merged
            branch_map[branch_id] = merged
            out_branches.append(branch_row)
        if out_branches:
            row["constitutive_branches"] = out_branches
        row["equation_ids"] = _merge_unique(
            _safe_list(row.get("equation_ids"))
            + [eq_id for branch_row in out_branches for eq_id in _safe_list(branch_row.get("governing_equation_ids"))]
        )
        out_models.append(row)
    if out_models:
        extracted["models"] = out_models

    out_claims: List[Dict[str, Any]] = []
    claims_backfilled = 0
    for claim in claims:
        claim_row = dict(claim)
        existing = _merge_unique(_safe_list(claim_row.get("governing_equation_ids")))
        symbol_matches = _infer_claim_equation_ids(claim_row, equation_rows)
        applies_to = _safe_dict(claim_row.get("applies_to"))
        branch_matches = branch_map.get(str(applies_to.get("branch_id") or "").strip(), [])
        if symbol_matches:
            merged = _merge_unique(existing + symbol_matches)
        else:
            merged = _merge_unique(existing + branch_matches)
        if not existing and merged:
            claims_backfilled += 1
        claim_row["governing_equation_ids"] = merged
        out_claims.append(claim_row)
    if out_claims:
        extracted["parameter_claims"] = out_claims
    return extracted, {
        "branches_backfilled": branches_backfilled,
        "claims_backfilled": claims_backfilled,
    }


def bind_model_equations(
    extracted_json: Dict[str, Any],
    *,
    paper_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    extracted = dict(extracted_json)
    models = [m for m in _safe_list(extracted.get("models")) if isinstance(m, dict)]
    if not models:
        return extracted, {"models": 0, "bound_models": 0, "bound_equations": 0}

    equation_rows = _load_equation_index(paper_dir)
    extracted, backfill_report = _backfill_equation_ids(extracted, equation_rows=equation_rows)
    models = [m for m in _safe_list(extracted.get("models")) if isinstance(m, dict)]
    equation_by_id = {
        _normalized_equation_row_id(row): row
        for row in equation_rows
        if _normalized_equation_row_id(row)
    }
    equation_lookup = _build_equation_lookup(equation_rows)

    evidence_objects = [row for row in _safe_list(extracted.get("evidence_objects")) if isinstance(row, dict)]

    bound_models = 0
    bound_equations = 0
    out_models: List[Dict[str, Any]] = []

    for idx, model in enumerate(models, start=1):
        row = dict(model)
        model_id = str(row.get("model_id") or "").strip() or f"model_{idx:03d}"
        row["model_id"] = model_id

        normalized_ids, resolved_equations, equation_evidence_ids, resolved_count = _resolve_equation_bundle(
            _safe_list(row.get("equation_ids")),
            equation_by_id=equation_by_id,
            equation_lookup=equation_lookup,
            owner_prefix="model",
            owner_id=model_id,
        )

        row["equation_ids"] = normalized_ids
        row["equations"] = resolved_equations

        branches = [b for b in _safe_list(row.get("constitutive_branches")) if isinstance(b, dict)]
        out_branches: List[Dict[str, Any]] = []
        for branch_idx, branch in enumerate(branches, start=1):
            branch_row = dict(branch)
            branch_id = str(branch_row.get("branch_id") or "").strip() or f"{model_id}_branch_{branch_idx:02d}"
            branch_row["branch_id"] = branch_id
            branch_eq_ids, branch_eqs, branch_ev_ids, branch_count = _resolve_equation_bundle(
                _safe_list(branch_row.get("governing_equation_ids")),
                equation_by_id=equation_by_id,
                equation_lookup=equation_lookup,
                owner_prefix="branch",
                owner_id=branch_id,
            )
            branch_row["governing_equation_ids"] = branch_eq_ids
            branch_row["governing_equations"] = branch_eqs
            resolved_count += branch_count
            out_branches.append(branch_row)
        if out_branches:
            row["constitutive_branches"] = out_branches

        branch_union_ids: List[str] = []
        for branch in out_branches:
            for eq_id in _safe_list(branch.get("governing_equation_ids")):
                eq_key = str(eq_id or "").strip()
                if eq_key and eq_key not in branch_union_ids:
                    branch_union_ids.append(eq_key)
        if branch_union_ids:
            merged_model_ids: List[str] = []
            for eq_id in normalized_ids + branch_union_ids:
                eq_key = str(eq_id or "").strip()
                if eq_key and eq_key not in merged_model_ids:
                    merged_model_ids.append(eq_key)
            normalized_ids = merged_model_ids
            resolved_equations = []
            for eq_id in normalized_ids:
                eq = equation_by_id.get(eq_id)
                if not eq:
                    continue
                resolved_equations.append({
                    "equation_id": eq_id,
                    "label": eq.get("label"),
                    "section_title": eq.get("section_title"),
                    "text": eq.get("text"),
                    "latex": eq.get("latex"),
                    "text_file": eq.get("text_file"),
                })
            row["equation_ids"] = normalized_ids
            row["equations"] = resolved_equations

        if resolved_equations:
            bound_models += 1
        bound_equations += resolved_count
        out_models.append(row)

    extracted["models"] = out_models

    claims = [c for c in _safe_list(extracted.get("parameter_claims")) if isinstance(c, dict)]
    out_claims: List[Dict[str, Any]] = []
    claim_eqs_by_model: Dict[str, List[str]] = {}
    for idx, claim in enumerate(claims, start=1):
        claim_row = dict(claim)
        claim_id = str(claim_row.get("claim_id") or "").strip() or f"claim_{idx:04d}"
        claim_row["claim_id"] = claim_id
        claim_eq_ids, claim_eqs, claim_ev_ids, claim_count = _resolve_equation_bundle(
            _safe_list(claim_row.get("governing_equation_ids")),
            equation_by_id=equation_by_id,
            equation_lookup=equation_lookup,
            owner_prefix="claim",
            owner_id=claim_id,
        )
        claim_row["governing_equation_ids"] = claim_eq_ids
        claim_row["governing_equations"] = claim_eqs
        bound_equations += claim_count
        model_id = str(_safe_dict(claim_row.get("applies_to")).get("model_id") or "").strip()
        if model_id and claim_eq_ids:
            claim_eqs_by_model.setdefault(model_id, [])
            claim_eqs_by_model[model_id] = _merge_unique(claim_eqs_by_model[model_id] + claim_eq_ids)
        out_claims.append(claim_row)
    if out_claims:
        extracted["parameter_claims"] = out_claims

    if out_models:
        refreshed_models: List[Dict[str, Any]] = []
        for model in out_models:
            row = dict(model)
            model_id = str(row.get("model_id") or "").strip()
            extra_ids = claim_eqs_by_model.get(model_id, [])
            if extra_ids:
                merged_ids = _merge_unique(_safe_list(row.get("equation_ids")) + extra_ids)
                resolved_equations = []
                for eq_id in merged_ids:
                    eq = equation_by_id.get(eq_id)
                    if not eq:
                        continue
                    resolved_equations.append({
                        "equation_id": eq_id,
                        "label": eq.get("label"),
                        "section_title": eq.get("section_title"),
                        "text": eq.get("text"),
                        "latex": eq.get("latex"),
                        "text_file": eq.get("text_file"),
                    })
                row["equation_ids"] = merged_ids
                row["equations"] = resolved_equations
            refreshed_models.append(row)
        extracted["models"] = refreshed_models

    extracted["evidence_objects"] = evidence_objects
    return extracted, {
        "models": len(out_models),
        "bound_models": bound_models,
        "bound_equations": bound_equations,
        "branches_backfilled": backfill_report["branches_backfilled"],
        "claims_backfilled": backfill_report["claims_backfilled"],
        "equation_sources": len(equation_rows),
    }
