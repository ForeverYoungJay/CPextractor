from __future__ import annotations

from typing import Any, Dict, List, Tuple
from postprocess.param_iter import iter_parameter_items


_CANONICAL_NAME_MAP = {
    "tau0": "crss_initial",
    "τ0": "crss_initial",
    "g0": "crss_initial",
    "crss0": "crss_initial",
    "tau_sat": "crss_saturation",
    "taus": "crss_saturation",
    "h0": "hardening_h0",
    "h1": "hardening_h1",
    "q": "latent_ratio_q",
    "qab": "interaction_matrix_qab",
    "m": "rate_sensitivity_m",
    "n": "exponent_n",
    "gamma0": "gamma0_ref",
    "gammadot0": "gamma0_ref",
    "γ̇0": "gamma0_ref",
}

_CANONICAL_NAME_ALIAS = {
    "critical resolved shear stress": "crss_initial",
    "critical shear strength": "crss_initial",
    "initial hardening modulus": "hardening_h0",
    "latent hardening coefficient": "latent_ratio_q",
    "rate sensitivity exponent": "exponent_n",
    "rate sensitivity": "exponent_n",
    "reference shear rate": "gamma0_ref",
}

_MECHANISM_MAP = {
    "slip": "all_slip",
    "all slip": "all_slip",
    "all_slip": "all_slip",
    "twin": "all_twin",
    "twinning": "all_twin",
    "all_twin": "all_twin",
    "all_mechanisms": "all_mechanisms",
    "basal": "basal_slip",
    "basal slip": "basal_slip",
    "prismatic": "prismatic_slip",
    "prismatic slip": "prismatic_slip",
    "pyramidal": "pyramidal_slip",
    "pyramidal slip": "pyramidal_slip",
}


def _clean(v: Any) -> str:
    return str(v or "").strip().lower()


def _normalize_canonical_name(p: Dict[str, Any]) -> bool:
    changed = False
    symbol_key = _clean(p.get("symbol")).replace(".", "").replace("-", "")
    name_key = _clean(p.get("canonical_name"))
    desc_key = _clean(p.get("description"))

    target = None
    if symbol_key in _CANONICAL_NAME_MAP:
        target = _CANONICAL_NAME_MAP[symbol_key]
    elif name_key in _CANONICAL_NAME_ALIAS:
        target = _CANONICAL_NAME_ALIAS[name_key]
    elif desc_key in _CANONICAL_NAME_ALIAS:
        target = _CANONICAL_NAME_ALIAS[desc_key]

    if target and p.get("canonical_name") != target:
        p["canonical_name"] = target
        changed = True
    return changed


def _normalize_mechanism(p: Dict[str, Any]) -> bool:
    applies = p.get("applies_to")
    if not isinstance(applies, dict):
        return False
    m = _clean(applies.get("mechanism"))
    if not m:
        return False
    target = _MECHANISM_MAP.get(m)
    if target and applies.get("mechanism") != target:
        applies["mechanism"] = target
        return True
    return False


def _slug(v: Any) -> str:
    s = str(v or "").strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _infer_mechanism_from_family_name(name: str) -> str | None:
    n = _slug(name)
    if not n:
        return None
    if "basal" in n:
        return "basal_slip"
    if "prismatic" in n:
        return "prismatic_slip"
    if "pyramidal" in n:
        return "pyramidal_slip"
    if "twin" in n:
        return "twinning"
    return None


def _family_key(f: Dict[str, Any]) -> str:
    return _slug(f.get("family_id") or f.get("family_name") or f.get("plane_direction"))


def _ensure_family_systems(families: List[Dict[str, Any]], family_prefix: str) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for i, fam in enumerate(families):
        if not isinstance(fam, dict):
            continue
        fid = str(fam.get("family_id") or "").strip()
        if not fid:
            base = _slug(fam.get("family_name")) or _slug(fam.get("plane_direction")) or f"{family_prefix}_{i+1}"
            fid = f"{family_prefix}_{base}"
            fam["family_id"] = fid

        systems = fam.get("systems", [])
        if not isinstance(systems, list):
            systems = []
        n = fam.get("num_systems")
        try:
            n = int(n) if n is not None else None
        except Exception:
            n = None

        if not systems and n and n > 0:
            systems = [{"system_id": f"{fid}_s{j+1}", "plane": None, "direction": None} for j in range(n)]
            fam["systems"] = systems
        elif systems:
            # Fill missing ids.
            for j, s in enumerate(systems):
                if isinstance(s, dict) and not s.get("system_id"):
                    s["system_id"] = f"{fid}_s{j+1}"
            fam["num_systems"] = len([s for s in systems if isinstance(s, dict)])

        by_key[_family_key(fam)] = fam
    return by_key


def _choose_family_for_param(applies: Dict[str, Any], slip_by_key: Dict[str, Dict[str, Any]], twin_by_key: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    fname = _slug(applies.get("family_id") or applies.get("family_name"))
    mech = _slug(applies.get("mechanism"))
    if fname:
        return slip_by_key.get(fname) or twin_by_key.get(fname)

    if mech in {"basal_slip", "prismatic_slip", "pyramidal_slip"}:
        for fam in slip_by_key.values():
            fm = _infer_mechanism_from_family_name(str(fam.get("family_name") or ""))
            if fm == mech:
                return fam
    if mech in {"twinning", "all_twin"}:
        if len(twin_by_key) == 1:
            return list(twin_by_key.values())[0]
    if mech in {"all_slip"} and len(slip_by_key) == 1:
        return list(slip_by_key.values())[0]
    return None


def normalize_parameters(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report = {
        "plastic_parameters_total": 0,
        "canonical_name_normalized": 0,
        "mechanism_normalized": 0,
        "phase_id_filled": 0,
        "family_mapping_filled": 0,
        "system_mapping_filled": 0,
        "scope_normalized": 0,
    }

    items = [it for _, it in iter_parameter_items(extracted_json)]
    if not items:
        return extracted_json, report

    dm = extracted_json.get("deformation_mechanisms", {}) if isinstance(extracted_json.get("deformation_mechanisms"), dict) else {}
    # Backward compatibility with old field name.
    st = extracted_json.get("slip_twin_systems", {}) if isinstance(extracted_json.get("slip_twin_systems"), dict) else {}
    mechanism_root = dm if dm else st
    if not dm and st:
        extracted_json["deformation_mechanisms"] = st
    if "slip_twin_systems" in extracted_json:
        extracted_json.pop("slip_twin_systems", None)
    slip_fams = mechanism_root.get("slip_families", []) if isinstance(mechanism_root.get("slip_families"), list) else []
    twin_fams = mechanism_root.get("twinning_families", []) if isinstance(mechanism_root.get("twinning_families"), list) else []
    slip_by_key = _ensure_family_systems(slip_fams, "slip")
    twin_by_key = _ensure_family_systems(twin_fams, "twin")

    report["plastic_parameters_total"] = len(items)
    for p in items:
        if not isinstance(p, dict):
            continue
        if _normalize_canonical_name(p):
            report["canonical_name_normalized"] += 1
        if _normalize_mechanism(p):
            report["mechanism_normalized"] += 1

        applies = p.get("applies_to")
        if not isinstance(applies, dict):
            applies = {}
            p["applies_to"] = applies

        fam = _choose_family_for_param(applies, slip_by_key, twin_by_key)
        if fam:
            if not applies.get("family_id"):
                applies["family_id"] = fam.get("family_id")
                report["family_mapping_filled"] += 1
            if not applies.get("family_name"):
                applies["family_name"] = fam.get("family_name") or fam.get("plane_direction")
                report["family_mapping_filled"] += 1

            system_ids = applies.get("system_ids")
            if not isinstance(system_ids, list) or not system_ids:
                systems = fam.get("systems", []) if isinstance(fam.get("systems"), list) else []
                system_ids = [str(s.get("system_id")) for s in systems if isinstance(s, dict) and s.get("system_id")]
                if system_ids:
                    applies["system_ids"] = system_ids
                    report["system_mapping_filled"] += 1

            if applies.get("system_count") is None:
                if isinstance(applies.get("system_ids"), list) and applies["system_ids"]:
                    applies["system_count"] = len(applies["system_ids"])
                    report["system_mapping_filled"] += 1
                elif fam.get("num_systems") is not None:
                    applies["system_count"] = fam.get("num_systems")
                    report["system_mapping_filled"] += 1

        # Scope normalization rules:
        # global -> no phase_id
        # phase -> phase_id
        # family -> phase_id + family_id
        # system -> phase_id + system_ids
        scope = _slug(applies.get("scope"))
        if scope not in {"global", "phase", "family", "system"}:
            if isinstance(applies.get("system_ids"), list) and applies.get("system_ids"):
                scope = "system"
            elif applies.get("family_id") or applies.get("family_name"):
                scope = "family"
            elif applies.get("phase_id"):
                scope = "phase"
            else:
                scope = "global"
            applies["scope"] = scope
            report["scope_normalized"] += 1
        else:
            applies["scope"] = scope

        if scope == "global":
            applies.pop("phase_id", None)
            applies.pop("family_id", None)
            applies.pop("family_name", None)
            applies.pop("system_ids", None)
            applies.pop("system_count", None)
        elif scope == "phase":
            if not applies.get("phase_id"):
                applies["phase_id"] = "phase_1"
                report["phase_id_filled"] += 1
            applies.pop("family_id", None)
            applies.pop("family_name", None)
            applies.pop("system_ids", None)
            applies.pop("system_count", None)
        elif scope == "family":
            if not applies.get("phase_id"):
                applies["phase_id"] = "phase_1"
                report["phase_id_filled"] += 1
            if not applies.get("family_id"):
                if fam and fam.get("family_id"):
                    applies["family_id"] = fam.get("family_id")
                elif applies.get("family_name"):
                    applies["family_id"] = f"family_{_slug(applies.get('family_name'))}"
                else:
                    applies["family_id"] = "family_1"
                report["family_mapping_filled"] += 1
            applies.pop("system_ids", None)
        elif scope == "system":
            if not applies.get("phase_id"):
                applies["phase_id"] = "phase_1"
                report["phase_id_filled"] += 1
            system_ids = applies.get("system_ids")
            if not isinstance(system_ids, list) or not system_ids:
                if fam and isinstance(fam.get("systems"), list):
                    system_ids = [str(s.get("system_id")) for s in fam["systems"] if isinstance(s, dict) and s.get("system_id")]
                else:
                    system_ids = []
                if system_ids:
                    applies["system_ids"] = system_ids
                    report["system_mapping_filled"] += 1
            if applies.get("system_count") is None and isinstance(applies.get("system_ids"), list) and applies["system_ids"]:
                applies["system_count"] = len(applies["system_ids"])
                report["system_mapping_filled"] += 1

    return extracted_json, report
