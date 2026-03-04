import argparse
from pathlib import Path
import yaml


def deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a config profile onto config.yaml")
    ap.add_argument("--base", default="config.yaml")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--output", default="config.yaml")
    args = ap.parse_args()

    base_path = Path(args.base)
    profile_path = Path(args.profile)
    out_path = Path(args.output)

    with base_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    with profile_path.open("r", encoding="utf-8") as f:
        profile = yaml.safe_load(f) or {}

    merged = deep_merge(base, profile)

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)

    print(f"Applied profile: {profile_path} -> {out_path}")


if __name__ == "__main__":
    main()
