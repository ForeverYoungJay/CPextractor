"""Record local release readiness and create a source-only reproducible snapshot."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.benchmark_protocol import fingerprint, read_manifest
from scripts.eval.common import save_json


def source_files():
    files = [ROOT / name for name in ("README.md", "config.example.yaml", "requirements.txt", "requirements.lock.txt", "requirements-ui.txt", "schema.sql", "docker-compose.yml")]
    for folder in ("db", "kg", "llm", "elsevier", "scopus", "postprocess", "pipelines", "scripts", "tests"):
        files.extend((ROOT / folder).rglob("*.py"))
    files.extend((ROOT / "docs").glob("*.md"))
    files.extend((ROOT / "pipelines/config_profiles").glob("*.yaml"))
    files.extend(ROOT / name for name in ("chatbot.py", "chatbot_ui.py"))
    return sorted(set(p for p in files if p.is_file()))


def check(manifest_path, outdir, run_tests=False, archive=False, validation_root=None):
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(manifest_path)
    blockers = []
    for paper in manifest["papers"]:
        if not (paper.get("annotation_status") == "adjudicated" and paper.get("exhaustive") is True
                and paper.get("reviewer_type") == "human_expert" and paper.get("reviewer") and paper.get("adjudicator")):
            blockers.append({"stage": "expert_gold", "doi": paper["doi"], "reason": "not exhaustively expert-adjudicated"})
    tests = {"status": "not_run"}
    if run_tests:
        proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                              cwd=ROOT, capture_output=True, text=True)
        log = output / "tests.log"
        log.write_text(proc.stdout + proc.stderr)
        tests = {"status": "passed" if proc.returncode == 0 else "failed", "returncode": proc.returncode,
                 "log_sha256": fingerprint(log)}
    if tests["status"] != "passed":
        blockers.append({"stage": "software_tests", "reason": tests["status"]})
    validation = Path(validation_root) if validation_root else output.parent
    evidence_files = {
        "frozen_test_evaluation": validation / "test/metrics/benchmark_claims.json",
        "confidence_gate_validation": validation / "test_confidence/confidence_gate_report.json",
        "completed_ablation_comparison": validation / "ablation_comparison/ablation_comparison.json",
    }
    for stage, path in evidence_files.items():
        data = json.loads(path.read_text()) if path.exists() else {}
        valid = data.get("publication_ready") is True
        if stage == "frozen_test_evaluation":
            valid = valid and data.get("universe", {}).get("split") == "test" and data.get("inputs", {}).get("manifest_sha256") == fingerprint(manifest_path)
        elif stage == "confidence_gate_validation":
            valid = valid and data.get("split") == "test" and data.get("status") == "evaluated"
        else:
            valid = valid and bool(data.get("rows")) and all(r.get("not_run_papers") == 0 for r in data.get("rows", []))
        if not valid:
            blockers.append({"stage": stage, "reason": "missing or unvalidated evidence", "expected_path": str(path)})
    if not (ROOT / "LICENSE").exists():
        blockers.append({"stage": "software_license", "reason": "No explicit LICENSE file selected by the project owner"})
    files = source_files()
    hashes = {str(p.relative_to(ROOT)): fingerprint(p) for p in files}
    env = {}
    for name in ("openai", "PyYAML", "psycopg", "beautifulsoup4", "lxml", "requests", "openpyxl"):
        try:
            env[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            env[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    report = {"software_tests": tests, "scientific_release_ready": not blockers, "blockers": blockers,
              "python": platform.python_version(), "dependencies": env, "git_head": git.stdout.strip(),
              "working_tree_dirty": bool(status.stdout.strip()), "manifest_sha256": fingerprint(manifest_path),
              "source_sha256": hashes, "snapshot_scope": "source code, tests and selected docs; excludes corpus, credentials and personal config"}
    if archive:
        path = output / "cpextractor-source-candidate.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:
                info = zipfile.ZipInfo(str(p.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, p.read_bytes())
        report["archive"] = {"path": str(path), "sha256": fingerprint(path)}
    save_json(output / "release_readiness.json", report)
    print(json.dumps({"software_tests": tests["status"], "scientific_release_ready": report["scientific_release_ready"],
                      "blocker_count": len(blockers), "archive": report.get("archive")}, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--archive", action="store_true")
    ap.add_argument("--require-ready", action="store_true")
    ap.add_argument("--validation-root", help="Directory containing test, test_confidence and ablation_comparison evidence")
    args = ap.parse_args()
    report = check(args.manifest, args.outdir, args.run_tests, args.archive, args.validation_root)
    if args.require_ready and not report["scientific_release_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
