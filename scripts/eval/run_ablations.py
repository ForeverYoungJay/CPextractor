"""Isolated, resumable, database-free single-factor ablation runner."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.benchmark_protocol import PROTOCOL_VERSION, read_manifest, fingerprint
from scripts.eval.common import save_json

VARIANTS = {
    "baseline": {},
    "committee": {"llm.evaluator_mode": "committee"},
    "rules_on": {"pipeline.skip_quality_checks": False},
    "double_pass": {"llm.two_pass_extraction": True},
    "extract_gpt41": {"llm.model_extract": "gpt-4.1"},
}
SOURCE_NAMES = ("paper.xml", "references.json", "sections", "tables", "equations")


def sanitized_config(config):
    config = copy.deepcopy(config)
    for section, keys in {"elsevier": ["api_key", "inst_token"], "db": ["password"]}.items():
        for key in keys:
            if section in config and key in config[section]:
                config[section][key] = None
    return config


def variant_config(config, name):
    config = sanitized_config(config)
    # Fix baseline toggles irrespective of the user's personal config.
    config.setdefault("llm", {}).update(evaluator_mode="single_judge", two_pass_extraction=False,
                                        enable_evaluator=True, evaluation_feedback_json=None)
    config.setdefault("pipeline", {})["skip_quality_checks"] = True
    for key, value in VARIANTS[name].items():
        section, field = key.split(".")
        config[section][field] = value
    return config


def source_hashes(directory):
    directory = Path(directory)
    result = {}
    for name in SOURCE_NAMES:
        path = directory / name
        paths = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path] if path.is_file() else []
        for p in paths:
            # Cloud placeholders can block indefinitely. A plan records the
            # inaccessible paper as blocked; execution must never omit its bytes.
            previous = None
            if hasattr(signal, "SIGALRM"):
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Source read timed out: {p}")
                previous = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
            try:
                result[str(p.relative_to(directory))] = fingerprint(p)
            finally:
                if previous is not None:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous)
    return result


def create_plan(config, manifest, output, variants, split, repeats=1):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    out = Path(output)
    plan_path = out / "plan.json"
    papers = [p for p in manifest["papers"] if p["split"] == split]
    if not papers:
        raise ValueError(f"No papers in split {split}")
    jobs = []
    for paper in papers:
        source_error = None
        try:
            hashes = source_hashes(paper["paper_dir"])
        except (TimeoutError, OSError) as exc:
            hashes = None
            source_error = str(exc)
        for variant in variants:
            cfg = variant_config(config, variant)
            for repeat in range(1, repeats+1):
                jobs.append({"doi": paper["doi"], "source_dir": paper["paper_dir"], "variant": variant,
                             "repeat": repeat, "split": split, "source_hashes": hashes,
                             "source_error": source_error,
                             "config": cfg, "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                             "output_dir": str(out / variant / f"repeat_{repeat}" / Path(paper["paper_dir"]).name)})
    plan = {"protocol_version": PROTOCOL_VERSION, "split": split, "repeats": repeats,
            "manifest": manifest, "jobs": jobs,
            "status": "planned", "publication_ready": False,
            "notes": "No network calls during planning. Outputs isolated from corpus and formal DB. Failure counts remain in denominators."}
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old != plan:
            raise ValueError("Plan already exists with different inputs; use a new output directory")
    else:
        save_json(plan_path, plan)
    return plan


def run_job(job):
    if job.get("source_error") or job.get("source_hashes") is None:
        raise ValueError(f"Unfrozen source input: {job.get('source_error') or job['doi']}")
    from llm.extractor import run_llm_on_paper_dir
    from llm.evaluator import run_llm_evaluation
    from postprocess.reference_resolver import load_references
    from postprocess.workflow import run_structure_normalization, run_evidence_linking, run_deterministic_validation, run_finalization
    from pipelines.decision_layer import build_ingest_gate_report
    directory = Path(job["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    status_file = directory / "experiment_status.json"
    if status_file.exists():
        old = json.loads(status_file.read_text())
        if old.get("config_sha256") == job["config_sha256"] and old.get("status") == "success":
            return old
    if source_hashes(job["source_dir"]) != job["source_hashes"]:
        raise ValueError(f"Source changed after plan freeze: {job['doi']}")
    for name in SOURCE_NAMES:
        src, dest = Path(job["source_dir"]) / name, directory / name
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dest)
    start = time.perf_counter()
    status = {"doi": job["doi"], "variant": job["variant"], "repeat": job["repeat"],
              "config_sha256": job["config_sha256"], "status": "running"}
    save_json(status_file, status)
    llm, pipeline = job["config"]["llm"], job["config"]["pipeline"]
    metrics = {}
    try:
        extracted_result = run_llm_on_paper_dir(
            paper_dir=str(directory), model_select=llm["model_select"], model_extract=llm["model_extract"],
            max_snippet_chars=int(llm["max_snippet_chars"]), max_context_chars=int(llm["max_context_chars"]),
            max_extract_retries=int(llm.get("max_extract_retries", 2)),
            two_pass_extraction=bool(llm["two_pass_extraction"]),
            compact_evidence=bool(llm.get("compact_evidence", True)),
            enable_source_enrichment=bool(llm.get("enable_source_enrichment", False)),
            direct_image_table_input=bool(llm.get("direct_image_table_input", True)),
            image_download_api_key=os.environ.get("ELSEVIER_API_KEY"))
        extracted = extracted_result["extracted"]
        metrics["extraction"] = extracted_result.get("metrics", {})
        save_json(directory / "materials_extracted.extractor_raw.json", extracted)
        ref = directory / "references.json"
        extracted, structure = run_structure_normalization(extracted, paper_dir=str(directory), doi_hint=job["doi"],
                                                           reference_map=load_references(str(ref)) if ref.exists() else None)
        extracted, evidence = run_evidence_linking(extracted, paper_dir=str(directory))
        evaluation, eval_metrics = run_llm_evaluation(
            paper_dir=str(directory), extracted_json=extracted, model_evaluate=llm["model_evaluate"],
            evaluator_mode=llm["evaluator_mode"], max_context_chars=int(llm.get("max_evaluate_context_chars", 18000)),
            parameter_limit=int(llm.get("evaluate_parameter_limit", 40)),
            field_batch_size=int(llm.get("evaluate_parameter_batch_size", 12)),
            per_evidence_chars=int(llm.get("evaluate_evidence_chars", 300)),
            max_retries=int(llm.get("max_evaluate_retries", 2)), evidence_report=evidence.get("evidence_grounding"),
            feedback_artifact_path=None)
        metrics["evaluation"] = eval_metrics
        extracted, validation = run_deterministic_validation(extracted, skip_quality_checks=pipeline["skip_quality_checks"])
        extracted, final = run_finalization(extracted, evaluation_report=evaluation, quality_report=validation.get("quality_checks"))
        gate = build_ingest_gate_report(evaluation_report=evaluation, confidence_report=final["confidence_fusion"],
                                        extracted_json=extracted, enabled=True, blocked_verdicts=["rejected"],
                                        min_document_confidence_score=float(pipeline.get("db_ingest_min_document_confidence_score", 65)))
        save_json(directory / "materials_extracted.json", extracted)
        save_json(directory / "llm_evaluation.json", evaluation)
        save_json(directory / "postprocess_report.json", {"record_id": job["doi"], **structure, **evidence, **validation,
                                                          **final, "ingest_gate": gate, "pipeline_metrics": metrics})
        status["status"] = "success"
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    status.update(seconds=round(time.perf_counter()-start, 3), metrics=metrics,
                  cost_note="Usage is provider-reported where available; failed/retried calls may be billed without returned usage.")
    save_json(status_file, status)
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.example.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    ap.add_argument("--split", choices=["development", "calibration", "test"], default="development")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--diagnostic", action="store_true", help="Allow AI/pending labels; never a publication ablation")
    args = ap.parse_args()
    import yaml
    manifest = read_manifest(args.manifest)
    plan = create_plan(yaml.safe_load(Path(args.config).read_text()), manifest, args.outdir, args.variants, args.split, args.repeats)
    if not args.execute:
        print(f"Planned {len(plan['jobs'])} isolated runs; no API calls made.")
        return
    if not args.diagnostic and any(p.get("reviewer_type") != "human_expert" or p.get("annotation_status") != "adjudicated"
                                   or not p.get("exhaustive") for p in manifest["papers"] if p["split"] == args.split):
        raise SystemExit("Benchmark lock incomplete; use --diagnostic only for explicitly non-publication experiments")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set; plan saved, no model calls executed")
    results = []
    for job in plan["jobs"]:
        result = run_job(job)
        results.append(result)
        save_json(Path(args.outdir) / "run_summary.json", results)
        print(f"{job['variant']} {job['doi']}: {result['status']}", flush=True)
    if any(r["status"] != "success" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
