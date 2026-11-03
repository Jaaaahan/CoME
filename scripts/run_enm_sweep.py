#!/usr/bin/env python3
"""Full-dataset MemoryGallery ENM tuning, scored by GPT-5.4 mini and logged to W&B.

Every trial uses all questions in the bundle; no development/test split is used.
Use `trial` directly or through a W&B sweep agent. --dry-run makes no API calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

# Also work when launched by `wandb agent` without an editable install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mm_memory_bench.core import iter_jsonl, read_json
from mm_memory_bench.judge import (
    JUDGE_PROTOCOL_VERSION, JUDGE_RUBRIC_SHA256, backend_from_env,
    _normalized_judgment, judge_predictions,
)

PROTOCOL = "enm-mem-gallery-full-sweep-2"
DEFAULTS = dict(query_threshold=.95, source_top_l=24, candidate_limit=48,
                semantic_k=8, query_table_size=20)
FIXED = dict(batch_size=2, max_rounds=10, max_llm_calls=10, decision_retries=3)
READER_DEFAULTS = dict(model="Qwen/Qwen3-VL-8B-Instruct", base_url="http://127.0.0.1:8000/v1",
                       image_max_edge=1280, max_model_len=32768, max_output_tokens=1000,
                       reader_ocr_budget_chars=320)
PATH_KEYS = ("bundle", "output_root", "observations", "checkpoint_root",
             "source_priors", "relation_store", "embedding_model")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def gallery_questions(bundle):
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("benchmark") not in ("Mem-Gallery", "MemEye"):
        raise ValueError("this sweep requires a canonical Mem-Gallery or MemEye bundle")
    path = bundle / manifest.get("tables", {}).get("questions", "questions.jsonl")
    if path.resolve() != (bundle / "questions.jsonl").resolve():
        raise ValueError("adaptive reader requires the canonical questions.jsonl path")
    rows = list(iter_jsonl(path))
    if not rows or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("questions must be nonempty with unique IDs")
    return rows


def evaluation_set(bundle):
    """Bind tuning to every question in the canonical bundle, without sampling."""
    rows = gallery_questions(bundle)
    ids = {r["question_id"] for r in rows}
    manifest = dict(scope="full_dataset", score_role="tuning_set_accuracy",
                    questions=len(ids), questions_sha256=digest(bundle / "questions.jsonl"),
                    by_type=dict(Counter(r.get("task", {}).get("subcategory", "unknown") for r in rows)))
    return ids, manifest


def load_settings(path):
    settings = read_json(path)
    if "split_dir" in settings:
        raise ValueError("remove split_dir from settings: this sweep now uses all bundle questions")
    if set(settings) - {*PATH_KEYS, "reader", "judge", "wandb"}:
        raise ValueError("unknown settings; credentials belong in environment variables")
    for key in PATH_KEYS:
        value = Path(settings[key]).expanduser()
        settings[key] = str((path.parent / value).resolve())
    reader = settings.get("reader", {})
    if set(reader) - READER_DEFAULTS.keys():
        raise ValueError("unknown reader settings; ENM budgets are fixed for this sweep")
    settings["reader"] = {**READER_DEFAULTS, **reader}
    judge = {"provider": "openai", "model": "gpt-5.4-mini", "base_url_env": "OPENAI_BASE_URL",
             "api_key_env": "OPENAI_API_KEY", "concurrency": 4, "retries": 3, "timeout_seconds": 120,
             **settings.get("judge", {})}
    if set(judge) - {"provider", "model", "base_url_env", "api_key_env", "concurrency", "retries", "timeout_seconds"}:
        raise ValueError("unknown judge settings; use environment variables for endpoint and API key")
    if judge["provider"] != "openai" or not judge["model"]:
        raise ValueError("expected openai with an explicit model identifier")
    if min(judge["concurrency"], judge["retries"], judge["timeout_seconds"]) <= 0:
        raise ValueError("judge concurrency, retries and timeout must be positive")
    settings["judge"] = judge
    return settings


def validate_parameters(params):
    if not -1 <= params["query_threshold"] <= 1:
        raise ValueError("query_threshold must be in [-1, 1]")
    if not 0 < params["source_top_l"] <= params["candidate_limit"]:
        raise ValueError("require 0 < source_top_l <= candidate_limit")
    if not 1 <= params["query_table_size"] <= 20 or params["semantic_k"] < 1:
        raise ValueError("require query_table_size in 1..20 and positive semantic_k")


def validate_artifacts(settings, ids, params):
    for name in ("checkpoint_root", "embedding_model"):
        if not Path(settings[name]).is_dir():
            raise ValueError(f"missing {name} directory")
    for name in ("observations", "source_priors", "relation_store"):
        if not Path(settings[name]).is_file():
            raise ValueError(f"missing {name} file")
    contexts = {r["context_id"] for r in gallery_questions(Path(settings["bundle"])) if r["question_id"] in ids}
    found = set()
    for row in iter_jsonl(Path(settings["relation_store"])):
        if row.get("record_type") == "unit_graph" and row.get("context_id") in contexts:
            if params["semantic_k"] > row.get("max_semantic_k", 0):
                raise ValueError("semantic_k exceeds stored max_semantic_k; rebuild the relation store")
            found.add(row["context_id"])
    if found != contexts:
        raise ValueError("missing tuning contexts in relation store")
    found = set()
    for row in iter_jsonl(Path(settings["source_priors"])):
        qid = row.get("question_id")
        if qid in ids:
            if qid in found:
                raise ValueError("duplicate tuning question in source priors")
            hits = {h["unit_id"] for h in row.get("unit_hits", [])}
            if len(hits) < params["source_top_l"]:
                raise ValueError(f"{qid}: source_top_l exceeds stored unit hits; regenerate priors or lower it")
            found.add(qid)
    if found != ids:
        raise ValueError("missing tuning questions in source priors")


def reader_command(settings, params, output):
    command = [sys.executable, str(ROOT / "scripts/run_graphmemix_adaptive_reader.py")]
    values = {k: settings[k] for k in ("bundle", "observations", "checkpoint_root", "source_priors",
                                      "relation_store", "embedding_model")}
    values.update(settings["reader"])
    values.update(params)
    values.update(FIXED)
    values.update(initial_limit=params["candidate_limit"], backend="standalone", output=str(output))
    for name, value in values.items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    return command + ["--resume"]


class RetryingJudge:
    def __init__(self, backend, retries):
        self.backend, self.retries, self.model = backend, retries, backend.model

    def judge(self, item):
        for attempt in range(self.retries):
            try:
                value = self.backend.judge(item)
                _normalized_judgment(value)
                return value
            except Exception:
                if attempt + 1 == self.retries:
                    raise
                time.sleep(min(2 ** attempt, 8))


def evaluate_trial(settings, trial_dir, expected_ids, backend):
    all_ids, _ = evaluation_set(Path(settings["bundle"]))
    if expected_ids != all_ids:
        raise ValueError("expected IDs must include every question in the bundle")
    path = trial_dir / "predictions.jsonl"
    predictions = list(iter_jsonl(path))
    ids = [r["question_id"] for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("predictions must cover exactly the full tuning dataset")
    summary = judge_predictions(backend, Path(settings["bundle"]), path, trial_dir / "judgments.jsonl",
        concurrency=settings["judge"]["concurrency"])
    if summary["failed_judgments"] or summary["total_predictions"] != len(expected_ids):
        raise RuntimeError("judge incomplete: resume this trial to retry failed items; objective was not logged")
    metrics = {"tune/judge_accuracy": summary["accuracy_conservative"],
               "tune/questions": len(ids), "tune/failed_judgments": 0,
               "tune/semantic_macro_accuracy": summary["semantic_question_macro_accuracy"]}
    for key in ("llm_calls", "visited_units", "elapsed_seconds", "query_rejection_count"):
        metrics["tune/mean_" + key] = sum(r["metadata"][key] for r in predictions) / len(ids)
    metrics["tune/answer_failure_rate"] = sum(
        bool(r["metadata"].get("error_type")) or r["metadata"].get("status") not in (None, "ok")
        for r in predictions) / len(ids)
    return metrics


def export_judge_input(directory, expected_ids, dataset):
    """Publish a small, hash-bound answer file after full generation completes."""
    source = directory / "predictions.jsonl"
    seen = set()
    output = directory / "judge-input.jsonl"
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(source):
            qid = row["question_id"]
            if qid in seen or qid not in expected_ids:
                raise ValueError("generation contains duplicate or unexpected question IDs")
            seen.add(qid)
            metadata = row["metadata"]
            compact = dict(question_id=qid, prediction=row["prediction"], metadata={
                key: metadata[key] for key in ("status", "error_type", "llm_calls", "visited_units",
                    "elapsed_seconds", "query_rejection_count") if key in metadata})
            compact["source_prediction_row_sha256"] = hashlib.sha256(json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            handle.write(json.dumps(compact, ensure_ascii=False) + "\n")
    if seen != expected_ids:
        raise ValueError("generation must finish all tuning questions before publishing")
    temporary.replace(output)
    ready = dict(protocol=PROTOCOL, status="generation_complete", questions=len(seen),
        questions_sha256=dataset["questions_sha256"], predictions_sha256=digest(source),
        judge_input_sha256=digest(output), contract_sha256=digest(directory / "contract.json"))
    write_json(directory / "generation-complete.json", ready)
    return ready


def trial(args):
    settings = load_settings(args.settings.resolve())
    params = {name: getattr(args, name) for name in DEFAULTS}
    validate_parameters(params)
    ids, dataset = evaluation_set(Path(settings["bundle"]))
    validate_artifacts(settings, ids, params)
    judge = settings["judge"]
    base_url = (os.environ.get(judge["base_url_env"]) or "https://api.openai.com/v1").rstrip("/")
    trial_id = args.run_id or os.environ.get("WANDB_RUN_ID") or uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", trial_id):
        raise ValueError("run-id may contain only letters, digits, underscores and hyphens")
    directory = Path(settings["output_root"]) / trial_id
    command = reader_command(settings, params, directory / "predictions.jsonl")
    if args.dry_run:
        print(json.dumps(dict(run_id=trial_id, tuning_questions=len(ids), evaluation_scope="full_dataset", parameters=params, fixed=FIXED,
                              command=command, judge_model=judge["model"]), indent=2))
        return
    generation_only = getattr(args, "generation_only", False)
    if generation_only and (args.wandb_mode or settings.get("wandb", {}).get("mode", "online")) != "disabled":
        raise ValueError("generation-only requires --wandb-mode disabled")
    if not generation_only:
        judge_backend = backend_from_env(model=judge["model"], base_url=base_url,
            api_key_env=judge["api_key_env"], timeout_seconds=judge["timeout_seconds"])
        if not judge_backend.api_key:
            raise ValueError(f"set {judge['api_key_env']} before running the judge")
    hashes = {name: digest(settings[name]) for name in ("observations", "source_priors", "relation_store")}
    for name in ("manifest", "questions", "memories", "assets"):
        suffix = ".json" if name == "manifest" else ".jsonl"
        hashes[name] = digest(Path(settings["bundle"]) / (name + suffix))
    # Include all local reader/judge source dependencies, not just the entry point.
    code = {str(p.relative_to(ROOT)): digest(p)
            for root in (ROOT / "src", ROOT / "scripts") for p in sorted(root.rglob("*.py"))}
    contract = dict(protocol=PROTOCOL, settings={k: v for k, v in settings.items() if k != "wandb"},
                    parameters=params, fixed=FIXED, initial_limit=params["candidate_limit"],
                    dataset=dataset, input_hashes=hashes, code_hashes=code,
                    judge_endpoint=base_url, judge_protocol=JUDGE_PROTOCOL_VERSION,
                    judge_rubric_sha256=JUDGE_RUBRIC_SHA256)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "trial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract_path = directory / "contract.json"
        if contract_path.exists() and read_json(contract_path) != contract:
            raise ValueError("trial contract changed; use a new run-id/output directory")
        write_json(contract_path, contract)
        wb = settings.get("wandb", {})
        mode = args.wandb_mode or wb.get("mode", "online")
        context = nullcontext(None)
        if mode != "disabled":
            import wandb
            context = wandb.init(project=wb.get("project", "enm-mem-gallery"), entity=wb.get("entity"),
                id=trial_id, resume="allow", mode=mode, dir=str(directory),
                config={**params, **FIXED, "initial_limit": params["candidate_limit"],
                        "judge_model": judge["model"], "judge_provider": judge["provider"],
                        "evaluation_scope": dataset["scope"], "score_role": dataset["score_role"],
                        "tuning_questions": len(ids), "questions_sha256": dataset["questions_sha256"],
                        "contract_sha256": digest(contract_path)})
        with context as run:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), str(ROOT / "scripts"),
                                                                       environment.get("PYTHONPATH")]))
            print(json.dumps(dict(event="enm_trial", run_id=trial_id, directory=str(directory),
                                  tuning_questions=len(ids), evaluation_scope="full_dataset", parameters=params)), flush=True)
            with (directory / "reader.log").open("a") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            ready = export_judge_input(directory, ids, dataset)
            if generation_only:
                print(json.dumps(dict(run_id=trial_id, generation=ready)), flush=True)
                return
            backend = RetryingJudge(judge_backend, judge["retries"])
            metrics = evaluate_trial(settings, directory, ids, backend)
            write_json(directory / "metrics.json", metrics)
            if run is not None:
                # Log the optimization metric once, only after every tuning item is judged.
                run.log(metrics)
            print(json.dumps(dict(run_id=trial_id, metrics=metrics), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("trial", help="run ENM, judge and report one parameter configuration")
    run.add_argument("--settings", type=Path, required=True)
    run.add_argument("--run-id", help="reuse the same ID to resume an interrupted trial")
    run.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    run.add_argument("--dry-run", action="store_true", help="validate inputs and print the reader command; no API calls")
    run.add_argument("--generation-only", action="store_true", help="generate all answers for later judging; requires disabled W&B")
    for name, default in DEFAULTS.items():
        run.add_argument("--" + name.replace("_", "-"), "--" + name, dest=name,
                         type=type(default), default=default)
    args = parser.parse_args()
    trial(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
