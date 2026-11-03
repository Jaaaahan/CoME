#!/usr/bin/env python3
"""Local ENM random search: GPU generation and resumable API judging on separate hosts."""
from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_enm_sweep import (
    DEFAULTS, FIXED, PROTOCOL, ROOT, RetryingJudge, backend_from_env, digest,
    evaluate_trial, evaluation_set, load_settings, read_json, trial, validate_parameters, write_json,
)

SEARCH_PROTOCOL = "enm-local-random-full-1"


def sample_parameters(space, count, seed):
    if set(space) != set(DEFAULTS):
        raise ValueError("search space must specify exactly the five ENM parameters")
    keys = sorted(space)
    candidates, seen = [], set()
    for values in itertools.product(*(space[key] for key in keys)):
        params = dict(zip(keys, values))
        validate_parameters(params)
        canonical = json.dumps(params, sort_keys=True)
        if params != DEFAULTS and canonical not in seen:
            candidates.append(params)
            seen.add(canonical)
    if not 1 <= count <= len(candidates):
        raise ValueError("random trial count exceeds the unique non-baseline configurations")
    return [dict(DEFAULTS), *random.Random(seed).sample(candidates, count)]


def create_plan(settings_path, search_path, output):
    search = read_json(search_path)
    settings = load_settings(settings_path.resolve())
    settings.setdefault("wandb", {})["mode"] = "disabled"
    rows = sample_parameters(search["parameters"], search["random_trials"], search["seed"])
    plan = dict(protocol=SEARCH_PROTOCOL, reader_protocol=PROTOCOL, seed=search["seed"],
        objective="tune/judge_accuracy", goal="maximize", settings=settings, fixed=FIXED,
        trials=[dict(index=i, id="baseline" if i == 0 else f"trial-{i:03d}", parameters=params)
                for i, params in enumerate(rows)])
    if output.exists() and read_json(output) != plan:
        raise ValueError("plan already exists with different settings; choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, plan)
    return plan


def read_plan(path):
    plan = read_json(path)
    if plan.get("protocol") != SEARCH_PROTOCOL or plan.get("reader_protocol") != PROTOCOL:
        raise ValueError("incompatible local search plan")
    for i, row in enumerate(plan["trials"]):
        expected = "baseline" if i == 0 else f"trial-{i:03d}"
        if row.get("index") != i or row.get("id") != expected:
            raise ValueError("invalid trial identity/order")
        validate_parameters(row["parameters"])
    return plan


def generate(plan_path, index, base_url):
    plan = read_plan(plan_path)
    if not 0 <= index < len(plan["trials"]):
        raise ValueError("trial index out of range")
    row = plan["trials"][index]
    settings = json.loads(json.dumps(plan["settings"]))
    settings["reader"]["base_url"] = base_url
    settings["wandb"]["mode"] = "disabled"
    directory = Path(settings["output_root"]) / row["id"]
    directory.mkdir(parents=True, exist_ok=True)
    settings_path = directory / "settings.json"
    if settings_path.exists() and read_json(settings_path) != settings:
        raise ValueError("trial settings changed; preserve the reader endpoint when resuming")
    write_json(settings_path, settings)
    trial(argparse.Namespace(settings=settings_path, run_id=row["id"], wandb_mode="disabled",
        dry_run=False, generation_only=True, **row["parameters"]))


def verify_download(directory, expected_ids, questions_hash, params, judge):
    ready = read_json(directory / "generation-complete.json")
    contract = read_json(directory / "contract.json")
    if (ready.get("status") != "generation_complete" or ready.get("protocol") != PROTOCOL
            or ready.get("questions") != len(expected_ids) or ready.get("questions_sha256") != questions_hash
            or ready.get("judge_input_sha256") != digest(directory / "predictions.jsonl")
            or ready.get("contract_sha256") != digest(directory / "contract.json")
            or contract.get("parameters") != params or contract.get("fixed") != FIXED
            or contract.get("dataset", {}).get("questions_sha256") != questions_hash
            or contract.get("settings", {}).get("judge") != judge
            or contract.get("judge_endpoint") != "https://api.openai.com/v1"):
        raise ValueError("downloaded answers do not match the full dataset, parameters or judge contract")
    return ready


def fetch_file(host, remote, local):
    temporary = local.with_suffix(local.suffix + ".download")
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                    f"{host}:{remote}", str(temporary)], check=True)
    temporary.replace(local)


def available_trials(host, remote_root, trials):
    # Paths/IDs come from the validated plan, and are shell quoted explicitly.
    command = "\n".join(
        f"if test -f {shlex.quote(str(remote_root / 'trials' / row['id'] / 'generation-complete.json'))}; "
        f"then printf '%s\\n' {shlex.quote(row['id'])}; fi" for row in trials)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
                            check=True, capture_output=True, text=True)
    return set(result.stdout.splitlines())


def leaderboard(output, plan):
    rows = []
    for entry in plan["trials"]:
        directory = output / "trials" / entry["id"]
        row = dict(entry)
        metrics = directory / "metrics.json"
        if metrics.exists():
            row.update(status="scored", metrics=read_json(metrics))
        elif (directory / "generation-complete.json").exists():
            row["status"] = "awaiting_judge"
        else:
            row["status"] = "awaiting_generation"
        rows.append(row)
    scored = sorted((row for row in rows if row["status"] == "scored"),
                    key=lambda row: (-row["metrics"]["tune/judge_accuracy"], row["index"]))
    report = dict(protocol=SEARCH_PROTOCOL, objective=plan["objective"], score_role="tuning_set_accuracy",
        judge_model=plan["settings"]["judge"]["model"], total_trials=len(rows), scored_trials=len(scored),
        best=scored[0] if scored else None, ranking=scored, trials=rows)
    write_json(output / "leaderboard.json", report)
    return report


def watch(args):
    plan = read_plan(args.plan)
    ids, dataset = evaluation_set(args.bundle)
    judge = plan["settings"]["judge"]
    backend = RetryingJudge(backend_from_env(model=judge["model"], base_url="https://api.openai.com/v1",
        api_key_env=judge["api_key_env"], timeout_seconds=judge["timeout_seconds"]), judge["retries"])
    if not backend.backend.api_key:
        raise ValueError("missing judge API key")
    args.output.mkdir(parents=True, exist_ok=True)
    contract = dict(protocol=SEARCH_PROTOCOL, plan_sha256=digest(args.plan),
        dataset=dataset, remote_host=args.ssh_host, remote_root=str(args.remote_root),
        judge=judge, implementation_sha256=digest(Path(__file__)))
    with (args.output / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / "watch-contract.json"
        if path.exists() and read_json(path) != contract:
            raise ValueError("watcher contract changed; use a new output directory")
        write_json(path, contract)
        settings = {**plan["settings"], "bundle": str(args.bundle.resolve())}
        while True:
            try:
                available = available_trials(args.ssh_host, args.remote_root, plan["trials"])
                for row in plan["trials"]:
                    directory = args.output / "trials" / row["id"]
                    if row["id"] not in available or (directory / "metrics.json").exists():
                        continue
                    directory.mkdir(parents=True, exist_ok=True)
                    try:
                        if not (directory / "download-verified.json").exists():
                            remote = args.remote_root / "trials" / row["id"]
                            for source, target in (("contract.json", "contract.json"),
                                    ("judge-input.jsonl", "predictions.jsonl"),
                                    ("generation-complete.json", "generation-complete.json")):
                                fetch_file(args.ssh_host, remote / source, directory / target)
                            ready = verify_download(directory, ids, dataset["questions_sha256"], row["parameters"], judge)
                            write_json(directory / "download-verified.json", ready)
                        else:
                            verify_download(directory, ids, dataset["questions_sha256"], row["parameters"], judge)
                        metrics = evaluate_trial(settings, directory, ids, backend)
                        write_json(directory / "metrics.json", metrics)
                        print(json.dumps(dict(event="trial_scored", trial=row["id"], metrics=metrics)), flush=True)
                    except Exception as exc:
                        # Do not lose earlier successful judgments or expose credentials in logs.
                        message = str(exc).replace(backend.backend.api_key, "[REDACTED]")
                        write_json(directory / "judge-error.json", dict(error_type=type(exc).__name__, message=message[:1000]))
                        print(json.dumps(dict(event="trial_judge_pending", trial=row["id"], error_type=type(exc).__name__)), flush=True)
                report = leaderboard(args.output, plan)
                write_json(args.output / "status.json", dict(phase="complete" if report["scored_trials"] == report["total_trials"] else "watching",
                    pid=os.getpid(), updated_at=time.time(), available_trials=len(available),
                    scored_trials=report["scored_trials"], total_trials=report["total_trials"]))
                if args.once or report["scored_trials"] == report["total_trials"]:
                    return
            except (OSError, subprocess.SubprocessError) as exc:
                print(json.dumps(dict(event="transfer_retry", error_type=type(exc).__name__)), flush=True)
                if args.once:
                    raise
            time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("plan")
    prepare.add_argument("--settings", type=Path, required=True)
    prepare.add_argument("--search", type=Path, default=ROOT / "config/enm_mem_gallery_local_search.json")
    prepare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("generate")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--base-url", required=True)
    judge = sub.add_parser("watch")
    judge.add_argument("--plan", type=Path, required=True)
    judge.add_argument("--bundle", type=Path, required=True)
    judge.add_argument("--ssh-host", default="4090cuhk-Storage")
    judge.add_argument("--remote-root", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--poll-seconds", type=int, default=120)
    judge.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        plan = create_plan(args.settings, args.search, args.output)
        print(json.dumps(dict(plan=str(args.output), trials=len(plan["trials"]), seed=plan["seed"])))
    elif args.command == "generate":
        generate(args.plan, args.index, args.base_url)
    else:
        if args.poll_seconds < 30:
            parser.error("poll-seconds must be at least 30")
        watch(args)


if __name__ == "__main__":
    main()
