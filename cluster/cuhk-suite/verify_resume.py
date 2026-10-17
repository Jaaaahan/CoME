"""Verify the frozen source and existing outputs before resuming inference."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mm_memory_bench.methods.memory_backend as backend

experiment = Path(os.environ["EXPERIMENT_ROOT"])
source = Path(os.environ["PROJECT_ROOT"])
data = Path(os.environ["DATA_ROOT"]) / "data/unified"
expected_hash = "2d8045d969183bc19593f8301223c8aa4167dd4ad0c2c140902d763eab901bfa"
assert Path(backend.__file__).resolve().is_relative_to(source.resolve())
assert backend.backend_hash("standalone") == expected_hash
for line in (source / "STANDALONE_MANIFEST.sha256").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected, relative
states = list((experiment / "artifacts/checkpoints").glob("*/contexts/*/state.json"))
assert len(states) == 37, len(states)
for path in states:
    assert json.loads(path.read_text())["config"]["core_sha256"] == expected_hash, str(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


counts = {}
for dataset, bundle, method, expected in [
    ("atm", "atm_bench", "graphmemix", 1044),
    ("atm", "atm_bench", "dense", 1044),
    ("memeye", "memeye", "graphmemix", 1855),
    ("memeye", "memeye", "dense", 337),
]:
    relative = (f"runs/qwen3vl8b/{dataset}/reader/predictions.jsonl"
                if method == "graphmemix" else f"runs/dense/{dataset}/predictions.jsonl")
    predictions = rows(experiment / relative)
    question_ids = {row["question_id"] for row in rows(data / bundle / "questions.jsonl")}
    ids = {row["question_id"] for row in predictions}
    assert len(predictions) == len(ids), relative
    if dataset == "memeye" and method == "dense":
        assert expected <= len(ids) <= len(question_ids), relative
    else:
        assert len(ids) == expected, relative
    assert ids <= question_ids
    if not (dataset == "memeye" and method == "dense"):
        assert ids == question_ids
    assert all(row.get("prediction", "").strip() and
               row.get("metadata", {}).get("status") != "error" for row in predictions)
    counts[f"{dataset}/{method}"] = len(ids)
# Exercise the original reader's entire resume validation using a copy of the
# saved answers and an empty question allowlist. This makes no model requests
# and never edits the live prediction file or relaxes its provenance checks.
with tempfile.TemporaryDirectory(prefix="reader-contract-", dir=experiment / "validation") as tmp:
    temp = Path(tmp)
    predictions_copy = temp / "predictions.jsonl"
    shutil.copyfile(experiment / "runs/dense/memeye/predictions.jsonl", predictions_copy)
    before = predictions_copy.read_bytes()
    allowlist = temp / "no-questions.txt"
    allowlist.write_text("")
    checked = subprocess.run([
        sys.executable, str(source / "scripts/run_memix_answers_from_retrieval.py"),
        "--bundle", str(data / "memeye"),
        "--retrieval", str(experiment / "runs/qwen3vl8b/memeye/atomic_retrieval.jsonl"),
        "--checkpoint-root", str(experiment / "artifacts/checkpoints/memeye"),
        "--backend", "standalone", "--output", str(predictions_copy),
        "--base-url", os.environ["BASE_URL"], "--model", "Qwen/Qwen3-VL-8B-Instruct",
        "--concurrency", "2", "--max-model-len", "32768",
        "--max-output-tokens", "1000", "--reasoning-effort", "none",
        "--overflow-image-max-edge", "2500", "--reader-ocr-budget-chars", "320",
        "--allow-variable-retrieval", "--resume", "--record-errors",
        "--question-ids", str(allowlist),
    ], cwd=source, check=True, text=True, capture_output=True)
    assert predictions_copy.read_bytes() == before
    reader_check = json.loads(checked.stdout.strip().splitlines()[-1])
    assert reader_check["new_answers"] == 0 and not reader_check["failures"]
report = {"source": str(source), "backend_sha256": expected_hash,
          "checkpoint_contexts_verified": len(states), "existing_answers": counts,
          "original_reader_resume_validation": "passed",
          "reader_base_url": os.environ["BASE_URL"],
          "resume_from_job": "78262", "job_id": os.environ["SLURM_JOB_ID"]}
output = experiment / "validation" / f"resume-{os.environ['SLURM_JOB_ID']}.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
