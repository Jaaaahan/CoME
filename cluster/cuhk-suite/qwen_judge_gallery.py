"""Judge frozen Mem-Gallery predictions with the released rubric and local Qwen."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
import urllib.request

from mm_memory_bench import judge as original


MODEL = "Qwen/Qwen3-VL-8B-Instruct"
REQUEST_SETTINGS = {
    "temperature": 0, "seed": 0, "max_tokens": 256,
    "response_format": {"type": "json_object"},
}


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class QwenJudge:
    model = MODEL

    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.calls = 0
        self.lock = threading.Lock()

    def judge(self, item):
        body = {
            "model": self.model, **REQUEST_SETTINGS,
            "messages": [
                {"role": "system", "content": original.JUDGE_RUBRIC},
                {"role": "user", "content": json.dumps(dict(item), ensure_ascii=False, sort_keys=True)},
            ],
        }
        failure = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    self.base_url + "/chat/completions",
                    data=json.dumps(body, ensure_ascii=False).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = json.load(response)
                choice = payload["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("judge response was truncated")
                parsed = original._json_from_text(choice["message"]["content"])
                original._normalized_judgment(parsed)
                with self.lock:
                    self.calls += 1
                    if self.calls % 100 == 0:
                        emit("judge_api_progress", successful_calls=self.calls)
                return parsed
            except Exception as exc:
                failure = exc
                if attempt < 2:
                    time.sleep(attempt + 1)
        raise RuntimeError(f"judge failed after 3 attempts: {failure}") from failure


def validate_output(path, questions, predictions):
    if not path.exists():
        return False
    try:
        records = read_rows(path)
    except (ValueError, OSError):
        return False
    ids = [str(row.get("question_id", "")) for row in records]
    if len(ids) != len(set(ids)) or set(ids) != set(questions):
        return False
    for record in records:
        qid = record["question_id"]
        prediction_hash = hashlib.sha256(json.dumps(
            predictions[qid], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if (record.get("status") != "ok" or record.get("judge_model") != MODEL
                or record.get("judge_protocol") != original.JUDGE_PROTOCOL_VERSION
                or record.get("judge_rubric_sha256") != original.JUDGE_RUBRIC_SHA256
                or record.get("prediction_row_sha256") != prediction_hash):
            return False
        if not isinstance(record.get("correct"), bool):
            return False
        if record.get("score") != float(record["correct"]):
            return False
    return True


def wait_ready(base_url, timeout):
    deadline = time.monotonic() + timeout
    attempts = 0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/models", timeout=5) as response:
                models = json.load(response)
            assert any(row["id"] == MODEL for row in models["data"]), models
            return models
        except Exception:
            if attempts % 4 == 0:
                emit("waiting_for_judge_service", timeout_seconds=timeout)
            attempts += 1
            time.sleep(15)
    raise TimeoutError("judge service did not become ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18262/v1")
    parser.add_argument("--wait-seconds", type=int, default=3600)
    parser.add_argument("--check-complete", action="store_true")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    experiment = Path(os.environ["EXPERIMENT_ROOT"])
    data = Path(os.environ["DATA_ROOT"]) / "data/unified"
    output = experiment / "judgments/qwen3vl8b-mem-gallery"
    output.mkdir(parents=True, exist_ok=True)
    # One writer; the afterany fallback starts only after the reuse job is gone.
    with (output / "judge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries = []
        inputs = {}
        for dataset, method in [("mem_gallery", "graphmemix"), ("mem_gallery", "dense")]:
            bundle = data / dataset
            prediction = (experiment / f"runs/qwen3vl8b/{dataset}/reader/predictions.jsonl"
                          if method == "graphmemix" else experiment / f"runs/dense/{dataset}/predictions.jsonl")
            qs = read_rows(bundle / "questions.jsonl")
            ps = read_rows(prediction)
            questions = {row["question_id"]: row for row in qs}
            predictions = {row["question_id"]: row for row in ps}
            expected = 1711
            assert len(qs) == len(ps) == len(questions) == len(predictions) == expected
            assert set(questions) == set(predictions)
            assert all(row.get("prediction", "").strip() and
                       row.get("metadata", {}).get("status") != "error" for row in ps)
            key = f"{dataset}/{method}"
            inputs[key] = {"predictions_sha256": digest(prediction),
                           "questions_sha256": digest(bundle / "questions.jsonl"), "count": expected}
            entries.append((key, bundle, prediction, output / dataset / f"{method}.jsonl", questions, predictions))

        weights = Path(os.environ["SUITE_ROOT"]) / "qwen-weights-reference.json"
        contract = {
            "label": "Qwen3-VL-8B Judge Accuracy (not paper GPT-5-mini J.Acc.)",
            "model": MODEL, "model_reference": json.loads(weights.read_text()),
            "judge_protocol": original.JUDGE_PROTOCOL_VERSION,
            "judge_rubric_sha256": original.JUDGE_RUBRIC_SHA256,
            "judge_module_sha256": digest(Path(original.__file__)),
            "runner_sha256": digest(Path(__file__)),
            "request_settings": REQUEST_SETTINGS, "inputs": inputs,
            "request_timeout_seconds": 600, "attempts_per_request": 3,
            "provider": "CUHK local vLLM", "answer_model": MODEL,
            "same_model_as_answer_generator": True, "official_gpt5mini_judge": False,
        }
        contract_path = output / "contract.json"
        if contract_path.exists():
            assert json.loads(contract_path.read_text()) == contract, "judge contract changed; use a separate output root"
        else:
            atomic_json(contract_path, contract)

        complete = all(validate_output(path, qs, ps) for _, _, _, path, qs, ps in entries)
        if args.check_complete:
            emit("judge_completion_check", complete=complete)
            return 0 if complete else 3

        if not complete:
            models = wait_ready(args.base_url.rstrip("/"), args.wait_seconds)
            backend = QwenJudge(args.base_url)
            probes = []
            for answer, expected in [("Paris", True), ("Rome", False)]:
                question = {"prompt": [{"type": "text", "text": "What is the capital of France?"}],
                            "answer": {"text": "Paris"}, "task": {"response_type": "text"}}
                actual = backend.judge(original.judge_item(question, answer))["correct"]
                assert actual is expected, (answer, actual)
                probes.append({"prediction": answer, "expected": expected, "actual": actual})
            atomic_json(output / f"startup-{os.environ['SLURM_JOB_ID']}.json", {
                "models": models, "base_url": args.base_url, "node": socket.gethostname(),
                "job_id": os.environ["SLURM_JOB_ID"], "probes": probes,
            })
            emit("judge_probes_passed", probes=probes)
            for key, bundle, prediction, path, qs, ps in entries:
                if validate_output(path, qs, ps):
                    emit("judge_reused_complete", dataset_method=key)
                    continue
                for attempt in range(3):
                    emit("judge_dataset_start", dataset_method=key, attempt=attempt + 1)
                    summary = original.judge_predictions(
                        backend, bundle, prediction, path, resume=True, concurrency=args.concurrency)
                    emit("judge_dataset_end", dataset_method=key, summary=summary)
                    if validate_output(path, qs, ps):
                        break
                else:
                    raise RuntimeError(f"{key}: invalid or failed judgments remain after retry passes")

        results = {}
        for key, _, _, path, qs, ps in entries:
            assert validate_output(path, qs, ps), key
            summary = original.summarize_judgments(read_rows(path))
            results[key] = {**summary, "judge_accuracy_percent": 100 * summary["accuracy_conservative"],
                            "judgments_sha256": digest(path)}
        report = {"contract": contract, "complete": True, "results": results,
                  "completed_at_unix": time.time(), "job_id": os.environ["SLURM_JOB_ID"]}
        atomic_json(output / "results.json", report)
        emit("QWEN_JUDGE_COMPLETE", results=results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
