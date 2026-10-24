import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from mm_memory_bench.core import BundleWriter, content_text
from mm_memory_bench.judge import OpenAICompatibleJudge, backend_from_env, judge_predictions


class FakeJudge:
    model = "fixture-judge"

    def __init__(self) -> None:
        self.items: list[Mapping[str, Any]] = []

    def judge(self, item: Mapping[str, Any]) -> Mapping[str, Any]:
        self.items.append(item)
        reference = str(item["reference_answer"]["text"])
        correct = str(item["prediction"]) == reference
        return {"correct": correct}


class JudgeTest(unittest.TestCase):
    def test_official_openai_judge_loads_project_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
            env_file.chmod(0o600)
            with patch("mm_memory_bench.judge._project_env_path", return_value=env_file):
                with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
                    backend = backend_from_env(model="gpt-5-mini", base_url="https://api.openai.com/v1",
                                               api_key_env="OPENAI_API_KEY", timeout_seconds=120)
            self.assertEqual(backend.api_key, "file-key")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
                backend = backend_from_env(model="gpt-5-mini", base_url="https://api.openai.com/v1",
                                           api_key_env="OPENAI_API_KEY", timeout_seconds=120)
            self.assertEqual(backend.api_key, "env-key")
            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
                backend = backend_from_env(model="local-judge", base_url="http://127.0.0.1:8000/v1",
                                           api_key_env="OPENAI_API_KEY", timeout_seconds=120)
            self.assertIsNone(backend.api_key)
            env_file.chmod(0o644)
            with patch("mm_memory_bench.judge._project_env_path", return_value=env_file):
                with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
                    with self.assertRaisesRegex(ValueError, "readable only by you"):
                        backend_from_env(model="gpt-5-mini", base_url="https://api.openai.com/v1",
                                         api_key_env="OPENAI_API_KEY", timeout_seconds=120)

    def test_gpt5_mini_request_omits_unsupported_temperature(self) -> None:
        response = json.dumps({"choices": [{"message": {"content": '{"correct": true}'}}]}).encode()
        with patch("mm_memory_bench.judge.urllib.request.urlopen") as open_url:
            open_url.return_value.__enter__.return_value.read.return_value = response
            result = OpenAICompatibleJudge(model="gpt-5-mini", api_key="test-key").judge({"prediction": "yes"})
            request = open_url.call_args.args[0]
            body = json.loads(request.data)
        self.assertEqual(result, {"correct": True})
        self.assertEqual(body["model"], "gpt-5-mini")
        self.assertNotIn("temperature", body)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")

    def test_question_id_allowlist_limits_reporting_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            with BundleWriter(bundle, {"benchmark": "fixture"}) as writer:
                writer.add_context({"context_id": "c", "benchmark": "fixture"})
                for question_id in ("q1", "q2"):
                    writer.add_question({
                        "question_id": question_id,
                        "context_id": "c",
                        "subset": "default",
                        "prompt": [content_text("question")],
                        "task": {"response_type": "text"},
                        "answer": {"text": question_id},
                        "evidence": [],
                        "memory_scope": {"mode": "all"},
                    })
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"question_id": "q1", "prediction": "q1"}) + "\n"
                + json.dumps({"question_id": "q2", "prediction": "wrong"}) + "\n",
                encoding="utf-8",
            )
            allowlist = root / "ids.txt"
            allowlist.write_text("q1\n", encoding="utf-8")
            backend = FakeJudge()
            summary = judge_predictions(
                backend,
                bundle,
                predictions,
                root / "judgments.jsonl",
                question_ids_path=allowlist,
            )
            self.assertEqual(summary["total_predictions"], 1)
            self.assertEqual(summary["reporting_track_size"], 1)
            self.assertEqual(summary["accuracy_conservative"], 1.0)
            self.assertEqual(len(backend.items), 1)

    def test_empty_and_failed_predictions_are_false_without_calling_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            with BundleWriter(bundle, {"benchmark": "fixture"}) as writer:
                writer.add_context({"context_id": "c", "benchmark": "fixture"})
                for question_id in ("q-empty", "q-error"):
                    writer.add_question({
                        "question_id": question_id,
                        "context_id": "c",
                        "subset": "default",
                        "prompt": [content_text("question")],
                        "task": {"response_type": "text"},
                        "answer": {"text": "Not mentioned."},
                        "evidence": [],
                        "memory_scope": {"mode": "all"},
                    })
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"question_id": "q-empty", "prediction": ""}) + "\n"
                + json.dumps({
                    "question_id": "q-error",
                    "prediction": "Not mentioned.",
                    "metadata": {"status": "error", "error_type": "runtime"},
                }) + "\n",
                encoding="utf-8",
            )
            backend = FakeJudge()
            summary = judge_predictions(
                backend, bundle, predictions, root / "judgments.jsonl"
            )
            self.assertEqual(summary["accuracy_conservative"], 0.0)
            self.assertEqual(backend.items, [])

    def test_uniform_judge_records_and_semantic_macro(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            with BundleWriter(bundle, {"benchmark": "fixture"}) as writer:
                writer.add_context({"context_id": "fixture:c0", "benchmark": "fixture"})
                for index, semantic_id, subset, answer in (
                    (1, "fixture:s1", "mcq", "A"),
                    (2, "fixture:s1", "mcq", "B"),
                    (3, "fixture:s2", "open", "blue"),
                ):
                    writer.add_question(
                        {
                            "question_id": f"fixture:q{index}",
                            "semantic_question_id": semantic_id,
                            "context_id": "fixture:c0",
                            "subset": subset,
                            "prompt": [content_text(f"question {index}")],
                            "task": {
                                "response_type": "choice" if subset == "mcq" else "text"
                            },
                            "choices": (
                                [
                                    {"choice_id": "A", "text": "alpha"},
                                    {"choice_id": "B", "text": "beta"},
                                ]
                                if subset == "mcq"
                                else []
                            ),
                            "answer": {"text": answer},
                            "evidence": [{"native_id": "PRIVATE_EVIDENCE"}],
                            "memory_scope": {"mode": "all"},
                            "metadata": {"evaluation_private": {"notes": "SECRET"}},
                        }
                    )
            predictions = root / "predictions.jsonl"
            rows = [
                {"question_id": "fixture:q1", "prediction": "A"},
                {"question_id": "fixture:q2", "prediction": "A"},
                {"question_id": "fixture:q3", "prediction": "blue"},
            ]
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output = root / "judgments.jsonl"
            backend = FakeJudge()
            summary = judge_predictions(
                backend, bundle, predictions, output, concurrency=2
            )

            self.assertEqual(summary["valid_judgments"], 3)
            self.assertEqual(summary["concurrency"], 2)
            self.assertAlmostEqual(summary["accuracy_conservative"], 2 / 3)
            self.assertAlmostEqual(summary["semantic_question_macro_accuracy"], 0.75)
            self.assertEqual(summary["by_subset"]["mcq"]["count"], 2)
            self.assertEqual(len(list(output.read_text(encoding="utf-8").splitlines())), 3)
            self.assertTrue(Path(summary["summary_path"]).is_file())
            self.assertTrue(all("evidence" not in item for item in backend.items))
            self.assertTrue(all("metadata" not in item for item in backend.items))
            judgment = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("explanation", judgment)
            self.assertNotIn("judge_response", judgment)

            # A resumed run keeps successful rows and does not call the judge again.
            resumed = FakeJudge()
            second = judge_predictions(resumed, bundle, predictions, output, resume=True)
            self.assertEqual(second["total_predictions"], 3)
            self.assertEqual(resumed.items, [])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)

            # Changing a prediction invalidates only its cached judgment.
            rows[0]["prediction"] = "B"
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            changed = FakeJudge()
            third = judge_predictions(changed, bundle, predictions, output, resume=True)
            self.assertEqual(third["total_predictions"], 3)
            self.assertEqual(len(changed.items), 1)
            self.assertEqual(changed.items[0]["prediction"], "B")


if __name__ == "__main__":
    unittest.main()
