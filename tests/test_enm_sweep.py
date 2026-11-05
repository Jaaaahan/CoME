import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import run_enm_sweep as sweep


class ENMSweepTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        sweep.write_json(self.bundle / "manifest.json", {"benchmark": "Mem-Gallery"})
        self.rows = [dict(question_id=f"q{i}", context_id="c", subset="default",
                          task={"subcategory": "VS" if i < 4 else "FR"},
                          prompt=[{"type": "text", "text": f"Question {i}"}],
                          answer={"text": "yes"}) for i in range(8)]
        self.write_rows(self.bundle / "questions.jsonl", self.rows)
        for name in ("memories", "assets"):
            self.write_rows(self.bundle / f"{name}.jsonl", [])

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def settings(self):
        for name in ("checkpoint_root", "embedding_model"):
            (self.root / name).mkdir()
        self.write_rows(self.root / "observations.jsonl", [])
        self.write_rows(self.root / "relations.jsonl", [dict(record_type="unit_graph", context_id="c", max_semantic_k=16)])
        self.write_rows(self.root / "priors.jsonl", [dict(question_id=r["question_id"],
            unit_hits=[{"unit_id": f"u{i}"} for i in range(32)]) for r in self.rows])
        settings = dict(bundle="bundle", output_root="trials",
            observations="observations.jsonl", source_priors="priors.jsonl", relation_store="relations.jsonl",
            checkpoint_root="checkpoint_root", embedding_model="embedding_model",
            judge={"concurrency": 1, "retries": 2}, wandb={"mode": "disabled"})
        path = self.root / "settings.json"
        sweep.write_json(path, settings)
        return path

    def args(self, path, **overrides):
        return argparse.Namespace(settings=path, run_id="baseline", dry_run=False,
                                  wandb_mode=None, **sweep.DEFAULTS, **overrides)

    def fake_reader(self, command, **kwargs):
        self.assertNotIn("--question-ids", command)
        bundle = Path(command[command.index("--bundle") + 1])
        ids = [r["question_id"] for r in sweep.gallery_questions(bundle)]
        path = Path(command[command.index("--output") + 1])
        self.assertEqual(command[command.index("--initial-limit") + 1],
                         command[command.index("--candidate-limit") + 1])
        self.assertEqual(command[command.index("--batch-size") + 1], "2")
        self.assertEqual(set(ids), {r["question_id"] for r in self.rows})
        if not path.exists():
            self.write_rows(path, [dict(question_id=q, prediction="yes", metadata=dict(
                llm_calls=4, visited_units=8, elapsed_seconds=2, query_rejection_count=1,
                status="ok")) for q in ids])

    def response(self, correct=True):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read.return_value = json.dumps({"choices": [{"message": {
            "content": json.dumps({"correct": correct})}}]}).encode()
        return response

    def test_full_dataset_includes_every_question_and_variant(self):
        for row in self.rows[:2]:
            row["semantic_question_id"] = "same-question"
        self.write_rows(self.bundle / "questions.jsonl", self.rows)
        # A previously generated local split must not restrict tuning.
        old_split = self.root / "split"
        old_split.mkdir()
        (old_split / "dev_ids.txt").write_text("q0\n")
        ids, manifest = sweep.evaluation_set(self.bundle)
        self.assertEqual(ids, {r["question_id"] for r in self.rows})
        self.assertEqual(manifest["questions"], 8)
        self.assertEqual(manifest["by_type"], {"VS": 4, "FR": 4})
        self.assertEqual(manifest["scope"], "full_dataset")
        self.assertEqual(manifest["score_role"], "tuning_set_accuracy")
        self.assertEqual(manifest["questions_sha256"], sweep.digest(self.bundle / "questions.jsonl"))

    def test_old_split_settings_require_explicit_migration(self):
        path = self.settings()
        settings = sweep.read_json(path)
        settings["split_dir"] = "split"
        sweep.write_json(path, settings)
        with self.assertRaisesRegex(ValueError, "remove split_dir"):
            sweep.load_settings(path)

    def test_h2hmem_full_dataset_uses_gpt54_mini(self):
        sweep.write_json(self.bundle / "manifest.json", {"benchmark": "H2HMem"})
        path = self.settings()
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://openai.invalid/v1", "OPENAI_API_KEY": "fixture-secret"}), \
                patch.object(sweep.subprocess, "run", side_effect=self.fake_reader), \
                patch("mm_memory_bench.judge.urllib.request.urlopen", return_value=self.response()) as request:
            sweep.trial(self.args(path))
        self.assertEqual(request.call_count, len(self.rows))
        self.assertTrue(all(json.loads(call.args[0].data)["model"] == "gpt-5.4-mini"
                            for call in request.call_args_list))
        metrics = sweep.read_json(self.root / "trials/baseline/metrics.json")
        self.assertEqual(metrics["tune/questions"], len(self.rows))

    def test_atm_full_dataset_uses_explicit_gpt5_mini_judge(self):
        sweep.write_json(self.bundle / "manifest.json", {"benchmark": "ATM-Bench"})
        path = self.settings()
        settings = sweep.read_json(path)
        settings["judge"]["model"] = "gpt-5-mini"
        sweep.write_json(path, settings)
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://openai.invalid/v1", "OPENAI_API_KEY": "fixture-secret"}), \
                patch.object(sweep.subprocess, "run", side_effect=self.fake_reader), \
                patch("mm_memory_bench.judge.urllib.request.urlopen", return_value=self.response()) as request:
            sweep.trial(self.args(path))
        self.assertEqual(request.call_count, len(self.rows))
        self.assertTrue(all(json.loads(call.args[0].data)["model"] == "gpt-5-mini"
                            for call in request.call_args_list))
        metrics = sweep.read_json(self.root / "trials/baseline/metrics.json")
        self.assertEqual(metrics["tune/questions"], len(self.rows))

    def test_memeye_full_dataset_keeps_all_variants(self):
        sweep.write_json(self.bundle / "manifest.json", {"benchmark": "MemEye"})
        for row in self.rows:
            row["semantic_question_id"] = "same-question"
        self.write_rows(self.bundle / "questions.jsonl", self.rows)
        ids, manifest = sweep.evaluation_set(self.bundle)
        self.assertEqual(ids, {row["question_id"] for row in self.rows})
        self.assertEqual(manifest["questions"], len(self.rows))

    def test_preflight_rejects_unsupported_graph_and_prior_limits(self):
        path = self.settings()
        settings = sweep.load_settings(path)
        ids, _ = sweep.evaluation_set(self.bundle)
        with self.assertRaisesRegex(ValueError, "max_semantic_k"):
            sweep.validate_artifacts(settings, ids, {**sweep.DEFAULTS, "semantic_k": 32})
        with self.assertRaisesRegex(ValueError, "stored unit hits"):
            sweep.validate_artifacts(settings, ids, {**sweep.DEFAULTS, "source_top_l": 40})
        with self.assertRaisesRegex(ValueError, "source_top_l"):
            sweep.validate_parameters({**sweep.DEFAULTS, "candidate_limit": 16})
        with self.assertRaisesRegex(ValueError, "1..20"):
            sweep.validate_parameters({**sweep.DEFAULTS, "query_table_size": 21})

    def test_trial_uses_gpt54_mini_and_resume_skips_successful_judgments(self):
        path = self.settings()
        args = self.args(path)
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://openai.invalid/v1", "OPENAI_API_KEY": "fixture-secret"}), \
                patch.object(sweep.subprocess, "run", side_effect=self.fake_reader), \
                patch("mm_memory_bench.judge.urllib.request.urlopen", return_value=self.response()) as request:
            sweep.trial(args)
            self.assertEqual(request.call_count, 8)
            body = json.loads(request.call_args.args[0].data)
            self.assertEqual(body["model"], "gpt-5.4-mini")
            self.assertNotIn("temperature", body)
            self.assertEqual(request.call_args.args[0].get_header("Authorization"), "Bearer fixture-secret")
            self.assertEqual(request.call_args.args[0].full_url, "https://openai.invalid/v1/chat/completions")
            sweep.trial(args)
            self.assertEqual(request.call_count, 8)
            with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://different.invalid/v1"}):
                with self.assertRaisesRegex(ValueError, "contract changed"):
                    sweep.trial(args)
        directory = self.root / "trials/baseline"
        self.assertEqual(sweep.read_json(directory / "metrics.json")["tune/judge_accuracy"], 1)
        self.assertNotIn("fixture-secret", (directory / "contract.json").read_text())

    def test_failed_judge_never_logs_objective_and_resume_repairs_it(self):
        path = self.settings()
        args = self.args(path)
        args.wandb_mode = "online"
        run = Mock()
        fake_wandb = SimpleNamespace(init=Mock(side_effect=lambda **kwargs: nullcontext(run)))
        # First item fails both attempts; other items succeed. Resume only that item.
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://openai.invalid/v1", "OPENAI_API_KEY": "test"}), \
                patch.dict(sys.modules, {"wandb": fake_wandb}), \
                patch.object(sweep.subprocess, "run", side_effect=self.fake_reader), \
                patch.object(sweep.time, "sleep"), \
                patch("mm_memory_bench.judge.urllib.request.urlopen",
                      side_effect=[ValueError("temporary"), ValueError("temporary"),
                                   *[self.response() for _ in range(7)]]) as request:
            with self.assertRaisesRegex(RuntimeError, "judge incomplete"):
                sweep.trial(args)
            run.log.assert_not_called()
            self.assertFalse((self.root / "trials/baseline/metrics.json").exists())
            request.side_effect = None
            request.return_value = self.response(False)
            sweep.trial(args)
            self.assertEqual(request.call_count, 10)
            self.assertEqual(run.log.call_count, 1)
            self.assertEqual(run.log.call_args.args[0]["tune/judge_accuracy"], .875)

    def test_official_endpoint_uses_existing_project_env_key(self):
        path = self.settings()
        env_file = self.root / ".env"
        env_file.write_text("OPENAI_API_KEY=fixture-local-secret\n")
        env_file.chmod(0o600)
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "", "OPENAI_API_KEY": ""}), \
                patch("mm_memory_bench.judge._project_env_path", return_value=env_file), \
                patch.object(sweep.subprocess, "run", side_effect=self.fake_reader), \
                patch("mm_memory_bench.judge.urllib.request.urlopen", return_value=self.response()) as request:
            sweep.trial(self.args(path))
        self.assertEqual(request.call_args.args[0].full_url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(request.call_args.args[0].get_header("Authorization"), "Bearer fixture-local-secret")
        contract = (self.root / "trials/baseline/contract.json").read_text()
        self.assertNotIn("fixture-local-secret", contract)
        self.assertEqual(json.loads(contract)["settings"]["judge"]["model"], "gpt-5.4-mini")

    def test_failed_answer_counts_as_wrong_and_missing_predictions_fail(self):
        path = self.settings()
        settings = sweep.load_settings(path)
        ids, _ = sweep.evaluation_set(self.bundle)
        directory = self.root / "trial"
        directory.mkdir()
        self.fake_reader(sweep.reader_command(settings, sweep.DEFAULTS, directory / "predictions.jsonl"))
        predictions = list(sweep.iter_jsonl(directory / "predictions.jsonl"))
        predictions[0]["metadata"].update(status="invalid_final_answer", error_type="InvalidFinalAnswer")
        self.write_rows(directory / "predictions.jsonl", predictions)
        backend = SimpleNamespace(model="fixture", judge=Mock(return_value={"correct": True}))
        metrics = sweep.evaluate_trial(settings, directory, ids, backend)
        self.assertEqual(metrics["tune/judge_accuracy"], .875)
        self.assertEqual(metrics["tune/answer_failure_rate"], .125)
        self.assertEqual(backend.judge.call_count, 7)
        self.write_rows(directory / "predictions.jsonl", predictions[:-1])
        with self.assertRaisesRegex(ValueError, "every question"):
            sweep.evaluate_trial(settings, directory, ids - {predictions[-1]["question_id"]}, backend)
        with self.assertRaisesRegex(ValueError, "exactly"):
            sweep.evaluate_trial(settings, directory, ids, backend)

    def test_dry_run_needs_no_api_key_and_makes_no_calls(self):
        path = self.settings()
        args = self.args(path)
        args.dry_run = True
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://openai.invalid/v1", "OPENAI_API_KEY": ""}), \
                patch.object(sweep.subprocess, "run") as reader, \
                patch("mm_memory_bench.judge.urllib.request.urlopen") as judge:
            sweep.trial(args)
            reader.assert_not_called()
            judge.assert_not_called()
        self.assertFalse((self.root / "trials").exists())

    def test_generation_only_exports_complete_answers_without_api_or_wandb(self):
        path = self.settings()
        args = self.args(path)
        args.generation_only = True
        with patch.object(sweep.subprocess, "run", side_effect=self.fake_reader), \
                patch.object(sweep, "backend_from_env") as backend, \
                patch.dict(sys.modules, {"wandb": None}):
            sweep.trial(args)
        backend.assert_not_called()
        directory = self.root / "trials/baseline"
        ready = sweep.read_json(directory / "generation-complete.json")
        self.assertEqual(ready["questions"], 8)
        self.assertEqual(ready["judge_input_sha256"], sweep.digest(directory / "judge-input.jsonl"))
        self.assertEqual(ready["predictions_sha256"], sweep.digest(directory / "predictions.jsonl"))
        self.assertFalse((directory / "metrics.json").exists())
        compact = list(sweep.iter_jsonl(directory / "judge-input.jsonl"))
        self.assertEqual(len(compact), 8)
        self.assertTrue(all("source_prediction_row_sha256" in row for row in compact))

    def test_incomplete_generation_does_not_publish_ready_marker(self):
        path = self.settings()
        settings = sweep.load_settings(path)
        directory = self.root / "trial"
        directory.mkdir()
        self.fake_reader(sweep.reader_command(settings, sweep.DEFAULTS, directory / "predictions.jsonl"))
        rows = list(sweep.iter_jsonl(directory / "predictions.jsonl"))
        self.write_rows(directory / "predictions.jsonl", rows[:-1])
        ids, dataset = sweep.evaluation_set(self.bundle)
        with self.assertRaisesRegex(ValueError, "finish all"):
            sweep.export_judge_input(directory, ids, dataset)
        self.assertFalse((directory / "generation-complete.json").exists())


if __name__ == "__main__":
    unittest.main()
