from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_enm_local_search as search
import test_enm_sweep as fixtures


class LocalSearchTest(unittest.TestCase):
    setUp = fixtures.ENMSweepTest.setUp
    write_rows = fixtures.ENMSweepTest.write_rows
    settings = fixtures.ENMSweepTest.settings
    fake_reader = fixtures.ENMSweepTest.fake_reader
    def search_plan(self):
        settings = self.settings()
        config = self.root / "search.json"
        config.write_text(json.dumps(dict(seed=42, random_trials=1, parameters={
            **{k: [v] for k, v in search.DEFAULTS.items()}, "candidate_limit": [48, 64]})))
        path = self.root / "plan.json"
        search.create_plan(settings, config, path)
        return path

    def test_sampling_is_unique_reproducible_and_contains_baseline(self):
        config = json.loads((search.ROOT / "config/enm_mem_gallery_local_search.json").read_text())
        rows = search.sample_parameters(config["parameters"], 24, 42)
        self.assertEqual(rows, search.sample_parameters(config["parameters"], 24, 42))
        self.assertEqual(rows[0], search.DEFAULTS)
        self.assertEqual(len({json.dumps(r, sort_keys=True) for r in rows}), 25)

    def test_watcher_scores_only_verified_complete_trials_and_resumes(self):
        plan_path = self.search_plan()
        plan = search.read_plan(plan_path)
        with patch.object(search.subprocess, "run", side_effect=self.fake_reader), redirect_stdout(io.StringIO()):
            search.generate(plan_path, 0, "http://127.0.0.1:8000/v1")
        original = self.root / "trials/baseline"
        destination = self.root / "scored"
        backend = SimpleNamespace(model="gpt-5.4-mini", api_key="fixture-key", judge=lambda item: {"correct": True})

        def fetch(host, remote, local):
            shutil.copy2(original / remote.name, local)

        args = SimpleNamespace(plan=plan_path, bundle=self.bundle, output=destination,
            ssh_host="fixture", remote_root=Path("/remote"), once=True, poll_seconds=120)
        with patch.object(search, "available_trials", return_value={"baseline"}), \
                patch.object(search, "fetch_file", side_effect=fetch) as download, \
                patch.object(search, "backend_from_env", return_value=backend), redirect_stdout(io.StringIO()):
            search.watch(args)
            search.watch(args)
            self.assertEqual(download.call_count, 3)
        report = search.read_json(destination / "leaderboard.json")
        self.assertEqual(report["scored_trials"], 1)
        self.assertEqual(report["best"]["id"], "baseline")
        self.assertEqual(report["best"]["metrics"]["tune/judge_accuracy"], 1.0)
        self.assertEqual(report["trials"][1]["status"], "awaiting_generation")

    def test_corrupt_download_never_gets_scored(self):
        plan_path = self.search_plan()
        plan = search.read_plan(plan_path)
        with patch.object(search.subprocess, "run", side_effect=self.fake_reader), redirect_stdout(io.StringIO()):
            search.generate(plan_path, 0, "http://127.0.0.1:8000/v1")
        directory = self.root / "trials/baseline"
        shutil.copy2(directory / "judge-input.jsonl", directory / "predictions.jsonl")
        with (directory / "predictions.jsonl").open("a") as handle:
            handle.write('{}\n')
        ids, dataset = search.evaluation_set(self.bundle)
        with self.assertRaisesRegex(ValueError, "do not match"):
            search.verify_download(directory, ids, dataset["questions_sha256"],
                                   plan["trials"][0]["parameters"], plan["settings"]["judge"])


if __name__ == "__main__":
    unittest.main()
