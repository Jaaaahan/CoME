#!/usr/bin/env python3
"""Audit and score the frozen unit20 package without altering original artifacts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mm_memory_bench.judge import (
    JUDGE_PROTOCOL_VERSION, JUDGE_RUBRIC, JUDGE_RUBRIC_SHA256,
    _json_from_text, _normalized_judgment, backend_from_env, judge_item, judge_predictions,
)


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inputs(package):
    result = {}
    for key, count in {"atm": 1044, "mem_gallery": 1711, "memeye": 1855, "h2hmem": 1982}.items():
        experiment = package / ("memeye-unit20-context-20261027" if key == "memeye" else "unit20-context-suite-20261027")
        bundle = experiment / "data" / ("atm_bench" if key == "atm" else key)
        predictions_path = experiment / key / "runs/observations-v1/qwen3vl8b" / key / "reader/predictions.jsonl"
        questions_list, predictions_list = rows(bundle / "questions.jsonl"), rows(predictions_path)
        questions = {r["question_id"]: r for r in questions_list}
        predictions = {r["question_id"]: r for r in predictions_list}
        assert len(questions_list) == len(questions) == len(predictions_list) == len(predictions) == count, key
        assert questions.keys() == predictions.keys(), key
        assert all(p["prediction"].strip() and p.get("metadata", {}).get("status") != "error" for p in predictions.values()), key
        result[key] = dict(experiment=experiment, bundle=bundle, predictions_path=predictions_path,
                           questions=questions, predictions=predictions)
    return result


def native(package, output, data):
    suite = package / "unit20-context-suite-20261027/source/cluster/cuhk-suite"
    sys.path.insert(0, str(suite / "atm_evaluator"))
    from memqa.utils.evaluator.evaluate_qa import deterministic_accuracy
    memeye = load_module("frozen_memeye_evaluator", suite / "memeye_evaluator.py")
    h2h = load_module("frozen_h2h_evaluator", suite / "h2h_evaluator/Lexical_metrics.py")
    provenance = json.loads((suite / "h2h_evaluator/provenance.json").read_text())
    assert digest(suite / "h2h_evaluator/Lexical_metrics.py") == provenance["sha256"]
    calculator = h2h.EvaluationMetricsCalculator(h2h.EnglishTextProcessor())
    # Require full NLTK tokenization; never silently fall back to whitespace splitting.
    import nltk
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords"):
        nltk.data.find(resource)
    summary = {}
    names = {"atm": "EM", "mem_gallery": "EM", "memeye": "M.EM", "h2hmem": "Recall"}
    for key, item in data.items():
        records = []
        for qid, q in item["questions"].items():
            if key == "memeye" and q["subset"] != "mcq":
                continue
            pred, ref = item["predictions"][qid]["prediction"], q["answer"]["text"]
            extra = {}
            if key == "atm":
                prompt = "\n".join(p["text"] for p in q["prompt"] if p["type"] == "text")
                score = float(deterministic_accuracy(ref, pred, prompt))
                extra["strict_normalized_em_diagnostic"] = float(memeye.score_open(pred, ref)[0])
            elif key == "mem_gallery":
                score = float(memeye.score_open(pred, ref)[0])
            elif key == "memeye":
                choice = memeye.extract_choice(pred, {c["choice_id"].upper() for c in q["choices"]})
                score = float(choice == ref.upper())
                extra["extracted_choice"] = choice
            else:
                p_tokens = set(calculator.text_processor.tokenize(pred))
                r_tokens = set(calculator.text_processor.tokenize(ref))
                score = calculator._recall(pred, ref)
                extra.update(matched_unique_tokens=len(p_tokens & r_tokens), reference_unique_tokens=len(r_tokens))
            records.append(dict(question_id=qid, semantic_question_id=q.get("semantic_question_id", qid),
                                subset=q.get("subset", "default"), category=q["task"]["subcategory"],
                                metric=names[key], score=score, **extra))
        path = output / f"{key}.native.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        groups = defaultdict(list)
        for r in records:
            groups[r["category"]].append(r["score"])
        summary[key] = dict(metric=names[key], questions=len(records), value=statistics.fmean(r["score"] for r in records),
                            percent=100 * statistics.fmean(r["score"] for r in records),
                            correct=int(sum(r["score"] for r in records)) if key != "h2hmem" else None,
                            by_category={k: dict(count=len(v), percent=100 * statistics.fmean(v)) for k, v in sorted(groups.items())})
        if key == "atm":
            summary[key]["strict_normalized_em_diagnostic_percent"] = 100 * statistics.fmean(r["strict_normalized_em_diagnostic"] for r in records)
        if key == "memeye":
            assert len(records) == 1484 and len({r["semantic_question_id"] for r in records}) == 371
        print(json.dumps(dict(event="native_complete", dataset=key, **summary[key])), flush=True)
    report = dict(created_at=stamp(), results=summary, definitions={
        "atm": "Project-native EM: ATM-Bench deterministic_accuracy across all 1,044 questions; this accepts normalized numeric/date and some substring/token matches, not strict string equality and not the official composite QS.",
        "mem_gallery": "normalize(prediction) == normalize(reference), nonempty; frozen MemEye evaluator's exact port of Mem-Gallery normalization.",
        "memeye": "Multiple-choice exact match after official extract_choice; all 1,484 rotations (371 semantic questions x 4). Excludes 371 open-ended rows.",
        "h2hmem": "Macro mean of |unique predicted tokens intersect unique reference tokens| / |unique reference tokens|, using official NLTK lowercase/tokenization/stopword filtering; 1,982 evidence-supported questions, not 2,236 official full-set questions."
    }, evaluator_hashes={str(p.relative_to(package)): digest(p) for p in [suite / "memeye_evaluator.py", suite / "h2h_evaluator/Lexical_metrics.py", *sorted((suite / "atm_evaluator").rglob("*.py"))]},
        nltk_version=nltk.__version__)
    write_json(output / "native-summary.json", report)


class AuditedJudge:
    model = "gpt-5-mini"

    def __init__(self, output):
        self.backend = backend_from_env(model=self.model, base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY", timeout_seconds=180)
        self.output = output
        self.lock = threading.Lock()
        self.count = 0

    def log(self, row):
        with self.lock:
            with (self.output / "api-audit.jsonl").open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row.get("status") == "ok":
                self.count += 1
                if self.count % 100 == 0:
                    print(json.dumps(dict(event="api_progress", completed_this_process=self.count, at=stamp())), flush=True)

    def judge(self, item):
        body = dict(model=self.model, response_format={"type": "json_object"}, messages=[
            {"role": "system", "content": JUDGE_RUBRIC},
            {"role": "user", "content": json.dumps(dict(item), ensure_ascii=False, sort_keys=True)}])
        encoded = json.dumps(body, ensure_ascii=False).encode()
        request_hash = hashlib.sha256(encoded).hexdigest()
        for attempt in range(7):
            start = time.monotonic()
            request = urllib.request.Request(self.backend.base_url + "/chat/completions", data=encoded,
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.backend.api_key}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = json.loads(response.read())
                    request_id = response.headers.get("x-request-id")
                    limits = {k: v for k, v in response.headers.items() if k.lower().startswith("x-ratelimit-")}
                value = _json_from_text(payload["choices"][0]["message"]["content"])
                _normalized_judgment(value)
                self.log(dict(at=stamp(), status="ok", attempt=attempt + 1, request_sha256=request_hash,
                              request_id=request_id, seconds=time.monotonic() - start, rate_limits=limits, response=payload))
                return value
            except Exception as exc:
                code = getattr(exc, "code", None)
                # Do not persist response/error text that might contain credential material.
                self.log(dict(at=stamp(), status="error", attempt=attempt + 1, request_sha256=request_hash,
                              error_type=type(exc).__name__, http_status=code, seconds=time.monotonic() - start))
                if code in (400, 401, 403, 404) or attempt == 6:
                    raise RuntimeError(f"API evaluation failed: {type(exc).__name__}, HTTP {code}") from None
                time.sleep(min(45, 2 ** attempt + random.random()))


def run_judge(output, data, concurrency, pilot):
    backend = AuditedJudge(output)
    def one(key):
        item = data[key]
        summary = judge_predictions(backend, item["bundle"], item["predictions_path"],
            output / f"{key}.gpt-5-mini.judgments.jsonl", concurrency=concurrency,
            max_items=1 if pilot else None,
            question_ids_path=item["bundle"] / "question_ids.txt" if key == "h2hmem" else None)
        print(json.dumps(dict(event="judge_complete", dataset=key, summary=summary)), flush=True)
        if summary["failed_judgments"]:
            raise RuntimeError(f"{key}: {summary['failed_judgments']} failed judgments; resume to retry")
        return summary
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, data))


def report(output, data):
    native_report = json.loads((output / "native-summary.json").read_text())
    audit = rows(output / "api-audit.jsonl")
    successes = [r for r in audit if r["status"] == "ok"]
    audited_scores = defaultdict(Counter)
    for r in successes:
        raw = _json_from_text(r["response"]["choices"][0]["message"]["content"])
        audited_scores[r["request_sha256"]][raw["correct"]] += 1
    checks = []
    results = {}
    for key, item in data.items():
        judged = rows(output / f"{key}.gpt-5-mini.judgments.jsonl")
        assert len(judged) == len(item["questions"])
        assert {r["question_id"] for r in judged} == item["questions"].keys()
        assert all(r["status"] == "ok" for r in judged)
        for r in judged:
            qid = r["question_id"]
            body = dict(model="gpt-5-mini", response_format={"type": "json_object"}, messages=[
                {"role": "system", "content": JUDGE_RUBRIC},
                {"role": "user", "content": json.dumps(judge_item(item["questions"][qid], item["predictions"][qid]["prediction"]), ensure_ascii=False, sort_keys=True)}])
            request_hash = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
            assert audited_scores[request_hash][r["correct"]] > 0, (key, qid, "no matching API response")
            audited_scores[request_hash][r["correct"]] -= 1
        original = json.loads((item["experiment"] / key / "metrics/summary.json").read_text())
        assert original["predictions_sha256"] == digest(item["predictions_path"])
        checks.append(dict(dataset=key, question_and_prediction_ids_exact=True, predictions_match_original_hash=True,
                           every_judgment_matches_raw_api_response=True, judgments=len(judged), failed_judgments=0))
        local = {r["question_id"]: r for r in rows(item["experiment"] / key / "metrics/local-qwen-judgments.jsonl")}
        correct = sum(r["correct"] for r in judged)
        local_correct = sum(r["correct"] for r in local.values())
        subsets = defaultdict(list)
        for r in judged:
            subsets[r["subset"]].append(r["score"])
        results[key] = dict(questions=len(judged), correct=correct, gpt_accuracy_percent=100 * correct / len(judged),
            local_qwen_accuracy_percent=100 * local_correct / len(local), delta_vs_local_qwen_pp=100 * (correct - local_correct) / len(judged),
            judge_disagreements=sum(r["correct"] != local[r["question_id"]]["correct"] for r in judged),
            failed_judgments=0, native=native_report["results"][key],
            gpt_by_subset={s: dict(count=len(v), percent=100 * statistics.fmean(v)) for s, v in subsets.items()})
    usage = Counter()
    for r in successes:
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[k] += r["response"]["usage"].get(k, 0)
    total = sum(r["questions"] for r in results.values())
    correct = sum(r["correct"] for r in results.values())
    value = dict(created_at=stamp(), judge_model_requested="gpt-5-mini", judge_models_returned=sorted({r["response"]["model"] for r in successes}),
        judge_protocol=JUDGE_PROTOCOL_VERSION, judge_rubric_sha256=JUDGE_RUBRIC_SHA256, results=results,
        overall=dict(questions=total, correct=correct, micro_accuracy_percent=100 * correct / total,
                     dataset_macro_accuracy_percent=statistics.fmean(r["gpt_accuracy_percent"] for r in results.values())),
        api_usage=dict(usage), successful_api_calls=len(successes), retried_api_errors=len(audit)-len(successes))
    write_json(output / "results.json", value)
    write_json(output / "validation.json", dict(created_at=stamp(), passed=True, datasets=checks,
        successful_api_responses=len(successes), consumed_api_responses=sum(r["questions"] for r in results.values()),
        extra_successful_api_responses=sum(sum(c.values()) for c in audited_scores.values())))
    import csv
    with (output / "results.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "questions", "gpt_correct", "gpt_accuracy_percent", "native_metric", "native_questions", "native_correct", "native_percent"])
        for key, r in results.items():
            n = r["native"]
            writer.writerow([key, r["questions"], r["correct"], r["gpt_accuracy_percent"], n["metric"], n["questions"], n["correct"], n["percent"]])
    labels = {"atm": "ATM-Bench", "mem_gallery": "Mem-Gallery", "memeye": "MemEye", "h2hmem": "H2HMem"}
    lines = ["# unit20 平铺版：GPT API 评估与原生指标", "", f"评估完成时间：{value['created_at']}。",
        "", "本报告评估 2026-10-27 unit20 + source context 600 的最终 Reader 输出，Reader 为 Qwen/Qwen3-VL-8B-Instruct。原始包内容保持不变。",
        "", f"Judge：OpenAI API `{', '.join(value['judge_models_returned'])}`，请求模型 `gpt-5-mini`；项目统一二元判分协议 `{JUDGE_PROTOCOL_VERSION}`。使用原有提示词与默认推理参数，每条记录独立判分。",
        "这属于项目统一 GPT 判分，不等同于四个 benchmark 各自原论文的完整官方评估协议。",
        "", "| 数据集 | GPT 正确 / 总数 | GPT 准确率 | 指定指标 | 指标分数 | 指标样本数 |",
        "|---|---:|---:|---|---:|---:|"]
    for key, r in results.items():
        n = r["native"]
        lines.append(f"| {labels[key]} | {r['correct']} / {r['questions']} | {r['gpt_accuracy_percent']:.2f}% | {n['metric']} | {n['percent']:.2f}% | {n['questions']} |")
    lines += ["", f"GPT 总计 {correct} / {total}，按全部记录加权准确率 {100 * correct / total:.2f}%；四数据集等权均值 {value['overall']['dataset_macro_accuracy_percent']:.2f}%。最终失败判分为 0。",
        "", "## 指标口径", "",
        "- ATM EM：沿用项目 `compute_native_metrics.py` 的 ATM `deterministic_accuracy`。包含数字、日期、货币、拒答归一化以及部分子串/词项匹配，不能解释成严格字符串相等，也不是 ATM 官方综合 QS。",
        f"  若采用与 Mem-Gallery 一致的严格归一化字符串匹配作为补充诊断，ATM 为 {results['atm']['native']['strict_normalized_em_diagnostic_percent']:.2f}%。",
        "- Mem-Gallery EM：按冻结 evaluator 的官方归一化移植实现计算，要求归一化后的完整答案相等且非空。",
        "- MemEye M.EM：按冻结 `extract_choice` 抽取选项，统计 371 个语义问题的 4 次选项轮换，共 1,484 条；不包含 371 条开放题。GPT 准确率则包含全部 1,855 条。",
        "- H2HMem Recall：官方词集合召回率，使用 NLTK 分词、小写化及停用词过滤后，逐题计算交集大小 / 参考答案词集合大小，再取题目均值；不是来源检索 Recall@K。范围为包内 evidence-supported track 的 1,982 题，不是官方 2,236 题全集。",
        "", "## 与包内本地 Qwen 判分对比", "", "| 数据集 | 本地 Qwen | GPT | GPT − Qwen | 判分不一致条数 |", "|---|---:|---:|---:|---:|"]
    for key, r in results.items():
        lines.append(f"| {labels[key]} | {r['local_qwen_accuracy_percent']:.2f}% | {r['gpt_accuracy_percent']:.2f}% | {r['delta_vs_local_qwen_pp']:+.2f} pp | {r['judge_disagreements']} |")
    lines += ["", "上述差异来自评分模型变化，Reader 预测完全相同。包中的历史 GraphMemix、旧 COME、unit10 未在本次重新运行 GPT 判分，不能直接用其 Qwen 分数宣称 GPT 下的改进。",
        "", "## 复核与文件", "",
        "`input-manifest.json` 保存题目和预测 SHA256；`validation.json` 记录题目覆盖及原始 API 回复逐条核验；`native-summary.json` 保存公式、分组与冻结 scorer 哈希；`*.native.jsonl` 保存逐题指标。",
        "`*.gpt-5-mini.judgments.jsonl` 保存逐题判分；`api-audit.jsonl` 保存请求摘要、API request ID、实际模型、原始回复及 token usage；不含 API key。`results.json` 和 `results.csv` 是汇总结果。",
        "", f"成功 API 调用：{len(successes)}；失败尝试：{len(audit)-len(successes)}；输入 tokens：{usage['prompt_tokens']:,}；输出 tokens（含推理）：{usage['completion_tokens']:,}。",
        "", "复现脚本为仓库 `scripts/evaluate_packaged_unit20.py`，本目录另存同一脚本快照；本次本地 scorer 使用 NLTK 3.9.2。",
        "运行 `--stage native` 可重算本地指标，`--stage judge --concurrency 32` 可续跑 GPT 判分，`--stage report` 重新核验与汇总。",
        ""]
    (output / "README.md").write_text("\n".join(lines))
    (output / "evaluate_packaged_unit20.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT / "results/unit20-flat-all-datasets-20261027")
    parser.add_argument("--stage", choices=["native", "pilot", "judge", "report"], required=True)
    parser.add_argument("--concurrency", type=int, default=16, help="Per-dataset API concurrency; four datasets in parallel")
    args = parser.parse_args()
    output = args.package / "gpt-evaluation-20261028"
    output.mkdir(parents=True, exist_ok=True)
    data = inputs(args.package)
    inventory = dict(created_at=stamp(), package=str(args.package), inputs={key: dict(
        questions=len(d["questions"]), question_file=str(d["bundle"] / "questions.jsonl"),
        question_sha256=digest(d["bundle"] / "questions.jsonl"), predictions_file=str(d["predictions_path"]),
        predictions_sha256=digest(d["predictions_path"])) for key, d in data.items()})
    manifest = output / "input-manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["inputs"] == inventory["inputs"], "Input artifacts changed"
    else:
        write_json(manifest, inventory)
    if args.stage == "native":
        native(args.package, output, data)
    elif args.stage in ("pilot", "judge"):
        run_judge(output, data, args.concurrency, args.stage == "pilot")
    else:
        report(output, data)


if __name__ == "__main__":
    main()
