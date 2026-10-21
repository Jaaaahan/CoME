#!/usr/bin/env python3
"""Run exact rooted-forest GraphMemix selection on cached verifier prizes."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_atm_retrieval import memory_sessions, metrics_for, read_jsonl
from graphmemix_core import load_adjacency, missing_atomic_floor
from mm_memory_bench.evidence_forest import solve_forest, solve_forest_greedy


def expit(value: float) -> float:
    """Numerically stable scalar sigmoid without importing SciPy."""
    if value >= 0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--verifier-scores", type=Path, required=True)
    parser.add_argument(
        "--source-priors", type=Path, required=True,
        help="Frozen Atomic source priors; defines the same missing-score floor used in calibration.",
    )
    parser.add_argument("--relation-store", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-mode", choices=("none", "explicit", "full"), action="append")
    parser.add_argument("--semantic-k", type=int, default=8)
    parser.add_argument("--edge-weight", type=float, action="append")
    parser.add_argument("--root-cost", type=float, action="append")
    parser.add_argument("--split", choices=("dev", "heldout100", "heldout", "all"), default="dev")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-prefix-limit", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--solver", choices=("exact", "greedy"), default="exact")
    args = parser.parse_args()
    modes = args.graph_mode or ["none", "explicit", "full"]
    edge_weights = args.edge_weight or [0.1, 0.2, 0.5, 1.0]
    root_costs = args.root_cost or [0.01, 0.05, 0.1, 0.2, 0.5]
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))["coefficients"]
    a, b, c = (float(calibration[key]) for key in ("a", "b", "c"))
    rows = read_jsonl(args.verifier_scores, "question_id")
    priors = read_jsonl(args.source_priors, "question_id")
    if args.split != "all":
        rows = {qid: row for qid, row in rows.items() if row.get("split") == args.split}
    if args.max_questions is not None:
        import hashlib
        ordered_qids = sorted(
            rows, key=lambda qid: hashlib.sha256(("graphmemix-pcsf-dev-v1\0" + qid).encode()).digest()
        )[:args.max_questions]
        rows = {qid: rows[qid] for qid in ordered_qids}
    questions = read_jsonl(args.bundle / "questions.jsonl", "question_id")
    sessions = memory_sessions(args.bundle / "memories.jsonl")
    adjacencies = {mode: load_adjacency(args.relation_store, mode, args.semantic_k) for mode in modes}
    variants: list[tuple[str, float, float]] = []
    for mode in modes:
        if mode == "none": variants.append((mode, 0.0, 0.0))
        else:
            variants.extend((mode, edge_weight, root_cost) for edge_weight in edge_weights for root_cost in root_costs)
    predictions: dict[str, dict[str, dict[str, Any]]] = {f"{m}_e{e:g}_k{r:g}": {} for m, e, r in variants}
    diagnostics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for done, (qid, row) in enumerate(sorted(rows.items()), 1):
        ids = list(row["candidate_ids"])
        atomic_values = row["atomic_scores"]
        if args.candidate_prefix_limit is not None:
            ids = ids[:args.candidate_prefix_limit]
            atomic_values = atomic_values[:args.candidate_prefix_limit]
        floor = missing_atomic_floor(priors[qid])
        verifier = {str(key): float(value) for key, value in row.get("verifier_scores", {}).items()}
        prizes = {
            memory_id: float(expit(a * (floor if atomic is None else float(atomic)) + b * verifier.get(memory_id, 0.0) + c))
            for memory_id, atomic in zip(ids, atomic_values)
        }
        for mode, edge_weight, root_cost in variants:
            label = f"{mode}_e{edge_weight:g}_k{root_cost:g}"
            if mode == "none":
                selected = sorted(ids, key=lambda mid: (-prizes[mid], ids.index(mid), mid))[:args.k]
                diagnostic = {"chosen_edges": 0, "components": args.k, "solver_seconds": 0.0}
            else:
                solver = solve_forest if args.solver == "exact" else solve_forest_greedy
                selected, diagnostic = solver(
                    ids, prizes, adjacencies[mode], k=args.k,
                    edge_weight=edge_weight, root_cost=root_cost,
                )
            predictions[label][qid] = {"question_id": qid, "context_id": row["context_id"], "retrieved_memory_ids": selected, "metadata": diagnostic}
            diagnostics[label].append(diagnostic)
        if done % 25 == 0 or done == len(rows):
            print(json.dumps({"event": "progress", "done": done, "total": len(rows)}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"protocol": "graphmemix-pcsf-1.0", "solver": args.solver, "split": args.split, "questions": len(rows), "variants": {}}
    for label, values in predictions.items():
        path = args.output_dir / f"{label}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in values.values(): handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metric = metrics_for((questions[qid] for qid in values), values, args.k, sessions)
        ds = diagnostics[label]
        metric.update({
            "mean_chosen_edges": float(np.mean([d.get("chosen_edges", 0) for d in ds])),
            "mean_components": float(np.mean([d.get("components", args.k) for d in ds])),
            "mean_solver_seconds": float(np.mean([d.get("solver_seconds", 0.0) for d in ds])),
            "p90_solver_seconds": float(np.quantile([d.get("solver_seconds", 0.0) for d in ds], 0.9)),
        })
        report["variants"][label] = metric
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
