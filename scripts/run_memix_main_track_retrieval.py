#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from contextlib import nullcontext
import time
from pathlib import Path
from typing import Any, Mapping

from mm_memory_bench.harness import _resolved_question
from mm_memory_bench.methods import (
    FaissFlatIPIndex,
    GMEQwen2VLEmbedder,
)
from mm_memory_bench.reader import BundleReader
from mm_memory_bench.come.fields import ObservationSnapshot
from mm_memory_bench.memory_scope import visible_ids
from mm_memory_bench.methods.memory_backend import backend_hash, backend_contract

from run_memix_retrieval_only import (
    SOURCE_PRIOR_PROTOCOL,
    append_jsonl,
    file_hash,
    hydrate_reranker,
    init_rerank_worker,
    model_manifest_hash,
    process_select,
    read_jsonl,
)


def select_dense(method, values):
    raw, resolved, detail = values
    started = time.perf_counter()
    selected, counts = method._select_evidence(resolved, source_detail=detail)
    return raw, selected, counts, time.perf_counter() - started


def checkpoint_states(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((root / "contexts").glob("*/state.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        context_id = str(state["context_id"])
        if context_id in result:
            raise ValueError(f"duplicate checkpoint for context: {context_id}")
        result[context_id] = path
    if not result:
        raise ValueError(f"no checkpoint states under {root}")
    return result


def read_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not values:
        raise ValueError(f"empty question allowlist: {path}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run clean Memix retrieval over a multi-context Main Track bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--memix-repo", type=Path, default=Path("vendor/memix-core"))
    parser.add_argument("--backend", choices=("memix", "standalone"), default="memix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-priors", type=Path, required=True)
    parser.add_argument("--question-ids", type=Path)
    parser.add_argument("--source-top-k", type=int, default=200)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()
    if args.processes <= 0 or args.source_top_k < 10:
        parser.error("--processes must be positive and --source-top-k must be at least 10")

    started = time.perf_counter()
    allowed = read_ids(args.question_ids)
    reader = BundleReader(args.bundle)
    observations = ObservationSnapshot.from_bundle(args.observations, reader) if args.observations else None
    batches = list(reader.iter_context_batches(question_ids=allowed))
    states = checkpoint_states(args.checkpoint_root)
    context_ids = {str(batch.context["context_id"]) for batch in batches}
    missing = context_ids - states.keys()
    if missing:
        raise ValueError(f"missing checkpoints for {len(missing)} contexts: {sorted(missing)}")
    expected_core = backend_hash(args.backend, args.memix_repo)
    for context_id in context_ids:
        state = json.loads(states[context_id].read_text(encoding="utf-8"))
        if observations:
            observations.require_contract(state["config"].get("observations"))
        elif state["config"].get("observations") is not None:
            raise ValueError("enhanced checkpoints require --observations")
        actual = str(state.get("config", {}).get("core_sha256", ""))
        if actual != expected_core:
            raise RuntimeError(f"Memix core mismatch for {context_id}: {actual} != {expected_core}")

    public_source_ids = {
        str(row["memory_id"]): str(row["source_id"])
        for line in (args.bundle / "memories.jsonl").open(encoding="utf-8")
        if (row := json.loads(line)).get("source_id") is not None
    }
    completed = read_jsonl(args.output, "question_id")
    priors = read_jsonl(args.source_priors, "question_id")
    context_by_question = {
        str(question["question_id"]): str(question["context_id"])
        for batch in batches for question in batch.questions
    }
    base_contract = {
        "source_prior_protocol": "graphmemix-standalone-prior-v1" if args.backend == "standalone" else SOURCE_PRIOR_PROTOCOL,
        "source_top_k": args.source_top_k,
        "legacy_empty_choices": False,
        **backend_contract(args.backend, args.memix_repo),
        "embedding_model_manifest_sha256": model_manifest_hash(args.embedding_model),
        "questions_sha256": file_hash(args.bundle / "questions.jsonl"),
    }
    if observations:
        base_contract["observations"] = observations.contract
    context_contracts: dict[str, dict[str, Any]] = {}
    for context_id in sorted(context_ids):
        state_path = states[context_id]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        context_contracts[context_id] = {
            **base_contract,
            "checkpoint_state_sha256": file_hash(state_path),
            "checkpoint_index_sha256": file_hash(
                state_path.parent / str(state["index_file"])
            ),
        }
    selected_ids = {
        str(question["question_id"])
        for batch in batches for question in batch.questions
    }
    if not set(completed).issubset(selected_ids) or not set(priors).issubset(selected_ids):
        raise ValueError("existing output or priors contain question IDs outside this run")
    incompatible_priors = [
        question_id for question_id, row in priors.items()
        if any(
            row.get(key) != value
            for key, value in context_contracts[
                context_by_question[question_id]
            ].items()
        )
    ]
    if incompatible_priors:
        raise ValueError(
            "source-prior cache was produced under a different context contract; "
            "use a new cache path"
        )
    incompatible_completed = [
        question_id for question_id, row in completed.items()
        if (
            not isinstance(row.get("metadata"), Mapping)
            or row["metadata"].get("source_prior_contract")
            != context_contracts[context_by_question[question_id]]
        )
    ]
    if incompatible_completed:
        raise ValueError(
            "retrieval output was produced under a different context contract; "
            "use a new output path"
        )

    embedder = GMEQwen2VLEmbedder(args.embedding_model)
    if observations:
        for context_id in context_ids:
            index_config = json.loads(states[context_id].read_text())["config"].get("observation_index", {})
            if index_config.get("embedding_model_identity") != getattr(embedder, "model_identity", None):
                raise ValueError("observation index embedding identity mismatch; rebuild checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.source_priors.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.output.exists() else "w"
    prior_mode = "a" if args.source_priors.exists() else "w"
    done = len(completed)
    total = len(selected_ids)
    with args.output.open(output_mode, encoding="utf-8") as output_handle, \
            args.source_priors.open(prior_mode, encoding="utf-8") as prior_handle:
        for context_index, batch in enumerate(batches, 1):
            context_id = str(batch.context["context_id"])
            pairs = [
                (raw, _resolved_question(reader, raw, visible_memory_ids=(
                    visible_ids(raw, batch.memories) if raw.get("memory_scope", {}).get("mode", "all") != "all" else None)))
                for raw in batch.questions
                if str(raw["question_id"]) not in completed
            ]
            if not pairs:
                continue
            state_path = states[context_id]
            state = json.loads(state_path.read_text(encoding="utf-8"))
            dimension = int(state["config"].get("embedding_dimension", 1536))
            if embedder.dimension != dimension:
                raise RuntimeError(f"embedding dimension mismatch for {context_id}")
            # Load the full vector state once for fresh GME query priors.
            method = hydrate_reranker(
                checkpoint_state=state_path,
                memix_repo=args.memix_repo,
                backend=args.backend,
                source_top_k=args.source_top_k,
                public_source_ids=public_source_ids,
            )
            method.embedder = embedder
            method._embedding_units = [dict(unit) for unit in state["embedding_units"]]
            for unit in method._embedding_units:
                source_id = public_source_ids.get(str(unit.get("memory_id", "")))
                if source_id is not None:
                    unit["source_id"] = source_id
            method.index = FaissFlatIPIndex(dimension)
            method.index.load(state_path.parent / str(state["index_file"]))

            details: list[dict[str, Any]] = []
            for raw, resolved in pairs:
                question_id = str(raw["question_id"])
                cached = priors.get(question_id)
                if cached is None:
                    query_started = time.perf_counter()
                    scope = visible_ids(raw, batch.memories)
                    detail = method._source_detail(resolved)
                    cached = {
                        "question_id": question_id,
                        "context_id": context_id,
                        "retrieval_ids": detail["retrieval_ids"],
                        "retrieval_scores": detail["retrieval_scores"],
                        "latency_seconds": time.perf_counter() - query_started,
                        "public_source_ids": True,
                        **context_contracts[context_id],
                    }
                    if raw.get("memory_scope", {}).get("mode", "all") != "all":
                        cached["visible_memory_ids"] = sorted(scope)
                    if observations:
                        cached["observation_hits"] = detail.get("observation_hits", {})
                    append_jsonl(prior_handle, cached)
                    priors[question_id] = cached
                details.append({
                    "retrieval_ids": list(cached["retrieval_ids"]),
                    "retrieval_scores": list(cached["retrieval_scores"]),
                })

            inputs = [(raw, resolved, detail) for (raw, resolved), detail in zip(pairs, details)]
            context = mp.get_context("spawn")
            with (context.Pool(
                processes=min(args.processes, len(inputs)),
                initializer=init_rerank_worker,
                initargs=(state_path, args.memix_repo, args.source_top_k, False, public_source_ids),
            ) if args.backend == "memix" else nullcontext()) as pool:
                selections = (
                    pool.imap(process_select, inputs, chunksize=1) if pool is not None
                    else (select_dense(method, values) for values in inputs)
                )
                for raw, selected, counts, latency in selections:
                    append_jsonl(output_handle, {
                        "question_id": raw["question_id"],
                        "semantic_question_id": raw.get("semantic_question_id", raw["question_id"]),
                        "context_id": context_id,
                        "subset": raw.get("subset", "default"),
                        "prediction": "",
                        "retrieved_memory_ids": selected,
                        "latency_seconds": latency,
                        "metadata": {
                            "method": "standalone-dense-v1" if args.backend == "standalone" else "memix-main-track-retrieval-only",
                            "source_top_k": args.source_top_k,
                            "public_source_ids": True,
                            "filter_counts": counts,
                            "source_prior_contract": context_contracts[context_id],
                        },
                    })
                    done += 1
                    if done % args.progress_interval == 0 or done == total:
                        print(json.dumps({
                            "event": "progress", "done": done, "total": total,
                            "context": context_index, "contexts": len(batches),
                            "elapsed_seconds": time.perf_counter() - started,
                        }), flush=True)
            method.index.reset()

    print(json.dumps({
        "event": "complete", "questions": total, "output": str(args.output),
        "source_priors": str(args.source_priors),
        "elapsed_seconds": time.perf_counter() - started,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
