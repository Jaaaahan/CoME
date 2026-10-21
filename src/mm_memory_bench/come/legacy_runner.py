"""CoME stages invoked by the existing GraphMemix release runner."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..harness import _resolved_memory, _resolved_question
from ..methods.backends import OpenAICompatibleQwenVL
from ..methods.base import GenerationConfig
from ..methods.gme import GMEQwen2VLEmbedder
from ..reader import BundleReader
from .legacy_pipeline import GraphMemixPipeline
from .legacy_models import MODEL_PROTOCOL
from .store import SourceCatalog, atomic_json, file_digest
from .types import CoMEConfig, PROTOCOL, QUERY_PROTOCOL, digest


def visible_ids(question: dict, memories: list[dict]) -> set[str]:
    scope = question.get("memory_scope", {"mode": "all"})
    all_ids = {str(m["memory_id"]) for m in memories}
    if scope.get("mode") == "all":
        return all_ids
    if scope.get("mode") == "prefix":
        cutoff = scope.get("max_sequence")
        if type(cutoff) is not int:
            raise ValueError("prefix memory scope needs max_sequence")
        return {str(m["memory_id"]) for m in memories if int(m["sequence"]) <= cutoff}
    if scope.get("mode") == "ids":
        ids = set(map(str, scope.get("memory_ids", [])))
        if not ids <= all_ids:
            raise ValueError("question references an unavailable memory")
        return ids
    raise ValueError("unknown memory scope")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("checkpoint", "observations", "reader"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--embedding-model")
    parser.add_argument("--question-ids", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    suite = config["benchmark_suite"]
    come = CoMEConfig(**config.get("come", {}))
    generation = GenerationConfig(model=args.model, base_url=args.base_url,
        max_model_len=suite["max_model_len"], max_output_tokens=suite["reader_max_output_tokens"],
        reasoning_effort=args.reasoning_effort)
    verifier_generation = GenerationConfig(**{**asdict(generation),
        "max_output_tokens": suite["verifier_max_output_tokens"]})
    embedder = GMEQwen2VLEmbedder(args.embedding_model or suite["embedding_model"])
    model = OpenAICompatibleQwenVL(verifier_generation)
    reader_model = OpenAICompatibleQwenVL(generation)
    bundle = BundleReader(args.bundle)
    wanted = set(args.question_ids.read_text().split()) if args.question_ids else None
    # Includes source bytes and model settings; resumption never mixes protocols.
    source_contract = {"protocol": PROTOCOL, "model_protocol": MODEL_PROTOCOL,
        "query_protocol": QUERY_PROTOCOL,
        "config": config, "generation": asdict(generation),
        "embedding_model": getattr(embedder, "model_identity", None),
        "sources": {table: file_digest(path) for table, path in bundle.table_paths.items()}}
    contract = digest(source_contract)
    manifest = args.output.with_suffix(".contract.json")
    completed: dict[str, dict] = {}
    if args.stage == "reader":
        if not come.isolate_queries:
            raise ValueError("benchmark reader requires isolate_queries=true")
        if args.output.exists():
            if not manifest.exists() or json.loads(manifest.read_text()).get("contract") != contract:
                raise ValueError("incompatible prediction resume contract")
            for line in args.output.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    qid = str(row["question_id"])
                    if qid in completed:
                        raise ValueError("duplicate prediction IDs")
                    if row.get("prediction"):
                        completed[qid] = row
        atomic_json(manifest, {"contract": contract, **source_contract})
        args.output.parent.mkdir(parents=True, exist_ok=True)

    answered = 0
    for batch in bundle.iter_context_batches(question_ids=wanted):
        memories = [_resolved_memory(bundle, memory, memory_view="raw") for memory in batch.memories]
        catalog = SourceCatalog(memories, chunk_chars=come.text_chunk_chars)
        root = args.artifact_root / "contexts" / digest(batch.context["context_id"])
        pipeline = GraphMemixPipeline(catalog, root, embedder, model, reader_model, generation, come)
        if args.stage == "checkpoint":
            retriever = pipeline._query_components(set(catalog.memories))[-1]
            atomic_json(root / "sources.json", {"protocol": PROTOCOL, "signature": catalog.signature,
                "nodes": len(retriever.nodes)})
        elif args.stage == "observations":
            info = pipeline.ingest()
            print(json.dumps({"context_id": batch.context["context_id"], **info}), flush=True)
        else:
            image_ids = {n.node_id for n in catalog.nodes.values() if n.kind == "image"}
            if not image_ids <= set(pipeline.ingestion["completed_images"]):
                raise ValueError("observations stage is incomplete; run it before reader")
            for question in batch.questions:
                qid = str(question["question_id"])
                if qid in completed:
                    if completed[qid].get("metadata", {}).get("source_signature") != catalog.signature:
                        raise ValueError("raw media changed since prediction checkpoint")
                    continue
                result = pipeline.answer(_resolved_question(bundle, question),
                    visible_memory_ids=visible_ids(question, batch.memories))
                if not str(result["prediction"]).strip():
                    raise ValueError(f"empty prediction for {qid}")
                result["metadata"]["source_signature"] = catalog.signature
                with args.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                answered += 1
                print(json.dumps({"question_id": qid, "new": answered,
                    "stop_reason": result["metadata"]["stop_reason"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
