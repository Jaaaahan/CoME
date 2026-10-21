"""Extract reusable observations from bundle sources, without loading questions or GME."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from ..methods.backends import OpenAICompatibleQwenVL
from ..methods.base import GenerationConfig
from ..reader import BundleReader
from .fields import ObservationSnapshot, manifest_path, raw_memories, write_snapshot
from .pipeline import ObservationPreprocessor, extraction_contract
from .types import PreprocessConfig, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent memory shards; each has its own resumable store")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = json.loads(args.config.read_text())
    preprocess = PreprocessConfig(**config.get("come", {}))
    settings = config.get("observations", {})
    generation = GenerationConfig(model=args.model, base_url=args.base_url,
        max_model_len=settings.get("max_model_len", 32768),
        max_output_tokens=settings.get("max_output_tokens", 8192),
        temperature=0, reasoning_effort=args.reasoning_effort)
    model = OpenAICompatibleQwenVL(generation)
    contract = extraction_contract(model, preprocess)
    memories = raw_memories(BundleReader(args.bundle))
    if args.output.exists() and manifest_path(args.output).exists():
        snapshot = ObservationSnapshot(args.output, memories)
        if snapshot.contract["extraction_contract"] != contract:
            raise ValueError("incompatible observation extraction contract; use a new output path")
        print(json.dumps({"status": "complete", "reused": True, "memories": len(memories)}))
        return 0
    def extract_shard(worker: int) -> list[dict]:
        # v1 source references are memory-local. Never split one memory across
        # workers or share mutable catalogs/stores/model diagnostics between them.
        shard_model = OpenAICompatibleQwenVL(generation)
        root = (args.artifact_root if args.workers == 1 else
                args.artifact_root / f"workers-{args.workers}" / f"worker-{worker:02d}")
        contexts = {}
        for memory in memories[worker::args.workers]:
            contexts.setdefault(str(memory.get("context_id", "")), []).append(memory)
        result = []
        for cid, sources in contexts.items():
            service = ObservationPreprocessor(sources, root / "contexts" / digest(cid), shard_model, preprocess)
            result.extend(service.ingest())
            print(json.dumps({"worker": worker, "context_id": cid, "memories": len(sources),
                              "usage": service.ingestion["usage"]}), flush=True)
        return result

    if args.workers == 1:
        rows = extract_shard(0)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = [row for shard in pool.map(extract_shard, range(args.workers)) for row in shard]
    write_snapshot(args.output, rows, extraction_contract=contract)
    ObservationSnapshot(args.output, memories)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
