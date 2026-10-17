"""Explicit provenance for the published Memix and standalone dense backends."""
from __future__ import annotations

import hashlib
from pathlib import Path

MEMIX_FILES = (
    "scripts/QA_Agent/MMRAG/memix_memory.py",
    "scripts/QA_Agent/MMRAG/iterative_reasoning_retrieval_eval.py",
    "scripts/QA_Agent/MMRAG/task_aware_cascade_retrieval_eval.py",
)


def backend_hash(backend: str, memix_repo: Path | None = None) -> str:
    if backend == "memix":
        if memix_repo is None:
            raise ValueError("the memix backend requires --memix-repo")
        root, files = Path(memix_repo), MEMIX_FILES
    elif backend == "standalone":
        root = Path(__file__).resolve().parents[1]
        files = (
            "methods/memory_backend.py", "methods/concrete_memix.py",
            "methods/gme.py", "methods/vector_index.py", "methods/media.py",
            "text_cleaning.py", "harness.py",
        )
    else:
        raise ValueError(f"unknown memory backend: {backend}")
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def backend_contract(backend: str, memix_repo: Path | None = None) -> dict[str, str]:
    identity = backend_hash(backend, memix_repo)
    if backend == "memix":
        return {"memix_core_sha256": identity}
    return {"memory_backend": "standalone-dense-v1", "backend_sha256": identity}
