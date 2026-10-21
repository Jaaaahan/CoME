"""Versioned, read-only observation sidecar interface. No retrieval or model calls."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from ..core import iter_jsonl
from .store import atomic_json, file_digest
from .types import (Fact, Observation, OBSERVATION_SCHEMA_VERSION,
                    OBSERVATION_PROTOCOL, OBSERVATION_ENCODING_PROTOCOL, digest)


def manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def source_digest(memory: Mapping[str, Any]) -> str:
    """Portable hash of public raw sources, including media bytes, never questions."""
    value = deepcopy(dict(memory))
    # Derived fields and agent metadata are not inputs to observation extraction.
    value.pop("metadata", None)
    for part in value.get("content", []):
        path = part.pop("path", None)
        if path:
            part["sha256"] = file_digest(path)
    return digest(value)


def raw_memories(reader) -> list[dict]:
    # Deliberately never call iter_context_batches: that iterator reads questions
    # and omits contexts with no questions.
    from ..harness import _resolved_memory
    return [_resolved_memory(reader, row, memory_view="raw")
            for row in iter_jsonl(reader.table_paths["memories"])]


def export_field(memory: dict, catalog, observations: Iterable[Observation]) -> dict:
    exported = []
    seen = set()
    for obs in sorted(observations, key=lambda o: o.id):
        if obs.memory_id != str(memory["memory_id"]):
            continue
        catalog.validate(obs)
        refs = []
        for nid in obs.source_refs:
            node = catalog.nodes[nid]
            if node.memory_id != obs.memory_id:
                raise ValueError("v1 observations require memory-local source references")
            ref = {"kind": node.kind, "memory_id": node.memory_id,
                   "part_index": node.location["part_index"]}
            if node.kind == "text":
                text = catalog.memories[node.memory_id]["content"][ref["part_index"]]["text"]
                ref.update(start=node.location["start"], end=node.location["end"],
                           source_version=digest(text), text=node.text)
            else:
                ref.update(asset_id=node.asset_id, source_version=node.version)
            refs.append(ref)
        key = digest([asdict(obs.fact), refs, obs.asset_version])
        if key in seen:
            continue
        seen.add(key)
        asset = catalog.assets[obs.asset_id]
        exported.append({**asdict(obs), "source_refs": refs,
                         "coordinate_system": "oriented_pixel_xyxy",
                         "image_size": [asset["width"], asset["height"]]})
    return {"schema_version": OBSERVATION_SCHEMA_VERSION,
            "producer_protocol": OBSERVATION_PROTOCOL,
            "source_digest": source_digest(memory), "observations": exported}


def validate_field(field: dict, memory: Mapping[str, Any]) -> dict:
    if set(field) != {"schema_version", "producer_protocol", "source_digest", "observations"}:
        raise ValueError("unknown or missing observation field keys")
    if (type(field["schema_version"]) is not int or
        field["schema_version"] != OBSERVATION_SCHEMA_VERSION or
        field["producer_protocol"] != OBSERVATION_PROTOCOL):
        raise ValueError("incompatible observation schema or producer protocol")
    if field["source_digest"] != source_digest(memory):
        raise ValueError(f"observation source changed: {memory['memory_id']}")
    if not isinstance(field["observations"], list):
        raise ValueError("observations must be a list, including when empty")
    mid = str(memory["memory_id"])
    parts = memory.get("content", [])
    ids, facts, result = set(), set(), []
    for obs in field["observations"]:
        required = {"id", "memory_id", "asset_id", "asset_version", "version", "description",
                    "fact", "coordinate_system", "image_size", "regions", "source_refs"}
        if not isinstance(obs, dict) or set(obs) != required:
            raise ValueError("unknown or missing observation keys")
        if (not isinstance(obs["id"], str) or not obs["id"] or obs["id"] in ids or
            obs["memory_id"] != mid or type(obs["version"]) is not int or obs["version"] < 1 or
            not isinstance(obs["description"], str) or not obs["description"].strip()):
            raise ValueError("invalid observation identity, version or description")
        ids.add(obs["id"])
        fact = Fact(**obs["fact"])
        if not fact.subject.startswith(obs["asset_id"] + ":"):
            raise ValueError("fact subject must use its image namespace")
        if obs["coordinate_system"] != "oriented_pixel_xyxy":
            raise ValueError("unsupported observation coordinate system; never guess coordinates")
        if not isinstance(obs["source_refs"], list) or not obs["source_refs"]:
            raise ValueError("observation needs independent source references")
        image_found, text_found = False, False
        for ref in obs["source_refs"]:
            if ref.get("memory_id") != mid:
                raise ValueError("v1 observations require memory-local source references")
            index = ref.get("part_index")
            if type(index) is not int or not 0 <= index < len(parts):
                raise ValueError("invalid source part reference")
            part = parts[index]
            if ref.get("kind") == "text":
                if set(ref) != {"kind", "memory_id", "part_index", "start", "end", "source_version", "text"}:
                    raise ValueError("invalid text source reference")
                text = part.get("text", "")
                start, end = ref["start"], ref["end"]
                if (part.get("type") not in {"text", "document", "table"} or
                    type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or
                    ref["source_version"] != digest(text) or ref["text"] != text[start:end]):
                    raise ValueError("stale or invalid original text reference")
                text_found = True
            elif ref.get("kind") == "image":
                if set(ref) != {"kind", "memory_id", "part_index", "asset_id", "source_version"}:
                    raise ValueError("invalid image source reference")
                aid = str(part.get("asset_id") or digest([mid, index]))
                if (part.get("type") != "image" or aid != obs["asset_id"] or
                    ref["asset_id"] != aid or ref["source_version"] != obs["asset_version"] or
                    file_digest(part["path"]) != obs["asset_version"]):
                    raise ValueError("stale or invalid image source reference")
                from PIL import Image, ImageOps
                with Image.open(part["path"]) as image:
                    size = list(ImageOps.exif_transpose(image).size)
                if obs["image_size"] != size:
                    raise ValueError("observation image size differs from oriented source")
                if not obs["regions"]:
                    raise ValueError("observation needs pixel regions")
                for box in obs["regions"]:
                    if (len(box) != 4 or any(type(v) is not int for v in box) or
                        not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])):
                        raise ValueError("observation region outside oriented pixel bounds")
                image_found = True
            else:
                raise ValueError("unsupported observation source kind")
        if not image_found or (fact.basis == "textual" and not text_found):
            raise ValueError("observation lacks required original image/text support")
        key = digest([obs["fact"], sorted(obs["source_refs"], key=digest), obs["asset_version"]])
        if key not in facts:
            result.append(deepcopy(obs))
            facts.add(key)
    return {**field, "observations": sorted(result, key=lambda o: o["id"])}


class ObservationSnapshot:
    """Fully validated snapshot. Attach returns copies; querying has no writer API."""

    def __init__(self, path: Path, memories: Iterable[Mapping[str, Any]]) -> None:
        path = Path(path)
        manifest = manifest_path(path)
        if not path.is_file() or not manifest.is_file():
            raise ValueError("observations stage is incomplete: missing sidecar or manifest")
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if (metadata.get("complete") is not True or metadata.get("schema_version") != OBSERVATION_SCHEMA_VERSION or
            metadata.get("producer_protocol") != OBSERVATION_PROTOCOL):
            raise ValueError("incomplete or incompatible observation manifest")
        if metadata.get("snapshot_sha256") != file_digest(path):
            raise ValueError("observation snapshot digest mismatch")
        self._fields = {}
        sources = {(str(m.get("context_id", "")), str(m["memory_id"])): m for m in memories}
        for row in iter_jsonl(path):
            if set(row) != {"context_id", "memory_id", "come"}:
                raise ValueError("invalid observation sidecar row")
            key = (str(row["context_id"]), str(row["memory_id"]))
            if key not in sources or key in self._fields:
                raise ValueError("duplicate or unavailable observation memory")
            self._fields[key] = validate_field(row["come"], sources[key])
        if self._fields.keys() != sources.keys():
            raise ValueError("observations stage is incomplete: every memory needs a row")
        source_hash = digest([[*key, field["source_digest"]] for key, field in sorted(self._fields.items())])
        if metadata.get("sources_sha256") != source_hash:
            raise ValueError("observation source snapshot mismatch")
        self.contract = {"schema_version": OBSERVATION_SCHEMA_VERSION,
                         "producer_protocol": OBSERVATION_PROTOCOL,
                         "snapshot_sha256": metadata["snapshot_sha256"],
                         "sources_sha256": source_hash,
                         "extraction_contract": metadata["extraction_contract"],
                         "encoding_protocol": OBSERVATION_ENCODING_PROTOCOL}

    @classmethod
    def from_bundle(cls, path: Path, reader):
        return cls(path, raw_memories(reader))

    def attach(self, memory: Mapping[str, Any]) -> dict:
        key = (str(memory.get("context_id", "")), str(memory.get("memory_id", memory.get("item_id", ""))))
        if key not in self._fields:
            raise ValueError(f"observation snapshot missing memory {key}")
        return {**memory, "metadata": {**memory.get("metadata", {}), "come": deepcopy(self._fields[key])}}

    def require_contract(self, actual: Mapping[str, Any] | None) -> None:
        if actual != self.contract:
            raise ValueError("observation snapshot contract mismatch; rebuild dependent artifacts")


def write_snapshot(path: Path, rows: list[dict], *, extraction_contract: dict) -> None:
    """Offline-only writer; publish the completion marker last."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for row in sorted(rows, key=lambda r: (r["context_id"], r["memory_id"])):
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    atomic_json(manifest_path(path), {
        "complete": True, "schema_version": OBSERVATION_SCHEMA_VERSION,
        "producer_protocol": OBSERVATION_PROTOCOL, "snapshot_sha256": file_digest(path),
        "sources_sha256": digest([[r["context_id"], r["memory_id"], r["come"]["source_digest"]]
                                  for r in sorted(rows, key=lambda r: (r["context_id"], r["memory_id"]))]),
        "extraction_contract": extraction_contract,
    })
