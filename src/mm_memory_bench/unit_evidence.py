"""Independent unit evidence views for verifiers, selection, and the Reader."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .memory_scope import visible_ids
from .observation_context import render_source_context
from .retrieval_units import unit_identity


UNIT_EVIDENCE_PROTOCOL = "graphmemix-unit-evidence-1"


def _field(memory, observations):
    return {**memory.get("metadata", {}).get("come", {}), "observations": observations}


class UnitEvidenceCatalog:
    """Resolve a unit without replacing it by its parent or sibling observations."""

    def __init__(self, memories: Mapping[str, Mapping[str, Any]]) -> None:
        self.memories = memories
        self.records: dict[str, dict[str, Any]] = {}
        self.owners: dict[str, str] = {}
        for mid, memory in memories.items():
            uid = unit_identity(mid)
            self.owners[uid] = mid
            self.records[uid] = {**memory, "unit_id": uid, "parent_memory_id": mid,
                "metadata": {**memory.get("metadata", {}), "come": _field(memory, [])}}
            for observation in memory.get("metadata", {}).get("come", {}).get("observations", []):
                uid = unit_identity(mid, observation["id"])
                if uid in self.records:
                    raise ValueError("duplicate observation evidence unit")
                self.owners[uid] = mid
                # Preserve bounded raw source context separately from the
                # observation. Do not inherit captions or sibling observations.
                metadata = {key: value for key, value in memory.get("metadata", {}).items()
                            if key not in {"derived", "come"}}
                metadata["come"] = _field(memory, [observation])
                self.records[uid] = {**memory, "unit_id": uid, "parent_memory_id": mid,
                    "observation_id": observation["id"], "kind": "observation",
                    "source_context": deepcopy(memory["observation_source_context"]),
                    "content": [{"type": "image", "asset_id": observation["asset_id"]}], "metadata": metadata}

    def visible(self, question: Mapping[str, Any]) -> set[str]:
        parents = visible_ids(question, self.memories.values())
        return {uid for uid, mid in self.owners.items() if mid in parents}

    def require(self, ids: Sequence[str], question: Mapping[str, Any] | None = None) -> None:
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evidence unit IDs")
        allowed = self.visible(question) if question is not None else self.owners.keys()
        if not set(ids) <= allowed:
            raise ValueError("evidence units are unknown or outside the question memory scope")

    def selection(self, ids: Sequence[str]) -> dict[str, Any]:
        self.require(ids)
        return {"retrieved_unit_ids": list(ids),
                # Original IDs are solely for benchmark source-level reporting.
                # No agent or subsequent selector consumes this projection.
                "retrieved_memory_ids": list(dict.fromkeys(self.owners[uid] for uid in ids))}


def selected_evidence_ids(row: Mapping[str, Any], *, unit_level: bool) -> list[str]:
    key = "retrieved_unit_ids" if unit_level else "retrieved_memory_ids"
    if unit_level and key not in row:
        raise ValueError("selection lacks unit IDs; regenerate unit-level selection")
    ids = list(map(str, row.get(key, [])))
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate selected evidence IDs")
    return ids


def reader_unit_records(records: Sequence[Mapping[str, Any]], embedding_units: Sequence[Mapping[str, Any]],
                        public_asset_paths: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Hydrate only metadata and pixels for each independent Reader unit."""
    observation_views = {unit_identity(str(u["memory_id"]), u["observation_id"]): u
                         for u in embedding_units if u.get("observation_id")}
    result = {}
    for record in records:
        mid = str(record["item_id"])
        uid = unit_identity(mid)
        parent = deepcopy(record)
        parent.update(item_id=uid, unit_id=uid, parent_memory_id=mid)
        parent["metadata"]["come"] = _field(record, [])
        result[uid] = parent
        for obs in record.get("metadata", {}).get("come", {}).get("observations", []):
            uid = unit_identity(mid, obs["id"])
            if uid not in observation_views:
                raise ValueError("observation evidence lacks its checkpoint image unit")
            view = observation_views[uid]
            if view.get("observation_version") != obs["version"]:
                raise ValueError("observation evidence version differs from checkpoint")
            path = public_asset_paths.get(obs["asset_id"], view.get("image"))
            if not path or not view.get("image_region"):
                raise ValueError("observation evidence lacks its source image region")
            if "observation_source_context" not in record:
                raise ValueError("observation evidence lacks validated source context; attach the current snapshot")
            context = deepcopy(record["observation_source_context"])
            result[uid] = {**record, "item_id": uid, "unit_id": uid, "parent_memory_id": mid,
                "observation_id": obs["id"], "modality": "image", "text": render_source_context(context),
                "source_context": context,
                "image_path": str(path), "video_path": None,
                "metadata": {**record.get("metadata", {}), "come": _field(record, [obs])},
                "content": [{"type": "image", "path": str(path), "asset_id": obs["asset_id"],
                             "image_region": list(view["image_region"]), "image_size": list(view["image_size"])}]}
    return result


def evidence_identity(record: Mapping[str, Any], fallback_id: str) -> str:
    if record.get("unit_id"):
        return f"unit_id={record['unit_id']}; memory_id={record['parent_memory_id']}"
    return f"memory_id={fallback_id}"
