"""Stable retrieval identities and source-grounded observation image views."""
from __future__ import annotations

import json
from typing import Any, Mapping


UNIT_GRAPH_PROTOCOL = "graphmemix-offline-unit-graph-2"
OBSERVATION_REPRESENTATION = "come_observation_multimodal"


def unit_identity(memory_id: str, observation_id: str | None = None) -> str:
    # JSON tuples avoid collisions even when source IDs contain delimiters.
    return json.dumps([str(memory_id), observation_id], ensure_ascii=False, separators=(",", ":"))


def unit_metadata(unit: Mapping[str, Any]) -> dict[str, Any]:
    mid = str(unit["memory_id"])
    oid = unit.get("observation_id")
    return {"unit_id": unit_identity(mid, oid), "memory_id": mid,
            "kind": "observation" if oid is not None else "memory",
            **({"observation_id": oid, "observation_version": unit["observation_version"]}
               if oid is not None else {})}


def observation_unit(memory: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    """One fused view of the description and the union of its oriented regions."""
    mid = str(memory["memory_id"])
    refs = [ref for ref in observation["source_refs"]
            if ref["kind"] == "image" and ref["asset_id"] == observation["asset_id"]]
    if len(refs) != 1 or refs[0]["memory_id"] != mid:
        raise ValueError("observation needs one memory-local source image")
    position = refs[0]["part_index"]
    sources = memory.get("observation_source_images")
    part = sources[str(position)] if sources is not None else memory["content"][position]
    if part.get("type") != "image" or not part.get("path"):
        raise ValueError("observation source image must be resolved before encoding")
    boxes = observation["regions"]
    if not boxes or observation["coordinate_system"] != "oriented_pixel_xyxy":
        raise ValueError("observation encoding needs oriented pixel regions")
    region = [min(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), max(b[3] for b in boxes)]
    return {"memory_id": mid, "text": observation["description"],
            "image": str(part["path"]), "image_region": region,
            "image_size": list(observation["image_size"]),
            "representation": OBSERVATION_REPRESENTATION,
            "observation_id": observation["id"], "observation_version": observation["version"]}
