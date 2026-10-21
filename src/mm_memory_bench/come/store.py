from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .types import BBox, Candidate, Observation, PROTOCOL, digest


def file_digest(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(handle)


def _hash_stream(handle) -> str:
    value = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class SourceCatalog:
    """Public raw sources with exact text offsets and oriented image coordinates."""

    def __init__(self, memories: Iterable[dict[str, Any]], *, chunk_chars: int = 800,
                 video_frames: int = 8) -> None:
        from PIL import Image, ImageOps

        self.memories = {str(m["memory_id"]): m for m in memories}
        self.nodes: dict[str, Candidate] = {}
        self.assets: dict[str, dict[str, Any]] = {}
        for mid, memory in self.memories.items():
            for part_index, part in enumerate(memory.get("content", [])):
                kind = part.get("type")
                if kind in {"text", "document", "table"} and part.get("text"):
                    text = str(part["text"])
                    start = 0
                    while start < len(text):
                        end = min(start + chunk_chars, len(text))
                        if end < len(text):
                            split = text.rfind("\n", start + chunk_chars // 2, end)
                            if split >= 0:
                                end = split + 1
                        span = text[start:end]
                        node_id = "text:" + digest([mid, part_index, start, end, span])[:24]
                        self.nodes[node_id] = Candidate(
                            node_id, "text", mid, span, fact_key=node_id,
                            version=digest(span), location={"part_index": part_index,
                                "start": start, "end": end, "source_id": memory.get("source_id"),
                                "speaker": memory.get("speaker", memory.get("role", "")),
                                "timestamp": memory.get("timestamp", "")})
                        start = end
                elif kind == "image" and part.get("path"):
                    aid = str(part.get("asset_id") or digest([mid, part_index]))
                    path = str(part["path"])
                    sha = file_digest(path)
                    with Image.open(path) as raw:
                        width, height = ImageOps.exif_transpose(raw).size
                    self.assets[aid] = {"path": path, "sha256": sha, "width": width,
                                        "height": height, "memory_id": mid, "type": "image"}
                    node_id = "image:" + digest([mid, aid])[:24]
                    self.nodes[node_id] = Candidate(
                        node_id, "image", mid, "Original image", asset_id=aid,
                        version=sha, fact_key=node_id, location={"part_index": part_index,
                            "source_id": memory.get("source_id")})
                elif kind == "video" and part.get("path"):
                    # Existing uniform frame sampler remains the video fallback.
                    aid = str(part.get("asset_id") or digest([mid, part_index]))
                    path = str(part["path"])
                    sha = file_digest(path)
                    self.assets[aid] = {"path": path, "sha256": sha, "memory_id": mid,
                                        "type": "video", "video_frames": video_frames}
                    node_id = "image:" + digest([mid, aid])[:24]
                    self.nodes[node_id] = Candidate(node_id, "image", mid,
                        "Original video (uniform sampled frames)", asset_id=aid,
                        version=sha, fact_key=node_id, location={"part_index": part_index,
                            "source_id": memory.get("source_id")})

    @property
    def signature(self) -> str:
        return digest({key: asdict(value) for key, value in sorted(self.nodes.items())})

    def subset(self, memory_ids: set[str]) -> SourceCatalog:
        if not memory_ids <= self.memories.keys():
            raise ValueError("query scope references unavailable memories")
        result = object.__new__(SourceCatalog)
        result.memories = {mid: self.memories[mid] for mid in memory_ids}
        result.nodes = {nid: n for nid, n in self.nodes.items() if n.memory_id in memory_ids}
        assets = {n.asset_id for n in result.nodes.values() if n.asset_id}
        result.assets = {aid: self.assets[aid] for aid in assets}
        return result

    def image_node(self, asset_id: str, memory_id: str) -> Candidate:
        return next(n for n in self.nodes.values()
                    if n.kind == "image" and n.asset_id == asset_id and n.memory_id == memory_id)

    def validate(self, observation: Observation) -> None:
        asset = self.assets.get(observation.asset_id)
        if not asset or asset["type"] != "image":
            raise ValueError("observation must refer to an available static image")
        self.image_node(observation.asset_id, observation.memory_id)
        if observation.asset_version != asset["sha256"]:
            raise ValueError("observation references a stale image version")
        if not observation.id or not observation.description.strip() or observation.version < 1:
            raise ValueError("invalid observation ID, description, or version")
        if not observation.regions:
            raise ValueError("an observation needs a region")
        for box in observation.regions:
            if len(box) != 4 or any(type(v) is not int for v in box):
                raise ValueError("bbox requires four integer pixel coordinates")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= asset["width"] and 0 <= y1 < y2 <= asset["height"]):
                raise ValueError(f"bbox {box!r} outside oriented image bounds "
                                 f"width={asset['width']}, height={asset['height']}")
        if not observation.fact.subject.startswith(observation.asset_id + ":"):
            raise ValueError(f"entity {observation.fact.subject!r} must start with "
                             f"the image-local prefix {observation.asset_id + ':'!r}")
        if any(ref not in self.nodes for ref in observation.source_refs):
            raise ValueError("observation source reference is unavailable")
        image_id = self.image_node(observation.asset_id, observation.memory_id).node_id
        if image_id not in observation.source_refs:
            raise ValueError("observation must cite its original image")
        if observation.fact.basis == "textual" and not any(
            self.nodes[ref].kind == "text" for ref in observation.source_refs
        ):
            raise ValueError("identity/ownership facts require original textual support")

    def candidate(self, observation: Observation) -> Candidate:
        self.validate(observation)
        textual = frozenset(ref for ref in observation.source_refs if self.nodes[ref].kind == "text")
        return Candidate(observation.id, "observation", observation.memory_id,
            observation.description, observation.asset_id, observation.regions,
            observation.source_refs,
            textual if observation.fact.basis == "textual" else frozenset(),
            observation.fact_key, digest(asdict(observation)),
            {"observation_version": observation.version, "fact": asdict(observation.fact)})


class ObservationStore:
    """Versioned JSONL; the last complete, validated version is indexed."""

    def __init__(self, path: Path | None, catalog: SourceCatalog,
                 initial: Iterable[Observation] = ()) -> None:
        self.path, self.catalog = path, catalog
        self.latest: dict[str, Observation] = {}
        for observation in initial:
            self._accept(observation)
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.pop("protocol", None) != PROTOCOL:
                        raise ValueError("incompatible observation protocol")
                    observation = Observation.from_dict(row)
                    try:
                        catalog.validate(observation)
                    except (ValueError, StopIteration):
                        # Missing/out-of-scope or changed sources are not searchable.
                        continue
                    self._accept(observation)

    def _accept(self, observation: Observation) -> None:
        self.catalog.validate(observation)
        previous = self.latest.get(observation.id)
        if previous:
            if previous.version > observation.version:
                raise ValueError("observation versions must increase")
            if previous.version == observation.version and previous != observation:
                raise ValueError("different observations share one ID and version")
        self.latest[observation.id] = observation

    def put(self, observation: Observation) -> bool:
        previous = self.latest.get(observation.id)
        if previous == observation:
            return False
        self._accept(observation)
        try:
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"protocol": PROTOCOL, **asdict(observation)}, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        except BaseException:
            if previous is None:
                self.latest.pop(observation.id, None)
            else:
                self.latest[observation.id] = previous
            raise
        return True

    def candidates(self) -> list[Candidate]:
        return [self.catalog.candidate(o) for o in self.latest.values()]


class CropCache:
    def __init__(self, root: Path, catalog: SourceCatalog, *, max_edge: int = 1280) -> None:
        self.root, self.catalog, self.max_edge = root, catalog, max_edge

    def image(self, asset_id: str, regions: tuple[BBox, ...] = ()) -> Path:
        from PIL import Image, ImageOps

        asset = self.catalog.assets[asset_id]
        # Detect replacement even within one process; never use a stale crop.
        sha = file_digest(asset["path"])
        if sha != asset["sha256"]:
            raise ValueError("image changed after source snapshot was opened")
        box = None
        if regions:
            box = (min(b[0] for b in regions), min(b[1] for b in regions),
                   max(b[2] for b in regions), max(b[3] for b in regions))
        key = digest(["oriented-union-v1", sha, box, self.max_edge])
        path = self.root / f"{key}.png"
        if not path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            with Image.open(asset["path"]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                if box:
                    if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                        raise ValueError("crop outside image bounds")
                    image = image.crop(box)
                image.thumbnail((self.max_edge, self.max_edge), Image.Resampling.LANCZOS)
                fd, temporary = tempfile.mkstemp(dir=self.root, suffix=".png")
                os.close(fd)
                try:
                    image.save(temporary, format="PNG")
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        return path

    def model_image(self, asset_id: str, regions: tuple[BBox, ...] = ()) -> tuple[Path, dict]:
        """Adapt only the input canvas; keep the source crop and its bbox exact.

        Qwen's image processor rejects aspect ratios greater than 200. Thin,
        valid regions therefore need neutral padding before tokenization and
        inference. Padding is never additional source evidence.
        """
        from PIL import Image, ImageOps

        source = self.image(asset_id, regions)
        with Image.open(source) as raw:
            width, height = raw.size
            target_width = max(width, (height + 199) // 200)
            target_height = max(height, (width + 199) // 200)
            if (width, height) == (target_width, target_height):
                return source, {}
            left, top = (target_width - width) // 2, (target_height - height) // 2
            padding = (left, top, target_width - width - left, target_height - height - top)
            key = digest(["model-aspect-pad-200-v1", file_digest(source), padding])
            path = self.root / f"{key}.png"
            if not path.exists():
                image = ImageOps.expand(raw.convert("RGB"), border=padding, fill=(127, 127, 127))
                fd, temporary = tempfile.mkstemp(dir=self.root, suffix=".png")
                os.close(fd)
                try:
                    image.save(temporary, format="PNG")
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        return path, {"policy": "model-aspect-pad-200-v1", "source_crop_size": [width, height],
            "canvas_size": [target_width, target_height], "padding_ltrb": list(padding)}
