from __future__ import annotations

import json
from typing import Iterable

from ..methods.backends import data_url
from ..methods.media import uniformly_sample_video
from .store import CropCache, SourceCatalog
from .types import Candidate, PreprocessConfig, digest


class SourceMedia:
    def __init__(self, catalog: SourceCatalog, crops: CropCache, config: PreprocessConfig) -> None:
        self.catalog, self.crops, self.config = catalog, crops, config
        self.media: dict[str, list[dict]] = {}

    def node_content(self, nodes: Iterable[Candidate]) -> list[dict]:
        content, emitted = [], set()
        for node in nodes:
            content.append({"type": "text", "text": json.dumps({
                "node_id": node.node_id, "kind": node.kind, "memory_id": node.memory_id,
                "description": node.text, "asset_id": node.asset_id,
                "regions": node.regions, "location": node.location,
                "source_refs": node.source_refs,
            }, ensure_ascii=False)})
            if not node.asset_id:
                continue
            asset = self.catalog.assets[node.asset_id]
            regions = node.regions if self.config.use_crops else ()
            # Union regions are one interaction crop; two different crops are not free.
            union = (() if not regions else (min(b[0] for b in regions), min(b[1] for b in regions),
                      max(b[2] for b in regions), max(b[3] for b in regions)))
            key = digest([asset["sha256"], union, self.config.image_max_edge])
            content.append({"type": "text", "text": f"Visual reference: {key}"})
            if key in emitted:
                continue
            emitted.add(key)
            if key not in self.media:
                if asset["type"] == "video":
                    images = uniformly_sample_video(asset["path"], asset["video_frames"],
                                                    max_edge=self.config.image_max_edge)
                    padding = {}
                else:
                    path, padding = self.crops.model_image(node.asset_id, regions)
                    images = [data_url(str(path))]
                self.media[key] = ([{"type": "text", "text": json.dumps({
                    "visual_input_padding": padding,
                    "note": "The gray border is artificial padding, not source evidence. "
                            "Original regions and source pixels are unchanged."})}] if padding else [])
                self.media[key].extend({"type": "image_url", "image_url": {"url": image}} for image in images)
            content.extend(self.media[key])
        return content

