"""Pure offline preprocessing. This service has no query or answer interface."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .fields import export_field, source_digest
from .models import MODEL_PROTOCOL, ObservationExtractor
from .source_media import SourceMedia
from .store import CropCache, ObservationStore, SourceCatalog, atomic_json
from .types import PreprocessConfig, OBSERVATION_PROTOCOL, digest


def extraction_contract(model, config: PreprocessConfig) -> dict:
    generation = getattr(model, "config", None)
    # Transport endpoint, reader, query, retrieval and embedding settings do not
    # affect the extraction contract. Only the extraction model and sampling do.
    settings = {key: getattr(generation, key, None) for key in
                ("model", "temperature", "max_output_tokens", "max_model_len", "reasoning_effort")}
    return {"producer_protocol": OBSERVATION_PROTOCOL, "model_protocol": MODEL_PROTOCOL,
            "model": settings, "preprocess": asdict(config)}


class ObservationPreprocessor:
    def __init__(self, memories: list[dict], root: Path, model,
                 config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.catalog = SourceCatalog(memories, chunk_chars=self.config.text_chunk_chars)
        self.root = root
        crops = CropCache(root / "crops", self.catalog, max_edge=self.config.image_max_edge)
        self.models = ObservationExtractor(model, self.catalog, SourceMedia(self.catalog, crops, self.config), self.config)
        self.contract = extraction_contract(model, self.config)
        signature = digest([self.contract, [source_digest(m) for m in memories]])
        self.manifest = root / "ingestion.json"
        self.ingestion = {"contract": signature, "completed_images": [], "usage": {}, "image_outcomes": {}}
        if self.manifest.exists():
            self.ingestion = json.loads(self.manifest.read_text())
            if self.ingestion.get("contract") != signature:
                raise ValueError("incompatible offline extraction snapshot; use a new artifact root")
        elif (root / "observations.jsonl").exists():
            raise ValueError("observations exist without extraction contract")
        self.store = ObservationStore(root / "observations.jsonl", self.catalog)

    def ingest(self) -> list[dict]:
        atomic_json(self.manifest, self.ingestion)
        completed = set(self.ingestion["completed_images"])
        prior_usage = dict(self.ingestion["usage"])
        failed_images = []
        for node in self.catalog.nodes.values():
            if node.kind != "image" or node.node_id in completed:
                continue
            for observation in self.models.observe(node, self.store):
                self.store.put(observation)
            outcome = dict(self.models.last_observation_report)
            # A valid empty response or a grounded subset may complete an image.
            # Failed grounding and all-invalid extraction remain resumable.
            if outcome.get("status") in {"grounding_failed", "partial"} or (
                outcome.get("errors") and outcome.get("status") != "validated_partial"
            ):
                failed_images.append(node.node_id)
                self.ingestion["image_outcomes"][node.node_id] = outcome
                self.ingestion["usage"] = {key: prior_usage.get(key, 0) + value
                                           for key, value in self.models.ingest_usage.items()}
                atomic_json(self.manifest, self.ingestion)
                print(json.dumps({"event": "ingestion_issue", "image": node.node_id,
                                  "status": outcome.get("status"), "accepted": outcome.get("accepted"),
                                  "rejected": outcome.get("rejected")}), flush=True)
                continue
            completed.add(node.node_id)
            self.ingestion["completed_images"] = sorted(completed)
            self.ingestion["image_outcomes"][node.node_id] = outcome
            self.ingestion["usage"] = {key: prior_usage.get(key, 0) + value
                                       for key, value in self.models.ingest_usage.items()}
            atomic_json(self.manifest, self.ingestion)
        if failed_images:
            raise ValueError(f"observation extraction incomplete for {len(failed_images)} images; "
                             f"first={failed_images[0]}; see ingestion.json image_outcomes")
        return [
            {
                "context_id": str(m.get("context_id", "")), 
                "memory_id": str(m["memory_id"]),
                "come": export_field(m, self.catalog, self.store.latest.values())
            }
                for m in self.catalog.memories.values()
        ]
