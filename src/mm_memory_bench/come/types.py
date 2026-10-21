from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


PROTOCOL = "graphmemix-come-1"
QUERY_PROTOCOL = "come-query-original-images-1"
BBox = tuple[int, int, int, int]

# Extraction and encoding have independent contracts. Query settings never
# invalidate a reusable observation snapshot.
OBSERVATION_SCHEMA_VERSION = 1
OBSERVATION_PROTOCOL = "come-observation-field-1"
OBSERVATION_ENCODING_PROTOCOL = "graphmemix-observation-description-1"


@dataclass(frozen=True)
class PreprocessConfig:
    observations_per_image: int = 8
    image_max_edge: int = 1280
    text_chunk_chars: int = 800
    use_crops: bool = True

    def __post_init__(self) -> None:
        for key in ("observations_per_image", "image_max_edge", "text_chunk_chars"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.use_crops) is not bool:
            raise ValueError("use_crops must be boolean")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"),
                                    default=lambda v: sorted(v) if isinstance(v, (set, frozenset)) else str(v)).encode()).hexdigest()


@dataclass(frozen=True)
class CoMEConfig:
    """Historical query experiment settings; new extraction uses PreprocessConfig."""
    observations_per_image: int = 8
    candidate_limit: int = 48
    anchor_count: int = 10
    anchors_per_candidate: int = 3
    proposal_k: int = 10
    source_top_k: int = 200
    max_inspection_rounds: int = 2
    inspection_images_per_round: int = 3
    max_work_calls: int = 32
    max_work_tokens: int = 160000
    verifier_batch_size: int = 6
    retrieval_batch_size: int = 16
    image_max_edge: int = 1280
    text_chunk_chars: int = 800
    edge_weight: float = 0.1
    root_cost: float = 0.2
    use_crops: bool = True
    use_dependencies: bool = True
    isolate_queries: bool = True  # Retained for config compatibility; queries never write observations.

    def __post_init__(self) -> None:
        for key in ("observations_per_image", "candidate_limit", "anchor_count",
                    "anchors_per_candidate", "proposal_k", "source_top_k",
                    "inspection_images_per_round", "max_work_calls", "max_work_tokens",
                    "verifier_batch_size", "retrieval_batch_size", "image_max_edge", "text_chunk_chars"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 <= self.max_inspection_rounds <= 2:
            raise ValueError("v1 permits at most two inspection rounds")
        if self.candidate_limit < 3 or self.proposal_k > self.candidate_limit:
            raise ValueError("candidate_limit must cover three routes and proposal_k")
        if self.anchors_per_candidate > self.anchor_count:
            raise ValueError("anchors_per_candidate exceeds anchor_count")
        if any(not math.isfinite(v) or v < 0 for v in (self.edge_weight, self.root_cost)):
            raise ValueError("forest costs must be finite and nonnegative")


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    value: str
    basis: str = "visual"

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v.strip() for v in
                   (self.subject, self.predicate, self.value)):
            raise ValueError("a fact needs subject, predicate, and value")
        if self.basis not in {"visual", "textual"}:
            raise ValueError("fact basis must be visual or textual")


@dataclass(frozen=True)
class Observation:
    id: str
    memory_id: str
    asset_id: str
    regions: tuple[BBox, ...]
    description: str
    source_refs: tuple[str, ...]
    version: int
    fact: Fact
    asset_version: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Observation:
        return cls(**{**row, "regions": tuple(tuple(b) for b in row["regions"]),
                      "source_refs": tuple(row["source_refs"]), "fact": Fact(**row["fact"])})

    @property
    def fact_key(self) -> str:
        # Values remain distinct: conflicting values must not be silently merged.
        return digest([self.memory_id, self.asset_id,
                       self.fact.subject, self.fact.predicate, self.fact.value])


@dataclass(frozen=True)
class Candidate:
    node_id: str
    kind: str
    memory_id: str
    text: str
    asset_id: str = ""
    regions: tuple[BBox, ...] = ()
    source_refs: tuple[str, ...] = ()
    requires: frozenset[str] = frozenset()
    fact_key: str = ""
    version: str = ""
    location: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    id: str
    description: str
    supports: frozenset[str]
    anchor_id: str
    utility: float
    fact_key: str


@dataclass(frozen=True)
class Edge:
    left: str
    right: str
    similarity: float
    role: str
    supports: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Gap:
    kind: str                 # target_missing | attribute_missing | ambiguous_binding
    description: str
    node_ids: tuple[str, ...] = ()


@dataclass
class Verification:
    prizes: dict[str, float] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    requires: dict[str, frozenset[str]] = field(default_factory=dict)


@dataclass
class QueryState:
    work_calls: int = 0
    work_tokens: int = 0
    completion_tokens: int = 0
    attempts: set[str] = field(default_factory=set)
    verification_cache: dict[str, Any] = field(default_factory=dict)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""

    def exhausted(self, config: CoMEConfig) -> bool:
        return self.work_calls >= config.max_work_calls or self.work_tokens >= config.max_work_tokens


@dataclass
class EvidenceDelta:
    candidates: list[Candidate] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)

    @property
    def has_progress(self) -> bool:
        return bool(self.candidates)
