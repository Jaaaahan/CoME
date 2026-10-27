"""GraphMemix's bounded evidence rendering policy for attached offline observations."""
from __future__ import annotations

import json
from typing import Mapping, Any

RENDER_PROTOCOL = "graphmemix-observation-evidence-2"
MAX_OBSERVATIONS = 4
SNIPPET_OBSERVATION_CHARS = 300
READER_OBSERVATION_CHARS = 1200


def render_observations(memory: Mapping[str, Any], *, max_chars: int,
                        max_observations: int = MAX_OBSERVATIONS) -> tuple[str, list[str]]:
    if max_chars <= 0 or max_observations <= 0:
        return "", []
    field = memory.get("metadata", {}).get("come", {})
    rows = sorted(field.get("observations", []), key=lambda o: o["id"])
    header = "Model observations (verify against sources; not ground truth):"
    lines, ids = [], []
    for obs in rows:
        refs, support = [], []
        for ref in obs["source_refs"]:
            if ref["kind"] == "image":
                refs.append(f"image={ref['asset_id']}")
            else:
                refs.append(f"text={ref['memory_id']}:{ref['part_index']}:{ref['start']}-{ref['end']}")
                support.append(ref.get("text", ""))
        if obs["fact"]["basis"] == "textual" and (not support or not all(support)):
            continue
        # Keep each observation and all its necessary original text atomic. A
        # truncated identity claim without its supporting source is never shown.
        line = f"[{obs['id']}] {obs['description']} ({'; '.join(refs)})"
        if support:
            line += " Original text: " + json.dumps(support, ensure_ascii=False)
        candidate = "\n".join([header, *lines, line])
        if len(candidate) > max_chars:
            continue
        lines.append(line)
        ids.append(obs["id"])
        if len(ids) >= max_observations:
            break
    return ("\n".join([header, *lines]) if lines else ""), ids


def rendering_contract() -> dict:
    from .unit_evidence import UNIT_EVIDENCE_PROTOCOL
    from .observation_context import SOURCE_CONTEXT_PROTOCOL, SOURCE_CONTEXT_TEXT_CHARS
    return {"protocol": RENDER_PROTOCOL, "max_observations": MAX_OBSERVATIONS,
            "evidence_identity": UNIT_EVIDENCE_PROTOCOL,
            "source_context": {"protocol": SOURCE_CONTEXT_PROTOCOL,
                               "original_text_chars": SOURCE_CONTEXT_TEXT_CHARS,
                               "budget": "separate-from-observation-snippet",
                               "truncation": "explicit-prefix-excerpt"},
            "snippet_observation_chars": SNIPPET_OBSERVATION_CHARS,
            "reader_observation_chars": READER_OBSERVATION_CHARS,
            "ordering": "observation_id", "text_support": "complete-or-omit"}


def snippet_observation_ids(memory: Mapping[str, Any], max_chars: int) -> list[str]:
    return render_observations(memory, max_chars=min(SNIPPET_OBSERVATION_CHARS, max_chars * 3 // 4))[1]
