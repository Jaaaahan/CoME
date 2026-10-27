"""Bounded original-source context shared by observation evidence consumers."""
from __future__ import annotations

import json
from typing import Any, Mapping


SOURCE_CONTEXT_PROTOCOL = "graphmemix-observation-source-context-1"
SOURCE_CONTEXT_TEXT_CHARS = 600


def source_context(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Read only public raw fields, never captions, answers or model summaries.

    Call on the validated raw memory, before any derived view replaces or adds
    content. The timestamp describes the record, not an inferred event time.
    """
    context = {
        target: str(memory[key])
        for key, target in (("timestamp", "recorded_timestamp"),
                            ("session_id", "session_id"), ("source_id", "source_id"),
                            ("round_id", "round_id"), ("speaker", "speaker"), ("role", "role"))
        if memory.get(key) is not None
    }
    original = "\n".join(str(part["text"]) for part in memory.get("content", [])
                         if part.get("type") == "text" and part.get("text"))
    context["original_text"] = original[:SOURCE_CONTEXT_TEXT_CHARS]
    context["original_text_truncated"] = len(original) > SOURCE_CONTEXT_TEXT_CHARS
    return context


def render_source_context(context: Mapping[str, Any]) -> str:
    return ("Source context (recorded_timestamp is record time, not inferred event time; "
            "original_text may be an excerpt):\n" + json.dumps(context, ensure_ascii=False))
