from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable

from ..methods.backends import estimate_multimodal_input_tokens
from ..methods.media import openai_content_from_parts, question_text
from .types import Candidate, Claim, Gap

from .source_media import SourceMedia


class PacketBuilder(SourceMedia):
    def messages(self, question: dict, nodes: Iterable[Candidate], claims: list[Claim],
                 gaps: list[Gap]) -> list[dict]:
        content = openai_content_from_parts([
            p for p in question.get("prompt", []) if p.get("type") in {"image", "video"}])
        instruction = (f"{question.get('instruction', '')}\nQuestion: {question_text(question)}\n"
            "Answer using this evidence packet. Observation descriptions are derived; inspect the pixels. "
            "Identity and ownership need the cited original text. Query claims are supported interpretations, "
            "not independent sources. Preserve unknowns for unresolved evidence gaps and cite node IDs.")
        if question.get("tool_mode") == "plan":
            instruction += "\nCandidate tools: " + json.dumps(question.get("tools", []), ensure_ascii=False)
        content.append({"type": "text", "text": instruction})
        content.extend(self.node_content(nodes))
        content.append({"type": "text", "text": json.dumps({
            "supported_claims": [{**asdict(c), "supports": sorted(c.supports)} for c in claims],
            "unresolved": [asdict(g) for g in gaps]}, ensure_ascii=False)})
        return [{"role": "user", "content": content}]

    @staticmethod
    def tools(question: dict):
        return question.get("tools") if question.get("tool_mode") != "plan" else None

    def estimate(self, question: dict, nodes: Iterable[Candidate], claims: list[Claim], gaps: list[Gap]) -> int:
        return estimate_multimodal_input_tokens(self.messages(question, nodes, claims, gaps), self.tools(question))
