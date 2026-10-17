"""Local Transformers transport for fixed-evidence Qwen3-VL experiments.

Keeps the existing reader's prompts and context checks, without a server's
image-count cap. Uses greedy decoding and never drops selected media.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from typing import Any, Mapping


class TransformersQwenTransport:
    def __init__(self, model_path: str, max_model_len: int = 32768):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        self.max_model_len = max_model_len
        self._prepared_key = None
        self._prepared = None

    def _prepare(self, messages):
        from PIL import Image

        key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if key == self._prepared_key:
            return self._prepared
        converted = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append(dict(part))
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/"):
                        raise ValueError("local reader accepts embedded images only")
                    raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                    with Image.open(io.BytesIO(raw)) as image:
                        parts.append({"type": "image", "image": image.convert("RGB")})
                else:
                    raise ValueError(f"unsupported local reader content: {part['type']}")
            converted.append({"role": message["role"], "content": parts})
        inputs = self.processor.apply_chat_template(
            converted, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        self._prepared_key, self._prepared = key, inputs
        return inputs

    def __call__(self, route: str, payload: Mapping[str, Any]):
        if payload.get("tools") or payload.get("response_format"):
            raise ValueError("local reader transport supports plain multimodal QA only")
        inputs = self._prepare(payload["messages"])
        count = int(inputs["input_ids"].shape[-1])
        if route == "/tokenize":
            return {"count": count}
        if route != "/chat/completions":
            raise ValueError(f"unsupported route: {route}")
        budget = int(payload.get("max_tokens", 1000))
        if count + budget > self.max_model_len:
            raise ValueError("local reader request exceeds the context window")
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs.to(self.model.device), max_new_tokens=budget,
                    do_sample=False, use_cache=True,
                )
            tokens = output[0, count:]
            text = self.processor.decode(tokens, skip_special_tokens=True)
            return {
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": count, "completion_tokens": len(tokens)},
            }
        finally:
            self._prepared_key, self._prepared = None, None
