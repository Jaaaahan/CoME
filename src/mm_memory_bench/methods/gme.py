from __future__ import annotations

import base64
import io
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _load_image(value: str, *, region=None, expected_size=None):
    from PIL import Image, ImageOps

    if value.startswith("data:"):
        payload = value.split(",", 1)[1]
        source = io.BytesIO(base64.b64decode(payload))
    else:
        source = value
    with Image.open(source) as original:
        # Preserve existing whole-memory embeddings; observations explicitly use
        # the oriented coordinate system exported by CoME.
        image = (ImageOps.exif_transpose(original) if region is not None else original).convert("RGB")
        if region is not None:
            if expected_size is None or list(image.size) != list(expected_size):
                raise ValueError("observation image size differs from oriented source")
            if (len(region) != 4 or any(type(v) is not int for v in region) or
                    not (0 <= region[0] < region[2] <= image.width and
                         0 <= region[1] < region[3] <= image.height)):
                raise ValueError("observation crop outside oriented image bounds")
            image = image.crop(tuple(region))
        return image


def _unit_image(unit: Mapping[str, Any]):
    return _load_image(str(unit["image"]), region=unit.get("image_region"),
                       expected_size=unit.get("image_size"))


class GMEQwen2VLEmbedder:
    """Adapter for GME-Qwen2-VL multimodal embeddings."""

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        *,
        dimension: int = 1536,
        image_batch_size: int = 4,
    ) -> None:
        if type(image_batch_size) is not int or image_batch_size < 1:
            raise ValueError("image_batch_size must be a positive integer")
        # An ingestion batch can contain many observation crops per memory.
        # Bound the neural image batch separately from memory ingestion; GME's
        # default of 32 images can exhaust a 24 GiB GPU at native resolution.
        self.image_batch_size = image_batch_size
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for GME") from exc
        self.torch = torch
        use_cuda = torch.cuda.is_available()
        # This adapter runs one model per process. Managed pools can inject
        # WORLD_SIZE without launching torchrun; "auto" then incorrectly enables
        # tensor parallelism in Transformers. Respect the process's selected GPU.
        device = torch.cuda.current_device() if use_cuda else "cpu"
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device_map={"": device},
            trust_remote_code=True,
        ).eval()
        self._dimension = dimension
        root = Path(model_name)
        if root.is_dir():
            digest = hashlib.sha256()
            for path in sorted(root.iterdir()):
                if path.is_file() and (path.suffix == ".safetensors" or path.name in {
                    "config.json", "modeling_gme_qwen2vl.py", "preprocessor_config.json",
                    "tokenizer.json", "tokenizer_config.json",
                }):
                    digest.update(path.name.encode())
                    with path.open("rb") as handle:
                        file_digest = hashlib.sha256()
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            file_digest.update(chunk)
                        digest.update(file_digest.digest())
            self.model_identity = digest.hexdigest()
        else:
            self.model_identity = f"{model_name}@{getattr(self.model.config, '_commit_hash', None)}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_units(self, units: Sequence[Mapping[str, Any]]) -> np.ndarray:
        rows: list[np.ndarray | None] = [None] * len(units)
        with self.torch.no_grad():
            text_indices = []
            image_indices = []
            fused_indices = []
            for index, unit in enumerate(units):
                text = str(unit.get("embedding_text", unit.get("text", ""))).strip()
                image_value = unit.get("image")
                if image_value and text:
                    fused_indices.append(index)
                elif image_value:
                    image_indices.append(index)
                else:
                    text_indices.append(index)
            groups = (
                (
                    fused_indices,
                    lambda ids: self.model.get_fused_embeddings(
                        texts=[
                            str(units[i].get("embedding_text", units[i].get("text", ""))).strip()
                            for i in ids
                        ],
                        images=[_unit_image(units[i]) for i in ids],
                        batch_size=self.image_batch_size,
                    ),
                ),
                (
                    image_indices,
                    lambda ids: self.model.get_image_embeddings(
                        images=[_unit_image(units[i]) for i in ids],
                        batch_size=self.image_batch_size,
                    ),
                ),
                (
                    text_indices,
                    lambda ids: self.model.get_text_embeddings(
                        texts=[
                            str(units[i].get("embedding_text", units[i].get("text", ""))).strip()
                            or " "
                            for i in ids
                        ]
                    ),
                ),
            )
            for indices, encode in groups:
                if not indices:
                    continue
                values = encode(indices).detach().float().cpu().numpy()
                for index, value in zip(indices, values):
                    rows[index] = value
        if any(row is None for row in rows):
            raise RuntimeError("GME failed to encode every retrieval unit")
        return np.asarray(rows, dtype=np.float32)
