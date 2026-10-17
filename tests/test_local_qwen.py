import base64
import io

from PIL import Image
import pytest

from mm_memory_bench.methods.local_qwen import TransformersQwenTransport


def test_local_reader_preserves_all_images_and_caches_preprocessing():
    captured = []

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            captured.append(messages)
            return {"prepared": True}

    # Test message conversion without loading model weights.
    transport = TransformersQwenTransport.__new__(TransformersQwenTransport)
    transport.processor = Processor()
    transport._prepared_key = transport._prepared = None
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (100, 50, 20)).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Inspect every image"},
        *[{"type": "image_url", "image_url": {"url": url}} for _ in range(12)],
    ]}]
    assert transport._prepare(messages) == {"prepared": True}
    transport._prepare(messages)
    assert len(captured) == 1
    assert len(captured[0][0]["content"]) == 13
    assert all(part["image"].size == (16, 16) for part in captured[0][0]["content"][1:])
    with pytest.raises(ValueError, match="unsupported"):
        transport._prepare([{"role": "user", "content": [{"type": "audio"}]}])
