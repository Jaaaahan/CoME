"""Lossless transport padding for images outside Qwen's aspect-ratio limit."""
import base64
from functools import lru_cache
import io
import math
from PIL import Image

POLICY = 'qwen-aspect-ratio-200-white-padding-v1'

@lru_cache(maxsize=64)
def pad_image_url(url):
    if not url.startswith('data:image/'):
        return url
    with Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1], validate=True))) as image:
        width, height = image.size
        if max(width, height) / min(width, height) <= 200:
            return url
        # Give the short dimension at least one merged Qwen3-VL patch.
        short = max(32, math.ceil(max(width, height) / 200))
        size = (max(width, short), max(height, short))
        padded = Image.new('RGB', size, 'white')
        padded.paste(image.convert('RGB'), ((size[0] - width) // 2, (size[1] - height) // 2))
        buffer = io.BytesIO()
        padded.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def pad_request_images(payload):
    """Modify only embedded image URLs; preserve text, tools and generation options."""
    for message in payload.get('messages', []):
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get('type') == 'image_url':
                image_url = part.get('image_url')
                if isinstance(image_url, dict) and isinstance(image_url.get('url'), str):
                    image_url['url'] = pad_image_url(image_url['url'])
    return payload
