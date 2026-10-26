"""Exercise the real GME adapter with a stand-in for neural model inference."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from mm_memory_bench.methods.gme import GMEQwen2VLEmbedder
from mm_memory_bench.retrieval_units import observation_unit


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class NeuralModel:
    def __init__(self):
        self.fused = []
        self.texts = []
        self.image_batches = []

    def get_fused_embeddings(self, *, texts, images, batch_size=32):
        self.fused.extend(zip(texts, images))
        return self.get_image_embeddings(images=images, batch_size=batch_size)

    def get_image_embeddings(self, *, images, batch_size=32):
        values = []
        # Emulate GME's native minibatching and a GPU that fits four images.
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            if len(batch) > 4:
                raise RuntimeError('image minibatch exceeds available memory')
            self.image_batches.append(len(batch))
            values.extend([[image.getpixel((0, 0))[0] + 1,
                            image.getpixel((0, 0))[2] + 1] for image in batch])
        return Tensor(values)

    def get_text_embeddings(self, *, texts):
        self.texts.extend(texts)
        return Tensor([[0., 1.] for _ in texts])


def encoder():
    adapter = object.__new__(GMEQwen2VLEmbedder)
    adapter.torch = SimpleNamespace(no_grad=nullcontext)
    adapter.model = NeuralModel()
    adapter.image_batch_size = 4
    return adapter


def test_observations_fuse_description_with_correct_exif_oriented_source_region(tmp_path):
    # After clockwise EXIF rotation, the blue region occupies y=20..60.
    image = Image.new('RGB', (60, 20), 'blue')
    image.paste('red', (0, 0, 20, 20))
    exif = image.getexif(); exif[274] = 6
    image.save(tmp_path / 'source.png', exif=exif)
    Image.new('RGB', (60, 20), 'green').save(tmp_path / 'unrelated.png')
    memory = {'memory_id': 'm', 'content': [
        {'type': 'image', 'path': str(tmp_path / 'unrelated.png')},
        {'type': 'text', 'text': 'Public accompanying text.'},
        {'type': 'image', 'path': str(tmp_path / 'source.png')},
    ]}
    observation = {'id': 'blue', 'asset_id': 'source', 'version': 1,
        'description': 'Two parts of a blue object.', 'image_size': [20, 60],
        'coordinate_system': 'oriented_pixel_xyxy',
        'regions': [[0, 20, 10, 40], [10, 40, 20, 60]],
        'source_refs': [{'kind': 'image', 'memory_id': 'm', 'part_index': 2, 'asset_id': 'source'}]}
    blue = observation_unit(memory, observation)
    red = observation_unit(memory, {**observation, 'id': 'red', 'description': 'A red object.',
                                   'regions': [[0, 0, 20, 20]]})
    adapter = encoder()
    values = adapter.encode_units([blue, {'text': 'whole memory'}, red])
    assert adapter.model.texts == ['whole memory']
    assert [(text, im.size, im.getpixel((0, 0))) for text, im in adapter.model.fused] == [
        ('Two parts of a blue object.', (20, 40), (0, 0, 255)),
        ('A red object.', (20, 20), (255, 0, 0))]
    assert np.array_equal(values, [[1, 256], [0, 1], [256, 1]])


def test_large_mixed_ingestion_preserves_all_units_in_order_with_bounded_image_batches(tmp_path):
    units = []
    expected = []
    for i in range(11):
        path = tmp_path / f'image-{i}.png'
        Image.new('RGB', (20, 20), (i, 0, 100 + i)).save(path)
        units.extend([{'text': f'observation {i}', 'image': str(path)},
                      {'text': f'text memory {i}'}, {'image': str(path)}])
        expected.extend([[i + 1, 101 + i], [0, 1], [i + 1, 101 + i]])
    adapter = encoder()
    values = adapter.encode_units(units)
    assert np.array_equal(values, expected)
    assert len(adapter.model.fused) == 11
    assert len(adapter.model.texts) == 11
    assert sum(adapter.model.image_batches) == 22
    assert max(adapter.model.image_batches) <= 4


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5])
def test_image_batch_size_must_be_a_positive_integer(batch_size):
    with pytest.raises(ValueError, match='image_batch_size'):
        GMEQwen2VLEmbedder(image_batch_size=batch_size)


@pytest.mark.parametrize('region,size,match', [
    ([0, 0, 21, 20], [20, 20], 'bounds'),
    ([0, 0, 20, 20], [30, 20], 'size'),
])
def test_bad_observation_geometry_is_rejected_instead_of_encoding_wrong_pixels(tmp_path, region, size, match):
    path = tmp_path / 'image.png'
    Image.new('RGB', (20, 20), 'blue').save(path)
    with pytest.raises(ValueError, match=match):
        encoder().encode_units([{'text': 'blue', 'image': str(path),
                                 'image_region': region, 'image_size': size}])
