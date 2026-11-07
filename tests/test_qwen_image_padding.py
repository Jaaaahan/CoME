import base64
import io
from pathlib import Path
import sys
import unittest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cluster/cuhk-suite'))
from qwen_image_padding import pad_image_url, pad_request_images

def encode(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

def decode(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1])))

class PaddingTest(unittest.TestCase):
    def test_thin_images_keep_all_pixels(self):
        for size in [(1280, 6), (6, 1280), (10000, 1)]:
            image = Image.new('RGB', size, (31, 67, 113))
            result = decode(pad_image_url(encode(image)))
            w, h = result.size
            self.assertLessEqual(max(w, h) / min(w, h), 200)
            x, y = (w-size[0])//2, (h-size[1])//2
            self.assertEqual(result.crop((x,y,x+size[0],y+size[1])).tobytes(), image.tobytes())
            self.assertEqual(result.getpixel((0,0)), (255,255,255))

    def test_ordinary_images_and_other_fields_unchanged(self):
        url = encode(Image.new('RGB',(1280,720)))
        payload = {'messages':[{'role':'user','content':[{'type':'text','text':'question'},
            {'type':'image_url','image_url':{'url':url,'detail':'high'}}]}],
            'tools':[{'type':'function'}], 'temperature':0}
        import copy
        before = copy.deepcopy(payload)
        self.assertEqual(pad_request_images(payload), before)

    def test_payload_and_idempotency(self):
        url = encode(Image.new('RGB',(1280,6)))
        payload = {'messages':[{'role':'system','content':'keep'}, {'role':'user','content':[
            {'type':'image_url','image_url':{'url':url}}]}]}
        padded = pad_request_images(payload)['messages'][1]['content'][0]['image_url']['url']
        self.assertNotEqual(padded, url)
        self.assertEqual(pad_image_url(padded), padded)

if __name__ == '__main__':
    unittest.main()
