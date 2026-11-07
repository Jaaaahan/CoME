import json
from pathlib import Path
from scripts.prepare_smmbench import convert
from mm_memory_bench.core import iter_jsonl


def test_raw_conversion_preserves_scope_document_order_and_keeps_gold_private(tmp_path):
    raw = tmp_path / 'raw'
    images = raw / 'Images'
    images.mkdir(parents=True)
    (images / 'a.png').write_bytes(b'image fixture')
    cluster = raw / 'Samples/cluster_1'
    cluster.mkdir(parents=True)
    rows = [
        dict(timestamp='2023-01-02 00:00:00', sender_name='A', content_type='image',
             content='', image_path='a.png', caption='Fig. abc\nSUPPLIED_VISUAL_DESCRIPTION'),
        dict(timestamp='2023-01-01 00:00:00', sender_name='B', content_type='json_evidence',
             image_path=None, caption='Doc. xyz\nSUPPLIED_DOCUMENT_SUMMARY', content=[
                 {'type': 'text', 'content': {'text': 'original document text'}},
                 {'type': 'image', 'content': {'image_path': 'a.png'}, 'caption': 'NESTED_DESCRIPTION'}]),
    ]
    (cluster / 'chat.json').write_text(json.dumps({'conversation': rows}))
    (cluster / 'QA_sample.json').write_text(json.dumps([
        {'id': 'q1', 'question': 'Which figure?', 'answer': 'SECRET_GOLD', 'evidence': {'private': 'SECRET_EVIDENCE'}}]))
    output = tmp_path / 'bundle'
    report = convert(raw, output)
    assert report['counts'] == dict(contexts=1, memories=2, assets=1, questions=1)
    memories = list(iter_jsonl(output / 'memories.jsonl'))
    assert memories[0]['content'][:2] == [
        {'type': 'text', 'text': 'Doc. xyz'}, {'type': 'text', 'text': 'original document text'}]
    assert memories[0]['content'][2]['type'] == 'image'
    assert memories[0]['memory_id'].endswith(':chat:1')
    assert memories[1]['content'][0]['text'] == 'Fig. abc'
    assert [r['sequence'] for r in memories] == [0, 1]
    public = (output / 'memories.jsonl').read_text()
    assert all(value not in public for value in ['SECRET_GOLD', 'SECRET_EVIDENCE', 'SUPPLIED_', 'NESTED_DESCRIPTION'])
    questions = list(iter_jsonl(output / 'questions.jsonl'))
    assert questions[0]['answer']['text'] == 'SECRET_GOLD'
    assert questions[0]['context_id'] == memories[0]['context_id']
    assert questions[0]['memory_scope'] == {'mode': 'all'}
