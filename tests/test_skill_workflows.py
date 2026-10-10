"""Exercise multi-step skill bookkeeping with actual rendered attachments."""
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from viewcraft.models import Usage
from viewcraft.runtime import run_static_item, select_visible
from viewcraft.schema import PublicItem
from viewcraft.views.config import Coordinates

ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def call(self, messages, **kwargs):
        text, images = [], []
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                text.append(content)
                continue
            for part in content:
                if part['type'] == 'text':
                    text.append(part['text'])
                elif part['type'] == 'image_url':
                    raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                    images.append(Image.open(io.BytesIO(raw)).convert('RGB'))
        self.calls.append(('\n'.join(text), images))
        return SimpleNamespace(content=json.dumps(next(self.actions)),
                               request_id='offline', usage=Usage())


def run(tmp_path, actions, n=1, steps=8):
    paths = []
    for i in range(n):
        path = tmp_path / f'page-{i}.png'
        Image.new('RGB', (128, 128), (100 + i, 40, 200)).save(path)
        paths.append(str(path))
    item = PublicItem('workflow', 'custom', 'Inspect the evidence.', tuple(paths),
                      meta={'source_labels': [f'report.pdf page {i+1}' for i in range(n)]})
    client = Client(actions + [{'action': 'answer', 'answer': 'done'}])
    coords = Coordinates.read_snapshot(ROOT / 'skill-library/pdf-reading')
    traj = run_static_item(item, coords, client,
        {'observations': steps, 'max_images_per_request': 8}, 'offline',
        arm='viewcraft', condition='normal', split='inference', answer_style='free_text')
    return client.calls, traj


@pytest.mark.parametrize('steps', [1, 8])
def test_working_note_survives_next_call_and_forced_answer(tmp_path, steps):
    calls, _ = run(tmp_path, [{'action': 'zoom', 'image': 1,
        'regions': [{'x0': .1, 'y0': .1, 'x1': .5, 'y1': .5}],
        'note': 'A: IDs 1-17 accepted; candidate 18 unresolved'}], steps=steps)
    assert 'A: IDs 1-17 accepted; candidate 18 unresolved' in calls[1][0]


def test_evicted_page_can_be_requested_again(tmp_path):
    calls, _ = run(tmp_path, [
        {'action': 'view', 'units': list(range(0, 16, 2))},
        {'action': 'view', 'units': [1]}], n=16)
    assert all(im.getpixel((0, 0))[0] != 101 for im in calls[1][1])
    assert any(im.getpixel((0, 0))[0] == 101 for im in calls[2][1])
    assert 'source unit 1' in calls[2][0]
    assert 'report.pdf page 2' in calls[2][0]


def test_marking_new_page_keeps_it_attached(tmp_path):
    calls, traj = run(tmp_path, [
        {'action': 'view', 'units': [0]},
        {'action': 'note', 'image': 8,
         'notes': [{'at': [.5, .5], 'text': 'field', 'id': 'A'}]}], n=16)
    attached = [im for im in calls[2][1] if im.getpixel((0, 0))[0] == 100]
    assert attached, 'newly viewed page disappeared after annotation'
    assert len(attached[0].getcolors(128 * 128)) > 1, 'page has no marker'
    assert len(traj.canvas_notes) == 1


def test_nearby_markers_keep_distinct_ids_and_allow_correction(tmp_path):
    notes = [{'at': [.5 + i*.002, .5], 'id': str(i), 'text': 'ok'} for i in range(6)]
    _, traj = run(tmp_path, [
        {'action': 'note', 'notes': notes},
        {'action': 'note', 'notes': [{'at': [.2, .3], 'id': '0', 'text': 'excluded'}]}])
    assert len(traj.canvas_notes) == 6
    corrected = next(n for n in traj.canvas_notes if n['id'] == '0')
    assert corrected['text'] == 'excluded' and corrected['x'] == .2


def test_initial_subset_labels_match_actual_pages(tmp_path):
    calls, _ = run(tmp_path, [], n=16)
    for line, im in zip([s for s in calls[0][0].splitlines() if s.startswith('  Image ')], calls[0][1]):
        unit = im.getpixel((0, 0))[0] - 100
        assert f'source unit {unit};' in line
        assert f'report.pdf page {unit+1}' in line


def test_many_crops_retain_source_overview(tmp_path):
    from viewcraft.views.render import render_crop, render_full
    from viewcraft.schema import Region
    path = tmp_path / 'source.png'
    Image.new('RGB', (200, 200), 'white').save(path)
    path = str(path)
    head = render_full(path, path, 128)
    crops = [render_crop(path, path, Region(.1, .1, .8, .8), 128) for _ in range(8)]
    coords = Coordinates.read_snapshot(ROOT / 'skill-library/dense-counting')
    visible = select_visible([head] + crops, coords, 8)
    assert len(visible) <= 8 and head in visible and crops[-1] in visible


def test_new_source_request_takes_priority_over_old_crops(tmp_path):
    from viewcraft.views.render import render_crop, render_full
    from viewcraft.schema import Region
    paths = []
    for name in ('old', 'new'):
        path = tmp_path / f'{name}.png'
        Image.new('RGB', (200, 200), 'white').save(path)
        paths.append(str(path))
    old, new = [render_full(p, p, 128) for p in paths]
    crops = [render_crop(paths[0], paths[0], Region(.1, .1, .8, .8), 128) for _ in range(8)]
    coords = Coordinates.read_snapshot(ROOT / 'skill-library/pdf-reading')
    visible = select_visible([old] + crops + [new], coords, 8)
    assert new in visible and len(visible) <= 8


def test_compose_preserves_requested_order_and_can_revisit_evicted_sheet(tmp_path):
    actions = [{'action': 'compose', 'units': [1, 0], 'cols': 2}]
    actions += [{'action': 'compose', 'units': [i, i+1], 'cols': 2} for i in range(8)]
    actions += [{'action': 'compose', 'units': [1, 0], 'cols': 2}]
    calls, _ = run(tmp_path, actions, n=10, steps=12)
    manifest = 'tile 1: source unit 1; report.pdf page 2; tile 2: source unit 0; report.pdf page 1'
    assert manifest in calls[1][0]
    assert manifest not in calls[9][0]
    assert manifest in calls[10][0]


def test_marker_correction_refreshes_existing_crop_from_clean_pixels(tmp_path):
    calls, _ = run(tmp_path, [
        {'action': 'zoom', 'image': 1, 'regions': [{'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1}]},
        {'action': 'note', 'image': 2, 'notes': [{'at': [.5, .5], 'id': 'A', 'text': 'yes'}]},
        {'action': 'note', 'image': 2, 'notes': [{'at': [.2, .2], 'id': 'A', 'text': 'no'}]}])
    crop = calls[3][1][-1]
    assert crop.crop((60, 56, 104, 74)).getcolors(128*128) == [(44*18, (100, 40, 200))]


@pytest.mark.parametrize('n', [1, 13])
def test_protocol_examples_address_available_sources_and_images(tmp_path, n):
    import re
    calls, _ = run(tmp_path, [], n=n)
    text, images = calls[0]
    for block in re.findall(r'```json\s*(.*?)```', text, re.DOTALL):
        action = json.loads(block)
        for unit in action.get('units', []):
            assert 0 <= unit < n, 'protocol teaches a nonexistent source unit'
        for note in action.get('notes', []):
            assert 1 <= note.get('image', 1) <= len(images)
