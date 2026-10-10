"""User-facing media inputs and full answers, without model API calls."""
from pathlib import Path
import json
import subprocess
import sys

import pytest
from PIL import Image

from viewcraft.skill_inputs import parse_pages, prepare_pdf, prepare_video, validate_times


def test_pages_preserve_document_order_and_reject_invalid_selection():
    assert parse_pages('3,1-2,2', 5) == [0, 1, 2]
    for value in ('0', '3-1', '6', '1,,2', 'x'):
        with pytest.raises(ValueError):
            parse_pages(value, 5)


def test_timestamps_require_finite_nonnegative_strict_order():
    validate_times([0, .5, 3], 3)
    for times in ([0, 0], [-1, 2], [float('nan'), 2], [0, float('inf')], [1, 0], [0]):
        with pytest.raises(ValueError):
            validate_times(times, 2)


def make_pdf(path):
    pages = [Image.new('RGB', (300, 400), color) for color in ('red', 'green', 'blue')]
    pages[0].save(path, save_all=True, append_images=pages[1:])


def test_pdf_selection_renders_actual_pages_with_source_identity(tmp_path):
    pytest.importorskip('pypdfium2')
    pdf = tmp_path / 'report.pdf'
    make_pdf(pdf)
    images, labels = prepare_pdf([pdf], tmp_path / 'pages', pages='2-3')
    assert len(images) == 2
    assert 'file page 2 of 3' in labels[0]
    assert 'report.pdf' in labels[1]
    with Image.open(images[1]) as image:
        r, g, b = image.getpixel((10, 10))[:3]
        assert b > 200 and r < 30 and g < 30
    with pytest.raises(ValueError, match='64'):
        parse_pages(None, 65)


def make_video(path):
    av = pytest.importorskip('av')
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 64, 'yuv420p'
        for color in ['red'] * 4 + ['blue'] * 4:
            frame = av.VideoFrame.from_image(Image.new('RGB', (64, 64), color))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_video_uses_real_timestamps_and_respects_requested_window(tmp_path):
    video = tmp_path / 'clip.mp4'
    make_video(video)
    images, times, labels = prepare_video(video, tmp_path / 'frames', start=.5, end=1.6, interval=.5, max_frames=20)
    assert times == pytest.approx([.5, 1., 1.5])
    assert len(images) == len(labels) == 3
    with Image.open(images[-1]) as image:
        r, g, b = image.getpixel((10, 10))
        assert b > 200 and r < 30
    with pytest.raises(ValueError, match='max-frames'):
        prepare_video(video, tmp_path / 'too-many', start=0, end=2, interval=.25, max_frames=2)
    with pytest.raises(ValueError):
        prepare_video(video, tmp_path / 'invalid', start=0, end=2, interval=0, max_frames=20)


def test_pdf_cli_check_needs_no_api_and_leaves_no_run_files(tmp_path):
    pytest.importorskip('pypdfium2')
    pdf = tmp_path / 'report.pdf'; make_pdf(pdf)
    output = tmp_path / 'output'
    root = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, str(root/'scripts/use_skill.py'), '--skill', 'pdf-reading',
                        '--pdf', str(pdf), '--pages', '2', '--question', 'Describe page 2.',
                        '--output', str(output), '--check'], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert '1 inputs' in p.stdout
    assert not output.exists()


def test_full_answer_retains_citations_and_table_rows(tmp_path):
    from types import SimpleNamespace
    from viewcraft.models import Usage
    from viewcraft.runtime import build_task_prompt, run_static_item
    from viewcraft.schema import PublicItem
    from viewcraft.views.config import Coordinates
    root = Path(__file__).resolve().parents[1]
    coords = Coordinates.read_snapshot(root/'skill-library/pdf-reading')
    image = tmp_path / 'page.png'
    Image.new('RGB', (64, 64), 'white').save(image)
    item = PublicItem('test', 'custom', 'Give a table with units and page references.', (str(image),))
    prompt = build_task_prompt(item, coords, 1, answer_style='free_text')
    assert 'bare value only' not in prompt
    answer = '| Item | Value | Source |\n| A | 20 kg | report.pdf, p. 2 |'
    class Client:
        model_id = 'offline-check'
        def call(self, messages, **kwargs):
            return SimpleNamespace(content=json.dumps({'action':'answer','answer':answer}),
                                   usage=Usage(input_tokens=0, output_tokens=0), wall_time_s=0,
                                   model_reported='offline-check', request_id='test', from_cache=False)
    trajectory = run_static_item(item, coords, Client(), {'observations':1, 'max_images_per_request':8},
                                 'test', arm='viewcraft', condition='normal', split='inference',
                                 answer_style='free_text')
    assert trajectory.answer_text == answer, trajectory.error


@pytest.mark.parametrize('record', [False, True])
def test_cli_saves_full_answer_and_records_only_when_requested(tmp_path, monkeypatch, record):
    import importlib.util
    from types import SimpleNamespace
    from viewcraft.models import Usage
    import viewcraft.models
    root = Path(__file__).resolve().parents[1]
    image = tmp_path / 'input.png'; Image.new('RGB', (64,64), 'white').save(image)
    out = tmp_path / 'result'
    answer = 'Visible count: 4.\nRegion A: 3; region B: 1.'
    class Client:
        model_id = 'offline-check'
        def __init__(self, role): pass
        def validate(self): pass
        def call(self, messages, **kwargs):
            prompt = '\n'.join(part['text'] for part in messages[-1]['content'] if part['type'] == 'text')
            assert 'Source unit 0: image 1: input.png' in prompt
            assert 'bare value only' not in prompt
            return SimpleNamespace(content=json.dumps({'action':'answer','answer':answer}),
                                   usage=Usage(input_tokens=0,output_tokens=0),wall_time_s=0,
                                   model_reported='offline-check',request_id='test',from_cache=False)
    monkeypatch.setattr(viewcraft.models, 'ModelClient', Client)
    monkeypatch.setenv('VIEWCRAFT_CONFIG', '')
    argv = ['use_skill.py','--skill','dense-counting','--images',str(image),'--question','Count by region.', '--output',str(out)]
    if record: argv.append('--record')
    monkeypatch.setattr(sys, 'argv', argv)
    spec = importlib.util.spec_from_file_location('skill_cli_test', root/'scripts/use_skill.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.main()
    assert (out/'answer.md').read_text().strip() == answer
    assert json.loads((out/'result.json').read_text())['answer'] == answer
    assert (out/'trajectory.json').exists() == record
    assert (out/'replay.html').exists() == record
    assert (out/'steps.json').exists() == record
