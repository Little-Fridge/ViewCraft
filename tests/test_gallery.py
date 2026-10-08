"""Gallery schema and replay fidelity checks; no model service required."""
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import yaml
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
SKILLS = sorted(p.name for p in (ROOT / "gallery").iterdir() if (p / "skill.yaml").exists())


@pytest.mark.parametrize("name", SKILLS)
def test_gallery_skill_and_case(name):
    from viewcraft.views.config import Coordinates
    from viewcraft.evolution.chain import Chain
    from viewcraft.evolution.propose import Candidate
    path = ROOT / "gallery" / name
    meta = yaml.safe_load((path / "skill.yaml").read_text())
    assert meta["id"] == name
    assert meta["answer_style"] in ("short_text", "point", "letter", "yes_no")
    coords = Coordinates.read_snapshot(path)
    Chain._assert_expressions_parse(None, Candidate("joint", coords, "gallery", "gallery"))
    request = json.loads((path / "case/request.json").read_text())
    assert "answer" not in request and "expected" not in request
    assert request["question"] and request["images"]
    for filename in request["images"]:
        with Image.open(path / "case" / filename) as image:
            image.verify()
    expected = json.loads((path / "case/expected.json").read_text())
    assert expected["answer"]
    assert (path / "case/generate.py").exists()


def test_recorder_preserves_real_attachments_and_text(tmp_path):
    from viewcraft.models import Usage
    from viewcraft.replay import RecordingClient, render_replay
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), "teal").save(buf, format="PNG")
    blob = buf.getvalue()
    seen = []
    class FakeClient:
        def call(self, messages, **kwargs):
            seen.append(kwargs)
            return SimpleNamespace(content='<actual response> "3"', model_reported="test-model",
                                   request_id="r1", usage=Usage(input_tokens=10, output_tokens=2),
                                   wall_time_s=0.1, from_cache=False)
    client = RecordingClient(FakeClient(), tmp_path)
    client.call([{"role":"user", "content":[{"type":"image_url", "image_url":{
        "url":"data:image/png;base64," + base64.b64encode(blob).decode()}}]}])
    assert seen[0]["use_cache"] is False
    assert (tmp_path / client.steps[0]["images"][0]).read_bytes() == blob
    assert client.steps[0]["response"] == '<actual response> "3"'
    render_replay(tmp_path, "Test", "Question?", {"answer":"3", "model":"test-model", "case_check":None})
    page = (tmp_path / "replay.html").read_text()
    assert "&lt;actual response&gt;" in page
    assert not (tmp_path / "replay.gif").exists()
    audit = ResourceAudit()
    audit.feed(page)
    assert not audit.external
    inline = [value for tag, key, value in audit.resources if tag == "img" and key == "src"]
    assert blob in [base64.b64decode(url.split(",", 1)[1]) for url in inline]


class ResourceAudit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources = []
        self.external = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href", "poster", "srcset", "data", "action") and value:
                self.resources.append((tag, key, value))
                if not value.startswith(("data:", "#")):
                    self.external.append((tag, key, value))
            if tag == "meta" and key == "http-equiv" and value.lower() == "refresh":
                self.external.append((tag, key, value))


@pytest.mark.parametrize("page", sorted((ROOT / "gallery").rglob("*.html")))
def test_exported_html_has_no_external_resources(page):
    parser = ResourceAudit()
    parser.feed(page.read_text())
    assert not parser.external, (page, parser.external)
    assert any(tag == "img" for tag, _, _ in parser.resources)
    assert 'fetch(' not in page.read_text()


def test_standalone_gallery_embeds_all_cases():
    parser = ResourceAudit()
    page = (ROOT / "gallery/index.html").read_text()
    parser.feed(page)
    for name in SKILLS:
        assert f'id="case-{name}"' in page
    assert '<video' not in page
    assert '<details' not in page
    assert '<button' not in page
    assert 'download=' not in page
    assert 'data:application/zip' not in page
    assert '@font-face' in page and 'data:font/ttf;base64,' in page
    assert page.count('class="coordinate"') == 30
