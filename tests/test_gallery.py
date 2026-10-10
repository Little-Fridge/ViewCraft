"""Skill library schema and replay fidelity checks; no model service required."""
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
SKILLS = sorted(p.name for p in (ROOT / "skill-library").iterdir() if (p / "skill.yaml").exists())


@pytest.mark.parametrize("name", SKILLS)
def test_library_skill(name):
    from viewcraft.views.config import Coordinates
    from viewcraft.evolution.chain import Chain
    from viewcraft.evolution.propose import Candidate
    path = ROOT / "skill-library" / name
    meta = yaml.safe_load((path / "skill.yaml").read_text())
    assert meta["id"] == name
    assert meta["answer_style"] in ("short_text", "point", "letter", "yes_no")
    coords = Coordinates.read_snapshot(path)
    Chain._assert_expressions_parse(None, Candidate("joint", coords, "gallery", "gallery"))
    assert meta["budget"]["observations"] > 0
    assert meta["budget"]["max_images_per_request"] == coords.psi.max_images_per_request
    assert coords.theta.text.strip()
    assert not (path / "case").exists()


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


def test_skill_cli_accepts_library_id(tmp_path):
    import subprocess
    import sys
    image = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), "white").save(image)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/use_skill.py"),
                             "--skill", "dense-counting", "--images", str(image),
                             "--question", "Count the visible targets.", "--check"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CHECK OK: dense-counting" in result.stdout


def test_skill_cli_requires_inputs():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(ROOT / "scripts/use_skill.py"),
                             "--skill", "dense-counting", "--check"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Supply --images and --question" in result.stderr
