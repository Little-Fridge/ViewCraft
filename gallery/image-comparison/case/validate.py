"""Offline fixture and ViewCraft configuration checks; never call a model."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
ROOT = SKILL.parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from PIL import Image, ImageChops, ImageDraw

from generate import CHANGE_BOXES, MUG_CENTERS, SIZE, base_scene, mug
from viewcraft.evolution.chain import Chain
from viewcraft.views.config import Coordinates
from viewcraft.views.layout import eval_layout_expr


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    coords = Coordinates.read_snapshot(SKILL)
    Chain._assert_expressions_parse(None, SimpleNamespace(coords=coords))
    metadata = yaml.safe_load((SKILL / "skill.yaml").read_text())
    assert metadata["input_type"] == "images"
    assert metadata["budget"] == {"observations": 6, "max_images_per_request": 6}
    assert coords.psi.protocol_contract == ""
    words = len(coords.theta.text.split())
    assert 300 <= words <= 600, words
    request = json.loads((HERE / "request.json").read_text())
    assert "answer" not in request and "expected" not in request
    expected = json.loads((HERE / "expected.json").read_text())
    assert expected == {"answer": "teal mug", "match": "contains"}
    paths = [HERE / name for name in request["images"]]
    assert len(paths) == 2 and all(p.is_file() for p in paths)
    before_hashes = [digest(p) for p in paths]
    subprocess.run([sys.executable, str(HERE / "generate.py")], check=True)
    hashes = [digest(p) for p in paths]
    assert before_hashes == hashes, "PNG regeneration changed file bytes"
    images = [Image.open(p).convert("RGB") for p in paths]
    assert all(im.size == SIZE for im in images)
    base = base_scene()
    for index, im in enumerate(images):
        reconstructed = base.copy()
        mug(reconstructed, MUG_CENTERS[index])
        assert ImageChops.difference(im, reconstructed).getbbox() is None
        empty_location = CHANGE_BOXES[1 - index]
        assert ImageChops.difference(im.crop(empty_location), base.crop(empty_location)).getbbox() is None
    difference = ImageChops.difference(*images)
    assert difference.getbbox() is not None
    allowed = Image.new("L", SIZE, 0)
    draw = ImageDraw.Draw(allowed)
    for x0, y0, x1, y1 in CHANGE_BOXES:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
        assert difference.crop((x0, y0, x1, y1)).getbbox() is not None
    outside = Image.composite(Image.new("RGB", SIZE), difference, allowed)
    assert outside.getbbox() is None, "An unrelated scene region changed"
    overviews = []
    for im in images:
        thumbnail = im.copy()
        thumbnail.thumbnail((coords.phi.overview_long_edge_px,) * 2)
        overviews.append(thumbnail)
    sheets = eval_layout_expr(coords.psi.layout_expr, overviews, {"n_media": 2})
    assert len(sheets) == 1 and max(sheets[0].size) <= 4096
    sheets[0].save(HERE / "input_preview.png")
    report = {
        "status": "passed",
        "scope": "offline asset and configuration validation only",
        "checks": ["Coordinates.read_snapshot", "Chain._assert_expressions_parse",
                   "actual two-source layout rendering", "skill word count",
                   "request and metadata structure", "byte-reproducible PNGs",
                   "same base scene with one translated mug",
                   "both empty coaster locations verified",
                   "all other scene pixels unchanged"],
        "skill_word_count": words,
        "config_hash": coords.config_hash,
        "image_size": list(SIZE),
        "preview_size": list(sheets[0].size),
        "allowed_change_regions_px": CHANGE_BOXES,
        "inputs": [{"file": p.name, "sha256": h} for p, h in zip(paths, hashes)],
        "model_run_performed": False,
        "model_replay": "pending main-agent run"
    }
    (HERE / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (SKILL / "CONFIG_HASH").write_text(coords.config_hash + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
