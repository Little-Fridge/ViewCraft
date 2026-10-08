"""Validate this gallery fixture locally; never invoke a model or generate a trace."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import io
import json
import sys

sys.dont_write_bytecode = True
SKILL = Path(__file__).resolve().parents[1]
ROOT = SKILL.parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from PIL import Image
from viewcraft.evolution.chain import Chain
from viewcraft.views.config import Coordinates
from viewcraft.views.exprs import render_pixel_expr
from viewcraft.views.render import render_full
from generate import main, TARGET_BOX, WIDTH, HEIGHT


def main_validate():
    coords = Coordinates.read_snapshot(SKILL)
    Chain._assert_expressions_parse(None, SimpleNamespace(coords=coords))
    words = len(coords.theta.text.split())
    assert 300 <= words <= 600, words
    assert coords.psi.layout_expr == "" and not coords.psi.context_ops
    assert not coords.phi.view_ops and coords.phi.grid is None
    assert coords.psi.keep_overview and not coords.psi.annotate_overview_boxes
    assert not coords.psi.protocol_contract
    metadata = yaml.safe_load((SKILL/"skill.yaml").read_text())
    assert metadata["answer_style"] == "point"
    assert metadata["budget"] == {"observations": 6, "max_images_per_request": 6}
    names = ["input.png", "request.json", "expected.json"]
    before = {n: sha256((SKILL/"case"/n).read_bytes()).hexdigest() for n in names}
    main()
    after = {n: sha256((SKILL/"case"/n).read_bytes()).hexdigest() for n in names}
    assert before == after, "Generator output is not deterministic"
    request = json.loads((SKILL/"case/request.json").read_text())
    assert "answer" not in request and "expected" not in request
    assert request["images"] == ["input.png"]
    expected = json.loads((SKILL/"case/expected.json").read_text())
    assert expected["match"] == "point"
    x, y = map(float, expected["answer"].split(","))
    x0, y0, x1, y1 = expected["region"]
    assert 0 <= x0 < x < x1 <= 1 and 0 <= y0 < y < y1 <= 1
    assert [x0*WIDTH, y0*HEIGHT, x1*WIDTH, y1*HEIGHT] == list(TARGET_BOX)
    assert (x*WIDTH, y*HEIGHT) == ((TARGET_BOX[0]+TARGET_BOX[2])/2, (TARGET_BOX[1]+TARGET_BOX[3])/2)
    with Image.open(SKILL/"case/input.png") as source:
        source = source.convert("RGB")
        assert source.size == (WIDTH, HEIGHT)
        transformed = render_pixel_expr(source, coords.phi.pixel_expr, {})
        assert transformed.tobytes() == source.tobytes(), "Pixel expression changed the image"
        rendered = render_full(SKILL/"case/input.png", "input", coords.phi.overview_long_edge_px)
        with Image.open(io.BytesIO(rendered.png_bytes)) as overview:
            assert overview.size == source.size and overview.tobytes() == source.tobytes()
    assert rendered.record.scale == 1.0 and not rendered.record.upscaled
    (SKILL/"CONFIG_HASH").write_text(coords.config_hash+"\n")
    report = {"snapshot": "passed", "expression_parse": "passed", "skill_words": words,
              "deterministic_regeneration": "passed", "source_dimensions": [WIDTH, HEIGHT],
              "source_geometry": "passed: identity RGB, full-resolution overview, no compositing",
              "point_region": expected["region"], "expected_answer": expected["answer"],
              "fixture_bounds": "passed", "sha256": after, "config_hash": coords.config_hash,
              "model_api_called": False, "model_replay": "pending central execution"}
    (SKILL/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main_validate()
