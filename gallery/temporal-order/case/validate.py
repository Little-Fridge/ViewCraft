"""Offline asset/configuration checks; never runs a model or reads credentials."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import yaml

CASE = Path(__file__).resolve().parent
SKILL = CASE.parent
REPO = SKILL.parent.parent
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from viewcraft.views.config import Coordinates
from viewcraft.evolution.chain import Chain
from viewcraft.views.layout import eval_layout_expr
from viewcraft.views.selection import eval_selection_expr
from generate import generate


def main():
    req = json.loads((CASE / "request.json").read_text())
    expected = json.loads((CASE / "expected.json").read_text())
    metadata = yaml.safe_load((SKILL / "skill.yaml").read_text())
    assert "answer" not in req and "expected" not in req
    assert 4 <= len(req["images"]) <= 6
    assert len(req["frame_times_s"]) == len(req["images"])
    assert req["frame_times_s"] == sorted(set(req["frame_times_s"]))
    assert expected == {"answer": "B", "match": "exact"}
    assert metadata["input_type"] == "frames"
    assert metadata["budget"] == {"observations": 6, "max_images_per_request": 6}
    coords = Coordinates.read_snapshot(SKILL)
    assert not coords.psi.protocol_contract
    Chain._assert_expressions_parse(None, SimpleNamespace(coords=coords))
    selected = eval_selection_expr(coords.phi.select_expr, len(req["images"]), {}, 6)
    assert selected == list(range(len(req["images"])))
    paths = [CASE / p for p in req["images"]] + [CASE / "input_preview.png"]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    generate()
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frames = []
    for p in paths[:-1]:
        with Image.open(p) as im:
            assert im.size == (1200, 900) and im.mode == "RGB"
            frames.append(im.copy())
    sheets = eval_layout_expr(coords.psi.layout_expr, frames, {"frame_times_s": req["frame_times_s"], "n_media": 5})
    assert len(sheets) == 1 and max(sheets[0].size) <= 4096
    sheets[0].save(CASE / "layout_preview.png")
    body = coords.theta.text.split("---", 2)[-1]
    word_count = len(body.split())
    assert 300 <= word_count <= 600, word_count
    (SKILL / "CONFIG_HASH").write_text(coords.config_hash + "\n")
    report = {
        "validation_type": "offline_assets_and_configuration",
        "status": "passed",
        "checks": ["Coordinates.read_snapshot", "Chain._assert_expressions_parse", "all_five_frames_selected", "five_timestamped_RGB_PNGs", "deterministic_regeneration_sha256", "layout_expression_rendered", "request_has_no_answer", "skill_word_count"],
        "skill_words": word_count,
        "layout_size": list(sheets[0].size),
        "config_hash": coords.config_hash,
        "input_sha256": before,
        "model_called": False,
        "model_evaluation": "pending main-agent run"
    }
    (CASE / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
