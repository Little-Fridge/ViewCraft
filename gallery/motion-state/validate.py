"""Offline artifact and runtime validation; never calls a model."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import yaml
from PIL import Image, ImageColor
from viewcraft.views.config import Coordinates
from viewcraft.evolution.chain import Chain
from viewcraft.views.layout import eval_layout_expr
from viewcraft.views.selection import eval_selection_expr


def main():
    coords = Coordinates.read_snapshot(HERE)
    Chain._assert_expressions_parse(None, SimpleNamespace(coords=coords))
    mapping = yaml.safe_load((HERE / "skill.yaml").read_text())
    assert mapping["input_type"] == "frames"
    assert mapping["budget"] == {"observations": 6, "max_images_per_request": 6}
    assert coords.psi.protocol_contract == ""
    words = len(coords.theta.text.split())
    assert 300 <= words <= 600, words
    case = HERE / "case"
    req = json.loads((case / "request.json").read_text())
    expected = json.loads((case / "expected.json").read_text())
    assert "answer" not in req and "expected" not in req
    assert len(req["images"]) == len(req["frame_times_s"]) == 5
    assert req["frame_times_s"] == sorted(set(req["frame_times_s"]))
    selected = list(eval_selection_expr(coords.phi.select_expr, 5, {}, 6))
    assert selected == list(range(5)), selected
    frames = [Image.open(case / name).convert("RGB") for name in req["images"]]
    assert all(im.size == (1200, 860) for im in frames)
    assert len({hashlib.sha256(im.tobytes()).hexdigest() for im in frames}) == 5
    colors = {"A": "#D65450", "B": "#397CD1", "C": "#E5A735", "D": "#339B77"}
    measured = {}
    for label, color in colors.items():
        positions = []
        for im in frames:
            # Exclude the header and object legend; measure body-color bounds.
            pixels = np.asarray(im)[222:716, 84:1117]
            ys, xs = np.where(np.all(pixels == ImageColor.getrgb(color), axis=2))
            assert len(xs) > 1000
            positions.append([(int(xs.min()) + int(xs.max())) / 2 + 84,
                              (int(ys.min()) + int(ys.max())) / 2 + 222])
        measured[label] = positions
    stationary = [label for label, points in measured.items() if all(p == points[0] for p in points)]
    assert ", ".join(stationary) == expected["answer"]
    assert measured["C"][0] == measured["C"][-1] and measured["C"][0] != measured["C"][2]
    sheet = eval_layout_expr(coords.psi.layout_expr, frames, {"frame_times_s": req["frame_times_s"], "n_media": 5})[0]
    sheet.save(case / "layout_preview.png")
    inputs = [case / name for name in req["images"]] + [case / "input_preview.png", case / "request.json", case / "expected.json"]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    spec = importlib.util.spec_from_file_location("motion_state_generate", case / "generate.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.main()
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    assert before == after, "Regeneration changed inputs"
    (HERE / "CONFIG_HASH").write_text(coords.config_hash + "\n")
    report = {"status": "passed", "scope": "offline artifact validation only", "model_run": "not performed",
              "snapshot_load": "passed", "expression_checks": "passed", "skill_word_count": words,
              "selected_frame_indices": selected, "frame_dimensions": [1200, 860],
              "frame_times_s": req["frame_times_s"], "measured_body_centers_px": measured,
              "stationary_labels": stationary, "expected_answer": expected["answer"],
              "deterministic_regeneration": "passed", "layout_dimensions": list(sheet.size),
              "config_hash": coords.config_hash, "asset_sha256": after}
    (HERE / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("asset_sha256", "measured_body_centers_px")}, indent=2))


if __name__ == "__main__":
    main()
