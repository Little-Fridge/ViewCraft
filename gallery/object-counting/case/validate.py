"""Offline validation of the snapshot and deterministic rendered counting case."""
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageColor
import yaml

HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
ROOT = SKILL.parents[1]
sys.path.insert(0, str(ROOT))

from viewcraft.evolution.chain import Chain
from viewcraft.views.config import Coordinates
from viewcraft.views.exprs import eval_pixel_expr, pixel_env
from viewcraft.views.selection import eval_selection_expr
from viewcraft.views.zoomgate import eval_zoom_gate, zoom_facts
from viewcraft.schema import Region
from generate import COLORS, PANELS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def orange_components(image):
    """Measure components using rendered pixels, independent of object metadata."""
    pixels = np.asarray(image).astype(np.int16)
    orange = np.array(ImageColor.getrgb(COLORS["orange"]))
    mask = np.max(np.abs(pixels-orange), axis=2) <= 5
    height, width = mask.shape
    components = []
    for y, x in zip(*np.nonzero(mask)):
        if not mask[y, x]:
            continue
        todo = deque([(int(x), int(y))])
        mask[y, x] = False
        area = 0
        x0 = x1 = int(x)
        y0 = y1 = int(y)
        while todo:
            px, py = todo.popleft()
            area += 1
            x0, x1 = min(x0, px), max(x1, px)
            y0, y1 = min(y0, py), max(y1, py)
            for nx, ny in ((px-1, py), (px+1, py), (px, py-1), (px, py+1)):
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx]:
                    mask[ny, nx] = False
                    todo.append((nx, ny))
        if area > 100:
            ratio = area / ((x1-x0+1)*(y1-y0+1))
            components.append((x0, y0, x1, y1, ratio))
    return components


def main():
    coords = Coordinates.read_snapshot(SKILL)
    Chain._assert_expressions_parse(None, SimpleNamespace(coords=coords))
    # Direct execution also surfaces errors the chain's permissive probe catches.
    probe = Image.new("RGB", (64, 64), (220, 110, 40))
    eval_pixel_expr(coords.phi.pixel_expr, pixel_env(probe, {}))
    assert eval_selection_expr(coords.phi.select_expr, 1, {}, 6) == [0]
    facts = zoom_facts(src_w=1440, src_h=1080, overview_px=1440,
                       crop_px=1440, region=Region(.1, .1, .3, .3),
                       step=0, steps_left=6, crops_so_far=0)
    assert eval_zoom_gate(coords.phi.zoom_gate_expr, facts)
    assert coords.psi.protocol_contract == ""
    assert coords.psi.max_images_per_request == 6
    meta = yaml.safe_load((SKILL / "skill.yaml").read_text())
    assert meta["id"] == "object-counting"
    assert meta["input_type"] == "image" and meta["answer_style"] == "short_text"
    assert meta["budget"] == {"observations": 6, "max_images_per_request": 6}
    assert 300 <= len(coords.theta.text.split()) <= 600

    files = [HERE / name for name in ("input.png", "request.json", "expected.json")]
    before = [digest(p) for p in files]
    for _ in range(2):
        subprocess.run([sys.executable, str(HERE / "generate.py")], check=True)
        assert [digest(p) for p in files] == before, "Generation is not deterministic"
    request = json.loads((HERE / "request.json").read_text())
    expected = json.loads((HERE / "expected.json").read_text())
    assert set(request) == {"question", "images", "options"}
    assert request["images"] == ["input.png"] and request["options"] == []
    assert "orange triangles" in request["question"]
    assert expected == {"answer": "7", "match": "numeric", "tolerance": 0}
    assert sum(len(panel) for panel in PANELS) == 36

    with Image.open(HERE / "input.png") as image:
        assert image.size == (1440, 1080) and image.mode == "RGB"
        components = orange_components(image)
    assert len(components) == 12, f"Expected 12 orange pieces, found {len(components)}"
    triangles = [c for c in components if 0.43 < c[4] < 0.57]
    others = [c for c in components if c not in triangles]
    assert all(c[4] > 0.70 for c in others), "An orange contour is ambiguous"
    counts = [0, 0, 0, 0]
    for x0, y0, x1, y1, _ in triangles:
        column = int((x0+x1)/2 > 720)
        row = int((y0+y1)/2 > 626)
        counts[row*2+column] += 1
    assert counts == [2, 1, 3, 1], counts
    assert len(triangles) == int(expected["answer"])
    stored_hash = SKILL / "CONFIG_HASH"
    if stored_hash.exists():
        assert stored_hash.read_text().strip() == coords.config_hash
    print(json.dumps({"validation": "passed", "api_calls": 0,
                      "png_sha256": before[0], "orange_components": len(components),
                      "orange_triangles": len(triangles), "panel_counts": counts,
                      "config_hash": coords.config_hash}, indent=2))


if __name__ == "__main__":
    main()
