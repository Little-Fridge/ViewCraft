"""ARC-AGI adapter (fchollet/ARC) -- the user's "let the model play a game"
idea, and a much better fit than synthetic logic puzzles.

Why ARC fits the mechanism. Each task gives a few input->output grid examples
and asks for the output of a test input. Two properties the synthetic tasks
lacked: (i) it is genuinely HARD -- frontier models score in the low double
digits on the public eval, so there is real headroom (a one-shot answer fails,
unlike zebra/sudoku which gemini one-shot); and (ii) the OUTPUT IS A GRID, so
the workspace is natural -- the model can sketch the transformation and draw
the output cell by cell on the canvas, which is exactly the visual-memory
mechanism (mark a cell's colour via a note).

The image shows the example pairs (input -> output) and the test input, plus a
BLANK answer grid the model draws on. The answer is the output grid as a
matrix of colour indices (0-9); the scorer parses it and exact-matches. Only
tasks whose output fits within MAX_DIM are kept, so drawing it stays
affordable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..paths import data_root
from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "arc"
MAX_DIM = 14          # keep output grids drawable/scorable tractably

# Standard ARC colour palette (0-9)
_PALETTE = {
    0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64),
    4: (255, 220, 0), 5: (170, 170, 170), 6: (240, 18, 190),
    7: (255, 133, 27), 8: (127, 219, 255), 9: (135, 12, 37),
}


def _draw_grid(d, grid, x0, y0, cell):
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            color = _PALETTE.get(int(v), (255, 255, 255))
            d.rectangle([x0 + c * cell, y0 + r * cell,
                         x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1],
                        fill=color, outline=(120, 120, 120))


def _render(path: Path, train: list, test_in: list) -> None:
    """Render example pairs + test input + a blank answer grid, clearly laid
    out (one example pair per row: input -> output; test below; blank answer
    grid to the right of the test)."""
    from PIL import Image, ImageDraw
    cell = 16
    row_gap = 16
    arrow_w = 26
    n_rows = len(train) + 1                     # examples + test row
    max_h = max([len(p["input"]) for p in train]
                + [len(p["output"]) for p in train] + [len(test_in)] + [MAX_DIM])
    in_w = max(len(p["input"][0]) for p in train)
    out_w = max(len(p["output"][0]) for p in train)
    w = 30 + in_w * cell + arrow_w + out_w * cell + 40 + MAX_DIM * cell
    h = 30 + n_rows * (max_h * cell + row_gap) + 20
    img = Image.new("RGB", (w, h), (252, 252, 250))
    d = ImageDraw.Draw(img)

    def grid_w(g):
        return len(g[0]) * cell

    y = 16
    for p in train:
        _draw_grid(d, p["input"], 20, y, cell)
        ax = 20 + grid_w(p["input"]) + 4
        d.text((ax, y + max_h * cell // 2 - 6), "->", fill=(40, 40, 40))
        _draw_grid(d, p["output"], ax + arrow_w, y, cell)
        y += max_h * cell + row_gap

    # test input (left) and the blank answer grid (right)
    d.text((20, y - 4), "TEST input:", fill=(40, 40, 40))
    _draw_grid(d, test_in, 20, y + 14, cell)
    ax = 20 + grid_w(test_in) + 40
    d.text((ax, y - 4), "YOUR answer grid (draw here):", fill=(160, 40, 40))
    for r in range(MAX_DIM):
        for c in range(MAX_DIM):
            d.rectangle([ax + c * cell, y + 14 + r * cell,
                         ax + (c + 1) * cell - 1, y + 14 + (r + 1) * cell - 1],
                        outline=(190, 190, 190))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def load(root: str | Path, seed: int = 20260911, split: str = "training",
         n_items: int = 300) -> list[Sample]:
    root = Path(root) / split
    files = sorted(root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no ARC json under {root}")
    samples: list[Sample] = []
    for f in files:
        t = json.loads(f.read_text())
        test = t.get("test") or []
        if not test or "output" not in test[0]:
            continue
        out = test[0]["output"]
        if len(out) > MAX_DIM or len(out[0]) > MAX_DIM:
            continue
        tid = f.stem
        img = data_root() / "arc" / "_imgs" / f"{tid}.png"
        if not img.exists():
            _render(img, t["train"], test[0]["input"])
        item_id = f"{DOMAIN}/{tid}"
        question = (
            "This is an ARC reasoning puzzle. The image shows example "
            "input->output grid pairs at the top, and a TEST input grid below. "
            "Work out the transformation rule from the examples, apply it to "
            "the test input, and produce the output grid. You may DRAW your "
            "working / the answer on the blank grid at the bottom. Colours are "
            "integers 0-9 (0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, "
            "6 magenta, 7 orange, 8 azure, 9 maroon).\n"
            "Give your final answer as the output grid: a JSON list of lists "
            "of integers, e.g. [[0,1,2],[3,4,5]]. Answer with ONLY the matrix.")
        public = PublicItem(item_id=item_id, domain=DOMAIN, question=question,
                            media_ids=(str(img),),
                            meta={"n_train": len(t["train"]),
                                  "out_shape": [len(out), len(out[0])]})
        private = PrivateItem(item_id=item_id, answer=out,
                              answer_type="grid", evidence_source="official",
                              meta={"output": out, "task_id": tid})
        samples.append(Sample(public=public, private=private,
                              group_key=tid))
        if len(samples) >= n_items:
            break
    print(f"[arc] kept {len(samples)} tasks (output<= {MAX_DIM}x{MAX_DIM}, "
          f"split={split})")
    return samples


def _parse_matrix(text: str):
    m = re.search(r"\[\s*\[.*\]\s*\]", text or "", re.DOTALL)
    if not m:
        return None
    try:
        g = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not (isinstance(g, list) and g and all(isinstance(r, list) for r in g)):
        return None
    return g


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    """Exact grid match (the official ARC metric): 1 iff the produced grid
    equals the gold output cell-for-cell."""
    gold = private.meta.get("output")
    got = _parse_matrix(answer_text)
    if gold is None or got is None:
        return 0.0
    try:
        return 1.0 if [[int(v) for v in r] for r in got] == \
                      [[int(v) for v in r] for r in gold] else 0.0
    except (TypeError, ValueError):
        return 0.0
