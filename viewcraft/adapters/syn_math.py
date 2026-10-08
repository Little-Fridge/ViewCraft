"""Synthetic MATH-as-visual-workspace domain (syn_math) -- the CL-Bench idea:
a NON-visual task (a 2D recurrence / DP table) whose natural human solution is
to WRITE intermediate cells on the grid, i.e. to use the image as a scratchpad.

Why this is the right test (and how it avoids the syn_state trap). syn_state
showed a strong reasoning model holds moderate SYMBOLIC state in one pass and
never needs the canvas. A 2D recurrence is different: each cell depends on its
upper and left neighbours, so the answer at the bottom-right requires holding
the WHOLE growing table, which is exactly what working memory cannot do and
what a hand-drawn table exists for. The image is the scratchpad: the model jots
each computed cell on the grid as it fills the table (the canvas), instead of
holding 30+ intermediate numbers in its head.

Zero download, deterministic per item. Difficulty = grid size (more cells to
hold). One image per item.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "syn_math"


def _rng_for(item_index: int, seed: int):
    import random
    h = int(hashlib.sha256(f"{seed}:math:{item_index}".encode()).hexdigest(), 16)
    return random.Random(h % (2 ** 32))


def _render(path: Path, row0: list[int], col0: list[int], n: int) -> None:
    """Draw the empty recurrence grid with row0 and col0 seeded."""
    from PIL import Image, ImageDraw
    cw = 78
    w, h = cw * n + 20, cw * n + 20
    img = Image.new("RGB", (w, h), (250, 250, 248))
    d = ImageDraw.Draw(img)
    for r in range(n):
        for c in range(n):
            x0, y0 = 10 + c * cw, 10 + r * cw
            d.rectangle([x0, y0, x0 + cw - 6, y0 + cw - 6],
                        outline=(60, 60, 60), width=2)
    for c, v in enumerate(row0):
        d.text((10 + c * cw + cw // 2 - 14, 10 + cw // 2 - 8), str(v),
               fill=(20, 20, 160))
    for r, v in enumerate(col0):
        if r == 0:
            continue
        d.text((10 + cw // 2 - 14, 10 + r * cw + cw // 2 - 8), str(v),
               fill=(20, 20, 160))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def load(root: str | Path, seed: int = 20260911, n_items: int = 240,
         n: int = 6) -> list[Sample]:
    """Generate (deterministically) recurrence-grid items.

    Rule (stated in the question): each cell = the cell above + the cell to its
    left. Answer = the bottom-right cell. Difficulty is in the filename.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for i in range(n_items):
        rng = _rng_for(i, seed)
        row0 = [rng.randint(1, 5) for _ in range(n)]
        col0 = [row0[0]] + [rng.randint(1, 5) for _ in range(n - 1)]
        # compute the true table
        T = [[0] * n for _ in range(n)]
        T[0] = list(row0)
        for r in range(n):
            T[r][0] = col0[r]
        for r in range(1, n):
            for c in range(1, n):
                T[r][c] = T[r - 1][c] + T[r][c - 1]
        answer = T[n - 1][n - 1]

        path = root / f"math_{i:04d}_n{n}.png"
        if not path.exists():
            _render(path, row0, col0, n)

        item_id = f"{DOMAIN}/{i:04d}"
        question = (
            "This is a table to fill. The top row and the left column are "
            "given. Every other cell equals the cell directly ABOVE it plus "
            "the cell directly to its LEFT. Work out the value of the "
            "bottom-right cell. Answer with the number only.")
        public = PublicItem(item_id=item_id, domain=DOMAIN, question=question,
                            media_ids=(str(path),), meta={"grid": n})
        private = PrivateItem(item_id=item_id, answer=answer,
                              answer_type="integer", evidence_source="none",
                              meta={"row0": row0, "col0": col0,
                                    "answer": answer})
        samples.append(Sample(public=public, private=private,
                              group_key=path.stem))
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    import re
    m = re.search(r"(-?\d+)", (answer_text or "").replace(",", ""))
    return 1.0 if m and int(m.group(1)) == int(private.answer) else 0.0
