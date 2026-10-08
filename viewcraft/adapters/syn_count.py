"""Synthetic hard-counting domain (syn_count) -- the canonical canvas task.

Why synthetic, and why this shape. The visual memory layer is meant to fix
STATE/MEMORY failures, not "cannot see" failures. So the objects here are
plainly visible (large enough to resolve at the overview budget) but NUMEROUS
and SIMILAR, which is exactly the regime where a model loses track of which
instances it has already counted and double-counts or skips. The failure is
working memory, and the fix the evolution should discover is mark-as-you-count
(drop a note on each counted dot). A real counting benchmark (CountBench) is
saturated at 2-10 objects and cannot dial the memory load; generating the
images lets us sweep N and read a dose-response curve (gain vs N), which is the
motivation figure the real benchmarks cannot give.

Zero download, fully deterministic per item (seeded by index), cached on disk
so re-runs reuse the same pixels. One image per item, so group == item and no
image spans splits.

Difficulty knobs (loader kwargs, for the dose-response probe):
  n_target_range  the count the model must produce (the memory load)
  n_distractors   similar-looking non-target dots (discrimination load)
  radius          dot size (kept clearly visible so it is a memory task)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "syn_count"

# Target red; distractors are muted and clearly NOT red, so the task is
# enumeration memory, not colour discrimination.
_TARGET = (220, 30, 30)
_DISTRACTORS = [(60, 90, 160), (90, 150, 90), (150, 130, 60), (140, 90, 150),
                (90, 90, 90), (170, 170, 170)]


def _rng_for(item_index: int, seed: int):
    import random
    h = int(hashlib.sha256(f"{seed}:{item_index}".encode()).hexdigest(), 16)
    return random.Random(h % (2 ** 32))


def _gen_image(path: Path, n_target: int, n_distr: int, radius: int,
               size: int, rng) -> None:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (245, 245, 242))
    d = ImageDraw.Draw(img)

    def place():
        return (rng.uniform(radius * 1.5, size - radius * 1.5),
                rng.uniform(radius * 1.5, size - radius * 1.5))

    for i in range(n_distr):
        x, y = place()
        c = _DISTRACTORS[i % len(_DISTRACTORS)]
        d.ellipse([x - radius, y - radius, x + radius, y + radius], fill=c)
    for _ in range(n_target):
        x, y = place()
        d.ellipse([x - radius, y - radius, x + radius, y + radius],
                  fill=_TARGET, outline=(120, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def load(root: str | Path, seed: int = 20260911, n_items: int = 240,
         n_target_range: tuple = (12, 40), n_distractors: int = 25,
         radius: int = 13, size: int = 1024) -> list[Sample]:
    """Generate (once, deterministically) and load n_items counting images."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    lo, hi = int(n_target_range[0]), int(n_target_range[1])
    for i in range(n_items):
        rng = _rng_for(i, seed)
        n_target = rng.randint(lo, hi)
        path = root / f"count_{i:04d}_n{n_target}.png"
        if not path.exists():
            _gen_image(path, n_target, n_distractors, radius, size, rng)
        item_id = f"{DOMAIN}/{i:04d}"
        question = ("How many red dots are in this image? "
                    "Answer with a number.")
        public = PublicItem(item_id=item_id, domain=DOMAIN, question=question,
                            media_ids=(str(path),),
                            # `count` is the answer and stays OUT of public.meta
                            # (answer-bearing keys are forbidden there).
                            meta={"object": "red dots"})
        private = PrivateItem(item_id=item_id, answer=n_target,
                              answer_type="integer", evidence_source="none",
                              meta={"count": n_target,
                                    "n_distractors": n_distractors})
        samples.append(Sample(public=public, private=private,
                              group_key=path.stem))
    return samples


def parse_count(raw: str) -> int | None:
    import re
    m = re.search(r"(\d+)", raw or "")
    return int(m.group(1)) if m else None


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    """Exact integer match; open-ended (no candidate set is shown)."""
    got = parse_count(answer_text)
    return 1.0 if got is not None and got == int(private.answer) else 0.0
