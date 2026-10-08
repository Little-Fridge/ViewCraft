"""Synthetic STATE-TRACKING domain (syn_state) -- the boldest test: a visual
memory layer for a NON-visual task.

Why this exists. The claim under test is that the visual canvas is a general
EXTERNAL MEMORY mechanism, not a trick for image tasks. So this domain is not a
vision task at all: it is WORKING-MEMORY state tracking. K slots each hold a
token; a text list of M swap operations permutes them; the model must report
which token ends in a target slot. The only way to be right is to hold the
whole permutation in memory across M steps -- the exact bottleneck the canvas
is built for. The image is just the medium of the memory: the model jots the
current token on each slot as it applies the swaps, so the state lives on the
canvas instead of in its head.

Zero download, deterministic per item, controllable memory load (K slots x M
swaps) for the dose-response probe (gain vs M). One image per item.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "syn_state"

_TOKENS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")   # no I/O (confusable)


def _rng_for(item_index: int, seed: int):
    import random
    h = int(hashlib.sha256(f"{seed}:state:{item_index}".encode()).hexdigest(), 16)
    return random.Random(h % (2 ** 32))


def _render(path: Path, slots: list[str], size_h: int = 240) -> None:
    from PIL import Image, ImageDraw
    n = len(slots)
    cw = 90                       # cell width
    w, h = cw * n + 20, size_h
    img = Image.new("RGB", (w, h), (250, 250, 248))
    d = ImageDraw.Draw(img)
    for i, tok in enumerate(slots):
        x0 = 10 + i * cw
        d.rectangle([x0, 60, x0 + cw - 8, 190], outline=(40, 40, 40), width=2)
        d.text((x0 + cw // 2 - 14, 30), f"slot {i + 1}", fill=(90, 90, 90))
        d.text((x0 + cw // 2 - 12, 110), tok, fill=(20, 20, 160))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def load(root: str | Path, seed: int = 20260911, n_items: int = 240,
         n_slots: int = 10, n_swaps: int = 30) -> list[Sample]:
    """Generate (once, deterministically) and load state-tracking items.

    Difficulty is in the filename so a harder config can never silently reuse a
    cached easier image. The default (10 slots x 30 swaps) is set so the baseline
    is NOT saturated: at 7x6 the executor tracked perfectly (raw 1.000) and at
    8x14 it still one-shot every item in a single reasoning pass (8/8, zero notes)
    -- a strong reasoning model holds moderate state in one pass, so the memory
    load must be pushed before the external canvas has anything to repair. At
    10x30 the baseline is ~0.75, the same movable-surface band as the study's
    other working domains.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for i in range(n_items):
        rng = _rng_for(i, seed)
        toks = rng.sample(_TOKENS, n_slots)
        state = list(toks)
        ops = []
        for _ in range(n_swaps):
            a, b = rng.sample(range(n_slots), 2)
            ops.append((a, b))
            state[a], state[b] = state[b], state[a]
        target = rng.randrange(n_slots)
        answer = state[target]

        path = root / f"state_{i:04d}_k{n_slots}_m{n_swaps}.png"
        if not path.exists():
            _render(path, toks)

        op_txt = "; ".join(f"swap slots {a + 1} and {b + 1}" for a, b in ops)
        item_id = f"{DOMAIN}/{i:04d}"
        question = (f"The tokens start in the slots exactly as shown. Apply "
                    f"these swaps in order: {op_txt}. After all of them, which "
                    f"token (letter) is in slot {target + 1}? Answer with the "
                    f"letter only.")
        public = PublicItem(item_id=item_id, domain=DOMAIN, question=question,
                            media_ids=(str(path),),
                            meta={"n_slots": n_slots, "n_swaps": n_swaps})
        private = PrivateItem(item_id=item_id, answer=answer,
                              answer_type="letter", evidence_source="none",
                              meta={"initial": toks, "ops": ops,
                                    "target_slot": target, "answer": answer})
        samples.append(Sample(public=public, private=private,
                              group_key=path.stem))
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    """Exact letter match (first standalone capital letter in the reply)."""
    import re
    want = str(private.answer).strip().upper()
    m = re.search(r"\b([A-Z])\b", (answer_text or "").strip())
    if not m:
        m = re.search(r"([A-Z])", (answer_text or "").strip())
    return 1.0 if m and m.group(1).upper() == want else 0.0
