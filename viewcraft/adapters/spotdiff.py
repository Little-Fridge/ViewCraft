"""Spot-the-Diff adapter (Lancelot53/spot-the-diff) -- CHANGE DETECTION between
natural image pairs, a perception bottleneck distinct from counting / video /
multi-image: compare two images and verify which regions actually changed.

Why it fits. Each item is an (image1, image2) pair plus human difference
sentences. The model must sweep both images, candidate-by-candidate, and verify
each claimed change region -- exactly the compare-and-mark regime the canvas
and side-by-side composition serve. Framed as MCQ ("which of these is a real
difference?"), with one true sentence and three distractors drawn from OTHER
items (plausible but wrong for this pair), so the model cannot answer from
caption priors alone.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "spotdiff"


def load(root: str | Path, seed: int = 20260911, n_items: int = 600) -> list[Sample]:
    import pyarrow.parquet as pq
    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")
    rows = []
    for p in pqs:
        rows.extend(pq.read_table(p).to_pylist())

    # pool of difference sentences for distractors
    pool = []
    for r in rows:
        for s in (r.get("sentences") or []):
            s = str(s).strip()
            if len(s) > 12:
                pool.append(s)
    rng = random.Random(seed)

    samples: list[Sample] = []
    for i, r in enumerate(rows):
        sents = [str(s).strip() for s in (r.get("sentences") or [])
                 if len(str(s).strip()) > 12]
        if not sents:
            continue
        b1 = (r.get("image1") or {}).get("bytes") if isinstance(
            r.get("image1"), dict) else r.get("image1")
        b2 = (r.get("image2") or {}).get("bytes") if isinstance(
            r.get("image2"), dict) else r.get("image2")
        if not b1 or not b2:
            continue
        p1 = write_media(DOMAIN, b1, f"{i}_1")
        p2 = write_media(DOMAIN, b2, f"{i}_2")
        truth = rng.choice(sents)
        distract = rng.sample(pool, 12)
        distract = [d for d in distract if d not in sents][:3]
        if len(distract) < 3:
            continue
        options = tuple([truth] + distract)
        order = list(range(4))
        rng.shuffle(order)
        options = tuple(options[j] for j in order)
        gold = "ABCD"[order.index(0)]
        item_id = f"{DOMAIN}/{i}"
        public = PublicItem(
            item_id=item_id, domain=DOMAIN,
            question=("These two images differ in some ways. Which of the "
                      "following is a REAL difference between them?"),
            media_ids=(p1, p2), options=options, meta={"n_images": 2})
        private = PrivateItem(item_id=item_id, answer=gold,
                              answer_type="letter", evidence_source="none",
                              meta={"truth": truth})
        samples.append(Sample(public=public, private=private,
                              group_key=f"pair_{i}"))
        if len(samples) >= n_items:
            break
    print(f"[spotdiff] kept {len(samples)} MCQ items")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    import re
    m = re.search(r"\b([A-D])\b", (answer_text or "").strip().upper())
    return 1.0 if m and m.group(1) == str(private.answer) else 0.0
