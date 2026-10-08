"""NABirds adapter (zguo0525/nabirds-dataset) -- FINE-GRAINED bird species
classification, 555 classes. The discriminative detail is tiny (wing bars, eye
rings, bill shape), so it is a zoom-into-the-diagnostic-part regime: the model
must find and read the small part rather than gist the bird.

MCQ with the true species plus three distractor species drawn from OTHER classes
in the dataset, so the model must verify against the image, not prior.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "nabirds"


def load(root: str | Path, seed: int = 20260911,
         n_items: int = 600) -> list[Sample]:
    import pyarrow.parquet as pq
    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")
    rows = []
    for p in pqs:
        rows.extend(pq.read_table(p).to_pylist())
    classes = sorted({str(r.get("label")) for r in rows})
    rng = random.Random(seed)

    samples: list[Sample] = []
    for i, r in enumerate(rows):
        blob = r.get("image")
        blob = blob.get("bytes") if isinstance(blob, dict) else blob
        if not blob:
            continue
        label = str(r.get("label"))
        distract = [c for c in rng.sample(classes, 8) if c != label][:3]
        if len(distract) < 3:
            continue
        media = write_media(DOMAIN, blob, str(i))
        options = tuple([f"species {label}"] + [f"species {c}" for c in distract])
        order = list(range(4))
        rng.shuffle(order)
        options = tuple(options[j] for j in order)
        gold = "ABCD"[order.index(0)]
        item_id = f"{DOMAIN}/{i}"
        public = PublicItem(item_id=item_id, domain=DOMAIN,
                            question=("Which bird species is shown in the "
                                      "image?"),
                            media_ids=(media,), options=options,
                            meta={})
        private = PrivateItem(item_id=item_id, answer=gold,
                              answer_type="letter", evidence_source="none",
                              meta={"label": label})
        samples.append(Sample(public=public, private=private,
                              group_key=str(i)))
        if len(samples) >= n_items:
            break
    print(f"[nabirds] kept {len(samples)} fine-grained MCQ items "
          f"({len(classes)} classes)")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    import re
    m = re.search(r"\b([A-D])\b", (answer_text or "").strip().upper())
    return 1.0 if m and m.group(1) == str(private.answer) else 0.0
