"""HR-Bench adapter (DreamMr/HR-Bench, 8K split) -- HIGH-RESOLUTION fine
perception, the "spatial accumulation" bottleneck the suite was missing.

Why it fits the mechanism (and how it differs from grounding tasks like
ScreenSpot, which we dropped). HR-Bench asks the model to READ fine detail on
high-resolution images -- a number above an entrance, text on a sign, a small
count -- detail that is illegible at the downscaled input resolution. That is
exactly where zoom/crop genuinely recovers evidence (measured in the study's
own evidence-size probe: zoom helps once a target is small enough), so the
strategy is real, not "just point at it". Category `single` = read one
high-res image; `cross` = relate details across instances. Letter answers.
"""
from __future__ import annotations

import base64
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "hrbench"


def load(root: str | Path, seed: int = 20260911) -> list[Sample]:
    import pyarrow.parquet as pq
    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")
    samples: list[Sample] = []
    for p in pqs:
        for i, r in enumerate(pq.read_table(p).to_pylist()):
            blob = r.get("image")
            if isinstance(blob, str):           # base64-encoded jpeg
                try:
                    blob = base64.b64decode(blob)
                except Exception:                                   # noqa: BLE001
                    continue
            elif isinstance(blob, dict):
                blob = blob.get("bytes")
            if not blob:
                continue
            media = write_media(DOMAIN, blob, str(r.get("index", i)))
            options = tuple(str(r.get(k) or "") for k in ("A", "B", "C", "D"))
            item_id = f"{DOMAIN}/{r.get('index', i)}"
            public = PublicItem(item_id=item_id, domain=DOMAIN,
                                question=str(r.get("question") or ""),
                                media_ids=(media,), options=options,
                                meta={"category": r.get("category"),
                                      "cycle_category": r.get("cycle_category")})
            private = PrivateItem(item_id=item_id,
                                  answer=str(r.get("answer") or "").strip().upper(),
                                  answer_type="letter", evidence_source="none",
                                  meta={"category": r.get("category")})
            samples.append(Sample(public=public, private=private,
                                  group_key=str(r.get("index", i))))
    print(f"[hrbench] kept {len(samples)} items")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    import re
    want = str(private.answer).strip().upper()
    m = re.search(r"\b([A-D])\b", (answer_text or "").strip().upper())
    return 1.0 if m and m.group(1) == want else 0.0
