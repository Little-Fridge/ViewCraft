"""MUIRBENCH adapter (MUIRBENCH/MUIRBENCH) -- MULTI-IMAGE reasoning, the task
type the main suite was missing. Every item carries 2-9 images; the question
requires relating them (difference spotting, counting across images, ordering,
attribute similarity, diagram/scene/action understanding...).

Why it belongs. The mechanism's distinctive strategies are exactly the
multi-image ones: composing several images side by side, marking candidates,
set-of-mark, and the visual memory canvas for tracking what has been compared.
Single-image (babyvision) and frame-pool video (mvbench) do not exercise the
"relate N independent images" decision at all. 12 task types, several directly
mechanism-friendly (Difference Spotting 340, Counting 234, Attribute
Similarity 196). Answers are option letters.
"""
from __future__ import annotations

import ast
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "muirbench"


def load(root: str | Path, seed: int = 20260911,
         tasks: tuple | None = None) -> list[Sample]:
    import pyarrow.parquet as pq
    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")
    samples: list[Sample] = []
    for p in pqs:
        for i, r in enumerate(pq.read_table(p).to_pylist()):
            task = str(r.get("task") or "")
            if tasks and task not in tasks:
                continue
            blobs = r.get("image_list") or []
            media = []
            for j, b in enumerate(blobs):
                b = b.get("bytes") if isinstance(b, dict) else b
                if b:
                    media.append(write_media(DOMAIN, b,
                                             f"{r.get('idx', i)}_{j}"))
            if not media:
                continue
            try:
                opts = ast.literal_eval(str(r.get("options") or "[]"))
            except (ValueError, SyntaxError):
                opts = []
            options = tuple(str(o) for o in opts)
            item_id = f"{DOMAIN}/{r.get('idx', i)}"
            public = PublicItem(item_id=item_id, domain=DOMAIN,
                                question=str(r.get("question") or ""),
                                media_ids=tuple(media), options=options,
                                meta={"task": task,
                                      "n_images": len(media),
                                      "image_relation": r.get("image_relation"),
                                      "image_type": r.get("image_type")})
            private = PrivateItem(item_id=item_id,
                                  answer=str(r.get("answer") or "").strip().upper(),
                                  answer_type="letter", evidence_source="none",
                                  meta={"task": task})
            samples.append(Sample(public=public, private=private,
                                  group_key=str(r.get("idx", i))))
    print(f"[muirbench] kept {len(samples)} items "
          f"({len({s.public.meta['task'] for s in samples})} task types)")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    import re
    want = str(private.answer).strip().upper()
    m = re.search(r"\b([A-Z])\b", (answer_text or "").strip().upper())
    return 1.0 if m and m.group(1) == want else 0.0
