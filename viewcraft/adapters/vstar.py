"""V*Bench adapter (D3): high-resolution visual search.

Released layout: one JSON per question next to its image, grouped in
category directories (direct_attributes / relative_position / OCR /
GPT4V-hard). Each JSON carries question, options, target_object names and
bbox in ABSOLUTE [x, y, w, h] source pixels.

The correct answer is not marked in the JSON: V*Bench's convention is that
options[0] is the correct one, and the official evaluator shuffles at
presentation time. We therefore store the answer text privately and present
options in a deterministic, per-item shuffled order — the presented order is
recorded so scoring maps back to the original option identity rather than
re-sorting.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Region, Sample

CATEGORIES = ("direct_attributes", "relative_position", "OCR", "GPT4V-hard")

# The official evaluator (vstar_bench_eval.py) iterates only these two
# categories, which together hold exactly the benchmark's 191 questions
# (115 + 76). OCR and GPT4V-hard ship in the same repo but are supplementary
# subsets outside the headline benchmark, so they are excluded from the main
# protocol and available for separate reporting.
OFFICIAL_CATEGORIES = ("direct_attributes", "relative_position")
IMAGE_EXTS = (".jpg", ".JPG", ".jpeg", ".png", ".PNG", ".webp")

# V*Bench ships the correct choice first in `options`.
CORRECT_INDEX_IN_RELEASE = 0


def _image_for(json_path: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        p = json_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def _regions_from_bbox(bboxes: list[list[float]], w: int,
                       h: int) -> tuple[Region, ...]:
    out = []
    for box in bboxes:
        x, y, bw, bh = box
        # released as absolute x, y, width, height
        out.append(Region.from_pixels((x, y, x + bw, y + bh), w, h))
    return tuple(out)


def load(root: str | Path, seed: int = 20260911,
         categories: tuple[str, ...] = OFFICIAL_CATEGORIES) -> list[Sample]:
    """Load V*Bench questions as public/private Sample pairs.

    Defaults to the 191-question official category set.
    """
    from PIL import Image

    root = Path(root)
    samples: list[Sample] = []

    for cat in categories:
        d = root / cat
        if not d.is_dir():
            continue
        for jp in sorted(d.glob("*.json")):
            ip = _image_for(jp)
            if ip is None:
                continue
            meta = json.loads(jp.read_text())
            options = list(meta.get("options") or [])
            if not options:
                continue
            answer_text = options[CORRECT_INDEX_IN_RELEASE]

            with Image.open(ip) as im:
                W, H = im.size

            item_id = f"vstar/{cat}/{jp.stem}"
            # Deterministic per-item presentation order, seeded by item id so
            # every arm and condition sees an identical ordering.
            order = list(range(len(options)))
            random.Random(f"{seed}:{item_id}").shuffle(order)
            presented = [options[i] for i in order]
            answer_position = order.index(CORRECT_INDEX_IN_RELEASE)

            public = PublicItem(
                item_id=item_id,
                domain="vstar",
                question=meta["question"],
                media_ids=(str(ip),),
                options=tuple(presented),
                meta={
                    "category": cat,
                    "image_width": W,
                    "image_height": H,
                    "n_options": len(presented),
                    # target NAMES come from the question text itself, so this
                    # is not extra information; it lets phi request a crop by
                    # name without revealing where the target is.
                    "target_names": tuple(meta.get("target_object") or ()),
                },
            )
            private = PrivateItem(
                item_id=item_id,
                answer=answer_text,
                answer_type="multiple_choice",
                evidence_regions=_regions_from_bbox(
                    meta.get("bbox") or [], W, H),
                evidence_source="official",
                evidence_verified=True,   # released bboxes; audited in pre-check
                meta={
                    "category": cat,
                    "answer_letter": chr(ord("A") + answer_position),
                    "answer_position": answer_position,
                    "presented_order": order,
                    "release_options": options,
                },
            )
            samples.append(Sample(public=public, private=private,
                                  group_key=f"{cat}/{jp.stem}"))
    return samples


def score(trajectory_answer: str, private: PrivateItem,
          public: PublicItem) -> float:
    """V*Bench accuracy under the ORIGINAL presented option mapping.

    Accepts either the option letter or the option text. Options are never
    re-sorted; the letter is resolved against the order actually shown.
    """
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(
        raw=trajectory_answer,
        options=list(public.options),
        correct_index=private.meta["answer_position"],
    )
