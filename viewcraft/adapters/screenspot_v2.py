"""ScreenSpot-v2 adapter (HongxinLi/ScreenSpot_v2) -- the SIMPLER grounding
sibling of ScreenSpot-Pro.

1272 instructions over 960x540-ish screenshots (windows/web/mobile), with
normalized bboxes and point-in-box scoring, exactly the ScreenSpot-Pro
contract. It exists because ScreenSpot-Pro's residual errors sit at the
model's intrinsic localisation floor (Q1 targets ~20px on 4K screens),
leaving nothing for view-strategy evolution to repair; this domain's
bigger targets move the task back into the regime where the view channel
-- not raw precision -- is the constraint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Region, Sample
from .media import write_media

DOMAIN = "screenspot_v2"

PREAMBLE = (
    "The attached image is a screenshot. Find the single UI element "
    "described below and give the point you would click to operate it.\n\n"
    "Instruction: ")


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
            blob = blob.get("bytes") if isinstance(blob, dict) else blob
            if not blob:
                continue
            try:
                bb = r["bbox"] if isinstance(r["bbox"], (list, tuple)) \
                    else json.loads(r["bbox"])
                x0, y0, x1, y1 = (float(v) for v in bb)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                continue
            media = write_media(DOMAIN, blob, str(r["file_name"]))
            sha = hashlib.sha256(blob).hexdigest()
            box = Region(x0, y0, x1, y1)
            item_id = f"{DOMAIN}/{i:04d}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN,
                question=PREAMBLE + str(r["instruction"]).strip(),
                media_ids=(media,),
                meta={"question_kind": "point",
                      "category": r.get("data_source"),
                      "ui_type": r.get("data_type")})
            private = PrivateItem(
                item_id=item_id,
                answer=f"{(x0 + x1) / 2:.4f}, {(y0 + y1) / 2:.4f}",
                answer_type="point", evidence_regions=(box,),
                evidence_source="official", evidence_verified=True,
                meta={"category": r.get("data_source"),
                      "ui_type": r.get("data_type")})
            samples.append(Sample(public=public, private=private,
                                  group_key=sha))
    print(f"[{DOMAIN}] kept {len(samples)} instructions over "
          f"{len({s.group_key for s in samples})} distinct images")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    from ..evaluation.scorers import point_in_box
    return point_in_box(
        raw=answer_text, boxes=private.evidence_regions,
        img_w=960, img_h=540)
