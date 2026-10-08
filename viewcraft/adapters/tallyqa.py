"""TallyQA-complex adapter (vikhyatk/tallyqa-test, complex split) -- counting
RARE / occluded objects in real images ("how many hoses are attached to the
cow's udders?"). Distinct from syn_count (many similar dots): here the target
is a few hard-to-see instances among clutter, so it is find-then-count with
occlusion -- the zoom-to-detail + mark regime, not mass enumeration.

Only the complex (is_simple=False) QA pairs are kept; the simple split is
near-saturated and would dilute the headroom. Exact integer match.
"""
from __future__ import annotations

import ast
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "tallyqa"


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
            qa = r.get("qa")
            if isinstance(qa, str):
                try:
                    qa = ast.literal_eval(qa)
                except (ValueError, SyntaxError):
                    continue
            for j, q in enumerate(qa or []):
                if q.get("is_simple"):
                    continue                       # complex split only
                media = write_media(DOMAIN, blob, f"{i}")
                item_id = f"{DOMAIN}/{i}_{j}"
                public = PublicItem(
                    item_id=item_id, domain=DOMAIN,
                    question=str(q.get("question") or "") +
                    " Answer with a number.",
                    media_ids=(media,), meta={"complex": True})
                private = PrivateItem(item_id=item_id,
                                      answer=str(q.get("answer")),
                                      answer_type="integer",
                                      evidence_source="none", meta={})
                samples.append(Sample(public=public, private=private,
                                      group_key=f"{i}"))
    print(f"[tallyqa] kept {len(samples)} complex counting items")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    import re
    m = re.search(r"(\d+)", (answer_text or "").strip())
    return 1.0 if m and int(m.group(1)) == int(private.answer) else 0.0
