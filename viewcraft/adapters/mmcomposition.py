"""MMComposition adapter (MMComposition/MMComposition, validation split).

THE TASK. One image plus a multiple-choice question probing one of 13
compositional capabilities (attribute/relation perception and reasoning,
counting, difference spotting, object interaction, ...), at four difficulty
levels. The validation split (1100 items) ships answers; the test split is
withheld and is never touched here.

TWO ANSWER SHAPES live in this benchmark and they are scored differently:
  * number_of_answers == "one"        (997 items): a single letter.
  * number_of_answers == "indefinite" (103 items): a SET of letters
    ("A,B,C"). Scored as exact set match -- all required letters and no
    extras -- which is the benchmark's own all-or-nothing convention for
    multi-select. Partial credit would inflate a model that names every
    option.

GROUP KEY IS THE IMAGE. Rows are one question per image on this split, but
keying on the picture (not the row) keeps the invariant if a later release
asks two questions of one image.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "mmcomposition"


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
                opts = json.loads(r["options"])
            except (TypeError, json.JSONDecodeError):
                continue
            letters = sorted(opts)
            options = tuple(str(opts[k]) for k in letters)
            media = write_media(DOMAIN, blob, str(r.get("filename") or i))
            item_id = f"{DOMAIN}/{i:04d}"
            multi = str(r.get("number_of_answers")) != "one"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN,
                question=str(r["question"]).strip(), media_ids=(media,),
                options=options,
                meta={"level": r.get("level"), "class": r.get("class"),
                      "org_dataset": r.get("org_dataset"),
                      "question_type": r.get("class"),
                      "multi_answer": multi})
            private = PrivateItem(
                item_id=item_id, answer=str(r["answer"]).strip(),
                answer_type="mcq_set" if multi else "mcq",
                evidence_source="official", meta={"level": r.get("level"),
                                                  "class": r.get("class")})
            samples.append(Sample(public=public, private=private,
                                  group_key=str(r.get("filename") or i)))
    n_multi = sum(1 for s in samples if s.public.meta["multi_answer"])
    print(f"[{DOMAIN}] kept {len(samples)} items "
          f"({n_multi} multi-answer)")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    """Exact letter match for single-answer; exact SET match for multi."""
    want = {w.strip().upper() for w in str(private.answer).split(",") if w.strip()}
    got = set(re.findall(r"\b([A-Z])\b", (answer_text or "").upper()))
    if not want or not got:
        return 0.0
    if len(want) == 1:
        return 1.0 if next(iter(want)) in got else 0.0
    return 1.0 if got == want else 0.0
