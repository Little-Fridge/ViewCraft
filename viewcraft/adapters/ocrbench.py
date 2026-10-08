"""OCRBench adapter (D7): reading text that is physically small on screen.

Why this domain is in the study: OCRBench's failure mode is legible-in-the-
original / illegible-after-downscale. That is exactly the failure ViewCraft's
phi is supposed to repair, so the domain can discriminate the mechanism rather
than merely rank models.

Released layout: one parquet, 1000 rows, image bytes embedded, `answer` a LIST
of acceptable strings.

Scoring follows the official OCRBench evaluator: a prediction is correct if any
accepted answer appears in it, case-insensitively, with two documented
exceptions carried over from the official script --- the two handwriting /
digit-string subsets strip whitespace from both sides before comparing, because
their ground truth contains formatting spaces that no model reproduces.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "ocrbench"

# The official evaluator special-cases these: whitespace inside the string is
# not meaningful for them, so it is removed on both sides before matching.
_STRIP_SPACE_SETS = {"HME100k", "ORAND"}

# Recognition subsets are single-word / single-expression reads; the VQA and
# KIE subsets are questions over a larger image.
RECOGNITION_TYPES = {
    "Regular Text Recognition", "Irregular Text Recognition",
    "Artistic Text Recognition", "Handwriting Recognition",
    "Digit String Recognition", "Non-Semantic Text Recognition",
    "Handwritten Mathematical Expression Recognition",
}


def load(root: str | Path, seed: int = 20260911,
         question_types: tuple[str, ...] | None = None) -> list[Sample]:
    """Load OCRBench rows as public/private pairs.

    `question_types` restricts to a subset. The default keeps the VQA and
    key-information subsets and drops the pure crop-recognition ones: those
    ship images whose long edge is already ~100 px, so there is no resolution
    to recover and a view mechanism cannot express anything on them. The
    exclusion is by IMAGE GEOMETRY, decided before any score was measured, and
    is reported as a protocol deviation.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")

    keep = (set(question_types) if question_types
            else {"Scene Text-centric VQA", "Doc-oriented VQA",
                  "Key Information Extraction"})

    samples: list[Sample] = []
    for p in pqs:
        for i, row in enumerate(pq.read_table(p).to_pylist()):
            qtype = row.get("question_type") or ""
            if keep and qtype not in keep:
                continue
            img = row.get("image") or {}
            blob = img.get("bytes") if isinstance(img, dict) else img
            if not blob:
                continue
            path = write_media(DOMAIN, blob, (img or {}).get("path", ""))
            answers = [str(a) for a in (row.get("answer") or []) if str(a)]
            if not answers:
                continue
            item_id = f"ocrbench/{row.get('dataset')}/{Path(path).stem[:16]}_{i}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN,
                question=str(row.get("question") or ""),
                media_ids=(path,),
                meta={"subset": row.get("dataset"), "question_type": qtype},
            )
            private = PrivateItem(
                item_id=item_id, answer=answers, answer_type="string_list",
                evidence_source="none",
                meta={"subset": row.get("dataset"), "question_type": qtype},
            )
            # group by IMAGE, so the same picture cannot straddle two splits
            samples.append(Sample(public=public, private=private,
                                  group_key=Path(path).stem))
    return samples


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official OCRBench containment match against any accepted answer."""
    pred = _norm(answer_text)
    if not pred:
        return 0.0
    subset = (private.meta or {}).get("subset")
    golds = private.answer if isinstance(private.answer, list) \
        else [private.answer]
    for g in golds:
        gold = _norm(str(g))
        if not gold:
            continue
        if subset in _STRIP_SPACE_SETS:
            if gold.replace(" ", "") in pred.replace(" ", ""):
                return 1.0
        elif gold in pred:
            return 1.0
    return 0.0
