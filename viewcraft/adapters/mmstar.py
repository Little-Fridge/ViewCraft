"""MMStar adapter (D8): multi-discipline multimodal reasoning, 1500 items.

Why in the study: MMStar was filtered so that every item genuinely needs the
image (the authors removed questions an LLM answers blind). That removes the
main confound in "does the view matter" --- if accuracy changes when the pixels
change, the pixels were load-bearing.

Released layout: one parquet; `question` embeds the option list as TEXT in one
of two dialects, `answer` is a bare option letter, `image` is raw bytes.

Options are presented in the order the release ships them and never re-sorted.
The gold letter is already spread across A-D in the release (429/447/309/315),
so there is nothing to de-bias by shuffling, and shuffling would only add a
mapping layer that could silently invert.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "mmstar"

# Dialect 1: "...\nOptions: A: foo, B: bar, C: baz, D: qux"
# Dialect 2: "...\nChoices:\n(A) foo\n(B) bar\n..."
_SPLIT_INLINE = re.compile(r"(?:^|,|\n)\s*([A-H])\s*[:.\)]\s+")
_SPLIT_PAREN = re.compile(r"(?:^|\n)\s*\(([A-H])\)\s*")
_HINT = re.compile(r"^\s*Hint:.*?\n", re.DOTALL)


def parse_question(raw: str) -> tuple[str, list[str], list[str]]:
    """Split the release's `question` into (stem, letters, option texts).

    Returns empty lists when no option block is found, so the caller drops the
    row rather than presenting a question with no choices.
    """
    text = raw or ""
    for marker, splitter in (("Options:", _SPLIT_INLINE),
                             ("Choices:", _SPLIT_PAREN),
                             ("Choices:", _SPLIT_INLINE)):
        i = text.find(marker)
        if i < 0:
            continue
        stem, seg = text[:i], text[i + len(marker):]
        parts = splitter.split(seg)
        letters = parts[1::2]
        texts = [t.strip().rstrip(",").strip() for t in parts[2::2]]
        if len(letters) >= 2 and len(letters) == len(texts):
            # The hint restates the answer format, which our own prompt already
            # specifies; keeping both produced contradictory instructions.
            stem = _HINT.sub("", stem)
            stem = re.sub(r"^\s*Question:\s*", "", stem.strip())
            return stem.strip(), letters, texts
    return text.strip(), [], []


def load(root: str | Path, seed: int = 20260911) -> list[Sample]:
    import pyarrow.parquet as pq

    root = Path(root)
    pqs = sorted(p for p in root.rglob("*.parquet")
                 if ".cache" not in p.parts)
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")

    samples: list[Sample] = []
    for p in pqs:
        for row in pq.read_table(p).to_pylist():
            stem, letters, texts = parse_question(row.get("question") or "")
            gold = str(row.get("answer") or "").strip().upper()
            if not letters or gold not in letters:
                # No usable option block, or a gold letter outside it. Either
                # way the item cannot be scored deterministically; dropping it
                # is reported in the manifest rather than guessed at.
                continue
            blob = row.get("image")
            if isinstance(blob, dict):
                blob = blob.get("bytes")
            if not blob:
                continue
            path = write_media(DOMAIN, blob,
                               (row.get("meta_info") or {}).get("image_path", ""))
            idx = int(row.get("index", 0))
            item_id = f"mmstar/{row.get('category', 'x').replace(' ', '_')}/{idx}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN, question=stem,
                media_ids=(path,), options=tuple(texts),
                meta={"category": row.get("category"),
                      "l2_category": row.get("l2_category"),
                      "source": (row.get("meta_info") or {}).get("source")},
            )
            private = PrivateItem(
                item_id=item_id, answer=texts[letters.index(gold)],
                answer_type="multiple_choice", evidence_source="none",
                meta={"category": row.get("category"),
                      "l2_category": row.get("l2_category"),
                      "answer_letter": gold,
                      "answer_position": letters.index(gold)},
            )
            samples.append(Sample(public=public, private=private,
                                  group_key=Path(path).stem))
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(
        raw=answer_text, options=list(public.options),
        correct_index=private.meta["answer_position"])
