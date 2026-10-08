"""HallusionBench adapter (D10): yes/no probes where the image contradicts the prior.

Why this domain is in the study: every other domain here rewards seeing MORE
detail, so a view mechanism can look good by simply spending pixels.
HallusionBench inverts the incentive. Its figures are deliberately edited to
contradict what a model already "knows" (a flipped chart axis, an illusion with
the illusion removed, a map with two countries swapped), so a model that answers
from its language prior rather than from the pixels scores near zero however
many views it requested. It therefore measures whether phi's views are actually
being READ, and it is the one domain in the set where a wrong-but-confident zoom
should be visibly punished.

Released layout: TWO parquets. Only `image-00000-of-00001.parquet` (951 rows) is
loaded here. The `non_image` parquet holds the text-only control condition ---
its rows carry no image at all, so there is nothing for a view mechanism to
crop, zoom, or select over and it cannot be run through any arm of this study.
Loading it would create rows whose view budget is vacuously satisfied and whose
accuracy differences measure only the language model. It is excluded by design,
not by oversight.

Questions are open yes/no and are presented with NO option list, matching the
official protocol: the benchmark's whole point is that the model must commit to
an affirmation or a denial in free text, and handing it "(A) yes (B) no" turns a
generation task into a two-way classification with a 50% floor.

Scoring deviation: the official evaluator calls GPT-4 to map a free-form reply
onto yes/no/unclear. This adapter uses a deterministic parser instead (no model
acts as judge anywhere in ViewCraft), and scores an unparseable reply 0.0. See
`score` for why that is the conservative choice and not a scoring bug.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "hallusionbench"

# Only the image split can be run through a view mechanism; see module docstring.
IMAGE_GLOB = "image-*.parquet"

# The release encodes gold as a STRING "1" (yes) / "0" (no), not an int or bool.
_GOLD = {"1": "yes", "0": "no"}


def load(root: str | Path, seed: int = 20260911,
         categories: tuple[str, ...] = ("VD", "VS")) -> list[Sample]:
    """Load the HallusionBench image split as public/private pairs.

    `categories` selects Visual Dependent ("VD", questions meaningless without
    the figure) and/or Visual Supplement ("VS", answerable from prior knowledge
    but contradicted by the figure). Both are kept by default.
    """
    import pyarrow.parquet as pq

    from .media import write_media

    root = Path(root)
    pqs = sorted(root.rglob(IMAGE_GLOB))
    if not pqs:
        raise FileNotFoundError(f"no {IMAGE_GLOB} under {root}")

    keep = set(categories)
    samples: list[Sample] = []

    for p in pqs:
        for i, row in enumerate(pq.read_table(p).to_pylist()):
            cat = str(row.get("category") or "")
            if keep and cat not in keep:
                continue
            gold_raw = str(row.get("gt_answer") or "").strip()
            gold = _GOLD.get(gold_raw)
            question = str(row.get("question") or "").strip()
            if gold is None or not question:
                # Gold outside {"0","1"} or an empty stem: not scoreable as a
                # yes/no item, so dropped rather than coerced to a default.
                continue

            img = row.get("image") or {}
            blob = img.get("bytes") if isinstance(img, dict) else img
            if not blob:
                continue
            filename = str(row.get("filename") or "")
            path = write_media(DOMAIN, blob,
                               (img or {}).get("path", "") or filename)

            sub = str(row.get("subcategory") or "")
            set_id = str(row.get("set_id") or "")
            figure_id = str(row.get("figure_id") or "")
            qid = str(row.get("question_id") or "")

            item_id = f"{DOMAIN}/{cat}/{sub}/{set_id}_{figure_id}_{qid}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN, question=question,
                media_ids=(path,),
                # No options: open yes/no, per the official protocol.
                meta={
                    "category": cat, "subcategory": sub,
                    "set_id": set_id, "figure_id": figure_id,
                    "question_id": qid,
                    # "visual_input" is a release field describing whether the
                    # figure is required ("1") or supplementary ("2"); it is a
                    # property of the item type, not of the answer.
                    "visual_input": str(row.get("visual_input") or ""),
                },
            )
            private = PrivateItem(
                item_id=item_id, answer=gold, answer_type="yes_no",
                evidence_source="none",
                meta={
                    "category": cat, "subcategory": sub,
                    "set_id": set_id, "figure_id": figure_id,
                    "question_id": qid,
                    "gt_answer_raw": gold_raw,
                    "gt_answer_details": row.get("gt_answer_details") or "",
                    "filename": filename,
                },
            )
            samples.append(Sample(public=public, private=private,
                                  group_key=group_key(cat, sub, set_id)))
    return samples


def group_key(category: str, subcategory: str, set_id: str) -> str:
    """The split unit: one edited/unedited FIGURE SET.

    HallusionBench's design is paired figures --- within a (category,
    subcategory, set_id) the release ships figure_id 0 and figure_id 1 (and
    occasionally 2) as the SAME scene with one fact altered, and asks the SAME
    question text of each with the gold answer flipped. In this parquet 411 of
    476 question texts recur across figure_ids of their own set, and 330 of
    those flip gold between the figures.

    So the split unit must be the set, NOT the individual figure: keying on
    figure_id would put "Is the right circle larger?" -> no in gate and the
    identical sentence -> yes in test. That is worse than ordinary leakage,
    because a model tuned on the gate half learns the answer that is wrong on
    the test half, and the measured effect would be anti-correlated with the
    mechanism under study.

    `subcategory` is part of the key because set_id restarts per subcategory:
    (VS, 0, 1) names four unrelated figures (chart, map, ocr, table). Omitting
    it would merge them into one group and coarsen the splits for no reason.
    """
    return f"{category}/{subcategory}/{set_id}"


# --- free-form yes/no parsing ------------------------------------------------

# An explicit declaration outranks everything else in the reply.
_DECLARED = re.compile(
    r"\b(?:answer|response|conclusion|verdict)\b\s*(?:is)?\s*[:\-]?\s*"
    r"[\"'*\(\[]*\s*(yes|no|true|false|correct|incorrect)\b",
    re.IGNORECASE)

# A reply that opens with the verdict, e.g. "No, the circles differ."
_LEADING = re.compile(
    r"^[\s\"'*\-\(\[]*(yes|no|yep|yeah|nope|true|false|correct|incorrect)\b",
    re.IGNORECASE)

_AFFIRM_WORDS = {"yes", "yep", "yeah", "true", "correct"}

# Phrasings that deny without using the word "no" --- "it is not", "there isn't",
# "that's incorrect", "neither circle is larger".
_NEGATIVE_PHRASE = re.compile(
    r"\b(?:is|are|was|were|does|do|did|has|have|had|can|could|would|will)"
    r"\s*n[o']t\b"
    r"|\bnot\s+(?:the\s+same|equal|larger|smaller|correct|true|present)\b"
    r"|\bno,\b|\bnot\b(?=[^.]*\?)"
    r"|\b(?:incorrect|false|untrue|neither|none of)\b"
    r"|\bthere\s+(?:is|are)\s+no\b"
    r"|\bcannot\s+be\s+(?:said|considered)\b",
    re.IGNORECASE)

# Refusals and hedges. These must be tested BEFORE the negative phrases,
# because English builds an epistemic refusal out of the same auxiliary
# negation as a denial: "I cannot tell" contains "cannot" and "I don't know"
# contains "don't", so the negative-phrase pattern below matches both and would
# silently convert "I have no idea" into a confident "no". On a benchmark whose
# gold skews negative that inflates the score of a model that declines to look.
_HEDGE = re.compile(
    r"\b(?:can|could|cannot|can't|unable\s+to)\s*(?:n[o']t)?\s*"
    r"(?:tell|say|determine|be\s+determined|know|answer|be\s+sure)\b"
    r"|\b(?:not|n[o']t)\s+(?:sure|certain|clear|possible\s+to\s+tell)\b"
    r"|\bno\s+idea\b|\bdon'?t\s+know\b|\bunclear\b|\bambiguous\b"
    r"|\bimpossible\s+to\s+(?:tell|say|determine)\b"
    r"|\bhard\s+to\s+(?:tell|say)\b"
    r"|\binsufficient\s+(?:information|detail)\b"
    r"|\bneed\s+(?:a\s+)?(?:closer|better|higher)\b"
    r"|\bcould\s+be\s+either\b|\bboth\s+yes\s+and\s+no\b",
    re.IGNORECASE)

_AFFIRM_PHRASE = re.compile(
    r"\b(?:that\s+is|that's|this\s+is|it\s+is|indeed|certainly|definitely)"
    r"\s+(?:correct|true|right|the\s+same|equal)\b"
    r"|\byes,\b"
    r"|\bthe\s+(?:answer|statement)\s+is\s+(?:correct|true)\b",
    re.IGNORECASE)


def parse_yes_no(raw: str) -> str | None:
    """Map a free-form reply to "yes" / "no", or None if genuinely undecidable.

    Resolution order is strongest-signal-first so that a stray word in the
    reasoning cannot outvote the stated verdict:
      1. an explicit "Answer: yes" style declaration
      2. a leading "Yes"/"No"
      3. an explicit hedge or refusal -> undecidable
      4. an unambiguous affirmative or negative phrase elsewhere in the reply
    A reply containing both kinds of phrase, and no declaration or leading
    verdict to break the tie, is undecidable -> None.

    The hedge test sits at step 3, after the two committed forms and before the
    loose phrase match: a reply that states a verdict and then qualifies it has
    still committed ("No, though I'd want a closer look"), while a reply whose
    only negation is the refusal itself has not ("I cannot tell").
    """
    text = (raw or "").strip()
    if not text:
        return None

    m = _DECLARED.search(text)
    if m:
        return "yes" if m.group(1).lower() in _AFFIRM_WORDS else "no"

    m = _LEADING.match(text)
    if m:
        return "yes" if m.group(1).lower() in _AFFIRM_WORDS else "no"

    if _HEDGE.search(text):
        return None

    neg = bool(_NEGATIVE_PHRASE.search(text))
    aff = bool(_AFFIRM_PHRASE.search(text))
    if neg and not aff:
        return "no"
    if aff and not neg:
        return "yes"
    # Both or neither: the reply hedged, refused, or answered a different
    # question. Guessing here would be fabrication.
    return None


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """HallusionBench per-question accuracy on an open yes/no reply.

    An undecidable reply scores 0.0 and is NEVER folded into the majority class.
    Gold in this split is 550 "no" to 401 "yes", so mapping unparseable replies
    to "no" would hand a model that hedges on every hard item ~58% of those
    items for free --- and the items a model hedges on are exactly the edited
    figures the benchmark is built to catch. That would manufacture the
    appearance of hallucination-resistance out of evasiveness, and it would bias
    the arms unequally, since a model given more views tends to produce longer,
    more heavily qualified replies. Scoring 0.0 keeps "did not commit" distinct
    from "committed correctly"; the rate of undecidable replies is reported
    separately so an arm cannot hide behind it.
    """
    parsed = parse_yes_no(answer_text)
    if parsed is None:
        return 0.0
    return 1.0 if parsed == str(private.answer) else 0.0
