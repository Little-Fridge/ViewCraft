"""SciFIBench Caption2Figure adapter: CROSS-FIGURE DISCRIMINATION.

Why this domain is in the study. Every perception domain already here asks
the model to find something INSIDE one picture -- a small object (vstar,
hrbench), a countable set (countbench, tallyqa), a moment in a video
(mvbench, nextgqa), a part of a bird (nabirds). Zooming is the answer to all
of them, and the zoom is the mechanism's most-exercised move. Caption2Figure
asks the opposite question: given one long scientific caption and FIVE
candidate figures from real arXiv papers, which figure is the caption
describing? The bottleneck is not resolution, it is COMPARISON -- the five
candidates are drawn from the same paper family, so they share axis style,
colour scheme and subject, and the discriminating evidence is a legend entry
or an axis label that only means something relative to the other four.

That makes it the sharpest available test of the mechanism's PSI half, the
half the current main table barely exercises:

  * The five figures cannot all be inspected at full resolution inside the
    image budget, so `psi.max_images_per_request`, `evict_policy` and
    `layout_expr` decide what the model gets to compare against what. Every
    single-image domain leaves those fields with nothing to do.
  * The comparison is not transitive-free: ruling out candidate C is
    evidence about A, so the visual working-memory canvas ("cross off
    options you have ruled out", runtime.py:520) has a real use here, as
    opposed to being a counting aid.
  * A zoom is still available and still useful (median figure long edge
    1704 px, so a crop genuinely recovers detail), but it costs a slot that
    a rival candidate could have occupied. That tradeoff -- detail versus
    breadth, under a fixed image cap -- is the decision the phi/psi split
    was built to express, and no domain in the main table poses it.

Released layout: five parquets under `data/`, split CS / General x
Caption2Figure / Figure2Caption. Only Caption2Figure is loaded (1000 items,
500 CS + 500 General): Figure2Caption ships long LaTeX captions as the
OPTIONS, which makes it a text-discrimination task wearing an image, and the
option text would then carry most of the signal.

Per row: `media` is a list of exactly 5 struct<bytes> (verified: 1000/1000
rows carry exactly five), `messages` is a JSON string holding a single user
turn whose `question` is the caption followed by "Which image best matches
the caption?" and whose `options` is always the fixed dict
{A: "A)  image0", ... E: "E)  image4"}, and `answer` is a bare letter.

CRITICAL ORDERING PROPERTY, and the reason media order is never touched: the
option letter is positional -- option A IS media[0], B IS media[1], and so
on. The release's own option strings say so literally ("A)  image0"). There
is no option TEXT that could recover the mapping if the images were
reordered or one were dropped, so a presentation change that permutes the
attached images silently relabels every answer. This is the same property
that makes BLINK the ordering canary, and it is stated here because the
adapter must never sort, dedupe or filter `media`.

The options presented to the executor are therefore rewritten from the
release's "A)  image0" into "candidate figure 1", and one sentence is
prepended to the caption stating that the five candidates arrive as the
whole-image views in that order. Keeping the release's wording would point
the model at a token "image0" that appears nowhere in what it is shown;
`describe_visible` labels what it sends as "Image 1: a FULL frame ..." and
never names a source. The added sentence is a true statement about the
harness, in the same class as BLINK's task preamble, not a hint about the
answer. Order is preserved exactly; only the wording changes, and the gold
index is the released letter's position, unmoved.

The standing hazard this creates is stated rather than hidden. Under a psi
that evicts a whole-image view, or orders crops first, "candidate figure 3"
stops naming the third attached image, and the item is then scored wrong for
a presentation reason. That is the same failure BLINK was added to detect
(registry note: "the only domain here where a psi ordering bug shows up as a
wrong answer rather than as a silent accuracy dip"), and the image cap below
is set so the search has to navigate it rather than be protected from it.

Gold letters are spread across the five positions in the release (A 201,
B 229, C 184, D 194, E 192), so there is nothing to de-bias by shuffling,
and shuffling would add a mapping layer that could silently invert.

`group_key` is the hash of the five-figure SET, not the row: 1000 rows use
only 671 distinct figure sets (the same five candidates are re-asked with a
different caption up to five times). Splitting by row would put the same
five figures in train and in test, which is the leak `mode: by_image`
exists to prevent.

Scoring is the official SciFIBench metric: exact multiple-choice accuracy
over the five candidates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "scifibench"

# Only the Caption2Figure half; see the module docstring for why
# Figure2Caption is excluded rather than merely unused.
GLOB = "*Caption2Figure*.parquet"

N_CANDIDATES = 5

# The release labels its options "A)  image0" .. "E)  image4", naming a token
# the executor never sees. These name the candidates in the order they are
# attached, which is the only handle the model is actually given.
OPTION_TEXTS = ("candidate figure 1", "candidate figure 2",
                "candidate figure 3", "candidate figure 4",
                "candidate figure 5")

# Stated once in the stem because nothing else tells the model that the
# whole-image views ARE the five candidates, in order. True of the harness
# and independent of the answer; compare BLINK, where the task preamble is
# likewise load-bearing and likewise comes from the question side.
PREAMBLE = ("The five candidate figures are attached as whole frames, in "
            "order: candidate figure 1 is the first, candidate figure 5 the "
            "last.\n\n")


def parse_row(row: dict) -> tuple[str, int] | None:
    """Return (caption question, gold index), or None if the row is unusable.

    A row is dropped rather than repaired when the messages blob is not the
    single-user-turn shape the release documents, or when the gold letter
    falls outside the five candidates: both would otherwise be scored against
    an option that does not exist.
    """
    try:
        msgs = json.loads(row.get("messages") or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(msgs, list) or len(msgs) != 1:
        return None
    question = str((msgs[0] or {}).get("question") or "").strip()
    if not question:
        return None
    gold = str(row.get("answer") or "").strip().upper()
    if len(gold) != 1:
        return None
    idx = ord(gold) - ord("A")
    if not 0 <= idx < N_CANDIDATES:
        return None
    return question, idx


def load(root: str | Path, seed: int = 20260911) -> list[Sample]:
    import pyarrow.parquet as pq

    root = Path(root)
    pqs = sorted(p for p in root.rglob(GLOB) if ".cache" not in p.parts)
    if not pqs:
        raise FileNotFoundError(f"no {GLOB} under {root}")

    samples: list[Sample] = []
    dropped = 0
    for p in pqs:
        # CS and General are separate releases with overlapping `id` values,
        # so provenance rides in the item_id via the file stem.
        subset = "cs" if p.name.startswith("CS_") else "general"
        for row in pq.read_table(p).to_pylist():
            parsed = parse_row(row)
            if parsed is None:
                dropped += 1
                continue
            question, gold_idx = parsed

            blobs = [b.get("bytes") if isinstance(b, dict) else b
                     for b in (row.get("media") or [])]
            # Exactly five, in the released order. A row with a missing
            # candidate cannot be scored positionally and is dropped, never
            # compacted -- compacting would shift every letter after the gap.
            if len(blobs) != N_CANDIDATES or not all(blobs):
                dropped += 1
                continue

            rid = str(row.get("id") or len(samples))
            media = [write_media(DOMAIN, b, f"{subset}_{rid}_{j}")
                     for j, b in enumerate(blobs)]

            # The five-figure SET, order-insensitive, is the unit that must
            # not straddle a split: 1000 rows reuse only 671 distinct sets.
            group_key = hashlib.sha256(
                "".join(sorted(Path(m).stem for m in media)).encode()
            ).hexdigest()[:16]

            item_id = f"{DOMAIN}/{subset}/{rid}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN, question=PREAMBLE + question,
                media_ids=tuple(media), options=OPTION_TEXTS,
                meta={"subset": subset, "n_images": len(media),
                      "category": row.get("category"),
                      "question_type": row.get("question_type")},
            )
            private = PrivateItem(
                item_id=item_id, answer=OPTION_TEXTS[gold_idx],
                answer_type="multiple_choice", evidence_source="none",
                meta={"subset": subset, "answer_position": gold_idx,
                      "answer_letter": chr(ord("A") + gold_idx)},
            )
            samples.append(Sample(public=public, private=private,
                                  group_key=group_key))

    print(f"[{DOMAIN}] kept {len(samples)} items, dropped {dropped}; "
          f"{len({s.group_key for s in samples})} distinct figure sets")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(
        raw=answer_text, options=list(public.options),
        correct_index=private.meta["answer_position"])
