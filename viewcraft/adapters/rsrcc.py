"""RSRCC (Google Research, 2026) adapter: BEFORE/AFTER satellite change reading.

The task. Two overhead captures of the same place at two times. The question
names a sub-region ("the front yard of the property at the northeastern curve
of the cul-de-sac") and asks what changed there: a tree planted or removed, a
building demolished or extended, a lot paved. Answering needs three things in
order -- find the named region in image 1, find the SAME region in image 2,
and decide whether the difference there is the one the options describe.

WHY THIS DOMAIN IS WORTH THE QUOTA: it is a PRE-REGISTERED TEST OF A CLAIM,
not another row in the table. Written 2026-09-17, before any RSRCC inference.

  Two "headroom" stories have been available for why co-evolution helps. The
  RESOLUTION story says the gain comes from recovering pixels the ordinary
  view throws away. The TASK story says it comes from the task being far from
  solved, whatever the pixels. Over the 11 evolved runs with paired test rows
  these separate cleanly: spearman(delta, baseline_acc) = -0.629, while
  spearman(delta, zoomable) = -0.083 -- task headroom predicts the gain and
  resolution headroom does not. (The negative half of that was already on
  record at tests/test_regressions.py:6210 on 5 domains with a different
  metric; what is new is the positive predictor.)

  RSRCC separates them BY CONSTRUCTION on the resolution axis. Every image in
  the release is 512 px square, and `phi.overview_long_edge_px` is set to 512
  for this domain (initial.py), so `render.py:_fit`'s `if scale > 1.0 and not
  allow_upscale` branch makes every crop return exactly the pixels the
  whole-image view already showed. Resolution headroom here is not "small",
  it is ZERO, and arithmetically so.

  CORRECTION, 2026-09-17, written after the first gate baseline came back and
  BEFORE any test-split number exists. The paragraph above originally
  continued "yet the task is far from solved". That was an ASSUMPTION, not a
  measurement, and the measurement does not support it: round-0 gate baseline
  on this domain is 0.854-0.896 on a 4-way question with a 0.25 floor. By the
  same table that motivated this test (EXPLORATION_FINDINGS §14), a 0.85
  baseline sits in the `baseline >= 0.75` bucket, whose four members average
  a delta of only +0.024. So TASK headroom here is small too.

  What that costs the design, stated plainly: the test is now ONE-SIDED. A
  gain at arithmetically zero resolution headroom still damages the
  resolution story. A null does NOT confirm the task story, because with
  little task headroom both stories predict little gain. RSRCC can falsify
  one claim; it cannot confirm the other. A domain that is simultaneously
  low-baseline and 512 px would be needed for that, and nothing in the
  corpus currently is.

  PREDICTION, recorded before the run and left unedited: the resolution story
  predicts NO gain on RSRCC; the task story predicts a gain, delivered
  through phi's pairing and psi's ordering rather than through
  magnification. Whichever way it falls it is reportable, and it falls on one
  run, scored once. The `locality` stratification below was declared before
  the run too, so the local/global breakdown may be reported alongside the
  headline without it being a post-hoc subset.

What phi and psi have left to do once zooming is inert is exactly the thing
this benchmark is about: the two images must be put side by side at matched
scale, the named region has to be isolated in BOTH, and the pair has to stay
in the context window together long enough to be compared. A crop that
isolates the cul-de-sac in image 1 but not image 2 is worse than no crop.

ORDERING HAZARD, stated rather than defended against. The two views are
positional -- view 1 is BEFORE, view 2 is AFTER -- and the stem says so.
Nothing in the option text recovers that mapping, so a psi that evicts a
whole-image view or orders crops ahead of sources silently inverts the
direction of every change ("a tree was planted" <-> "a tree was removed"),
and several option sets contain both directions. This is the same property
BLINK and MMAD were added to expose, and it is left in place because
navigating it is part of what the search is being asked to do.

MIXED CHANCE FLOOR -- MEASURED, AND IT FORCED TWO DECISIONS. Both were taken
on the loaded corpus BEFORE any RSRCC inference was run, and both are recorded
here with the number that forced them rather than folded in quietly.

  1. THE Yes/No HALF IS DROPPED. Of the 1891 binary rows extracted, 1827
     have the answer "Yes" -- the string "Yes" scores 0.966 on that half
     with no image at all, and answering the first option scores 0.692 over
     the whole release. The generator appears only to have asked about changes that did
     happen. A half where a constant beats every model is not a perception
     task, it is a prior, and any "gain" measured across it would be noise on
     a saturated subset. Only the 4-way rows are kept. This is an exclusion
     for validity, decided on a measurement, not a split tuned to a result.

  2. THE 4-WAY OPTIONS ARE PERMUTED PER ITEM. Even on the 4-way half the
     release's gold letters are {A 751, B 996, C 86, D 2} over 1835 rows:
     the true answer is almost never C or D, so the majority letter alone
     scores 0.543. After permutation the subsample's gold positions are
     {403, 370, 372, 355} over 1500 items, i.e. a 0.269 majority class
     against a 0.25 floor. Left
     alone, the search could raise the score by evolving a theta that leans
     on the letter prior -- a shortcut that would be credited as a perception
     gain. Each item's options are therefore shuffled by a permutation keyed
     on the item id, so the gold position is uniform by construction and the
     same permutation is produced on every machine and every run. Gold is
     carried as the index INTO THE PERMUTED LIST, which is the list the
     executor is shown and the list the scorer indexes, so there is no second
     labelling to fall out of sync.

  What permuting does NOT fix, stated plainly: the distractors are still
  written by the same generator, so an option phrased as a non-change ("It
  remained exactly the same") may remain recognisable as a distractor from
  text alone. That is a property of the release, and it inflates every arm
  equally.

A CAVEAT THAT IS NOT REMOVED. A minority of rows are globally phrased --
"which of the following best characterizes the relationship between the
scenes", with options like "High degree of visual parity and scene
consistency". Those are answerable from a gestalt impression and need no
localisation at all, so they dilute the mechanism this domain was chosen to
exercise. They are NOT filtered out: the filter would be a threshold chosen
by me on question wording, and the standing rule here is that splits and
metrics are not tuned. They are flagged in `meta["locality"]` so the dilution
can be reported instead of hidden.

`group_key` is the IMAGE PAIR, not the row: the release asks several questions
about the same two photographs, so splitting by row would put the same place
in train and in test.

Scoring is exact multiple-choice accuracy.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "rsrcc"

PREAMBLE = (
    "Two overhead images of the same location are attached. The FIRST is the "
    "BEFORE capture and the SECOND is the AFTER capture; they are aligned to "
    "the same area, and the question is about what changed between them.\n\n"
)

# Phrases that name a place inside the frame. Used ONLY to label an item for
# reporting -- nothing is dropped on the strength of it, and no split, gate or
# metric reads it. Deliberately literal rather than clever: a fuzzy matcher
# here would be a tuning knob on the benchmark's composition.
LOCALITY_MARKERS = (
    "section of", "side of", "corner", "located", "near the", "at the end",
    "in the north", "in the south", "in the east", "in the west",
    "northeast", "northwest", "southeast", "southwest",
    "upper", "lower", "left side", "right side", "top of", "bottom of",
    "adjacent", "front yard", "backyard", "along the", "next to",
)


def locality_of(question: str) -> str:
    q = question.lower()
    return "local" if any(m in q for m in LOCALITY_MARKERS) else "global"


def load(root: str | Path, seed: int = 20260911,
         n_items: int = 1500) -> list[Sample]:
    import pyarrow.parquet as pq

    root = Path(root)
    meta = root / "rsrcc_meta.parquet"
    if not meta.exists():
        raise FileNotFoundError(
            f"{meta} missing -- run scripts/prep_rsrcc.py first")
    rows = pq.read_table(meta).to_pylist()

    # Deterministic subsample that preserves the locality mix. Seed-independent
    # by design: the benchmark's composition should not move between runs, only
    # the split assignment should. Binary rows are dropped at the door -- see
    # the module docstring for the 0.966 constant-answer measurement.
    strata: dict[str, list] = {}
    n_binary = 0
    for r in rows:
        if int(r["n_options"]) != 4:
            n_binary += 1
            continue
        strata.setdefault(locality_of(r["question"]), []).append(r)

    total = sum(len(v) for v in strata.values())
    keep = []
    for _, group in sorted(strata.items()):
        group.sort(key=lambda r: hashlib.sha256(
            f"{r['pair']}|{r['question']}".encode()).hexdigest())
        keep.extend(group[: max(1, round(n_items * len(group) / total))])

    samples: list[Sample] = []
    dropped_media = 0
    for r in keep:
        before, after = root / r["before"], root / r["after"]
        # Both files must exist here, not at render time: every crop re-reads
        # the original file, so a path that resolves only at load would fail
        # mid-trajectory instead of at the door.
        if not before.exists() or not after.exists():
            dropped_media += 1
            continue
        options = tuple(json.loads(r["options"]))
        gold = int(r["gold_index"])
        loc = locality_of(r["question"])
        item_id = (f"{DOMAIN}/{r['pair']}/"
                   f"{hashlib.sha256(r['question'].encode()).hexdigest()[:8]}")
        # Break the release's letter prior. The permutation is a pure function
        # of the item id, so it is identical on every machine, every run and
        # every arm -- the four arms see the same option order, which is what
        # makes the paired comparison a comparison of the arms and not of two
        # shufflings. `gold` is re-derived as the index into the permuted list,
        # i.e. the only list that exists downstream.
        order = list(range(len(options)))
        random.Random(f"rsrcc-perm|{item_id}").shuffle(order)
        options = tuple(options[i] for i in order)
        gold = order.index(gold)
        kind = "mc4"
        public = PublicItem(
            item_id=item_id, domain=DOMAIN,
            question=PREAMBLE + r["question"].strip(),
            media_ids=(str(before), str(after)), options=options,
            meta={"question_kind": kind, "category": kind, "locality": loc,
                  "n_images": 2, "n_options": len(options)},
        )
        private = PrivateItem(
            item_id=item_id, answer=options[gold],
            answer_type="multiple_choice", evidence_source="none",
            meta={"answer_position": gold, "question_kind": kind,
                  "locality": loc},
        )
        # The PLACE, not the question, is what must not straddle a split.
        samples.append(Sample(public=public, private=private,
                              group_key=r["pair"]))

    n_local = sum(1 for s in samples if s.public.meta["locality"] == "local")
    print(f"[{DOMAIN}] kept {len(samples)} 4-way items "
          f"(dropped {n_binary} Yes/No rows -- 0.966 constant-answer, see "
          f"module docstring; {dropped_media} missing media); "
          f"{len({s.group_key for s in samples})} distinct image pairs; "
          f"{n_local} local / {len(samples)-n_local} global")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(
        raw=answer_text, options=list(public.options),
        correct_index=private.meta["answer_position"])
