"""Sizing a split in QUESTIONS when the plan is written in GROUPS.

THE MISMATCH
------------
`configs/experiment.yaml` specifies splits in GROUPS -- documents, videos,
scenes -- so that no group straddles two splits and no evidence leaks between
them. But COST scales with QUESTIONS, and one video carries ~5 of them. Both
numbers are called "test", so the discrepancy is invisible in the config:

    VSI      plan says test: 64 groups  ->  1367 questions   (21x)
    NExT-GQA plan says test: 120 groups ->   695 questions   (5.8x)
    ChartQA  plan says test: 160 groups ->   245 questions   (1.5x)

This already bit the gate once: "gate: 80" realised 466 questions on VSI, the
full plan came to 27.7 hours against a 20-hour budget, and the fix was
applied inside `EvolutionChain.__init__`. The test stage has its OWN sample
selection and never got it, so the same defect reappeared at the other entry
point -- the recurring "invariant enforced at 1 of N entry points" class.

Hence this module, which is the single place the group->question cap is
applied. Both entry points call it.

WHY CAPPING IS LEGITIMATE HERE
------------------------------
  * whole GROUPS are kept or dropped, so group-level disjointness survives
    and no group is half-measured
  * the order is fixed before any result is seen, never a post-hoc subset
    chosen after learning which items an arm got right
  * it is applied identically to every arm and condition, so it cannot move
    a comparison in any direction
"""
from __future__ import annotations

import fcntl
import json
import os
from collections import defaultdict
from pathlib import Path


def _category(sample) -> str:
    """The question-type bucket a sample belongs to, if its id encodes one.

    Item ids in this study look like `<domain>/<category>/<id>` (BLINK's
    `Art_Style`, VSI's `obj_appearance_order`, ...). Where that structure is
    absent the whole id is one bucket, which degrades this to the old
    behaviour rather than guessing.
    """
    iid = str(getattr(sample.public, "item_id", ""))
    parts = iid.split("/")
    return parts[1] if len(parts) >= 3 else ""


def cap_by_groups(samples: list, cap: int, label: str = "",
                  freeze_path=None) -> list:
    """Keep whole groups until `cap` questions, SPREAD ACROSS CATEGORIES.

    Taking a sorted PREFIX was the obvious implementation and it silently
    deleted whole question types: item ids begin with the category, so
    sorting clusters them, and VSI's cap of 1367 -> 241 dropped every
    `obj_appearance_order` question. The test set then measured a different
    task mixture than the benchmark defines, and any domain-level number
    computed from it is not comparable to published results.

    It also skewed partial runs: BLINK-hard's first 44 completed items were
    all `Art_Style` (0.850) and `Functional_Correspondence` (0.750), so its
    baseline read 0.818 and fell to 0.567 once `IQ_Test` (0.333) and
    `Multi-view_Reasoning` (0.308) arrived. Nothing was wrong with either
    number; they were means over different task mixtures.

    So groups are taken to cover categories. Selection stays deterministic
    and score-blind -- it depends only on ids and group sizes, both fixed
    before any arm runs -- while the kept subset keeps every category the
    full split has.

    `freeze_path` MAKES THE CHOSEN SET PERMANENT, and passing it is
    mandatory for any split a run will resume into. Changing this function
    changes which items it picks; when rounds 3-4 of VSI were resumed after
    the category fix, the gate became a DIFFERENT 100 questions than rounds
    1-2 had used (the ledger grew from 99 to 177 distinct gate items and the
    round-0 baseline read 0.485 instead of 0.418). Scores across those
    rounds were then not comparable, and the resumed rounds had to be
    discarded. A frozen split must survive every future improvement to the
    sampler, so it is written once and read back thereafter.
    """
    # A FROZEN SPLIT WINS OVER THE EARLY EXIT.
    #
    # `len(samples) <= cap` used to return everything before the freeze was
    # consulted. That is wrong whenever the cap grows or the sample count
    # shrinks after freezing: the run would silently widen its own split and
    # compare rounds measured on different question sets -- the exact defect
    # freezing exists to prevent. If a freeze file exists it is authoritative,
    # whatever the cap now says.
    frozen_exists = (freeze_path is not None
                     and Path(freeze_path).exists())
    if not frozen_exists and (not cap or len(samples) <= cap):
        return samples

    # SERIALISE THE DECISION ACROSS ARMS.
    #
    # Atomic writes stop a reader seeing half a file, but not two arms each
    # computing a subset and overwriting the other. `_choose` is
    # deterministic, so today they would agree -- but the whole point of
    # freezing is that a later change to `_choose` must NOT alter a live
    # split, and without a lock the arm that happens to run first after such
    # a change would silently redefine it for the others.
    #
    # A lock file next to the freeze makes the first arm decide and the rest
    # read what it decided.
    if freeze_path is not None:
        lock_path = Path(freeze_path).with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as _lk:
            fcntl.flock(_lk.fileno(), fcntl.LOCK_EX)
            try:
                return _cap_locked(samples, cap, label, freeze_path)
            finally:
                fcntl.flock(_lk.fileno(), fcntl.LOCK_UN)
    return _cap_locked(samples, cap, label, None)


def _cap_locked(samples: list, cap: int, label: str, freeze_path) -> list:
    """The body of `cap_by_groups`, run under the freeze lock."""
    # Replay a previously frozen choice, if one exists.
    if freeze_path is not None:
        fp = Path(freeze_path)
        if fp.exists():
            want = set(json.loads(fp.read_text()))
            kept = [s for s in samples if s.public.item_id in want]
            missing = want - {s.public.item_id for s in kept}
            if missing:
                # This also catches a run_id being reused across
                # `--type-group` values: a frozen gate for the `metric`
                # group names items the `relational` loader never yields,
                # and silently keeping the intersection would compare two
                # different question sets under one run id.
                raise RuntimeError(
                    f"{fp} names {len(missing)} items this loader did not "
                    f"produce, e.g. {sorted(missing)[:3]}; the frozen split "
                    f"and the data no longer agree (reusing a run_id across "
                    f"--type-group values will do this)")
            if label:
                print(f"{label} replayed frozen cap: {len(kept)} questions "
                      f"({len({s.group_key for s in kept})} groups)")
            return kept

    kept = _choose(samples, cap)

    if freeze_path is not None:
        # ATOMIC WRITE. Arms start concurrently and all of them reach this
        # function, so a plain `write_text` lets one arm read the file while
        # another is still writing it: vsi_G_relational/viewcraft died on
        # `Expecting value: line 1 column 1 (char 0)` -- an empty read of a
        # file that is complete on disk now.
        #
        # Writing to a per-process temp file and renaming makes the reader
        # see either the old state or the new one, never a partial one,
        # because rename within a directory is atomic.
        fp = Path(freeze_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(sorted(s.public.item_id for s in kept), indent=1)
        tmp = fp.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(payload)
        os.replace(tmp, fp)
    if label:
        all_cats = {_category(s) for s in samples}
        covered = {_category(s) for s in kept}
        miss = all_cats - covered
        note = f", MISSING {sorted(miss)}" if miss else ""
        print(f"{label} capped {len(samples)} -> {len(kept)} questions "
              f"({len({s.group_key for s in kept})} groups, "
              f"{len(covered)}/{len(all_cats)} categories{note})")
    return kept


def _choose(samples: list, cap: int) -> list:
    """Greedy set-cover on categories, then fill. Deterministic."""
    by_group: dict[str, list] = defaultdict(list)
    for s in samples:
        by_group[s.group_key].append(s)
    for g in by_group.values():
        g.sort(key=lambda x: x.public.item_id)

    cats_of = {g: {_category(s) for s in rows}
               for g, rows in by_group.items()}

    # A group is NOT one category: a VSI scene carries 21 questions spanning
    # 2-10 question types. Bucketing each group by its first sample's
    # category and round-robining over those buckets still missed types --
    # the first attempt at this fix kept only 4 of VSI's 10. So choose at
    # each step the group adding the most MISSING categories, breaking ties
    # by group_key for determinism.
    kept: list = []
    covered: set = set()
    remaining = sorted(by_group)

    while remaining and len(kept) < cap:
        best = max(
            remaining,
            key=lambda g: (len(cats_of[g] - covered), -len(by_group[g])))
        if not (cats_of[best] - covered):
            break
        remaining.remove(best)
        covered |= cats_of[best]
        kept.extend(by_group[best])

    # Fill the rest in deterministic order; whole groups only, so a document
    # is never scored on half its questions.
    for g in remaining:
        if len(kept) >= cap:
            break
        kept.extend(by_group[g])
    return kept
