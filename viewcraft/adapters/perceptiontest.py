"""Perception Test adapter: multiple-choice video QA, frames already on disk.

WHY THIS DOMAIN, AND WHY IT COST NOTHING TO ADD
-----------------------------------------------
Domain choice has been this study's real bottleneck. Two criteria were
tried and one was actively misleading: "view headroom" (does the pipeline
discard pixels?) turned out ANTI-correlated with whether the mechanism
helped, while "wrong items that have more than one source unit" -- the
working surface -- put both positive domains at 34-37 and every inert
domain at 0-10 (ChartQA single-image: 0; BLINK and MP-DocVQA: 10).

So the search for new domains was run on measurable cost, via the HF API
rather than by reading papers. The long-video sets are all out of reach in
the time available (LongVideoBench gated + 162 GB, Video-MME 101 GB, MLVU
430 GB, EgoSchema 106 GB). What made Perception Test reachable is an
accident worth recording: MVBench's video archive ships the SOURCE video
sets of several other benchmarks, and `video/perception/` already holds 792
Perception Test clips. Only the question file was missing, and it is
**0.3 MB**. 4,892 of the 19,140 released validation questions resolve
against those 792 local clips, so this domain is a 0.3 MB download standing
in for a 110 GB one.

WHY IT SUITS THE MECHANISM
--------------------------
  * every question is MULTIPLE CHOICE over released options, so the metric
    is accuracy with no judge, and there is no open-numeric answer to
    trigger the refusal collapse that destroyed VSI's `metric` group
    (100% refusals, 0.507 -> 0.000)
  * questions are about physical events over time -- "is the camera moving
    or static", "what happens after", counterfactuals -- so the answer is
    generally not in any single frame and `select_expr` choosing 8 of 32
    is a real decision
  * the release carries TWO orthogonal labels per question, `area`
    (physics / abstraction / semantics / memory) and `reasoning`
    (descriptive / explanatory / predictive / counterfactual), which is
    exactly the structure needed to test whether requirement groups want
    different sampling. On our local subset: physics 1923, abstraction
    1501, semantics 1266, memory 202; descriptive 4229, explanatory 322,
    predictive 229, counterfactual 112.

METRIC
------
Official Perception Test multiple-choice accuracy: one point for selecting
the released `answer_id`. `match_multiple_choice` resolves a declared letter
first, then exact option text, then unique containment. UNMODIFIED, and the
same scorer every other multiple-choice domain here uses.

ONE DECLARED CHOICE
-------------------
The frame pool is 32 uniformly sampled frames, the same as mvbench, nextgqa
and vsi, and for the same reason: the pool and the per-request image cap are
DIFFERENT quantities, and holding the pool at the 8-image cap would leave
frame selection a single legal answer. `uniform(8)` over 32 reproduces the
naive behaviour exactly, so the control is reproducible to the index.

The released options include an explicit "I don't know" on many questions.
It is kept in release order and never stripped: removing it would change the
benchmark's own chance floor.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "perceptiontest"
DEFAULT_N_FRAMES = 32

# Where the questions and the (already present) videos live.
from ..paths import data_root

DATA = data_root()
QUESTION_GLOB = str(DATA / "lmms-lab_PerceptionTest_Val" / "**" / "*.parquet")
VIDEO_ROOT = DATA / "OpenGVLab_MVBench" / "video" / "perception"


def _index_local_videos() -> dict:
    """video_name -> path, for the clips already on disk."""
    out = {}
    for p in glob.glob(str(VIDEO_ROOT / "**" / "*"), recursive=True):
        if p.endswith((".mp4", ".webm", ".avi", ".mkv")):
            out[os.path.splitext(os.path.basename(p))[0]] = Path(p)
    return out


def load(root: str | Path = DATA, seed: int = 20260911,
         n_frames: int = DEFAULT_N_FRAMES,
         areas: tuple[str, ...] | None = None,
         reasoning: tuple[str, ...] | None = None,
         limit: int | None = None) -> list[Sample]:
    """Load Perception Test validation MC questions with local videos.

    `seed` is accepted for interface uniformity and deliberately UNUSED:
    frame sampling is the deterministic uniform-midpoint scheme, so two
    loads with different seeds are identical, and a split plan is the only
    place randomness belongs.

    `areas` / `reasoning` restrict to a subset of the release's own labels,
    which is how the requirement-group domains are defined without copying
    any data.
    """
    import pyarrow.parquet as pq

    from .mvbench import _sample

    files = sorted(glob.glob(QUESTION_GLOB, recursive=True))
    if not files:
        raise FileNotFoundError(f"{DOMAIN}: no question parquet under "
                                f"{QUESTION_GLOB}")
    rows = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())

    local = _index_local_videos()
    keep_area = set(areas) if areas else None
    keep_reason = set(reasoning) if reasoning else None

    samples: list[Sample] = []
    n_no_video = n_unscoreable = n_filtered = 0

    for r in rows:
        vname = str(r.get("video_name") or "").strip()
        question = str(r.get("question") or "").strip()
        options = [str(o) for o in (r.get("options") or [])]
        area = str(r.get("area") or "")
        reason = str(r.get("reasoning") or "")

        if keep_area and area not in keep_area:
            n_filtered += 1
            continue
        if keep_reason and reason not in keep_reason:
            n_filtered += 1
            continue

        # The gold is an INDEX into the released options. A row whose index
        # is out of range is unscoreable and is dropped rather than guessed.
        try:
            gold = int(r.get("answer_id"))
        except (TypeError, ValueError):
            n_unscoreable += 1
            continue
        if not question or len(options) < 2 or not 0 <= gold < len(options):
            n_unscoreable += 1
            continue

        vpath = local.get(vname)
        if vpath is None:
            # The released set is 19,140 questions over ~2,000 clips; we
            # hold the 792 clips MVBench shipped. A question whose clip is
            # absent is counted and skipped, never faked from another clip.
            n_no_video += 1
            continue

        try:
            paths, seconds, info = _sample(
                "perception", f"{vname}.mp4", vpath, n_frames, None, None)
        except Exception:                                    # noqa: BLE001
            # A clip that will not decode is a property of the release, not
            # a candidate's failure.
            n_no_video += 1
            continue

        qid = r.get("question_id")
        item_id = f"{DOMAIN}/{area or 'unknown'}/{vname}_{qid}"
        public = PublicItem(
            item_id=item_id, domain=DOMAIN, question=question,
            media_ids=tuple(paths),
            options=tuple(options),          # release order, never re-sorted
            meta={
                "video": vname,
                # BOTH released labels. `question_type_name` is the key
                # `type_groups.group_of_sample` reads; carrying `area` and
                # `reasoning` separately keeps the other axis analysable.
                "question_type_name": area,
                "area": area,
                "reasoning": reason,
                "tag": list(r.get("tag") or []),
                # TRUE total frame count of the decoded clip, never the
                # number sampled: a selection expression reads the pool
                # size, an analysis needs the real one.
                "n_frames": int(info.get("n_frames") or 0),
                "n_frames_pooled": len(paths),
                "seconds": list(seconds),
                "fps": info.get("fps"),
                "duration_s": info.get("duration"),
                "sampling_scheme": "uniform_midpoint_v1",
            },
        )
        private = PrivateItem(item_id=item_id, answer=gold,
                              meta={"answer_text": options[gold],
                                    "area": area, "reasoning": reason})
        samples.append(Sample(public=public, private=private,
                              group_key=vname, split=""))
        if limit and len(samples) >= limit:
            break

    print(f"[{DOMAIN}] {len(samples)} questions over "
          f"{len({s.group_key for s in samples})} local clips "
          f"(skipped {n_no_video} with no local video, "
          f"{n_unscoreable} unscoreable"
          + (f", {n_filtered} outside the requested labels" if n_filtered
             else "") + ")")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official Perception Test metric: multiple-choice accuracy.

    No model acts as judge. `match_multiple_choice` resolves a declared
    letter first, then exact option text, then unique containment -- the
    same resolution order every other multiple-choice domain here uses.
    """
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(answer_text, list(public.options),
                                 int(private.answer))
