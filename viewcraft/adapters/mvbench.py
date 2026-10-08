"""MVBench adapter (D15): deciding WHICH frames of a clip to look at.

Why this domain is in the study: it is the video counterpart to the document
domains, and the mechanism's selection coordinate needs a temporal source as
well as a paginated one. Every question here is about something that happens
OVER TIME -- an action's direction, the order two things occurred in, whether
a state changed -- so the answer is not recoverable from any single frame,
and `phi.select_expr` choosing 8 frames out of a 32-frame pool is a real
decision with a real cost when it is made badly.

HONEST LIMITATION, stated up front because it bears on how much weight this
domain can carry: MVBench clips are SHORT (typically 5-35 s), so the frame
pool is an adapter choice rather than a property of an hour-long video. This
is a weaker multi-unit source than a genuine long-video benchmark. It is
here because the long-video sets could not be landed in the available time
-- LongVideoBench is gated (gated=auto, 401 without a token) and 162 GB,
Video-MME is 101 GB, MLVU 430 GB, EgoSchema 106 GB -- and MVBench at 17 GB
is the largest-coverage video benchmark that downloads cleanly. The decision
and its reason are recorded in scripts/download_wave4.py.

What it still buys, and why it is not just a worse VSI-Bench: VSI questions
are SPATIAL (how far, how big, what room) and are answered from a
walkthrough where most frames show the same static scene, so a mediocre
frame choice is often survivable. MVBench questions are TEMPORAL, and the
20 sub-tasks are explicitly built so that a model which samples the wrong
instants cannot recover -- "moving_direction" and "action_sequence" are
unanswerable from frames that miss the transition. Frame choice is
load-bearing in a way it is not on a static scene.

Released layout: `json/<task>.json`, one file per sub-task, 200 questions
each, 4,000 total. A row is

    {video, question, candidates, answer}

plus, on some tasks, `start`/`end` (a time window within a longer source
video) and, on `episodic_reasoning`, `subtitle`/`fps`/`show_name`. `answer`
is the literal TEXT of the correct candidate, not a letter or an index, so
the gold index is recovered by matching the answer string against
`candidates` -- and an item whose answer does not appear in its own
candidate list is dropped rather than guessed at.

The videos live in SEPARATE per-source archives (`video/star.zip`,
`video/clevrer.zip`, ...) whose names do not match the task names, so each
task is mapped to its source directory by locating its `video` paths on
disk. A task whose archive is absent is SKIPPED with a printed count rather
than failing the load, so the domain is usable with a subset of the
archives.

FRAME POOL, AND WHAT IS CAPPED
------------------------------
`PublicItem.media_ids` carries the sampled frame PNGs and `meta` carries the
true clip shape: `n_frames` is the TRUE total frame count of the decoded
clip (never the number sampled), with `n_frames_pooled`, `frame_indices`
(into the source) and `seconds` parallel to media_ids.

ONE CAP, DECLARED: the pool is `DEFAULT_N_FRAMES` = 32 uniformly sampled
frames, decided for the same reason the registry gives for nextgqa and vsi
-- the pool and the per-request image cap are DIFFERENT quantities, and
holding the pool at the 8-image cap is what would leave frame selection with
a single legal answer. 32 gives `select_expr` a real choice; psi's 8-image
cap then decides how many of them reach one request. Sampling is
`vsi.uniform_frame_indices` (midpoints of 32 equal bins), reused rather than
reimplemented so a capped pool is exactly what a `uniform(32)` expression
would produce and remains a legitimate control.

On tasks that ship `start`/`end`, the clip is the SOURCE video and the
window is a property of the question. The window is recorded in meta as
`clip_start_s` / `clip_end_s` and frames are sampled ACROSS THE WINDOW, not
across the whole source file: sampling the whole file would spend most of a
32-frame budget outside the interval the question is about, which is a
sampling bug rather than a hard question.

SPLIT UNIT
----------
`group_key` is the SOURCE VIDEO (`<task_source>/<video path>`), not the
question. Several questions can be drawn from one video, and letting one
video appear in both gate and test would let a strategy tune on frames it is
later measured on. The task name is part of the key only where two sources
could otherwise collide on a bare filename.

SCORING: `mvbench_accuracy_v1`
------------------------------
Plain multiple-choice accuracy over the released candidates
(`scorers.match_multiple_choice`), which is the official MVBench metric --
the benchmark is 100% multiple choice and its leaderboard reports mean
accuracy over the 20 tasks. The metric is UNMODIFIED.

ONE DECLARED DEVIATION: the official evaluation prompts the model with a
fixed template that appends the lettered options to the question and asks
for "Best option: (", then reads the next character. ViewCraft owns the
prompt (that is the theta coordinate it evolves), so that template is not
used; options are presented by the runtime's own answer-style machinery and
the reply is resolved by `match_multiple_choice`, which takes a declared
letter first, then exact option text, then unique containment. This is
strictly more permissive on FORMAT than reading one character, and it is
applied identically to every arm and condition. No LLM judges anything.

Candidate order is the RELEASE's order and is never re-sorted: the gold is
spread across positions in the release, and shuffling would only add a
mapping layer that could silently invert.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "mvbench"

from ..paths import project_root

ROOT = project_root()
DATA = ROOT / "data" / "OpenGVLab_MVBench"
MEDIA_ROOT = ROOT / "data" / "_media" / DOMAIN

# Pool of 32, send budget 8. See "FRAME POOL, AND WHAT IS CAPPED": the pool
# and the per-request cap are different quantities, and holding the pool at
# the cap leaves frame selection one legal answer.
DEFAULT_N_FRAMES = 32

# The per-source archives disagree on nesting depth, so every directory down
# to this depth is indexed as a candidate base. 3 covers the deepest observed
# layout (Moments_in_Time_Raw/videos/validation/<class>/).
MAX_INDEX_DEPTH = 3
# tvqa ships 37k extracted JPEG frames instead of clips; capping the index
# keeps one pathological archive from dominating the scan.
MAX_INDEX_FILES = 20000


def _index_video_dirs(video_root: Path) -> dict[str, set[str]]:
    """Map each candidate BASE DIRECTORY to the relative paths under it.

    The task json names a video by a path relative to its own source
    archive ("left/4504_frame52.mp4", "video_validation/video_12485.mp4"),
    and the archive name is not derivable from the task name, so the mapping
    is discovered from disk rather than hardcoded.

    The archives do not agree on nesting depth -- `ssv2_video/` holds its
    clips directly, while `clevrer/` wraps them in `video_validation/`,
    `star/` in `Charades_v1_480/` and `Moments_in_Time_Raw/` in
    `videos/validation/`. Indexing only the top-level directories therefore
    matched 2 of 20 tasks. Every directory down to `MAX_INDEX_DEPTH` is
    indexed as a candidate base instead, so a task resolves against whichever
    depth its paths were written relative to.
    """
    idx: dict[str, set[str]] = {}
    if not video_root.is_dir():
        return idx

    bases: list[Path] = []
    for d in sorted(p for p in video_root.iterdir() if p.is_dir()):
        bases.append(d)
        for sub in d.rglob("*"):
            if not sub.is_dir():
                continue
            if len(sub.relative_to(d).parts) <= MAX_INDEX_DEPTH:
                bases.append(sub)

    for b in bases:
        rels = set()
        for f in b.rglob("*"):
            if f.is_file():
                rels.add(str(f.relative_to(b)))
            if len(rels) > MAX_INDEX_FILES:
                # tvqa ships 37k extracted JPEG frames rather than clips;
                # indexing every one of them buys nothing and costs minutes.
                break
        if rels:
            idx[str(b.relative_to(video_root))] = rels
    return idx


def load(root: str | Path = DATA, seed: int = 20260911,
         n_frames: int = DEFAULT_N_FRAMES,
         tasks: tuple[str, ...] | None = None,
         limit: int | None = None,
         sample_frames_now: bool = True) -> list[Sample]:
    """Load MVBench as public/private Sample pairs.

    `seed` is accepted for interface uniformity and is deliberately UNUSED:
    frame sampling is the deterministic uniform-midpoint scheme, so two
    loads with different seeds are identical and a split plan is the only
    place randomness belongs.

    `tasks` restricts to a subset of the 20 sub-task names; the default
    keeps all of them, which is the official headline configuration. Tasks
    whose video archive is not on disk are skipped with a printed count, so
    the domain loads with a partial download instead of failing.
    """
    from .vsi import probe_video, sample_frames, uniform_frame_indices  # noqa: F401

    root = Path(root)
    json_dir = root / "json"
    video_root = root / "video"
    if not json_dir.is_dir():
        raise FileNotFoundError(f"{DOMAIN}: no json/ under {root}")

    keep = set(tasks) if tasks else None
    disk = _index_video_dirs(video_root)

    samples: list[Sample] = []
    missing_tasks: list[str] = []
    n_no_gold = 0
    n_missing_video = 0

    for jf in sorted(json_dir.glob("*.json")):
        task = jf.stem
        if keep and task not in keep:
            continue
        try:
            rows = json.loads(jf.read_text())
        except (ValueError, OSError):
            continue
        if not rows:
            continue

        # Which source archive holds this task's videos? The one that
        # contains the most of its referenced paths.
        best_dir, best_hits = None, 0
        for dname, rels in disk.items():
            hits = sum(1 for r in rows if str(r.get("video") or "") in rels)
            if hits > best_hits:
                best_dir, best_hits = dname, hits
        if not best_dir:
            # Archive not downloaded. Skipped with a count rather than
            # raising: the domain stays usable on a partial download.
            missing_tasks.append(task)
            continue

        src_root = video_root / best_dir
        for i, r in enumerate(rows):
            vid = str(r.get("video") or "").strip()
            question = str(r.get("question") or "").strip()
            answer = str(r.get("answer") or "").strip()
            cands = [str(c) for c in (r.get("candidates") or [])]
            if not vid or not question or not answer or len(cands) < 2:
                n_no_gold += 1
                continue
            # `answer` is the TEXT of the right candidate; the gold index is
            # recovered by matching it. An answer absent from its own
            # candidate list is unscoreable and is dropped, not guessed.
            if answer not in cands:
                n_no_gold += 1
                continue
            gold_idx = cands.index(answer)

            vpath = src_root / vid
            if not vpath.is_file():
                n_missing_video += 1
                continue

            # Tasks that ship start/end ask about a WINDOW of a longer
            # source video. Frames are sampled across that window; sampling
            # the whole file would spend most of the budget outside the
            # interval the question is about.
            start = r.get("start")
            end = r.get("end")
            try:
                start = float(start) if start is not None else None
                end = float(end) if end is not None else None
            except (TypeError, ValueError):
                start = end = None

            scene = f"{best_dir}/{vid}"
            if sample_frames_now:
                try:
                    paths, seconds, info = _sample(
                        best_dir, vid, vpath, n_frames, start, end)
                except Exception:                             # noqa: BLE001
                    # A clip that will not decode is a property of the
                    # release, not a candidate's failure: counted and
                    # skipped rather than crashing the load.
                    n_missing_video += 1
                    continue
            else:
                paths, seconds, info = (), (), {"n_frames": 0, "fps": 0.0,
                                               "duration": 0.0}

            item_id = f"{DOMAIN}/{task}/{Path(vid).stem}_{i}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN, question=question,
                media_ids=tuple(paths),
                # Release order, never re-sorted.
                options=tuple(cands),
                meta={
                    "task": task,
                    "source": best_dir,
                    "video": vid,
                    # TRUE total frame count of the decoded clip, never the
                    # number sampled. A selection expression reads the pool
                    # size, but an analysis needs the real one.
                    "n_frames": int(info.get("n_frames") or 0),
                    "n_frames_pooled": len(paths),
                    "seconds": list(seconds),
                    "fps": info.get("fps"),
                    "duration_s": info.get("duration"),
                    "clip_start_s": start,
                    "clip_end_s": end,
                    "sampling_scheme": "uniform_midpoint_v1",
                },
            )
            private = PrivateItem(
                item_id=item_id,
                # The INDEX, so the scorer never has to re-derive it.
                answer=gold_idx,
                answer_type="option_index",
                # MVBench releases no temporal span for the evidence on most
                # tasks, so the oracle / probe conditions cannot run here.
                evidence_source="none",
                meta={"task": task, "source": best_dir,
                      "answer_text": answer, "n_options": len(cands)},
            )
            # Group by the SOURCE VIDEO: several questions can share one
            # video, and one video in both gate and test would let a
            # strategy tune on frames it is later measured on.
            samples.append(Sample(public=public, private=private,
                                  group_key=scene))
            if limit and len(samples) >= limit:
                break
        if limit and len(samples) >= limit:
            break

    print(f"[{DOMAIN}] {len(samples)} questions over "
          f"{len({s.group_key for s in samples})} videos; "
          f"skipped {len(missing_tasks)} task(s) with no video archive "
          f"({', '.join(missing_tasks[:6])}{'...' if len(missing_tasks) > 6 else ''}), "
          f"{n_no_gold} unscoreable rows, {n_missing_video} missing/undecodable clips")
    return samples


def _sample(source: str, vid: str, vpath: Path, n_frames: int,
            start: float | None, end: float | None):
    """Sample `n_frames` frames, honouring a start/end window when present.

    Delegates to `vsi.sample_frames` for the whole-file case so the two
    video domains share one decoder, one cache layout and one sampling
    scheme. The windowed case is handled here because vsi has no windows.
    """
    from .vsi import probe_video, sample_frames, uniform_frame_indices

    safe = vid.replace("/", "__")
    if start is None or end is None or end <= start:
        return sample_frames(f"mvbench_{source}", safe, vpath, n_frames)

    # Windowed: map the time interval to frame indices, then reuse the same
    # uniform-midpoint scheme over that sub-range.
    import os
    import shutil
    import subprocess

    info = probe_video(vpath)
    fps = float(info.get("fps") or 0.0) or 1.0
    total = int(info.get("n_frames") or 0)
    i0 = max(0, int(round(start * fps)))
    i1 = min(total - 1, int(round(end * fps))) if total else int(round(end * fps))
    if i1 <= i0:
        return sample_frames(f"mvbench_{source}", safe, vpath, n_frames)

    span = i1 - i0 + 1
    rel = uniform_frame_indices(span, min(n_frames, span))
    indices = [i0 + k for k in rel]
    seconds = tuple(round(i / fps, 4) for i in indices)

    d = MEDIA_ROOT / source / safe / f"w{i0}_{i1}_n{len(indices)}"
    manifest = d / "manifest.json"
    paths = tuple(str(d / f"frame_{k:04d}.png") for k in range(len(indices)))
    if manifest.exists() and all(Path(p).exists() for p in paths):
        return paths, seconds, info

    d.mkdir(parents=True, exist_ok=True)
    stage = d / f".stage.{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    try:
        from .vsi import FFMPEG
        sel = "+".join(rf"eq(n\,{i})" for i in indices)
        subprocess.run(
            [FFMPEG, "-v", "error", "-nostdin", "-i", str(vpath),
             "-vf", f"select='{sel}'", "-vsync", "0",
             str(stage / "frame_%04d.png")],
            check=True, capture_output=True)
        produced = sorted(stage.glob("frame_*.png"))
        if len(produced) != len(indices):
            raise RuntimeError(
                f"{DOMAIN}: {source}/{vid} expected {len(indices)} frames, "
                f"ffmpeg wrote {len(produced)}")
        for k, src in enumerate(produced):
            src.replace(d / f"frame_{k:04d}.png")
        tmp = manifest.with_name(f".manifest.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({
            "source": source, "video": vid, "indices": indices,
            "seconds": list(seconds), "window": [i0, i1],
            "scheme": "uniform_midpoint_v1"}, indent=2))
        tmp.replace(manifest)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return paths, seconds, info


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official MVBench metric: plain multiple-choice accuracy.

    `match_multiple_choice` resolves a declared letter first, then exact
    option text, then unique containment -- see the DEVIATION paragraph in
    the module docstring. No model acts as judge.
    """
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(answer_text, list(public.options),
                                 int(private.answer))
