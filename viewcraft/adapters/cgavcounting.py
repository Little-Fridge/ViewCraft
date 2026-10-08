"""CG-AV-Counting adapter (CG-Bench): counting with CROSS-F FRAME de-duplication
in minute-to-hour-long videos.

Why this domain is in the study: it is the defining home fixture of the
canvas's "mark what you have already counted" usage. Every question asks HOW
MANY of something occurs in a long video -- an event recurring over time
("how many times does the man fall down"), an object appearing across
non-contiguous shots, or an attribute class spanning several instances --
and the released counts are annotated so that the SAME instance seen in two
frames is counted ONCE. A model that cannot carry an enumeration state
across frames double-counts reappearing instances and under-counts ones it
never re-finds; the visual-memory canvas (pin a marker on each counted
target, on the frame where it was found) is exactly the operation the
benchmark's de-duplication semantics rewards. TallyQA/syn_count exercise
counting WITHIN one image; this is the temporal counterpart where the count
only exists across frames.

Released layout (HF repo CG-Bench/CG-AV-Counting, metadata snapshot): one
annotation file `cg-av-counting.json`, a JSON list of rows

    {video, question, answer, category, type, clue, query_interval}

  * `video` -- filename under `cg_videos_720p/` (e.g. "xxxx.mp4"); the
    SOURCE video, minutes to an hour plus.
  * `answer` -- the integer count, identical for all three task modes.
  * `category` -- the counting type: "event" | "object" | "attribute".
    Recorded as meta["count_type"].
  * `clue` -- grounding annotations (the official clue_acc gold):
      event:     [{"start": sec, "end": sec}, ...]
      object:    [{"timestamp": sec, "bbox": [x0,y0,x1,y1]}, ...]
      attribute: [[{timestamp, bbox}, ...], ...]   (a list of CLUSTERS)
  * `query_interval` -- [start_s, end_s]; the ref-mode clip is the source
    video cropped to this window, shipped as
    `ref_videos/<video_stem>_<start:.2f>_<end:.2f>.mp4`.

THREE TASK MODES (from the official VLMEvalKit integration, which the row
schema above is taken from): "long" asks the count over the FULL video,
"ref" over the pre-cut query-interval clip, and "clue" asks for the
grounding itself (timestamps / per-frame boxes) and is scored by the
official WCS/IFA clustering metrics. This adapter emits numeric-count items
for all three modes (mode in meta["task_mode"]); the clue-mode grounding is
carried in PRIVATE meta["clue"] so a later clue_acc-style analysis can score
grounding without touching this scorer.

WHERE THE CLUE LIVES, AND WHY: `clue` is answer-bearing -- for event and
object rows the count is exactly len(clue), so placing it in PublicItem.meta
would hand the executor the answer. It therefore goes to PrivateItem.meta
("clue"), alongside `answer`. PublicItem.meta carries only the neutral
count_type / task_mode / video shape.

FRAME POOL, AND WHAT IS CAPPED
------------------------------
`DEFAULT_N_FRAMES` = 32 uniformly sampled frames, for the same reason
mvbench/nextgqa/vsi give: the pool and psi's per-request image cap are
different quantities, and holding the pool at the cap would leave
`phi.select_expr` with a single legal answer. On minute-to-hour video a
32-frame pool is also the OFFICIAL direct-input protocol of the benchmark:
the reference integration's long/ref modes decode N interior frames at
uniform stride (`int(i * len/(N+1))`) when the model is not video-native, so
a uniform pool is what the published numbers were produced with, not an
adapter invention. Sampling delegates to `vsi.sample_frames` (one ffmpeg
`select` pass, manifest-cached, shared with mvbench/vsi) rather than a
private cv2/PyAV decoder, so all video domains share one decoder, one cache
layout and one uniform-midpoint scheme -- a capped pool is exactly what a
`uniform(32)` expression produces and remains a legitimate control.

VIDEO FILES ARE SELECTIVELY DOWNLOADED
--------------------------------------
The source videos are large (720p, minutes-to-hours each; the full repo is
on the order of 100 GB) and are NOT mirrored up front. Following the mvbench
precedent, a row whose video file is absent is SKIPPED with a printed count
rather than failing the load, so the domain is usable with any subset of the
archive. `video_manifest()` emits the list of video files a given mode set
needs (JSON: mode, video, relpath, present) -- the download plan greps that
manifest, pulls only the chosen files into `cg_videos_720p/` (or
`ref_videos/`), and re-runs the load. `require_video=False` loads the full
annotation with empty media_ids for exactly this planning step.

SPLIT UNIT
----------
`group_key` is the SOURCE VIDEO filename. Several questions -- and up to
three task modes -- share one video; one video in both gate and test would
let a strategy tune on frames it is later measured on.

SCORING: `cgavcounting_exact_count_v1`
--------------------------------------
Numeric exact match, tallyqa-style: the first integer in the reply
(`\\d+`) must equal the gold count. This is the strict half of the official
long/ref metric -- the release additionally reports off-by-one accuracy and
a capped MAE/RMSE, which are aggregate tolerances rather than per-item
scores and can be recomputed offline from the recorded predictions if
needed. The official clue-mode WCS/IFA grounding metrics are NOT this
scorer; the clue gold is stored for that optional analysis.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample, TimeSpan

DOMAIN = "cgavcounting"

from ..paths import project_root

ROOT = project_root()
DATA = ROOT / "data" / "cgavcounting"

# Pool of 32, send budget owned by psi. See "FRAME POOL, AND WHAT IS CAPPED".
DEFAULT_N_FRAMES = 32

MODES = ("long", "ref", "clue")

# Candidate directories for the full-length source videos, most official
# first. The HF repo ships them under cg_videos_720p/; a selective download
# may drop them at the root instead.
_VIDEO_DIR_CANDIDATES = ("cg_videos_720p", "videos", "")
_REF_DIR_CANDIDATES = ("ref_videos", "")


def _find_annotation(root: Path) -> Path:
    """Locate the annotation file. `cg-av-counting.json` is the release name;
    any other top-level *.json / *.jsonl whose rows look like annotation
    rows (question + answer + video) is accepted so a partial snapshot with
    a differently-named dump still loads.
    """
    named = root / "cg-av-counting.json"
    if named.is_file():
        return named
    for p in sorted(root.glob("*.json")) + sorted(root.glob("*.jsonl")):
        rows = _read_rows(p)
        if rows and all(k in rows[0] for k in ("question", "answer", "video")):
            return p
    raise FileNotFoundError(
        f"{DOMAIN}: no annotation json/jsonl under {root} "
        f"(expected cg-av-counting.json)")


def _read_rows(path: Path) -> list[dict]:
    try:
        if path.suffix == ".jsonl":
            rows = [json.loads(l) for l in path.read_text().splitlines()
                    if l.strip()]
        else:
            rows = json.loads(path.read_text())
    except (ValueError, OSError):
        return []
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _resolve_video(root: Path, name: str, candidates: tuple[str, ...]) -> Path | None:
    """Find `name` under one of the candidate dirs, by relative path first,
    then by basename (a selective download may flatten the layout)."""
    name = name.strip()
    if not name:
        return None
    for d in candidates:
        p = (root / d / name) if d else (root / name)
        if p.is_file():
            return p
    base = Path(name).name
    for d in candidates:
        b = (root / d) if d else root
        if not b.is_dir():
            continue
        hit = b / base
        if hit.is_file():
            return hit
    return None


def _ref_clip_name(video: str, query_interval) -> str:
    """ref_videos/<stem>_<start:.2f>_<end:.2f>.mp4 -- the reference
    integration's `get_output_filename`, kept verbatim so a file downloaded
    by the official pipeline resolves here."""
    stem = Path(video).stem
    try:
        start, end = float(query_interval[0]), float(query_interval[1])
    except (TypeError, ValueError, IndexError):
        return ""
    return f"{stem}_{start:.2f}_{end:.2f}.mp4"


def _count_type(row: dict) -> str:
    cat = str(row.get("category") or row.get("type") or "").strip().lower()
    return cat if cat in ("event", "object", "attribute") else (cat or "unknown")


def load(root: str | Path = DATA, seed: int = 20260911,
         n_frames: int = DEFAULT_N_FRAMES,
         modes: tuple[str, ...] = ("long",),
         limit: int | None = None,
         sample_frames_now: bool = True,
         require_video: bool = True) -> list[Sample]:
    """Load CG-AV-Counting as public/private Sample pairs.

    `seed` is accepted for interface uniformity and deliberately UNUSED:
    frame sampling is the deterministic uniform-midpoint scheme, so a split
    plan is the only place randomness belongs.

    `modes` selects which of the official task modes to emit -- "long"
    (count over the full video, the default and the headline protocol),
    "ref" (count over the pre-cut query-interval clip) and "clue" (same
    count question, with the grounding gold carried in private meta for an
    optional clue_acc analysis). One row yields one sample per requested
    mode.

    A row whose video file is absent is skipped with a printed count (the
    mvbench partial-download precedent): the source archive is large and is
    mirrored selectively, so absence is a download-plan state, not an error.
    Pass `require_video=False` to load every annotated row with empty
    media_ids -- the planning step that decides which videos to pull (see
    `video_manifest`).
    """
    root = Path(root)
    ann = _find_annotation(root)
    rows = _read_rows(ann)

    keep_modes = tuple(m for m in modes if m in MODES)
    if not keep_modes:
        raise ValueError(f"{DOMAIN}: modes must be a subset of {MODES}")

    type_dist: Counter = Counter()
    mode_dist: Counter = Counter()
    samples: list[Sample] = []
    n_no_gold = 0
    n_missing_video = 0
    n_undecodable = 0
    videos_seen: set[str] = set()
    videos_present: set[str] = set()

    for i, r in enumerate(rows):
        video = str(r.get("video") or "").strip()
        question = str(r.get("question") or "").strip()
        ctype = _count_type(r)
        type_dist[ctype] += 1

        try:
            gold = int(float(str(r.get("answer")).strip()))
        except (TypeError, ValueError):
            n_no_gold += 1
            continue
        if not video or not question:
            n_no_gold += 1
            continue

        clue = r.get("clue")
        qi = r.get("query_interval")
        try:
            qi = [float(qi[0]), float(qi[1])] if qi is not None else None
        except (TypeError, ValueError, IndexError):
            qi = None

        for mode in keep_modes:
            if mode == "ref":
                rel = _ref_clip_name(video, qi)
                if not rel:
                    n_no_gold += 1
                    continue
                vpath = _resolve_video(root, rel, _REF_DIR_CANDIDATES)
                media_rel = f"ref_videos/{rel}"
            else:
                vpath = _resolve_video(root, video, _VIDEO_DIR_CANDIDATES)
                media_rel = f"cg_videos_720p/{video}"

            videos_seen.add(media_rel)
            if vpath is None:
                n_missing_video += 1
                if require_video:
                    # Partial download: counted, skipped, never fatal.
                    continue
                paths, seconds, info = (), (), {}
            else:
                videos_present.add(media_rel)
                if sample_frames_now:
                    from .vsi import sample_frames
                    scene = Path(media_rel).stem.replace("/", "__")
                    try:
                        paths, seconds, info = sample_frames(
                            DOMAIN, scene, vpath, n_frames)
                    except Exception:                         # noqa: BLE001
                        # A clip that will not decode is a property of the
                        # release, not a candidate's failure.
                        n_undecodable += 1
                        if require_video:
                            continue
                        paths, seconds, info = (), (), {}
                else:
                    paths, seconds, info = (), (), {}

            mode_dist[mode] += 1
            item_id = f"{DOMAIN}/{mode}/{Path(video).stem}_{i}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN,
                question=question + " Answer with a number.",
                media_ids=tuple(paths),
                meta={
                    "task_mode": mode,
                    "count_type": ctype,
                    "video": video,
                    # TRUE shape of the decoded source, never the number
                    # sampled: a selection expression reads the pool size,
                    # an analysis needs the real one.
                    "n_frames": int(info.get("n_frames") or 0),
                    "n_frames_pooled": len(paths),
                    "seconds": list(seconds),
                    "fps": info.get("fps"),
                    "duration_s": info.get("duration"),
                    "clip_start_s": qi[0] if (mode == "ref" and qi) else None,
                    "clip_end_s": qi[1] if (mode == "ref" and qi) else None,
                    "sampling_scheme": "uniform_midpoint_v1",
                },
            )
            # Official temporal evidence exists only for clue-mode EVENT
            # rows (start/end per occurrence); object/attribute clues are
            # per-frame boxes in pixel coords of the clue frames, which are
            # not normalized Regions of the pooled frames, so they stay raw
            # in meta rather than being faked into evidence_regions.
            spans = ()
            if mode == "clue" and ctype == "event" and isinstance(clue, list):
                try:
                    spans = tuple(
                        TimeSpan(float(c["start"]), float(c["end"]))
                        for c in clue if isinstance(c, dict))
                except (KeyError, TypeError, ValueError):
                    spans = ()
            private = PrivateItem(
                item_id=item_id,
                answer=gold,
                answer_type="integer",
                evidence_spans=spans,
                evidence_source="official" if spans else "none",
                # Answer-bearing: len(clue) IS the count for event/object
                # rows, so this must never cross into the public item.
                meta={"task_mode": mode, "count_type": ctype,
                      "clue": clue, "raw_type": r.get("type")},
            )
            # Group by the SOURCE VIDEO: several questions -- and up to
            # three modes -- share one video, and one video in both gate
            # and test would let a strategy tune on frames it is later
            # measured on.
            samples.append(Sample(public=public, private=private,
                                  group_key=video))
            if limit and len(samples) >= limit:
                break
        if limit and len(samples) >= limit:
            break

    n_v = len(videos_seen)
    avail = f"{len(videos_present)}/{n_v} ({100.0 * len(videos_present) / n_v:.1f}%)" \
        if n_v else "0/0"
    print(f"[{DOMAIN}] {len(samples)} questions "
          f"(modes {dict(mode_dist)}; count types {dict(type_dist)}); "
          f"over {len({s.group_key for s in samples})} videos; "
          f"video files available {avail}; skipped {n_no_gold} unscoreable rows, "
          f"{n_missing_video} missing videos, {n_undecodable} undecodable")
    return samples


def video_manifest(root: str | Path = DATA,
                   modes: tuple[str, ...] = MODES) -> list[dict]:
    """The video files a mode set needs, for the selective download plan.

    One entry per (mode, video file), each
    {"mode", "video", "relpath", "present"} where `relpath` is the path the
    file must land at relative to the data root (`cg_videos_720p/<name>.mp4`
    or `ref_videos/<clip>.mp4`) and `present` says whether it is already
    there. The download plan pulls the entries with present=false, at any
    sampling of rows it likes, and re-runs `load`.
    """
    root = Path(root)
    rows = _read_rows(_find_annotation(root))
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        video = str(r.get("video") or "").strip()
        if not video:
            continue
        try:
            int(float(str(r.get("answer")).strip()))
        except (TypeError, ValueError):
            continue                       # unscoreable row needs no file
        qi = r.get("query_interval")
        for mode in modes:
            if mode == "ref":
                rel = _ref_clip_name(video, qi)
                if not rel:
                    continue
                relpath = f"ref_videos/{rel}"
                present = _resolve_video(root, rel, _REF_DIR_CANDIDATES) is not None
            else:
                relpath = f"cg_videos_720p/{video}"
                present = _resolve_video(root, video, _VIDEO_DIR_CANDIDATES) is not None
            key = (mode, relpath)
            if key in seen:
                continue
            seen.add(key)
            out.append({"mode": mode, "video": video,
                        "relpath": relpath, "present": present})
    return out


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Numeric exact match, tallyqa-style: the first integer in the reply
    must equal the gold count. No model acts as judge."""
    import re
    m = re.search(r"(\d+)", (answer_text or "").strip())
    return 1.0 if m and int(m.group(1)) == int(private.answer) else 0.0
