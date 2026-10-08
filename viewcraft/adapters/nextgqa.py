"""NExT-GQA adapter (D4): visually-grounded multiple-choice video QA.

Released layout (data/nextgqa/annotations/):
  * `test.csv` / `val.csv` -- QA rows. One row per (video_id, qid) with the
    question, the five options in columns a0..a4, `answer` as the correct
    option TEXT (not an index), a `type` tag (CW/CH/TN/TC/TP) and the video's
    `frame_count`/`width`/`height`. These are a SUBSET of NExT-QA: the authors
    dropped questions answerable from global video content and the whole
    descriptive group, because neither can be temporally grounded.
  * `gsub_test.json` -- temporal grounding. Keyed by video_id, holding
    `duration` (seconds, rounded to an int), `fps`, and
    `location[qid] = [[start, end], ...]`: one or more disjoint evidence
    segments per question.
  * `frame2time_*.json` / `upbd_*.json` -- 32 sampled video timestamps per
    video. Both are the authors' own uniform sampling grid used by their
    baselines' upper-bound computation. They are NOT needed to interpret
    gsub and are deliberately unused here (we sample our own grid); they are
    documented so a reader does not mistake them for a required join.
  * `map_vid_vidorID.json` -- maps video_id to the `<dir>/<video_id>` VidOR
    path. Unused: our archive stores videos flat as `NExTVideo/<video_id>.mp4`.

TIME UNIT (verified empirically, not assumed)
---------------------------------------------
`gsub.location` spans are in SECONDS. Three independent checks, all of which
a frame-index reading would fail:
  1. Over the full test split, all 6,600 spans satisfy `end <= duration`,
     where `duration` is the per-video field in gsub. Frame indices would run
     to ~30x that bound.
  2. gsub's `duration` and `fps` agree with ffprobe on the real mp4s (e.g.
     video 10109006686: gsub 35s / 29.97fps vs probe 34.74s / 29.97fps).
     Spans therefore live on the same scale as a quantity ffprobe confirms is
     seconds.
  3. Span endpoints carry one decimal (1.2, 5.8, 29.7) and are non-integral,
     which is the shape of a seconds annotation, not of a frame counter.
So NO frame->second conversion is applied to gsub. `frame2time_*.json` would
be the tool for that conversion if the spans had been frame indices; the
verification above is what rules that out.

Because gsub's `duration` is rounded to a whole second while the true
duration is not, ~7% of released spans end up to ~0.08s past the real end of
the video (e.g. video 10149430394: span (15.0, 22.6) against a true 22.53s).
Spans are therefore CLAMPED to the ffprobe duration, which is the timeline
our sampled frame timestamps live on. The clamp is sub-frame at 29.97fps and
cannot move a temporal IoU appreciably, but leaving it out would put evidence
outside the video the executor is shown.

Option order: options are presented in RELEASE order a0..a4, never re-sorted
and never shuffled. This differs from our V*Bench adapter, which must shuffle
because V*Bench encodes the answer positionally (options[0] is always
correct). NExT-GQA has no such leak -- the gold index is near-uniform over
the five slots (1113/1105/1133/1136/1066 on test) -- so release order is
already an unbiased presentation, and preserving it keeps our per-item option
indices comparable with the official evaluator's.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample, TimeSpan

DOMAIN = "nextgqa"

from ..paths import project_root

ROOT = project_root()
DATA = ROOT / "data" / "nextgqa"
ANNOTATIONS = DATA / "annotations"

# Source videos: the 6.1 GB archive is never unpacked wholesale; only the
# members a loaded row set needs are extracted here.
VIDEO_ZIP = ROOT / "data" / "nextqa_videos" / "videos.zip"
VIDEO_ROOT = ROOT / "data" / "nextqa_videos" / "extracted"
ZIP_PREFIX = "NExTVideo"

# Emitted media (sampled frames) live under the shared _media tree.
MEDIA_ROOT = ROOT / "data" / "_media" / DOMAIN

# Not on PATH inside the craft env; absolute paths are required.
FFMPEG = os.environ.get("VIEWCRAFT_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("VIEWCRAFT_FFPROBE") or shutil.which("ffprobe") or "ffprobe"

DEFAULT_N_FRAMES = 8
SAMPLING_SCHEME = "uniform_time_midpoint_v1"

# NExT-QA's question-type tags. Descriptive ("D*") types are absent by
# construction: NExT-GQA excludes them as ungroundable.
QUESTION_TYPES = {
    "CW": "causal_why",
    "CH": "causal_how",
    "TN": "temporal_next",
    "TC": "temporal_current",
    "TP": "temporal_previous",
}

N_OPTION_COLUMNS = 5


# ---------------------------------------------------------------------------
# annotations


def read_rows(split: str = "test") -> list[dict]:
    """QA rows for a split, sorted by (video_id, qid) for a stable order."""
    path = ANNOTATIONS / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{DOMAIN}: missing {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: (str(r["video_id"]), int(r["qid"])))
    return rows


def read_gsub(split: str = "test") -> dict:
    """Temporal grounding annotations for a split."""
    path = ANNOTATIONS / f"gsub_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{DOMAIN}: missing {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# source videos: extract only what the loaded rows need


def ensure_videos(video_ids: set[str]) -> dict[str, str]:
    """Extract the needed mp4s from videos.zip into data/nextqa_videos/extracted/.

    The archive holds 1,570 videos (6.1 GB) while the test split touches 990,
    and a smoke run touches a handful, so members are extracted individually
    instead of unpacking the archive. Idempotent and parallel-safe: a member
    already present at the archive's recorded size is skipped, and each write
    goes to a pid-scoped temp name that is renamed into place, so a concurrent
    reader never opens a half-written mp4. A short read is raised rather than
    published, because a truncated mp4 would probe as a shorter video and
    silently shift every frame timestamp.
    """
    if not VIDEO_ZIP.exists():
        raise FileNotFoundError(f"{DOMAIN}: missing archive {VIDEO_ZIP}")

    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    out = {str(v): str(VIDEO_ROOT / f"{v}.mp4") for v in video_ids}
    need = [v for v, p in out.items() if not Path(p).exists()]
    if not need:
        return out

    with zipfile.ZipFile(VIDEO_ZIP) as zf:
        names = set(zf.namelist())
        for vid in sorted(need):
            member = f"{ZIP_PREFIX}/{vid}.mp4"
            if member not in names:
                raise FileNotFoundError(
                    f"{DOMAIN}: {member} not in {VIDEO_ZIP.name}")
            dest = Path(out[vid])
            size = zf.getinfo(member).file_size
            if dest.exists() and dest.stat().st_size == size:
                continue
            tmp = dest.with_name(f".{vid}.{os.getpid()}.tmp")
            with zf.open(member) as src, open(tmp, "wb") as fh:
                shutil.copyfileobj(src, fh, 1 << 20)
            if tmp.stat().st_size != size:
                tmp.unlink(missing_ok=True)
                raise IOError(f"{DOMAIN}: short read extracting {member}")
            tmp.replace(dest)
    return out


# ---------------------------------------------------------------------------
# probing


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr.strip()[:400]}")
    return proc.stdout


def probe_video(path: str | Path) -> dict:
    """Duration, fps, frame count and size of one video, via ffprobe.

    Cached next to the mp4 as `<video>.probe.json`: the loader probes once per
    video even though many questions share it. `nb_frames` is trusted when the
    container reports it and recomputed from duration x fps otherwise, since a
    missing count would collapse the sampler to a single frame.
    """
    path = Path(path)
    cached = path.with_suffix(".probe.json")
    if cached.exists():
        try:
            return json.loads(cached.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    raw = _run([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ])
    info = json.loads(raw)
    if not info.get("streams"):
        raise RuntimeError(f"{DOMAIN}: no video stream in {path}")
    st = info["streams"][0]
    num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
    fps = (float(num) / float(den)) if float(den or 0) else 0.0
    duration = float(info.get("format", {}).get("duration") or 0.0)

    nb = st.get("nb_frames")
    n_frames = int(nb) if nb not in (None, "", "N/A") else 0
    if n_frames <= 0:
        n_frames = int(round(duration * fps)) if fps > 0 else 0
    if n_frames <= 0:
        raise RuntimeError(f"{DOMAIN}: cannot determine frame count for {path}")
    if duration <= 0:
        raise RuntimeError(f"{DOMAIN}: non-positive duration for {path}")

    out = {
        "duration": duration, "fps": fps, "n_frames": n_frames,
        "width": int(st.get("width") or 0), "height": int(st.get("height") or 0),
        "codec": st.get("codec_name") or "",
    }
    tmp = cached.with_name(f".{cached.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(cached)
    return out


# ---------------------------------------------------------------------------
# frame sampling


def uniform_frame_indices(n_frames: int, n_sample: int) -> list[int]:
    """`n_sample` frame indices at the midpoints of equal bins over the video.

    index_i = floor((i + 0.5) * n_frames / n_sample), clipped to the last
    frame. Midpoints rather than a linspace including both endpoints, for two
    reasons specific to this dataset:

      * NExT-QA clips are short (13-180s) and often open or close on a fade,
        a title card or camera motion; spending two of only eight frames on
        the exact first and last frame wastes a quarter of the budget on the
        least informative content.
      * every sampled timestamp then falls strictly INSIDE (0, duration), so
        a frame time can never sit outside the timeline that the grounding
        spans are annotated on.

    The scheme is a pure function of (n_frames, n_sample), so it is
    deterministic and reproducible with no seed to record.
    """
    if n_frames <= 0:
        raise ValueError(f"{DOMAIN}: n_frames={n_frames}")
    if n_sample <= 0:
        raise ValueError(f"{DOMAIN}: n_sample={n_sample}")
    if n_sample >= n_frames:
        return list(range(n_frames))
    idx = [int((i + 0.5) * n_frames / n_sample) for i in range(n_sample)]
    return [min(i, n_frames - 1) for i in idx]


def _frame_dir(video_id: str, n_sample: int) -> Path:
    return MEDIA_ROOT / str(video_id) / f"n{n_sample}"


def sample_frames(video_id: str, video_path: str | Path,
                  n_sample: int = DEFAULT_N_FRAMES,
                  ) -> tuple[tuple[str, ...], tuple[float, ...], dict]:
    """Uniformly sample `n_sample` frames to disk. Returns (paths, seconds, info).

    Frames are written as PNG FILES under
    data/_media/nextgqa/<video_id>/n<N>/ because the view renderer
    (views/render.py) re-reads the source path whenever it makes a crop -- an
    in-memory array or an open video handle could not be zoomed into that way.
    PNG, not JPEG, so a crop is not compounding a second lossy re-encode on
    top of the mp4's own compression.

    `decord` is not installed in this environment, so decoding goes through
    ffmpeg as a subprocess. All frames come from ONE ffmpeg pass driven by a
    `select` filter over the target indices; seeking per frame would re-read
    the file N times.

    CACHED, idempotent and parallel-safe. `manifest.json` is written last and
    acts as the completion marker, so a second call returns identical paths
    without re-decoding, and a run interrupted mid-decode is not mistaken for
    a finished one. Decoding happens in a pid-scoped staging directory whose
    frames are moved into their final names only once the expected count is
    confirmed, so two concurrent workers on the same video cannot interleave
    partial output.
    """
    info = probe_video(video_path)
    indices = uniform_frame_indices(info["n_frames"], n_sample)
    fps = info["fps"] or 1.0
    duration = info["duration"]
    # The timestamp of the frame actually decoded, clipped to the probed
    # duration so a timestamp can never exceed the video the model is shown.
    seconds = tuple(round(min(i / fps, duration), 4) for i in indices)

    d = _frame_dir(video_id, n_sample)
    manifest = d / "manifest.json"
    paths = tuple(str(d / f"frame_{k:04d}.png") for k in range(len(indices)))

    if manifest.exists():
        try:
            man = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            man = None
        if (man and man.get("indices") == indices
                and man.get("scheme") == SAMPLING_SCHEME
                and all(Path(p).exists() for p in paths)):
            return paths, tuple(man["seconds"]), info

    d.mkdir(parents=True, exist_ok=True)
    stage = d / f".stage.{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    try:
        sel = "+".join(rf"eq(n\,{i})" for i in indices)
        _run([
            FFMPEG, "-v", "error", "-nostdin", "-i", str(video_path),
            "-vf", f"select='{sel}'", "-vsync", "0",
            str(stage / "frame_%04d.png"),
        ])
        produced = sorted(stage.glob("frame_*.png"))
        if not produced:
            raise RuntimeError(
                f"{DOMAIN}: {video_id} yielded no frames at all")
        if len(produced) != len(indices):
            # ffmpeg writes one frame short when the requested index lands
            # on or past the last decodable frame (a truncated or
            # off-by-one-length video). That is a property of THAT VIDEO,
            # not a fault in the request: raising here aborted the entire
            # domain load, so one bad video in 990 stopped NExT-GQA from
            # running at all.
            #
            # Keep what was decoded and trim `indices`/`seconds` to match,
            # so the frame files, their timestamps and the recorded count
            # stay consistent. The shortfall is printed rather than
            # silently absorbed.
            print(f"[{DOMAIN}] {video_id}: asked {len(indices)} frames, "
                  f"ffmpeg wrote {len(produced)}; using {len(produced)}")
            indices = indices[: len(produced)]
            # `seconds` and `paths` were derived from the pre-trim indices
            # above, so they must be recomputed or a frame would be
            # labelled with another frame's timestamp -- which is exactly
            # the class of defect that made C_ground judge pixels nobody
            # saw. Rebuild both from the trimmed list.
            seconds = tuple(round(min(i / fps, duration), 4)
                            for i in indices)
            paths = tuple(str(d / f"frame_{k:04d}.png")
                          for k in range(len(indices)))
        # ffmpeg numbers its output 1..N; republish 0-based so file order
        # matches the order of `indices` and `seconds` (chronological).
        for k, src in enumerate(produced):
            src.replace(d / f"frame_{k:04d}.png")
        tmp = manifest.with_name(f".manifest.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({
            "video_id": str(video_id), "n_sample": n_sample,
            "indices": indices, "seconds": list(seconds),
            "duration": duration, "fps": info["fps"],
            "n_frames": info["n_frames"], "scheme": SAMPLING_SCHEME,
        }, indent=2))
        tmp.replace(manifest)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    return paths, seconds, info


# ---------------------------------------------------------------------------
# grounding spans


def _spans_for(gsub: dict, video_id: str, qid: str,
               duration: float) -> tuple[tuple[TimeSpan, ...], bool]:
    """Gold evidence spans in seconds, clamped to [0, duration].

    Returns (spans, was_clamped). See the module docstring: gsub annotates
    against a duration rounded to a whole second, so a span can end a few
    hundredths past the true end of the video. Clamping keeps evidence on the
    same timeline as our frame timestamps; the adjustment is sub-frame.

    `duration` must be the PUBLISHED duration (`meta["duration_s"]`), not the
    raw ffprobe float. Clamping to full precision and then rounding the
    endpoint to 4 decimals can leave a span a hair beyond a duration published
    at 3 decimals, which trips a legitimate "evidence inside the video" check
    for a purely cosmetic reason.
    """
    entry = gsub.get(str(video_id)) or {}
    raw = (entry.get("location") or {}).get(str(qid))
    if not raw:
        return (), False

    spans: list[TimeSpan] = []
    clamped = False
    for pair in raw:
        start, end = float(pair[0]), float(pair[1])
        cs = min(max(start, 0.0), duration)
        ce = min(max(end, 0.0), duration)
        if (cs, ce) != (start, end):
            clamped = True
        if ce < cs:
            cs, ce = ce, cs
        # Round first, then re-clamp: rounding can push an endpoint back over
        # the boundary we just clamped it to.
        cs = min(max(round(cs, 4), 0.0), duration)
        ce = min(max(round(ce, 4), 0.0), duration)
        spans.append(TimeSpan(start_s=cs, end_s=ce))
    return tuple(spans), clamped


# ---------------------------------------------------------------------------
# loading


def load(root: str | Path = DATA, seed: int = 20260911,
         split: str = "test", n_frames: int = DEFAULT_N_FRAMES,
         limit: int | None = None,
         video_ids: tuple[str, ...] | None = None,
         sample_frames_now: bool = True) -> list[Sample]:
    """Load NExT-GQA questions as public/private Sample pairs.

    Defaults to the TEST split: 5,553 questions over 990 videos, every one of
    which has a gsub grounding entry and an mp4 in the archive.

    `seed` is accepted for interface uniformity with the other adapters and is
    intentionally unused: option order is the dataset's own release order and
    frame sampling is a deterministic function of the video, so this loader has
    no stochastic component to seed.

    `video_ids` / `limit` restrict the row set, and only the videos the
    surviving rows need are extracted and decoded -- that is what lets the
    smoke test materialize a handful of videos instead of 990. Each needed
    video is decoded once even when many questions share it.
    """
    root = Path(root)
    rows = read_rows(split)
    gsub = read_gsub(split)

    if video_ids:
        keep = {str(v) for v in video_ids}
        rows = [r for r in rows if str(r["video_id"]) in keep]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return []

    videos = ensure_videos({str(r["video_id"]) for r in rows})

    # One probe/decode per video, shared by every question on that video.
    frame_cache: dict[str, tuple] = {}

    samples: list[Sample] = []
    for r in rows:
        vid = str(r["video_id"])
        qid = str(r["qid"])
        qtype = str(r["type"])
        vpath = videos[vid]
        item_id = f"{DOMAIN}/{vid}_{qid}"

        if vid not in frame_cache:
            if sample_frames_now:
                frame_cache[vid] = sample_frames(vid, vpath, n_frames)
            else:
                info = probe_video(vpath)
                idx = uniform_frame_indices(info["n_frames"], n_frames)
                fps = info["fps"] or 1.0
                d = _frame_dir(vid, n_frames)
                frame_cache[vid] = (
                    tuple(str(d / f"frame_{k:04d}.png")
                          for k in range(len(idx))),
                    tuple(round(min(i / fps, info["duration"]), 4)
                          for i in idx),
                    info,
                )
        paths, seconds, info = frame_cache[vid]
        # The single published duration. Frame timestamps and evidence spans
        # are both bounded by THIS value, so "evidence lies inside the video"
        # is checkable against the number the consumer actually sees.
        duration = round(info["duration"], 3)

        # Release order a0..a4, preserved. Never re-sorted, never shuffled.
        options = tuple(str(r[f"a{i}"]) for i in range(N_OPTION_COLUMNS)
                        if r.get(f"a{i}") is not None)
        if len(options) < 2:
            raise ValueError(f"{item_id}: {len(options)} option(s)")

        # The release gives the correct option as TEXT, not an index.
        answer_text = str(r["answer"])
        matches = [i for i, o in enumerate(options) if o == answer_text]
        if not matches:
            raise ValueError(
                f"{item_id}: answer {answer_text!r} is not one of {options}")
        # Duplicate option texts would make the position ambiguous; the first
        # match is the gold slot and the ambiguity is recorded privately.
        answer_position = matches[0]

        spans, clamped = _spans_for(gsub, vid, qid, duration)

        public = PublicItem(
            item_id=item_id,
            domain=DOMAIN,
            question=str(r["question"]),
            media_ids=paths,
            options=options,
            meta={
                "video_id": vid,
                "qid": qid,
                "question_type": qtype,
                "question_type_name": QUESTION_TYPES.get(qtype, qtype),
                "duration_s": duration,
                "frame_times_s": [float(min(s, duration)) for s in seconds],
                "n_frames_sampled": len(paths),
                "video_fps": round(info["fps"], 4),
                "video_n_frames": info["n_frames"],
                "frame_width": info["width"],
                "frame_height": info["height"],
                "sampling_scheme": SAMPLING_SCHEME,
                "n_options": len(options),
                "answer_format": "option_letter",
            },
        )
        private = PrivateItem(
            item_id=item_id,
            answer=answer_text,
            answer_type="multiple_choice",
            evidence_spans=spans,
            evidence_source="official" if spans else "none",
            evidence_verified=bool(spans),
            meta={
                "answer_position": answer_position,
                "answer_letter": chr(ord("A") + answer_position),
                "question_type": qtype,
                "question_type_name": QUESTION_TYPES.get(qtype, qtype),
                "video_id": vid,
                "qid": qid,
                "scorer": "match_multiple_choice",
                "metric": "qa_accuracy",
                "release_options": list(options),
                "gsub_duration_s": (gsub.get(vid) or {}).get("duration"),
                "gsub_fps": (gsub.get(vid) or {}).get("fps"),
                "spans_clamped_to_duration": clamped,
                "duplicate_option_text": len(matches) > 1,
                "n_evidence_spans": len(spans),
            },
        )
        samples.append(Sample(public=public, private=private, split=split,
                              group_key=vid))
    return samples


# ---------------------------------------------------------------------------
# scoring


def score(trajectory_answer: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Plain multiple-choice QA accuracy -- NOT the official Acc@GQA.

    This is a deliberate, declared deviation from the source benchmark:

      * NExT-GQA's headline metric is Acc@GQA, which counts a question correct
        only if the answer is right AND the model's predicted temporal span
        overlaps the gold evidence with IoU >= 0.5. Answering correctly while
        pointing at the wrong moment scores zero there.
      * Our main protocol never asks the executor for a time span, so no
        predicted span exists to compute that IoU from. Reporting Acc@GQA
        would require inventing a span on the model's behalf.
      * We therefore report PLAIN QA ACCURACY and label it as such: the
        registry names this scorer `nextgqa_qa_accuracy_v1`, and the number is
        never presented as Acc@GQA. It is an upper bound on Acc@GQA, since
        every Acc@GQA-correct question is also QA-correct.

    The gold spans are still loaded into `PrivateItem.evidence_spans`, so a
    grounding-aware condition that does elicit a span can compute the official
    joint metric with `scorers.temporal_iou` and
    `scorers.grounded_qa_accuracy` without re-reading the annotations.

    Accepts either the option letter or the option text, resolved against the
    release order actually shown; options are never re-sorted.
    """
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(
        raw=trajectory_answer,
        options=list(public.options),
        correct_index=private.meta["answer_position"],
    )


def score_grounded(trajectory_answer: str, predicted_span: tuple[float, float],
                   private: PrivateItem, public: PublicItem,
                   iou_threshold: float = 0.5) -> dict:
    """The official Acc@GQA, for conditions that DO elicit a time span.

    Not the registry scorer: the main protocol has no predicted span. Returns
    the three numbers separately (QA accuracy, grounding IoU, joint Acc@GQA)
    so an accuracy and a grounding score can never be averaged into one
    headline.
    """
    from ..evaluation.scorers import grounded_qa_accuracy, temporal_iou

    qa = score(trajectory_answer, private, public)
    gold = [(s.start_s, s.end_s) for s in private.evidence_spans]
    iou = temporal_iou(predicted_span, gold) if gold else 0.0
    return {
        "qa_accuracy": qa,
        "temporal_iou": iou,
        "acc_at_gqa": grounded_qa_accuracy(qa, iou, iou_threshold),
        "has_gold_span": bool(gold),
    }
