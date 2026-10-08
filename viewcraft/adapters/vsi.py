"""VSI-Bench adapter (D5): visual-spatial intelligence from egocentric video.

Released layout: `test.jsonl` (5,130 rows) plus three zips of scene
walkthrough videos (`arkitscenes.zip`, `scannet.zip`, `scannetpp.zip`), one
mp4 per scene. A row is
    {"id", "dataset", "scene_name", "question_type", "question",
     "ground_truth", "options"}
and `options` is null exactly when the question is numeric.

TWO METRICS, NEVER ONE NUMBER
-----------------------------
VSI-Bench is a mixed benchmark and its own protocol scores the halves
differently:

  * NUMERIC types (`options` is null) -- object_counting,
    object_abs_distance, object_size_estimation, room_size_estimation --
    are scored with Mean Relative Accuracy (`scorers.mra`), a mean over ten
    relative-error thresholds. It is NOT 1 - relative_error.
  * MULTIPLE-CHOICE types (`options` present) -- object_rel_distance,
    object_rel_direction_{easy,medium,hard}, route_planning,
    obj_appearance_order -- are scored with plain accuracy
    (`scorers.match_multiple_choice`).

MRA and accuracy are different quantities on the same [0, 1] axis and a
plain mean over all 5,130 items silently mixes them. Every item therefore
records `metric` ("MRA" or "accuracy") and `scorer` in
`PrivateItem.meta`, so an aggregate can be broken down by metric.
**Never report a single VSI-Bench mean without stating the numeric/MC
composition of the set it was computed over**: the two subsets are 2,640
numeric and 2,490 MC items in the full release, and any filtering (by
question type, by scene-level split, by the `pruned` column) moves that
ratio and therefore moves the blended number for reasons that have nothing
to do with model quality. `aggregate()` below returns the split-out form.

OPTION PRESENTATION
-------------------
The release ships options already lettered ("A. chair", "B. stool", ...)
and `ground_truth` is the bare letter. The letter prefix is stripped so the
option text is the bare answer: `match_multiple_choice` falls back to
option-text matching, and a reply of "chair" must be able to match the
option "chair" rather than the string "A. chair". Position carries the
letter, so nothing is lost. Options are presented in release order and
never re-sorted -- the gold letter is already spread across the release
(A=716, B=719, C=603, D=452) so shuffling would only add a mapping layer
that could silently invert.

SPLIT UNIT
----------
`group_key` is the SCENE (`dataset/scene_name`), not the question, so one
scene's video cannot appear in two splits. This is a real constraint, not a
formality: scene counts per question type are wildly uneven (288 scenes
total, but `obj_appearance_order` draws its 618 questions from only 32
scenes). A scene-level split is coarse for that type and any per-type
scene-level split must be checked for having enough scenes to fill its
folds.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "vsi"

from ..paths import project_root

ROOT = project_root()
DATA = ROOT / "data" / "vsibench"
# Extracted scene mp4s. Kept out of _media/ because these are source videos,
# not emitted media, and they are extracted on demand per loaded row set.
VIDEO_ROOT = ROOT / "data" / "vsibench_videos"
MEDIA_ROOT = ROOT / "data" / "_media" / DOMAIN

FFMPEG = os.environ.get("VIEWCRAFT_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("VIEWCRAFT_FFPROBE") or shutil.which("ffprobe") or "ffprobe"

SOURCES = ("arkitscenes", "scannet", "scannetpp")

# `options` is null for exactly these types in the release; the adapter keys
# off the row's own `options` field rather than this list, which exists to
# make the intended partition auditable and to catch release drift.
NUMERIC_TYPES = frozenset({
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
})
MC_TYPES = frozenset({
    "object_rel_distance",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "route_planning",
    "obj_appearance_order",
})

DEFAULT_N_FRAMES = 32

_LETTER_PREFIX = re.compile(r"^\s*([A-Z])\s*[.):]\s*")
# A leading number in a free-form reply, incl. "1,234", ".5", "3.2e1".
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------------------
# scene videos: extract only what the loaded rows need


def _zip_member(dataset: str, scene: str) -> str:
    return f"{dataset}/{scene}.mp4"


def ensure_videos(scenes: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Extract the (dataset, scene) mp4s that are actually needed.

    The three zips total ~5.7 GB and hold 512 videos, while a loaded row set
    may touch far fewer scenes, so members are extracted individually rather
    than by unpacking whole archives. Extraction is idempotent: a scene whose
    file already exists at the archive's recorded size is skipped, and writes
    go to a temp name and are renamed into place so a concurrent reader never
    opens a half-written mp4.
    """
    out: dict[tuple[str, str], str] = {}
    by_ds: dict[str, list[str]] = {}
    for ds, scene in scenes:
        by_ds.setdefault(ds, []).append(scene)

    for ds, scene_list in by_ds.items():
        zpath = DATA / f"{ds}.zip"
        if not zpath.exists():
            raise FileNotFoundError(f"{DOMAIN}: missing archive {zpath}")
        dest_dir = VIDEO_ROOT / ds
        dest_dir.mkdir(parents=True, exist_ok=True)

        pending = []
        for scene in sorted(set(scene_list)):
            dest = dest_dir / f"{scene}.mp4"
            out[(ds, scene)] = str(dest)
            pending.append((scene, dest))

        need = [(s, d) for s, d in pending if not d.exists()]
        if not need:
            continue

        with zipfile.ZipFile(zpath) as zf:
            names = set(zf.namelist())
            for scene, dest in need:
                member = _zip_member(ds, scene)
                if member not in names:
                    raise FileNotFoundError(
                        f"{DOMAIN}: {member} not in {zpath.name}")
                size = zf.getinfo(member).file_size
                if dest.exists() and dest.stat().st_size == size:
                    continue
                tmp = dest.with_name(f".{scene}.{os.getpid()}.tmp")
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
    """Duration, fps and frame count of one scene video, via ffprobe.

    `nb_frames` is trusted when the container reports it and recomputed from
    duration x fps otherwise, because a missing count would silently collapse
    the sampler to a single frame.
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
    """`n_sample` frame indices spread uniformly over the whole walkthrough.

    Midpoints of `n_sample` equal bins: index_i = floor((i + 0.5) * n / N).
    Midpoints rather than endpoints because a walkthrough's first and last
    frames are the least informative ones (camera still being raised, or
    already lowered) and a linspace including both spends two of a small
    budget there. The scheme depends only on (n_frames, n_sample), so it is
    deterministic and reproducible without storing a seed.
    """
    if n_frames <= 0:
        raise ValueError(f"{DOMAIN}: n_frames={n_frames}")
    if n_sample <= 0:
        raise ValueError(f"{DOMAIN}: n_sample={n_sample}")
    if n_sample >= n_frames:
        return list(range(n_frames))
    idx = [int((i + 0.5) * n_frames / n_sample) for i in range(n_sample)]
    return [min(i, n_frames - 1) for i in idx]


def _frame_dir(dataset: str, scene: str, n_sample: int) -> Path:
    return MEDIA_ROOT / dataset / scene / f"n{n_sample}"


def sample_frames(dataset: str, scene: str, video_path: str | Path,
                  n_sample: int = DEFAULT_N_FRAMES,
                  ) -> tuple[tuple[str, ...], tuple[float, ...], dict]:
    """Uniformly sample `n_sample` frames to disk. Returns (paths, seconds, info).

    Frames are written as PNG files under data/_media/vsi/<dataset>/<scene>/
    n<N>/ because the view renderer re-reads the source FILE whenever it makes
    a crop -- an in-memory array or a video handle could not be zoomed into
    that way.

    CACHED and idempotent. A `manifest.json` is written last and acts as the
    completion marker, so a second call returns the identical paths without
    invoking ffmpeg, and a run interrupted mid-decode is not mistaken for a
    finished one. All frames are decoded in ONE ffmpeg pass using a `select`
    filter over the target indices; seeking separately per frame would re-read
    the file N times.
    """
    info = probe_video(video_path)
    indices = uniform_frame_indices(info["n_frames"], n_sample)
    fps = info["fps"] or 1.0
    seconds = tuple(round(i / fps, 4) for i in indices)

    d = _frame_dir(dataset, scene, n_sample)
    manifest = d / "manifest.json"
    paths = tuple(str(d / f"frame_{k:04d}.png") for k in range(len(indices)))

    if manifest.exists():
        try:
            man = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            man = None
        if (man and man.get("indices") == indices
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
        if len(produced) != len(indices):
            raise RuntimeError(
                f"{DOMAIN}: {dataset}/{scene} expected {len(indices)} frames, "
                f"ffmpeg wrote {len(produced)}")
        # ffmpeg numbers its output 1..N; republish as 0-based to match the
        # order of `indices` and `seconds`.
        for k, src in enumerate(produced):
            src.replace(d / f"frame_{k:04d}.png")
        tmp = manifest.with_name(f".manifest.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({
            "dataset": dataset, "scene_name": scene, "n_sample": n_sample,
            "indices": indices, "seconds": list(seconds),
            "duration": info["duration"], "fps": info["fps"],
            "n_frames": info["n_frames"], "scheme": "uniform_midpoint_v1",
        }, indent=2))
        tmp.replace(manifest)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    return paths, seconds, info


# ---------------------------------------------------------------------------
# loading


def _strip_letter(option: str) -> str:
    return _LETTER_PREFIX.sub("", option or "").strip()


def _pruned_ids() -> set[str]:
    p = DATA / "pruned_ids.txt"
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}


def read_rows(root: str | Path = DATA) -> list[dict]:
    """The canonical annotation set: `test.jsonl`, all 5,130 rows.

    The release recommends loading the FULL set and using the `pruned` column
    to report full and Debiased-v1 numbers from one run, rather than
    evaluating a subset directly -- so the pruned flag is attached here as
    metadata and no row is dropped by default.
    """
    path = Path(root) / "test.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    pruned = _pruned_ids()
    for r in rows:
        r["pruned"] = str(r["id"]) in pruned
    return rows


def load(root: str | Path = DATA, n_frames: int = DEFAULT_N_FRAMES,
         question_types: tuple[str, ...] | None = None,
         datasets: tuple[str, ...] | None = None,
         subset: str = "full",
         limit: int | None = None,
         sample_frames_now: bool = True) -> list[Sample]:
    """Load VSI-Bench rows as public/private Sample pairs.

    `subset`: "full" (all rows, each tagged with its pruned flag), "debiased"
    (the 2,362 rows the v1 pilot retained) or "pruned" (the 2,768 it removed).
    v1 is a manual pilot that materially changes the question-type mix, so it
    is a separate artifact and never the default.

    Only the scene videos the surviving rows need are extracted, and each
    needed scene is decoded once even when many questions share it.
    """
    root = Path(root)
    rows = read_rows(root)

    if subset == "debiased":
        rows = [r for r in rows if not r["pruned"]]
    elif subset == "pruned":
        rows = [r for r in rows if r["pruned"]]
    elif subset != "full":
        raise ValueError(f"{DOMAIN}: unknown subset {subset!r}")

    if question_types:
        keep = set(question_types)
        rows = [r for r in rows if r["question_type"] in keep]
    if datasets:
        keep = set(datasets)
        rows = [r for r in rows if r["dataset"] in keep]
    rows.sort(key=lambda r: r["id"])
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return []

    scenes = {(r["dataset"], r["scene_name"]) for r in rows}
    videos = ensure_videos(scenes)

    # One decode per scene, shared by every question on that scene.
    frame_cache: dict[tuple[str, str], tuple] = {}

    samples: list[Sample] = []
    for r in rows:
        ds, scene = r["dataset"], r["scene_name"]
        qt = r["question_type"]
        key = (ds, scene)
        vpath = videos[key]

        if key not in frame_cache:
            if sample_frames_now:
                frame_cache[key] = sample_frames(ds, scene, vpath, n_frames)
            else:
                info = probe_video(vpath)
                idx = uniform_frame_indices(info["n_frames"], n_frames)
                fps = info["fps"] or 1.0
                d = _frame_dir(ds, scene, n_frames)
                frame_cache[key] = (
                    tuple(str(d / f"frame_{k:04d}.png") for k in range(len(idx))),
                    tuple(round(i / fps, 4) for i in idx),
                    info,
                )
        paths, seconds, info = frame_cache[key]

        raw_options = list(r.get("options") or [])
        is_mc = bool(raw_options)
        gold_raw = str(r["ground_truth"]).strip()
        item_id = f"{DOMAIN}/{qt}/{scene}_{r['id']}"

        if is_mc:
            options = tuple(_strip_letter(o) for o in raw_options)
            if len(options) < 2:
                raise ValueError(f"{item_id}: {len(options)} option(s)")
            pos = ord(gold_raw[:1].upper()) - ord("A")
            if not (0 <= pos < len(options)):
                raise ValueError(
                    f"{item_id}: gold {gold_raw!r} outside 0..{len(options) - 1}")
            answer = options[pos]
            answer_type = "multiple_choice"
            private_meta = {
                "scorer": "match_multiple_choice",
                "metric": "accuracy",
                "answer_letter": chr(ord("A") + pos),
                "answer_position": pos,
                "release_options": raw_options,
            }
        else:
            options = ()
            try:
                answer = float(gold_raw)
            except ValueError as exc:
                raise ValueError(
                    f"{item_id}: numeric gold {gold_raw!r} not a float") from exc
            answer_type = "numeric"
            private_meta = {
                "scorer": "mra",
                "metric": "MRA",
                "gold_text": gold_raw,
            }

        private_meta.update({
            "question_type": qt, "dataset": ds, "scene_name": scene,
            "source_id": r["id"], "pruned": r["pruned"],
        })

        public = PublicItem(
            item_id=item_id,
            domain=DOMAIN,
            question=r["question"],
            media_ids=paths,
            options=options,
            meta={
                "dataset": ds,
                "scene_name": scene,
                "question_type": qt,
                "duration_s": round(info["duration"], 3),
                "frame_seconds": seconds,
                "n_frames_sampled": len(paths),
                "video_fps": round(info["fps"], 4),
                "video_n_frames": info["n_frames"],
                "frame_width": info["width"],
                "frame_height": info["height"],
                "sampling_scheme": "uniform_midpoint_v1",
                # Answer FORMAT only, never the answer: numeric questions need
                # a bare number, and the executor cannot guess that from the
                # question text alone.
                "answer_format": ("option_letter" if is_mc else "number"),
                "n_options": len(options),
            },
        )
        private = PrivateItem(
            item_id=item_id,
            answer=answer,
            answer_type=answer_type,
            # VSI-Bench ships no per-question temporal or spatial evidence: the
            # question is about the whole walkthrough. Claiming evidence spans
            # we derived ourselves would misreport the source.
            evidence_source="none",
            evidence_verified=False,
            meta=private_meta,
        )
        samples.append(Sample(public=public, private=private,
                              group_key=f"{ds}/{scene}"))
    return samples


# ---------------------------------------------------------------------------
# scoring


def extract_number(raw: str) -> str | None:
    """Pull the answer number out of a free-form reply.

    The official numeric protocol compares a bare number, so a reply of
    "about 4 tables" has to be reduced before MRA can apply. The LAST number
    is taken, because a reply that reasons before answering ("2 chairs and 3
    stools, so 5") ends on its conclusion. Comparison itself is left entirely
    to `scorers.mra`.
    """
    if not raw:
        return None
    cands = [m.group(0) for m in _NUMBER.finditer(raw)
             if m.group(0) not in ("", "-", "+", ".", "-.", "+.")]
    cands = [c for c in cands if re.search(r"\d", c)]
    if not cands:
        return None
    return cands[-1].replace(",", "")


def score(trajectory_answer: str, private: PrivateItem,
          public: PublicItem) -> float:
    """VSI-Bench per-item score: MRA for numeric, accuracy for MC.

    The returned floats are NOT commensurable across the two branches --
    `private.meta["metric"]` says which one this is, and aggregation must
    group by it (see `aggregate`).
    """
    from ..evaluation.scorers import match_multiple_choice, mra

    if private.answer_type == "multiple_choice":
        return match_multiple_choice(
            raw=trajectory_answer,
            options=list(public.options),
            correct_index=private.meta["answer_position"],
        )

    pred = extract_number(trajectory_answer)
    if pred is None:
        return 0.0
    return mra(pred, private.meta["gold_text"])


def aggregate(scored: list[tuple[float, PrivateItem]]) -> dict:
    """Break scores down by metric and question type; never blend blindly.

    Returns the two headline numbers separately (`mra_mean` over numeric
    items, `accuracy_mean` over MC items) plus per-type means and the item
    counts each was computed over. `overall_unweighted_mean` is included
    because the VSI-Bench paper reports one, but it is labelled and carries
    its own composition so a reader can see what was mixed.
    """
    by_metric: dict[str, list[float]] = {}
    by_type: dict[str, list[float]] = {}
    for s, priv in scored:
        by_metric.setdefault(priv.meta["metric"], []).append(s)
        by_type.setdefault(priv.meta["question_type"], []).append(s)

    def mean(v: list[float]) -> float | None:
        return (sum(v) / len(v)) if v else None

    num = by_metric.get("MRA", [])
    acc = by_metric.get("accuracy", [])
    return {
        "mra_mean": mean(num),
        "n_numeric": len(num),
        "accuracy_mean": mean(acc),
        "n_multiple_choice": len(acc),
        "per_question_type": {
            k: {"mean": mean(v), "n": len(v),
                "metric": "MRA" if k in NUMERIC_TYPES else "accuracy"}
            for k, v in sorted(by_type.items())
        },
        "overall_unweighted_mean": mean(num + acc),
        "overall_mixes_metrics": bool(num and acc),
        "composition_note": (
            f"blends {len(num)} MRA items with {len(acc)} accuracy items"
            if num and acc else "single-metric set"),
    }
