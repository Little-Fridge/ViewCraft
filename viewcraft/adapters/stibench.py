"""STI-Bench adapter: accumulating a path through space, frame after frame.

Why this domain is in the study: it is the purest SPATIO-TEMPORAL
ACCUMULATION benchmark in the set. MVBench asks whether something changed;
STI-Bench asks how far the camera travelled, how fast it moved, which way it
turned, and what trajectory it traced -- quantities that are only
recoverable by INTEGRATING over many frames, never by reading one. That is
the exact failure surface the canvas exists for: a model that can jot "the
path so far" as it goes (mark where it has been, accumulate displacement)
has a mechanism a frame-sampler alone does not. The "canvas as a record of
the road travelled" condition is this domain's home turf.

The domain's second value is structural: every row carries one of the eight
OFFICIAL task types, split by the authors into two tiers --

  Static Understanding:  Dimensional Measurement, Spatial Relation,
                         3D Video Grounding
  Dynamic Understanding: Displacement & Path Length, Speed & Acceleration,
                         Ego-Centric Orientation, Trajectory Description,
                         Pose Estimation

so "temporal accumulation vs static measurement" can be decomposed from a
single run without pooling unlike questions. `meta["task_type"]` carries the
official name verbatim; a bucket analysis groups on it.

RELEASED LAYOUT (HF MINT-SJTU/STI-Bench snapshot, verified against the repo
tree and qa.parquet itself, 2026-09):

    qa.parquet                     2,064 rows over 369 videos
    data/test-00000-of-00001.parquet   same table, datasets-style config
    video.zip                      371 clips, flat as video/<name>.mp4
                                   (zip central directory read over HTTP)

One row is

    {Video, Source, Task, QType, Question, Prompt, time_start, time_end,
     Candidates: {A..E}, Answer, Answer Detail, ID, scene}

with Source in {ScanNet, Waymo, Omni6DPose} (scene is its alias: indoor /
outdoor / desktop), QType always "Single Choice", FIVE candidates A-E, and
`Answer` the correct LETTER. Video basenames are unique across the three
sources (369 unique names == 369 unique (Source, name) pairs, verified), so
the bare filename is a safe identity and `group_key`.

OFFICIAL PROTOCOL (github.com/MINT-SJTU/STI-Bench openai_test.py, fetched
2026-09): up to 30 evenly spaced frames taken over the WHOLE video
(stride = total/30 from frame 0), then a prompt of

    "From {time_start} seconds to {time_end} seconds. {Question}"
    + lettered options, "Please output only the option you choose!"

The window prefix is added UNCONDITIONALLY, even for the 920/2,064 rows
(45%) where time_end <= time_start -- those are the static tier, where the
degenerate window marks the instant the question is about. This adapter
keeps the prefix unconditional for comparability with the published
numbers. Sampling follows mvbench/vsi instead: DEFAULT_N_FRAMES = 32 by
uniform-midpoint over the whole video, so a capped pool is exactly what a
`uniform(32)` select_expr produces and the control survives. 30-strided vs
32-midpoint is an immaterial whole-video uniform difference and is DECLARED
here rather than hidden.

WHY NOT WINDOWED SAMPLING: mvbench samples across a released start/end
window because its windows are real. Here 45% of windows are empty, and the
official protocol itself samples the whole video and merely STATES the
window in the prompt. Sampling the window would crash on the static tier
and deviate from the reference numbers on the rest.

PITFALLS, all verified against the release:
  * `Answer Detail` is NOT the gold text -- it disagrees with
    Candidates[Answer] on 1,014/2,064 rows (rounding, unit formatting,
    rewritten phrasing). The LETTER is authoritative; gold index is the
    letter's position among the non-empty candidates. Answer Detail is kept
    in private meta for audit only.
  * Candidates can be None: row 817 ships A filled, B-E null, answer "B" --
    unscoreable by construction. Rows whose gold letter names an empty
    candidate, or with fewer than two non-empty candidates, are DROPPED
    with a printed count, never guessed.
  * `ID` is not unique (max 14; it resets per scene/task block). Item
    identity is the video stem plus the row's ordinal in file order.
  * `Prompt` (an "Object description: ..." string on 825 rows) is NOT part
    of the official prompt -- openai_test.py never reads it. Kept in meta,
    never prepended.
  * The TSV fallback (lmms-lab-si/EASI-Leaderboard-Data STIBench.tsv, the
    vlmeval route) carries the same content under lowercase names
    (video/question/options/answer/task_type/...); it is read only when no
    parquet is present.

MISSING DATA IS NOT AN ERROR: the snapshot may still be downloading, and
video.zip (1.0 GB) may be present but unextracted (this adapter does not
unzip -- the loader must stay cheap and side-effect free beyond the frame
cache; extraction is the download script's job, as in vlmeval's
download_stibench). A question whose clip is absent is skipped with a
printed count, so the domain works on whatever subset has arrived.

SCORING: `stibench_accuracy_v1` -- plain multiple-choice accuracy over the
released candidates (`scorers.match_multiple_choice`: declared letter, then
exact option text, then unique containment), the official metric (the
benchmark is 100% single choice; the leaderboard reports mean accuracy,
broken down by the eight task types). ONE DECLARED DEVIATION, identical to
mvbench's: the official harness appends the lettered options to a fixed
template and reads the reply's first character; ViewCraft owns the prompt
(the theta coordinate it evolves), so options are presented by the
runtime's answer-style machinery and the reply is resolved by the more
permissive matcher, applied identically to every arm and condition. No LLM
judges anything. Candidate order is the release's A-E order, never
re-sorted.

SPLIT UNIT: `group_key` is the video basename. Several questions share one
clip (2,064 over 369), and one clip in both gate and test would let a
strategy tune on frames it is later measured on.
"""
from __future__ import annotations

import string
from collections import Counter
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "stibench"

from ..paths import project_root

ROOT = project_root()
DATA = ROOT / "data" / "stibench"

# Pool of 32, send budget 8 -- the same split mvbench/nextgqa/vsi use, for
# the same reason: the pool and the per-request cap are different
# quantities, and holding the pool at the cap leaves frame selection a
# single legal answer. Official STI-Bench uses 30 strided frames; see the
# docstring for why 32-midpoint is the declared equivalent here.
DEFAULT_N_FRAMES = 32

# The eight official task types, in the paper's static/dynamic order. Not
# used to filter -- the release is trusted to name its own tasks -- but kept
# as the canonical bucket list for analysis and for the loader's sanity
# warning when an unexpected name appears.
TASK_TYPES = (
    "Dimensional Measurement",
    "Spatial Relation",
    "3D Video Grounding",
    "Displacement & Path Length",
    "Speed & Acceleration",
    "Ego-Centric Orientation",
    "Trajectory Description",
    "Pose Estimation",
)

_LETTERS = string.ascii_uppercase


# ---------------------------------------------------------------------------
# QA table reading: parquet (snapshot) first, TSV (vlmeval route) as fallback


def _find_qa_file(root: Path) -> Path | None:
    """Locate the question table under the snapshot root.

    `qa.parquet` is the canonical table; `data/test-*.parquet` is the same
    content in the datasets-library layout; any `*.tsv` is the EASI /
    vlmeval fallback. In that order, first hit wins.
    """
    for cand in (root / "qa.parquet",):
        if cand.is_file():
            return cand
    tests = sorted((root / "data").glob("test-*.parquet")) \
        if (root / "data").is_dir() else []
    if tests:
        return tests[0]
    tsvs = sorted(p for p in root.glob("*.tsv") if p.is_file())
    return tsvs[0] if tsvs else None


def _read_rows(qa: Path) -> list[dict]:
    """Normalize one parquet/TSV row set to plain dicts with lower keys."""
    if qa.suffix == ".parquet":
        import pyarrow.parquet as pq
        records = pq.read_table(qa).to_pandas().to_dict("records")
    else:
        import csv
        with qa.open(newline="") as f:
            records = list(csv.DictReader(f, delimiter="\t"))

    out = []
    for rec in records:
        low = {str(k).strip().lower(): v for k, v in rec.items()}
        # Candidates: a struct dict {A:.., E:..} in parquet, a literal-list
        # string in the TSV. Both become an ordered [(letter, text)] list.
        cands = low.get("candidates")
        if isinstance(cands, dict):
            pairs = [(L, cands.get(L)) for L in _LETTERS if L in cands]
        else:
            raw = low.get("options")
            opts = _parse_options(raw)
            pairs = list(zip(_LETTERS, opts)) if opts else []
        out.append({
            "video": str(low.get("video") or "").strip(),
            "question": str(low.get("question") or "").strip(),
            "object_prompt": str(low.get("prompt") or "").strip(),
            "time_start": _as_float(low.get("time_start")),
            "time_end": _as_float(low.get("time_end")),
            "pairs": [(L, str(t).strip()) for L, t in pairs
                      if t is not None and str(t).strip()],
            "answer": str(low.get("answer") or "").strip(),
            "answer_detail": str(low.get("answer detail") or "").strip(),
            "task_type": str(low.get("task") or low.get("task_type")
                             or "").strip(),
            "source": str(low.get("source") or "").strip(),
            "scene": str(low.get("scene") or "").strip(),
        })
    return out


def _parse_options(raw) -> list[str]:
    """The TSV's `options` cell: a Python-literal list of strings."""
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    if not raw:
        return []
    import ast
    try:
        v = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return []
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else []


def _as_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _gold_index(row: dict) -> int | None:
    """Position of the gold among the NON-EMPTY candidates, or None.

    `Answer` is a letter in the release; the letter's slot can itself be
    empty (see PITFALLS), so the index is taken over the kept pairs, not
    the raw A-E slots. A TSV row may instead carry the option TEXT as its
    answer; that is resolved by exact match, mvbench-style.
    """
    letters = [L for L, _ in row["pairs"]]
    texts = [t for _, t in row["pairs"]]
    ans = row["answer"]
    if len(ans) == 1 and ans in letters:
        return letters.index(ans)
    if len(ans) == 1 and ans in _LETTERS:
        return None            # letter names an absent/empty candidate
    if ans in texts:
        return texts.index(ans)
    return None


# ---------------------------------------------------------------------------
# loading


def load(root: str | Path = DATA, seed: int = 20260911,
         n_frames: int = DEFAULT_N_FRAMES,
         tasks: tuple[str, ...] | None = None,
         limit: int | None = None,
         sample_frames_now: bool = True) -> list[Sample]:
    """Load STI-Bench as public/private Sample pairs.

    `seed` is accepted for interface uniformity and is deliberately UNUSED:
    frame sampling is the deterministic uniform-midpoint scheme, so two
    loads with different seeds are identical and a split plan is the only
    place randomness belongs.

    `tasks` restricts to a subset of the eight official task-type names
    (exact strings, see TASK_TYPES) for bucket-specific DomainSpecs; the
    default keeps all of them, the official headline configuration.

    Questions whose video has not arrived (zip unextracted or partial
    download) are skipped with a printed count, so the domain loads on
    whatever subset is on disk.
    """
    from .vsi import sample_frames  # noqa: F401 -- one decoder, one cache

    root = Path(root)
    qa = _find_qa_file(root)
    if qa is None:
        raise FileNotFoundError(
            f"{DOMAIN}: no qa.parquet, data/test-*.parquet or *.tsv under "
            f"{root} -- the snapshot has not landed yet")

    rows = _read_rows(qa)
    keep = set(tasks) if tasks else None

    samples: list[Sample] = []
    n_no_gold = 0
    n_missing_video = 0
    n_filtered = 0
    seen_tasks: set[str] = set()

    for i, row in enumerate(rows):
        task = row["task_type"]
        if keep and task not in keep:
            n_filtered += 1
            continue
        seen_tasks.add(task)
        vid = row["video"]
        if not vid or not row["question"] or len(row["pairs"]) < 2:
            n_no_gold += 1
            continue
        gold_idx = _gold_index(row)
        if gold_idx is None:
            # Unscoreable by construction (e.g. gold letter names an empty
            # candidate). Dropped, not guessed.
            n_no_gold += 1
            continue

        vpath = root / "video" / vid
        if not vpath.is_file():
            alt = root / vid
            vpath = alt if alt.is_file() else vpath
        if not vpath.is_file():
            # Zip unextracted or clip not yet downloaded: counted and
            # skipped so a partial snapshot stays loadable.
            n_missing_video += 1
            continue

        options = tuple(t for _, t in row["pairs"])
        start, end = row["time_start"], row["time_end"]
        # Official prompt shape, unconditional prefix included (see the
        # docstring: 45% of windows are degenerate and the official harness
        # states them anyway).
        if start is not None and end is not None:
            question = (f"From {start} seconds to {end} seconds. "
                        + row["question"])
        else:
            question = row["question"]

        stem = Path(vid).stem
        if sample_frames_now:
            try:
                paths, seconds, info = sample_frames(
                    DOMAIN, stem, vpath, n_frames)
            except Exception:                             # noqa: BLE001
                # A clip that will not decode is a property of the release,
                # not a candidate's failure: counted and skipped rather
                # than crashing the load.
                n_missing_video += 1
                continue
        else:
            paths, seconds, info = (), (), {"n_frames": 0, "fps": 0.0,
                                            "duration": 0.0}

        item_id = f"{DOMAIN}/{stem}/{i}"
        public = PublicItem(
            item_id=item_id, domain=DOMAIN, question=question,
            media_ids=tuple(paths),
            options=options,                      # release A-E order, kept
            meta={
                # THE analysis bucket: official task type, verbatim.
                "task_type": task,
                "source": row["source"],
                "scene": row["scene"],
                "video": vid,
                "time_start_s": start,
                "time_end_s": end,
                "object_prompt": row["object_prompt"],
                # TRUE total frame count of the decoded clip, never the
                # number sampled.
                "n_frames": int(info.get("n_frames") or 0),
                "n_frames_pooled": len(paths),
                "seconds": list(seconds),
                "fps": info.get("fps"),
                "duration_s": info.get("duration"),
                "sampling_scheme": "uniform_midpoint_v1",
            },
        )
        private = PrivateItem(
            item_id=item_id,
            # The INDEX, so the scorer never has to re-derive it.
            answer=gold_idx,
            answer_type="option_index",
            # STI-Bench releases no evidence localization: the time window
            # is a property of the question, stated in the official prompt,
            # not released evidence -- and it is degenerate on the static
            # tier -- so oracle / probe conditions cannot run here.
            evidence_source="none",
            meta={"task_type": task,
                  "answer_letter": row["pairs"][gold_idx][0],
                  "answer_text": row["pairs"][gold_idx][1],
                  "answer_detail": row["answer_detail"],
                  "n_options": len(options)},
        )
        # Split unit is the VIDEO: 2,064 questions share 369 clips.
        samples.append(Sample(public=public, private=private,
                              group_key=vid))
        if limit and len(samples) >= limit:
            break

    dist = Counter(s.public.meta["task_type"] for s in samples)
    unknown = set(dist) - set(TASK_TYPES)
    dist_s = ", ".join(f"{t}:{n}" for t, n in
                       sorted(dist.items(), key=lambda kv: -kv[1]))
    print(f"[{DOMAIN}] {len(samples)} questions over "
          f"{len({s.group_key for s in samples})} videos from {qa.name}; "
          f"{n_missing_video} missing/unextracted clips, "
          f"{n_no_gold} unscoreable rows"
          + (f", {n_filtered} filtered by tasks" if keep else ""))
    print(f"[{DOMAIN}] task_type distribution: {dist_s or 'none'}")
    if unknown:
        print(f"[{DOMAIN}] WARNING: task types outside the official eight: "
              f"{sorted(unknown)}")
    return samples


# ---------------------------------------------------------------------------
# scoring


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official STI-Bench metric: plain multiple-choice accuracy.

    `match_multiple_choice` resolves a declared letter first, then exact
    option text, then unique containment -- see the DEVIATION paragraph in
    the module docstring. No model acts as judge.
    """
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(answer_text, list(public.options),
                                 int(private.answer))


# ---------------------------------------------------------------------------
# smoke: python -m viewcraft.adapters.stibench


def _smoke_real(limit: int = 12) -> bool:
    """Load the real snapshot (whatever subset has arrived) and self-check.

    Returns True when the QA table was found. Missing clips are the
    expected state mid-download and are reported by load() itself.
    """
    if _find_qa_file(DATA) is None:
        return False
    samples = load(DATA, limit=limit, sample_frames_now=True)
    _check_scoring(samples)
    return True


def _check_scoring(samples: list[Sample]) -> None:
    """score(gold letter) == 1.0 on every loaded sample."""
    bad = 0
    for s in samples:
        letter = _LETTERS[int(s.private.answer)]
        got = score(letter, s.private, s.public)
        # ...and the option text must resolve too (unique containment path)
        got_text = score(s.private.meta["answer_text"], s.private, s.public)
        if got != 1.0 or got_text != 1.0:
            bad += 1
            print(f"[{DOMAIN}] SMOKE FAIL {s.public.item_id}: "
                  f"letter->{got} text->{got_text}")
    print(f"[{DOMAIN}] gold-scoring check: {len(samples) - bad}/"
          f"{len(samples)} score 1.0 by letter and by option text")
    if bad:
        raise SystemExit(1)


def _smoke_synthetic() -> None:
    """No data yet: verify parse + frame sampling + scoring on fake rows.

    Builds a two-video / three-question snapshot in a temp dir -- one
    parquet row per parse hazard the real release exhibits (degenerate
    window, five candidates, a video that never arrived) -- plus a TSV
    copy to exercise the fallback reader.
    """
    import subprocess
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    from .vsi import FFMPEG

    tmp = Path(tempfile.mkdtemp(prefix="stibench_smoke_"))
    (tmp / "video").mkdir()
    for name, color in (("fake_a.mp4", "red"), ("fake_b.mp4", "blue")):
        subprocess.run(
            [FFMPEG, "-v", "error", "-nostdin", "-f", "lavfi",
             "-i", f"color={color}:duration=3:size=64x64:rate=10",
             "-y", str(tmp / "video" / name)],
            check=True, capture_output=True)

    def cands(**kw):
        return {L: kw.get(L) for L in "ABCDE"}

    rows = [
        # dynamic tier, real window, gold C
        {"Video": "fake_a.mp4", "Source": "ScanNet",
         "Task": "Displacement & Path Length", "QType": "Single Choice",
         "Question": "What is the camera's trajectory length? (unit: m)",
         "Prompt": "", "time_start": 0.0, "time_end": 2.0,
         "Candidates": cands(A="1.0m", B="2.0m", C="3.0m", D="4.0m",
                             E="5.0m"),
         "Answer": "C", "Answer Detail": "3.0m", "ID": 0,
         "scene": "indoor"},
        # static tier, DEGENERATE window, gold E (last slot)
        {"Video": "fake_a.mp4", "Source": "ScanNet",
         "Task": "Spatial Relation", "QType": "Single Choice",
         "Question": "What is left of the box?", "Prompt": "",
         "time_start": 1.0, "time_end": 1.0,
         "Candidates": cands(A="w", B="x", C="y", D="z", E="the chair"),
         "Answer": "E", "Answer Detail": "the chair", "ID": 1,
         "scene": "indoor"},
        # second video, so group_key can be checked
        {"Video": "fake_b.mp4", "Source": "Waymo",
         "Task": "Speed & Acceleration", "QType": "Single Choice",
         "Question": "Average speed? (unit: m/s)", "Prompt": "",
         "time_start": 0.0, "time_end": 3.0,
         "Candidates": cands(A="1", B="2", C="3", D="4", E="5"),
         "Answer": "A", "Answer Detail": "1", "ID": 0, "scene": "outdoor"},
        # clip never arrived: must be SKIPPED, not fatal
        {"Video": "missing.mp4", "Source": "Waymo",
         "Task": "Pose Estimation", "QType": "Single Choice",
         "Question": "Pose?", "Prompt": "", "time_start": 0.0,
         "time_end": 1.0,
         "Candidates": cands(A="p", B="q", C="r", D="s", E="t"),
         "Answer": "B", "Answer Detail": "q", "ID": 2, "scene": "outdoor"},
        # unscoreable: gold letter names an empty candidate (real row 817)
        {"Video": "fake_b.mp4", "Source": "Waymo",
         "Task": "Trajectory Description", "QType": "Single Choice",
         "Question": "Trajectory?", "Prompt": "", "time_start": 0.0,
         "time_end": 3.0,
         "Candidates": cands(A="only option"),
         "Answer": "B", "Answer Detail": "q", "ID": 3, "scene": "outdoor"},
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp / "qa.parquet")

    samples = load(tmp, sample_frames_now=True)
    assert len(samples) == 3, f"expected 3 samples, got {len(samples)}"
    assert {s.group_key for s in samples} == {"fake_a.mp4", "fake_b.mp4"}
    assert all(len(s.public.options) == 5 for s in samples)
    assert all(s.public.meta["n_frames_pooled"] ==
               min(DEFAULT_N_FRAMES, s.public.meta["n_frames"])
               for s in samples)
    # degenerate window still gets the official unconditional prefix
    assert any("From 1.0 seconds to 1.0 seconds." in s.public.question
               for s in samples)
    _check_scoring(samples)

    # TSV fallback reader: same content under vlmeval's lowercase names.
    (tmp / "qa.parquet").unlink()
    (tmp / "STIBench.tsv").write_text(
        "index\tvideo\tquestion\toptions\tanswer\ttask_type\t"
        "time_start\ttime_end\n"
        "0\tfake_a.mp4\tHow far?\t['1.0m', '2.0m', '3.0m', '4.0m', "
        "'5.0m']\tC\tDisplacement & Path Length\t0.0\t2.0\n"
        "1\tmissing.mp4\tGone?\t['p', 'q']\tA\tSpeed & Acceleration\t"
        "0.0\t1.0\n")
    tsv_samples = load(tmp, sample_frames_now=False)
    assert len(tsv_samples) == 1, \
        f"TSV fallback: expected 1 sample, got {len(tsv_samples)}"
    assert tsv_samples[0].private.answer == 2
    _check_scoring(tsv_samples)
    print(f"[{DOMAIN}] synthetic smoke OK (parquet + TSV fallback)")


if __name__ == "__main__":
    if not _smoke_real():
        print(f"[{DOMAIN}] no snapshot under {DATA}; "
              f"running synthetic smoke instead")
        _smoke_synthetic()
