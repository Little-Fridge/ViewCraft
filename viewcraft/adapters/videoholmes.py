"""VideoHolmes adapter: multi-clue cumulative reasoning over suspense shorts.

WHY THIS DOMAIN IS IN THE STUDY
-------------------------------
VideoHolmes (TencentARC/Video-Holmes) is 270 suspense short films of 1-10
minutes with 1,837 human-written multiple-choice questions. What separates
it from the other video domains here is the SHAPE of the evidence: a
question is answered by accumulating several clues that are distributed
across the WHOLE film -- an early seemingly-innocent detail, a mid-film
contradiction, a late reveal -- rather than by finding one decisive moment.
MVBench asks about something that happens over a few seconds inside a 5-35 s
clip; a VideoHolmes question about motive or causality is unanswerable until
the clues are jointly in context, and no single frame or short window
carries the answer. That is precisely the regime where the mechanism's
selection and composition coordinates are load-bearing: `phi.select_expr`
must keep the clue-bearing frames of a 32-frame pool alive under psi's
8-image per-request cap, and a strategy that spends the budget on even
coverage can miss the one frame that resolves the contradiction. The
released `Question Type` tags let the analysis bin gains by reasoning
category instead of reporting one pooled number.

RELEASED LAYOUT (HF TencentARC/Video-Holmes)
--------------------------------------------
One QA file plus the videos, which may be on disk in any of three shapes
depending on how the snapshot landed:

  * `test_Video-Holmes.json` -- a list of 1,837 items, each:
        {"video ID", "Question ID", "Question Type", "Question",
         "Options": {"A": text, "B": text, ... up to "F"},
         "Answer": "<letter>", "Explanation": ...}
    The option count VARIES per item (three to six, keys A..F), unlike the
    fixed four/five of mvbench/nextgqa.
  * videos as `video/<video ID>.mp4` or `videos/<video ID>.mp4`, or still
    packed as `videos.zip` (members may be nested; matched by basename), or
    as HF `video_pkl/*.pkl` archives (each pickle a {name: bytes} dict).

`Answer` is the gold LETTER, not the option text -- the opposite convention
from MVBench. The loader therefore resolves the letter against the released
option keys; it also tolerates a release variant that writes the answer as
"(A)" / "A." / the option TEXT, resolving those back to the letter. An item
whose answer cannot be resolved against its own options is unscoreable and
is DROPPED with a printed count, never guessed.

OFFICIAL PROTOCOL, AND WHAT IS KEPT
-----------------------------------
The official evaluation is plain multiple-choice accuracy: the prompt lists
the lettered options and the reply is resolved to a letter (exact letter
first, then text matching; VLMEval's `resolve_videoholmes_answer`). For
models that take frames rather than a native video stream, the official
direct-input setting feeds UNIFORMLY SAMPLED frames spanning the whole film
-- which is exactly what this adapter writes as the media pool, so the
32-frame pool is the official direct frame-input representation, not an
adapter invention. As in mvbench/nextgqa, ViewCraft owns the prompt (that
is the theta coordinate), so the official fixed template with its
<think>/<answer> tags is not used; replies resolve through
`scorers.match_multiple_choice`, the same parser the other MCQ domains use.
That parser is strictly more permissive on FORMAT than reading one letter,
is applied identically to every arm, and judges nothing with an LLM.

FRAME POOL: `vsi.sample_frames` is reused (one decoder, one cache layout,
one uniform-midpoint scheme) so a 32-frame pool is exactly what a
`uniform(32)` selection expression would produce and remains a legitimate
control. `meta` carries the TRUE clip shape (`n_frames` = total decoded
frames, never the number pooled) alongside `n_frames_pooled` and the
per-frame `seconds`. One decode per VIDEO, shared by all its questions.
NOTE on scale: these films run 1-10 min, so 32 frames is far sparser
relative to the content than on MVBench's 5-35 s clips -- one frame per
~2-19 s. That sparsity is deliberate (same pool as the other video domains,
for comparability), and it is what makes clue coverage a real decision.

SPLIT UNIT: `group_key` is the video ID. A film carries ~7 questions, and
letting one film appear in both gate and test would let a strategy tune on
footage it is later measured on.

KNOWN PITFALLS (all handled, all printed rather than silent):
  * The data root may still be MID-DOWNLOAD: the QA json can be present
    while only some mp4s have landed, or videos may still be packed in
    `videos.zip` / `video_pkl/`. The loader extracts needed members from a
    zip or unwraps needed pickles when that is the only source, and a video
    that is simply absent is SKIPPED WITH A COUNT -- the domain loads on a
    partial download instead of failing. A clip that will not decode is a
    property of the release, counted and skipped the same way.
  * Option count varies 3-6 (A..F): nothing here may assume four options.
  * `Explanation` is answer-bearing rationale. It rides on PrivateItem.meta
    ONLY; putting it in the public meta would leak the solution.
  * No temporal evidence spans are released, so the oracle / probe
    conditions cannot run here (`evidence_source="none"`).

SCORING: `videoholmes_accuracy_v1`
----------------------------------
Official metric: plain MCQ accuracy, resolved by
`scorers.match_multiple_choice` -- declared letter first, then exact option
text, then unique containment. Same resolution tolerance as mvbench and
nextgqa; the gold index comes from PrivateItem.meta["answer_position"] so
the scorer never re-derives it. `PrivateItem.answer` holds the gold LETTER
("C"), matching the release's own convention.

SMOKE
-----
    cd <repo root> && python -m viewcraft.adapters.videoholmes [LIMIT]

Runs the parse/score self-checks unconditionally; if the QA file has landed
it then loads the release (LIMIT rows if given) and asserts
score(gold letter) == 1.0 on every loaded sample.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "videoholmes"

from ..paths import project_root

ROOT = project_root()
DATA = ROOT / "data" / DOMAIN

QA_FILE = "test_Video-Holmes.json"
# Candidate video locations, in the order they are trusted. The HF snapshot
# lands either extracted mp4s or the packed archives; all three are handled.
VIDEO_SUBDIRS = ("video", "videos")
VIDEO_ZIP = "videos.zip"
PKL_SUBDIR = "video_pkl"

# Pool of 32, send budget 8 -- the same split as mvbench/nextgqa/vsi. The
# pool and the per-request cap are DIFFERENT quantities: holding the pool at
# the cap would leave frame selection with a single legal answer.
DEFAULT_N_FRAMES = 32
SAMPLING_SCHEME = "uniform_midpoint_v1"

# Options keys are letters, at most A..F in the release; resolved in letter
# order, which is the release's presentation order and is never re-sorted.
_LETTERS = "ABCDEFGHIJ"


# ---------------------------------------------------------------------------
# QA rows and gold resolution


def read_rows(root: str | Path = DATA) -> list[dict]:
    """The released QA list, 1,837 items over 270 films (test set; the
    benchmark ships no other split)."""
    path = Path(root) / QA_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{DOMAIN}: no {QA_FILE} under {root} -- the snapshot has not "
            f"landed yet (download may still be running)")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{DOMAIN}: {path} is not a JSON list")
    return rows


def _options_of(item: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(letters, texts) of an item's options, in release letter order."""
    opts = item.get("Options") or {}
    if not isinstance(opts, dict):
        return (), ()
    letters = [k for k in _LETTERS if k in opts]
    return tuple(letters), tuple(str(opts[k]) for k in letters)


def _resolve_gold(answer: str, letters: tuple[str, ...],
                  texts: tuple[str, ...]) -> int | None:
    """Gold option index, or None if the answer cannot be resolved.

    The release writes `Answer` as the bare letter ("C"). Tolerated variants
    are "(C)" / "C." / "C)" decorations and the option TEXT itself; anything
    else is unscoreable and the caller drops the item rather than guessing.
    """
    a = str(answer or "").strip()
    if not a or not letters:
        return None
    # Bare or decorated single letter: C, (C), C., C), "C: ..."
    head = a.strip("()[] ")
    if head in letters:
        return letters.index(head)
    if len(a) >= 2 and a[0] in letters and a[1] in ".)]:、":
        return letters.index(a[0])
    # The answer written as the option TEXT (MVBench's convention, in case a
    # release variant uses it). A text matching two options is ambiguous and
    # resolves to neither -- that item is dropped.
    matches = [i for i, t in enumerate(texts) if t.strip() == a]
    if len(matches) == 1:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# videos: extracted mp4s first, then the packed archives


def _video_dirs(root: Path) -> list[Path]:
    return [root / sub for sub in VIDEO_SUBDIRS if (root / sub).is_dir()]


def find_video(root: Path, vid: str) -> Path | None:
    for d in _video_dirs(root):
        p = d / f"{vid}.mp4"
        if p.is_file():
            return p
    return None


def _extract_from_zip(root: Path, need: set[str]) -> None:
    """Extract needed mp4s from videos.zip into <root>/video/.

    Members are matched by BASENAME because the archive's nesting is not
    part of the QA schema. Each member goes to a pid-scoped temp name that
    is renamed into place only after the byte count matches the archive
    record, so a concurrent reader never opens a half-written mp4 and a
    short read is never published (a truncated mp4 would probe as a shorter
    film and silently shift every frame timestamp).
    """
    zp = root / VIDEO_ZIP
    if not zp.is_file():
        return
    out_dir = root / "video"
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zp) as zf:
        by_base = {}
        for info in zf.infolist():
            base = os.path.basename(info.filename)
            if base.endswith(".mp4"):
                by_base.setdefault(base, info)
        for vid in sorted(need):
            info = by_base.get(f"{vid}.mp4")
            if info is None:
                continue
            dest = out_dir / f"{vid}.mp4"
            if dest.exists() and dest.stat().st_size == info.file_size:
                continue
            tmp = dest.with_name(f".{vid}.{os.getpid()}.tmp")
            with zf.open(info) as src, open(tmp, "wb") as fh:
                shutil.copyfileobj(src, fh, 1 << 20)
            if tmp.stat().st_size != info.file_size:
                tmp.unlink(missing_ok=True)
                raise IOError(f"{DOMAIN}: short read extracting {vid}")
            tmp.replace(dest)


def _unwrap_from_pkl(root: Path, need: set[str]) -> None:
    """Write needed mp4s out of HF video_pkl/*.pkl archives into <root>/video/.

    Each pickle is a {video_name: bytes} dict. Only names in `need` are
    written; the full unwrap (vlmeval's behaviour) would publish ~270 mp4s
    when a smoke run touches a handful.
    """
    pkl_dir = root / PKL_SUBDIR
    if not pkl_dir.is_dir():
        return
    import pickle
    out_dir = root / "video"
    out_dir.mkdir(exist_ok=True)
    remaining = {v for v in need
                 if not (out_dir / f"{v}.mp4").is_file()}
    for pf in sorted(pkl_dir.glob("*.pkl")):
        if not remaining:
            break
        try:
            with open(pf, "rb") as fh:
                blob = pickle.load(fh)  # noqa: S301 -- benchmark release
        except Exception:                               # noqa: BLE001
            # One corrupt archive must not abort the load; its videos are
            # simply counted missing downstream.
            print(f"[{DOMAIN}] unreadable pickle {pf.name}; skipping it")
            continue
        if not isinstance(blob, dict):
            continue
        for name, content in blob.items():
            vid = str(name)
            if vid not in remaining:
                continue
            dest = out_dir / f"{vid}.mp4"
            tmp = dest.with_name(f".{vid}.{os.getpid()}.tmp")
            tmp.write_bytes(content)
            tmp.replace(dest)
            remaining.discard(vid)


def ensure_videos(root: Path, video_ids: set[str]) -> dict[str, Path | None]:
    """Resolve each needed video ID to an mp4 path, or None if unavailable.

    Tolerant by design: whatever the snapshot has landed so far is used, the
    packed archives are mined only for what is needed, and what is genuinely
    absent comes back as None for the caller to count and skip.
    """
    root = Path(root)
    out: dict[str, Path | None] = {v: find_video(root, v) for v in video_ids}
    missing = {v for v, p in out.items() if p is None}
    if missing:
        _extract_from_zip(root, missing)
        out.update({v: find_video(root, v) for v in missing})
    missing = {v for v, p in out.items() if p is None}
    if missing:
        _unwrap_from_pkl(root, missing)
        out.update({v: find_video(root, v) for v in missing})
    return out


# ---------------------------------------------------------------------------
# loading


def load(root: str | Path = DATA, seed: int = 20260911,
         n_frames: int = DEFAULT_N_FRAMES,
         limit: int | None = None,
         video_ids: tuple[str, ...] | None = None,
         sample_frames_now: bool = True) -> list[Sample]:
    """Load VideoHolmes as public/private Sample pairs.

    `seed` is accepted for interface uniformity and is deliberately UNUSED:
    option order is the release's letter order and frame sampling is the
    deterministic uniform-midpoint scheme, so this loader has no stochastic
    component to seed.

    `limit` / `video_ids` restrict the row set, and only the videos the
    surviving rows need are resolved and decoded -- that is what lets a
    smoke run materialize a handful of films instead of 270. Each needed
    film is decoded once even though ~7 questions share it.

    Tolerant mode: films whose mp4 has not landed (or will not decode) are
    skipped with a printed count, so the domain loads on a partial download
    instead of failing. Questions whose answer cannot be resolved against
    their own options are dropped with a printed count, never guessed.
    """
    from .vsi import sample_frames  # shared decoder/cache/sampling scheme

    root = Path(root)
    rows = read_rows(root)

    if video_ids:
        keep = {str(v) for v in video_ids}
        rows = [r for r in rows if str(r.get("video ID") or "") in keep]
    if limit is not None:
        rows = rows[:limit]

    # Resolve gold BEFORE touching any video, so unscoreable rows do not
    # trigger an extraction or decode.
    parsed: list[tuple[dict, str, tuple[str, ...], tuple[str, ...], int]] = []
    n_no_gold = 0
    for r in rows:
        vid = str(r.get("video ID") or "").strip()
        question = str(r.get("Question") or "").strip()
        letters, texts = _options_of(r)
        gold_idx = _resolve_gold(r.get("Answer"), letters, texts)
        if not vid or not question or len(letters) < 2 or gold_idx is None:
            n_no_gold += 1
            continue
        parsed.append((r, vid, letters, texts, gold_idx))
    if not parsed:
        print(f"[{DOMAIN}] 0 usable questions "
              f"({n_no_gold} unscoreable of {len(rows)} rows)")
        return []

    videos = ensure_videos(root, {vid for _, vid, _, _, _ in parsed})

    # One probe/decode per film, shared by every question on it. A film that
    # fails to decode caches None so its other questions are counted, not
    # re-attempted.
    frame_cache: dict[str, tuple | None] = {}

    samples: list[Sample] = []
    n_missing_video = 0
    for i, (r, vid, letters, texts, gold_idx) in enumerate(parsed):
        vpath = videos.get(vid)
        if vpath is None:
            n_missing_video += 1
            continue
        if vid not in frame_cache:
            if sample_frames_now:
                try:
                    frame_cache[vid] = sample_frames(
                        DOMAIN, vid, vpath, n_frames)
                except Exception:                         # noqa: BLE001
                    # A film that will not decode is a property of the
                    # release, not a candidate's failure: counted and
                    # skipped rather than crashing the load.
                    frame_cache[vid] = None
            else:
                # Parse-only mode (used when the media tree must not be
                # written): no frames, and the true shape is unknown.
                frame_cache[vid] = ((), (), {"n_frames": 0, "fps": 0.0,
                                             "duration": 0.0})
        entry = frame_cache[vid]
        if entry is None:
            n_missing_video += 1
            continue
        paths, seconds, info = entry

        qid = str(r.get("Question ID") or "").strip() or f"{vid}_{i}"
        item_id = f"{DOMAIN}/{qid}"
        qtype = str(r.get("Question Type") or "").strip()

        public = PublicItem(
            item_id=item_id, domain=DOMAIN, question=question,
            media_ids=tuple(paths),
            # Release letter order (A..F as released), never re-sorted.
            options=texts,
            meta={
                "video_id": vid,
                "question_id": qid,
                "question_type": qtype,
                # TRUE total frame count of the decoded film, never the
                # number pooled. A selection expression reads the pool size;
                # an analysis needs the real one.
                "n_frames": int(info.get("n_frames") or 0),
                "n_frames_pooled": len(paths),
                "seconds": list(seconds),
                "fps": info.get("fps"),
                "duration_s": info.get("duration"),
                "sampling_scheme": SAMPLING_SCHEME,
                "n_options": len(texts),
                "answer_format": "option_letter",
            },
        )
        private = PrivateItem(
            item_id=item_id,
            # The gold LETTER, matching the release's own convention. The
            # index the scorer needs rides in meta so it is never re-derived.
            answer=letters[gold_idx],
            answer_type="option_letter",
            # No temporal evidence spans are released, so the oracle / probe
            # conditions cannot run here.
            evidence_source="none",
            meta={
                "answer_position": gold_idx,
                "answer_letter": letters[gold_idx],
                "question_id": qid,
                "question_type": qtype,
                "video_id": vid,
                # Answer-bearing rationale: PRIVATE ONLY. In public meta it
                # would hand the executor the solution.
                "explanation": str(r.get("Explanation") or ""),
                "scorer": "match_multiple_choice",
                "metric": "qa_accuracy",
                "n_options": len(texts),
            },
        )
        # Group by the FILM: ~7 questions share one video, and one film in
        # both gate and test would let a strategy tune on footage it is
        # later measured on.
        samples.append(Sample(public=public, private=private, split="test",
                              group_key=vid))

    n_avail = sum(1 for p in videos.values() if p is not None)
    print(f"[{DOMAIN}] {len(samples)} questions over "
          f"{len({s.group_key for s in samples})} films; "
          f"{n_no_gold} unscoreable rows, "
          f"{n_missing_video} questions skipped (film missing/undecodable; "
          f"{n_avail}/{len(videos)} needed films on disk)")
    return samples


# ---------------------------------------------------------------------------
# scoring


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official VideoHolmes metric: plain multiple-choice accuracy.

    `match_multiple_choice` resolves a declared letter first, then exact
    option text, then unique containment -- the same tolerance mvbench and
    nextgqa use, and strictly more permissive on FORMAT than the official
    read-one-letter check. No model acts as judge.
    """
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(
        raw=answer_text, options=list(public.options),
        correct_index=private.meta["answer_position"])


# ---------------------------------------------------------------------------
# smoke


def _selfcheck() -> None:
    """Parse/score checks that need no data on disk."""
    letters, texts = ("A", "B", "C", "D"), ("red", "green", "blue", "cyan")
    assert _resolve_gold("C", letters, texts) == 2
    assert _resolve_gold("(C)", letters, texts) == 2
    assert _resolve_gold("C.", letters, texts) == 2
    assert _resolve_gold("blue", letters, texts) == 2      # text convention
    assert _resolve_gold("Z", letters, texts) is None      # unknown letter
    assert _resolve_gold("", letters, texts) is None
    assert _resolve_gold("reddish", letters, texts) is None
    # Options count varies A..F: a 6-option item must resolve F.
    six_l = ("A", "B", "C", "D", "E", "F")
    six_t = ("o1", "o2", "o3", "o4", "o5", "o6")
    assert _resolve_gold("F", six_l, six_t) == 5

    pub = PublicItem(item_id=f"{DOMAIN}/smoke", domain=DOMAIN,
                     question="q?", media_ids=(), options=texts, meta={})
    prv = PrivateItem(item_id=f"{DOMAIN}/smoke", answer="C",
                      answer_type="option_letter", evidence_source="none",
                      meta={"answer_position": 2})
    assert score("C", prv, pub) == 1.0
    assert score("The answer is (C).", prv, pub) == 1.0
    assert score("blue", prv, pub) == 1.0                  # text resolution
    assert score("A", prv, pub) == 0.0
    print(f"[{DOMAIN}] self-check passed (gold resolution + score)")


def _smoke(limit: int | None) -> None:
    import statistics

    _selfcheck()
    try:
        samples = load(limit=limit)
    except FileNotFoundError as e:
        # QA file not landed yet: the parse/score self-check above is what
        # validates the adapter logic until the snapshot completes.
        print(f"[{DOMAIN}] data not on disk yet ({e}); "
              f"self-check only")
        return
    if not samples:
        return

    groups = {s.group_key for s in samples}
    sizes: dict[str, int] = {}
    for s in samples:
        sizes[s.group_key] = sizes.get(s.group_key, 0) + 1
    types: dict[str, int] = {}
    durs = []
    for s in samples:
        t = str(s.public.meta.get("question_type") or "?")
        types[t] = types.get(t, 0) + 1
        d = s.public.meta.get("duration_s")
        if d:
            durs.append(float(d))
    print(f"\n[{DOMAIN}] smoke over {len(samples)} questions, "
          f"{len(groups)} films")
    print(f"questions per film : min {min(sizes.values())} "
          f"median {statistics.median(sizes.values()):g} "
          f"max {max(sizes.values())}")
    print("question types     : "
          + ", ".join(f"{k}={v}" for k, v in sorted(types.items())))
    if durs:
        durs.sort()
        print(f"film duration s    : min {durs[0]:g} "
              f"median {statistics.median(durs):g} max {durs[-1]:g}")
    pooled = {s.public.meta["n_frames_pooled"] for s in samples}
    print(f"frames pooled      : {sorted(pooled)}")

    n_bad = 0
    for s in samples:
        # The gold letter, as the executor would emit it, must score 1.0.
        if score(str(s.private.answer), s.private, s.public) != 1.0:
            n_bad += 1
            print(f"  GOLD SCORED WRONG: {s.public.item_id} "
                  f"answer={s.private.answer!r} options={s.public.options}")
    assert n_bad == 0, f"{n_bad} gold answers failed to score 1.0"
    print(f"score(gold letter) : 1.0 on all {len(samples)} items -- OK")


if __name__ == "__main__":
    import sys
    _smoke(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
