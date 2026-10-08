"""OpenEQA HM-EQA adapter: first-person EPISODIC memory over a home tour.

WHY THIS DOMAIN IS IN THE STUDY
-------------------------------
OpenEQA (Majumdar et al., 2024) asks questions about a space the agent has
SEEN -- a walkthrough of a real home -- the way a person would ask someone
who lives there: "where did I leave the remote?", "what colour is the sofa
in the study?", "is the dishwasher open?". The answer is distributed across
the episode: no single frame carries it, and the model must recall what was
seen WHERE. That is precisely the regime the canvas's "map marking" gesture
was built for -- jotting "keys on the hallway table" at the right spot as
the tour goes by, then reading the map back at question time. If the visual
workspace is external memory anywhere, it is here; this is the textbook
home ground for the mechanism, complementing mvbench (what HAPPENED in a
short clip) with where-things-ARE across a long first-person tour.

WHY HM-EQA ONLY
---------------
OpenEQA ships two halves: HM-EQA (557 questions over 63 HM3D scenes) and
ScanNet-EQA (1079 questions over 165 ScanNet scenes). HM3D frames download
directly (a single public archive, hm-eqa-frames-hq); ScanNet scene data is
behind the ScanNet terms-of-use APPLICATION -- per-user registration and
manual approval -- which could not be completed in the available time. The
HM half keeps all seven official question categories represented (63
scenes, ~9 questions per scene), so the split costs coverage, not
diversity. If ScanNet access is granted later, the loader below already
keys episodes by the `episode_history` prefix ("hm3d-v0/" vs "scannet-v0/")
-- widening the prefix filter is the only change needed.

OFFICIAL PROTOCOL (LLM-Match), AND WHAT THIS ADAPTER KEEPS
----------------------------------------------------------
OpenEQA answers are FREE-FORM SHORT TEXT and the official metric is
LLM-Match: an independent GPT judge is shown the question, the ground-truth
answer and the model's answer, and marks the model answer "correct" or
"incorrect"; the headline number is the fraction marked correct. It is a
BINARY judgement, not a 1-5 scale (the paper's 1-5 ratings are a human
study; the released automatic metric is the correct/incorrect match). This
adapter keeps:

  * the question text VERBATIM -- no option list, no format instruction
    appended (the official prompt adds only a generic "answer concisely"
    style line, and ViewCraft owns the prompt through its own answer-style
    machinery);
  * the binary correct/incorrect judgement, scored by the OPTIMIZER model
    (gpt-6-astra) acting as an independent judge, never the executor --
    judge != answerer, the same pattern as clbench's rubric judge;
  * answer_style="short_text", matching the free-form nature of the task.

ONE DECLARED DEVIATION: the judge prompt is REWRITTEN (question / gold /
model answer, reply with JSON {"label": "correct"|"incorrect"}) rather than
the official template verbatim, because the judge here is a different model
family and the JSON contract is machine-checked. The grading SEMANTICS --
correct iff the answer matches the gold's meaning, extra hedging tolerated,
wrong or vague marked incorrect -- follow the official rubric.

FRAME POOL, AND WHAT IS CAPPED
------------------------------
Each HM-EQA episode ships as a directory of extracted RGB frames (a dense
tour, typically hundreds of frames). `media_ids` carries `DEFAULT_N_FRAMES`
= 32 frames sampled by `vsi.uniform_frame_indices` (midpoints of equal
bins, deterministic, no seed) -- the same pool size and scheme as
nextgqa/vsi/mvbench, and for the same reason: the pool and the per-request
image cap (8) are DIFFERENT quantities, and a pool held at the cap leaves
phi.select_expr with exactly one legal answer. `meta` records the TRUE
total frame count (`n_frames`), the pooled count and the sampled source
indices.

SPLIT UNIT
----------
`group_key` is the EPISODE ID. A scene is asked ~9 questions; letting one
episode straddle gate and test would let a strategy tune its map-marking on
frames it is later measured on.

Episodes whose frame directory is not yet on disk are SKIPPED with a
printed count (the frames archive may still be extracting when this first
runs), so the domain degrades to the available episodes instead of failing
the load.

SCORERS
-------
`score`        -- official-protocol LLM-Match, judge = optimizer, 0.0/1.0.
`score_offline`-- judge-free quick self-check: exact match after
                  normalization (case, punctuation, articles). Deliberately
                  STRICTER than LLM-Match -- a paraphrase the judge would
                  accept scores 0 here -- so it is a smoke-test floor, never
                  a reported metric.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "openeqa"

# Pool of 32, send budget 8. See "FRAME POOL, AND WHAT IS CAPPED": the pool
# and the per-request cap are different quantities.
DEFAULT_N_FRAMES = 32

# Only the HM3D half of OpenEQA -- see "WHY HM-EQA ONLY". The released
# `episode_history` field is "<dataset>/<episode_id>", e.g.
# "hm3d-v0/000-hm3d-BFRyYbPCCPE".
KEEP_PREFIX = "hm3d"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

_JUDGE_CLIENT = None


def _judge():
    """Lazily build the judge client -- the OPTIMIZER, not the executor, so
    the grader is independent of the model being graded."""
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        from ..models import ModelClient
        _JUDGE_CLIENT = ModelClient("optimizer")
    return _JUDGE_CLIENT


def _load_questions(root: Path) -> list[dict]:
    """Read the released question list, whichever local copy is present.

    Resolution order: the released JSON next to the frames, the HF parquet
    snapshot, then the download staging area (`data/_dl/`). Rows are the
    release's dicts: {question, answer, category, question_id,
    episode_history}.
    """
    j = root / "open-eqa-v0.json"
    if j.is_file():
        return list(json.loads(j.read_text()))
    pqs = sorted(root.rglob("*.parquet"))
    if pqs:
        import pandas as pd
        rows: list[dict] = []
        for pq in pqs:
            rows.extend(pd.read_parquet(pq).to_dict("records"))
        return rows
    staged = root.parent / "_dl" / "open-eqa-v0.json"
    if staged.is_file():
        return list(json.loads(staged.read_text()))
    raise FileNotFoundError(
        f"{DOMAIN}: no open-eqa-v0.json or parquet under {root} "
        f"(also tried {staged})")


def _index_episode_dirs(root: Path) -> dict[str, Path]:
    """Map episode_id -> frame directory.

    The extraction layout of the frames archive is not fixed by the release
    (frames may sit at <root>/<episode_id>/ or one wrapper deeper, e.g.
    <root>/frames/<episode_id>/), so directories are indexed by basename
    down to depth 2 and resolved by name rather than hardcoded.
    """
    idx: dict[str, Path] = {}
    if not root.is_dir():
        return idx
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith((".", "_")):
            continue
        idx.setdefault(d.name, d)
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            idx.setdefault(sub.name, sub)
    return idx


def _episode_frames(ep_dir: Path) -> list[Path]:
    """All frame images of an episode, in filename order (tour order)."""
    return sorted(
        (p for p in ep_dir.rglob("*")
         if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda p: str(p.relative_to(ep_dir)))


def load(root: str | Path, seed: int = 20260911,
         n_frames: int = DEFAULT_N_FRAMES,
         limit: int | None = None) -> list[Sample]:
    """Load HM-EQA as public/private Sample pairs.

    `seed` is accepted for interface uniformity and is deliberately UNUSED:
    frame sampling is the deterministic uniform-midpoint scheme, so two
    loads with different seeds are identical and a split plan is the only
    place randomness belongs.
    """
    from .vsi import uniform_frame_indices

    root = Path(root)
    rows = _load_questions(root)
    ep_dirs = _index_episode_dirs(root)

    samples: list[Sample] = []
    missing_eps: list[str] = []
    n_empty_ep = 0
    n_bad_row = 0

    for r in rows:
        ep_hist = str(r.get("episode_history") or "")
        if not ep_hist.startswith(KEEP_PREFIX):
            continue
        ep_id = ep_hist.split("/")[-1].strip()
        question = str(r.get("question") or "").strip()
        gold = str(r.get("answer") or "").strip()
        qid = str(r.get("question_id") or "").strip()
        category = str(r.get("category") or "").strip()
        if not ep_id or not question or not gold:
            n_bad_row += 1
            continue

        ep_dir = ep_dirs.get(ep_id)
        if ep_dir is None:
            # Frames for this episode have not landed yet: counted and
            # skipped, so a partial extraction still loads.
            if ep_id not in missing_eps:
                missing_eps.append(ep_id)
            continue
        frames = _episode_frames(ep_dir)
        if not frames:
            n_empty_ep += 1
            if ep_id not in missing_eps:
                missing_eps.append(ep_id)
            continue

        idx = uniform_frame_indices(len(frames), min(n_frames, len(frames)))
        pooled = [str(frames[i]) for i in idx]

        item_id = f"{DOMAIN}/{qid or f'{ep_id}-{len(samples)}'}"
        public = PublicItem(
            item_id=item_id, domain=DOMAIN, question=question,
            media_ids=tuple(pooled),
            meta={
                "category": category,
                "episode_id": ep_id,
                # TRUE total frame count of the episode on disk, never the
                # number sampled.
                "n_frames": len(frames),
                "n_frames_pooled": len(pooled),
                "frame_indices": list(idx),
                "sampling_scheme": "uniform_midpoint_v1",
            })
        private = PrivateItem(
            item_id=item_id, answer=gold, answer_type="string",
            # OpenEQA releases no frame-level evidence for a question, so
            # the oracle / probe conditions cannot run here.
            evidence_source="none",
            meta={"category": category, "episode_id": ep_id,
                  "question_id": qid, "episode_history": ep_hist})
        samples.append(Sample(public=public, private=private,
                              group_key=ep_id))
        if limit and len(samples) >= limit:
            break

    from collections import Counter
    cat_dist = Counter(s.public.meta["category"] for s in samples)
    print(f"[{DOMAIN}] {len(samples)} HM-EQA questions over "
          f"{len({s.group_key for s in samples})} episodes; "
          f"categories: {dict(sorted(cat_dist.items()))}")
    if missing_eps or n_bad_row or n_empty_ep:
        print(f"[{DOMAIN}] skipped {len(missing_eps)} episode(s) with no "
              f"frames on disk ({', '.join(missing_eps[:5])}"
              f"{'...' if len(missing_eps) > 5 else ''}), "
              f"{n_bad_row} malformed rows, {n_empty_ep} empty frame dirs")
    return samples


_LLM_MATCH_PROMPT = (
    "You are grading an AI assistant's free-form answer to a question about "
    "a home the assistant toured. You are given the QUESTION, the "
    "ground-truth GOLD answer, and the assistant's MODEL answer.\n\n"
    "Mark the model answer \"correct\" if it matches the MEANING of the gold "
    "answer: equivalent wording, synonyms and reasonable paraphrases count, "
    "and extra correct detail does not hurt. Mark it \"incorrect\" if it "
    "contradicts the gold, names a different object/place/attribute, is too "
    "vague to distinguish the gold answer, or says it cannot answer. When "
    "the gold accepts several phrasings, any one of them is correct.\n\n"
    "QUESTION: {question}\n"
    "GOLD ANSWER: {gold}\n"
    "MODEL ANSWER: {answer}\n\n"
    "Reply with a JSON object: {{\"label\": \"correct\"}} or "
    "{{\"label\": \"incorrect\"}}")


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    """Official LLM-Match protocol: an independent judge (the OPTIMIZER,
    never the executor) marks the answer correct/incorrect; returns 1.0/0.0.
    """
    ans = (answer_text or "").strip()
    gold = str(private.answer or "").strip()
    if not ans or not gold:
        return 0.0
    prompt = _LLM_MATCH_PROMPT.format(
        question=public.question[:2000], gold=gold[:1000], answer=ans[:4000])
    try:
        from ..models import user_message
        resp = _judge().call([user_message(prompt)], max_tokens=256,
                             use_cache=False)
        m = re.search(r"\{.*\}", resp.content or "", re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
        return 1.0 if str(obj.get("label", "")).strip().lower() == "correct" \
            else 0.0
    except Exception:                                       # noqa: BLE001
        return 0.0


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    t = _PUNCT.sub(" ", str(text).lower())
    return " ".join(_ARTICLES.sub(" ", t).split())


def score_offline(answer_text: str, private: PrivateItem,
                  public: PublicItem) -> float:
    """Judge-free self-check: normalized exact match, 1.0/0.0.

    STRICTER than LLM-Match (a correct paraphrase scores 0), so it is a
    floor for smoke tests -- `score_offline(gold)` must be 1.0 -- and never
    a reported metric.
    """
    gold = str(private.answer or "")
    return 1.0 if _normalize(answer_text) == _normalize(gold) and \
        _normalize(gold) else 0.0
