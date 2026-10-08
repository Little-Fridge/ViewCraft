"""CharXiv adapter (D13): reasoning over a real scientific figure.

Why this domain is in the study: it is the hardest SINGLE-IMAGE domain
available, and it is here as difficulty insurance rather than as a test of
the selection half of the mechanism. Every other image domain in the set is
saturated on this executor (raw 0.76-0.96), which leaves an optimizer four to
twenty wrong items per hundred and nothing to repair. CharXiv's reasoning
split is published at ~0.40 for frontier models, so there is real headroom.

WHAT IT CAN AND CANNOT EXERCISE -- stated plainly, because it bears on how
the result should be read. The mechanism evolves (a) WHICH source units are
shown, (b) how each is rendered, and (c) how they are composed. CharXiv ships
ONE figure per question, so `phi.select_expr` has exactly one legal answer
and (a) and (c) are inert: only rendering is under test. That is a weaker
exercise of the mechanism than any multi-page or multi-frame domain, and a
flat result here would be much less informative than a flat result on
mpdocvqa. It earns its place only on difficulty.

What makes the figures hard is that they are REAL arXiv figures, not
generated plots: multi-panel layouts (`num_subplots` up to dozens), unlabelled
sub-panels referred to positionally ("subplot (c)"), legends that overlap the
data, and values that must be read off an axis rather than from a data
label. The reasoning questions are written to require a derived quantity --
a comparison, a count of crossings, a difference between two series -- so a
model that merely captions the figure fails.

Released layout: `val.parquet`, 1,000 rows, one row per FIGURE. Each row
carries FOUR descriptive questions (`descriptive_q1..q4` are integer question
TEMPLATE IDS, with `descriptive_a1..a4` their answers) and exactly ONE
reasoning question (`reasoning_q` / `reasoning_a`). `image` is a
struct<bytes, path> that embeds the figure, so images.zip is not needed to
load the domain.

ONLY THE REASONING QUESTION IS LOADED, one item per figure, 1,000 items.
The descriptive half is deliberately excluded: its questions are not released
as text at all, only as template ids that must be rehydrated from the
authors' separate template file, and the CharXiv paper reports the two halves
as SEPARATE headline numbers (descriptive is the easy half, ~0.8, and folding
it in would raise the baseline out of the useful band and hide the reasoning
signal). `descriptive_*` columns are left unread rather than silently mixed.

VAL SPLIT ONLY. `test.parquet` ships alongside it but its answers are
withheld for the leaderboard, so a test-backed domain would score every item
against an empty gold -- the same trap `blink` documents.

SPLIT UNIT
----------
`group_key` is the FIGURE. 979 of the 1,000 rows are distinct arXiv papers
and each figure carries exactly one reasoning question, so grouping by figure
is already almost one-item-per-group; it is done explicitly anyway so that
the handful of figures drawn from the same paper cannot straddle gate and
test.

SCORING: `charxiv_reasoning_match_v1`
-------------------------------------
Per-item binary, unweighted mean.

ONE DECLARED DEVIATION, and it is the significant one: the official CharXiv
protocol scores the reasoning split with GPT-4o AS A JUDGE, prompted with the
question, the gold and the model's reply. No model acts as judge anywhere in
ViewCraft, because a judged score cannot be reproduced from the ledger, so
the comparison here is DETERMINISTIC (`_match` below): the reply is reduced
to a span, then compared numerically when the gold is a number (exact after
unit cleaning) and by normalized exact match otherwise, with a containment
fallback for the short-label golds ("(b) OPT", "lambda_L = 0.13") where the
model's span legitimately carries the label plus its description.

Direction of the error is knowable and is stated rather than hidden: a
deterministic comparison is STRICTER than an LLM judge, which accepts
paraphrases ("the blue one" for "Joint-CNN"). Reported CharXiv numbers here
are therefore a LOWER BOUND on the official metric and are not comparable to
the published leaderboard. They remain fully comparable ACROSS ARMS, which
is what this study measures, because the identical comparison is applied to
every arm and condition.

`reasoning_a_type` (released as 1-4) is carried through to
`private.meta["answer_type_released"]` so a per-type breakdown can be taken
later, but it is NOT used to branch the comparison: the observed golds mix
numbers and short labels within the same type id, so branching on it would
be a guess about the release's semantics rather than a faithful reading.
Numeric-vs-text is decided from the GOLD ITSELF, which is unambiguous.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "charxiv"

# val only: test answers are withheld for the leaderboard.
VAL_NAME = "val.parquet"

_ANSWER_DECL = re.compile(
    r"^[\s>*#\-`'\"]*(?:final\s+)?answer\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE)


def load(root: str | Path, seed: int = 20260911,
         limit: int | None = None) -> list[Sample]:
    """Load the CharXiv val reasoning questions as public/private pairs.

    `seed` is accepted for interface uniformity and is deliberately UNUSED:
    this adapter makes no random choice, so two loads are identical.

    One item per figure (the single reasoning question). The four descriptive
    questions on each row are not loaded -- see the module docstring.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    p = root / VAL_NAME
    if not p.is_file():
        cands = sorted(root.rglob(VAL_NAME))
        if not cands:
            raise FileNotFoundError(f"{DOMAIN}: no {VAL_NAME} under {root}")
        p = cands[0]

    cols = ["image", "category", "year", "original_id", "figure_path",
            "num_subplots", "reasoning_q", "reasoning_a", "reasoning_a_type",
            "reasoning_q_source"]

    samples: list[Sample] = []
    skipped = 0
    for i, row in enumerate(pq.read_table(p, columns=cols).to_pylist()):
        question = str(row.get("reasoning_q") or "").strip()
        gold = str(row.get("reasoning_a") or "").strip()
        if not question or not gold:
            # No question or no gold: unscoreable, dropped not guessed.
            skipped += 1
            continue

        blob = row.get("image")
        if isinstance(blob, dict):
            blob = blob.get("bytes")
        if not blob:
            skipped += 1
            continue
        fig = str(row.get("figure_path") or f"fig_{i}")
        path = write_media(DOMAIN, blob, Path(fig).name)

        category = str(row.get("category") or "unk").strip() or "unk"
        item_id = f"{DOMAIN}/{category}/{Path(fig).stem}_{i}"
        public = PublicItem(
            item_id=item_id, domain=DOMAIN, question=question,
            media_ids=(path,),
            meta={
                "category": category,
                "year": str(row.get("year") or ""),
                "original_id": str(row.get("original_id") or ""),
                # Panel count of the figure. Neutral shape information, the
                # single-image analogue of n_pages: a strategy may reasonably
                # render a 12-panel figure differently from a 1-panel one.
                "num_subplots": int(row.get("num_subplots") or 0),
                # Exactly one source unit per question. Recorded explicitly
                # so an analysis never has to infer from len(media_ids)
                # whether this domain could exercise selection at all.
                "n_units": 1,
            },
        )
        private = PrivateItem(
            item_id=item_id, answer=gold, answer_type="string",
            # CharXiv releases no bounding box for the evidence, so the
            # oracle / probe conditions cannot run here. Declared as "none"
            # rather than faked from the whole figure.
            evidence_source="none",
            meta={
                "category": category,
                # Released type id, carried for a post-hoc per-type
                # breakdown. NOT used to branch the comparison -- see the
                # module docstring.
                "answer_type_released": row.get("reasoning_a_type"),
                "question_source_released": row.get("reasoning_q_source"),
                "num_subplots": int(row.get("num_subplots") or 0),
            },
        )
        # Group by the FIGURE: one reasoning question per figure, but a few
        # figures come from the same paper, and grouping explicitly keeps a
        # paper from straddling gate and test.
        samples.append(Sample(public=public, private=private,
                              group_key=Path(path).stem))
        if limit and len(samples) >= limit:
            break

    print(f"[{DOMAIN}] {len(samples)} reasoning questions "
          f"(skipped {skipped} without a question, gold or figure)")
    return samples


def extract_span(raw: str) -> str:
    """Reduce a free-form reply to the span the comparison is made on.

    Prefers an "Answer:" declaration anchored to the START OF A LINE, so the
    word appearing inside prose does not outrank a real declaration; the last
    declaration wins when a model corrects itself. Otherwise the reply's last
    non-empty line, because a model that reasons before answering ends on its
    conclusion. Markdown emphasis is removed only when it WRAPS the whole
    span, so an asterisk that is part of the answer survives.
    """
    s = (raw or "").strip()
    if not s:
        return ""
    decls = _ANSWER_DECL.findall(s)
    span = decls[-1].strip() if decls else ""
    if not span:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        span = lines[-1] if lines else s
    for mark in ("**", "*", "`"):
        while (len(span) > 2 * len(mark) and span.startswith(mark)
               and span.endswith(mark)):
            span = span[len(mark):-len(mark)].strip()
    return span.strip()


def _norm(s: str) -> str:
    """Lowercase, collapse whitespace, drop a trailing period."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower().rstrip(".")


def _num(s: str):
    """Parse a bare number, tolerating commas, %, $ and a unit suffix."""
    t = (s or "").strip().replace(",", "").replace("%", "").replace("$", "")
    m = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
    if not m:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _match(pred_span: str, gold: str) -> float:
    """Deterministic stand-in for the official GPT-4o judge.

    Numeric golds compare as numbers (so "94" and "94.0" agree and a model
    that writes "94 crossings" is reduced to 94 first). Everything else is a
    normalized exact match, with a CONTAINMENT fallback in one direction
    only -- gold inside prediction -- because CharXiv's short-label golds
    ("(c)", "(b) OPT") are routinely answered as "subplot (c)" or
    "(b) OPT, the orange line". The reverse direction is deliberately not
    allowed: a prediction inside the gold would let a bare "2" match the
    gold "2.5" and credit a wrong answer.
    """
    p, g = _norm(pred_span), _norm(gold)
    if not p:
        return 0.0
    if p == g:
        return 1.0

    gn = _num(g)
    if gn is not None:
        pn = _num(p)
        if pn is None:
            # Pull a trailing number out of "about 94 crossings".
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                              p.replace(",", ""))
            pn = float(nums[-1]) if nums else None
        if pn is None:
            return 0.0
        return 1.0 if abs(pn - gn) < 1e-6 else 0.0

    # Text gold: require the whole gold to appear in the reply, and require
    # it to be a reasonably specific string, so a one-character gold cannot
    # be matched by accident.
    if len(g) >= 2 and g in p:
        return 1.0
    return 0.0


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Deterministic CharXiv reasoning match. No model acts as judge.

    Stricter than the official GPT-4o judge by construction, so the number
    is a lower bound on the published metric and is stated as such in the
    module docstring. Applied identically to every arm and condition.
    """
    return _match(extract_span(answer_text), str(private.answer))
