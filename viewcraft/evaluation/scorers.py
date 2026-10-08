"""Deterministic scorers. No model acts as judge here.

Each scorer follows the source benchmark's own definition. Where ViewCraft
must deviate from an official protocol, the deviation is documented in the
function docstring and surfaced in RESULTS.md rather than hidden.
"""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower().rstrip(".")


def extract_choice_letter(raw: str, n_options: int) -> int | None:
    """Find a option letter (A..) in a free-form reply. None if absent.

    Looks for an explicit 'Answer: X' first, then a standalone letter, so a
    letter appearing inside prose does not outrank the declared answer.
    """
    letters = "".join(chr(ord("A") + i) for i in range(n_options))
    m = re.search(rf"answer\s*[:：]\s*\(?([{letters}])\)?\b", raw or "",
                  re.IGNORECASE)
    if m:
        return ord(m.group(1).upper()) - ord("A")
    m = re.search(rf"(?:^|[\s(\[])([{letters}])(?:[)\].,:]|$)", raw or "",
                  re.IGNORECASE | re.MULTILINE)
    if m:
        return ord(m.group(1).upper()) - ord("A")
    return None


def match_multiple_choice(raw: str, options: list[str],
                          correct_index: int) -> float:
    """1.0 if the reply selects the correct option, else 0.0.

    Resolution order: declared letter, then exact option text, then unique
    substring containment. Options are evaluated in the order presented; they
    are never re-sorted.
    """
    n = len(options)
    idx = extract_choice_letter(raw, n)
    if idx is not None and 0 <= idx < n:
        return 1.0 if idx == correct_index else 0.0

    r = _norm(raw)
    if not r:
        return 0.0

    exact = [i for i, o in enumerate(options) if _norm(o) == r]
    if len(exact) == 1:
        return 1.0 if exact[0] == correct_index else 0.0

    contained = [i for i, o in enumerate(options) if _norm(o) and _norm(o) in r]
    if len(contained) == 1:
        return 1.0 if contained[0] == correct_index else 0.0

    return 0.0


# ---------------------------------------------------------------------------
# ChartQA relaxed accuracy


def relaxed_accuracy(pred: str, gold: str, tolerance: float = 0.05) -> float:
    """ChartQA relaxed correctness.

    Numeric answers are correct within a 5% RELATIVE tolerance (the official
    definition); non-numeric answers require an exact normalized match. A
    fixed absolute epsilon would be wrong and is deliberately not used.
    """
    p, g = _norm(pred), _norm(gold)
    gnum, pnum = _to_float(g), _to_float(p)
    if gnum is not None and pnum is not None:
        if gnum == 0:
            return 1.0 if pnum == 0 else 0.0
        return 1.0 if abs(pnum - gnum) / abs(gnum) <= tolerance else 0.0
    return 1.0 if p == g else 0.0


def _to_float(s: str) -> float | None:
    if s is None:
        return None
    t = s.strip().replace(",", "").replace("%", "").replace("$", "")
    try:
        return float(t)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# ANLS -- the official metric of the DocVQA family (DocVQA, MP-DocVQA,
# InfographicVQA, DUDE)


def _levenshtein(a: str, b: str) -> int:
    """Edit distance, iterative two-row DP.

    Written out rather than pulled from a library because ANLS is the
    headline metric of three domains here and it must not silently change
    when an optional dependency is absent or upgraded.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,       # insertion
                           prev[j - 1] + (ca != cb)))   # substitution
        prev = cur
    return prev[-1]


def anls(pred: str, golds, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity, official DocVQA definition.

    For one question with a list of acceptable gold strings:

        NL(p, g) = edit_distance(p, g) / max(len(p), len(g))
        s        = max over golds of (1 - NL)
        ANLS     = s if s >= threshold else 0.0

    The THRESHOLD IS LOAD-BEARING and is the reason this is not just a
    similarity: answers closer than 0.5 score their similarity (so OCR-level
    slips are partially credited), and anything further away scores a hard
    zero rather than a small positive. Dropping the threshold would make a
    wholly wrong answer worth ~0.2 on average and inflate every reported
    number.

    Comparison is on lowercased, whitespace-collapsed strings, which is what
    the official evaluation does; it is NOT otherwise normalized (no
    punctuation stripping, no number canonicalisation), because the released
    gold lists already enumerate the accepted surface forms.

    `golds` is a list because the DocVQA family releases MULTIPLE accepted
    answers per question ("University of California", "university of
    california, san diego"); scoring against only the first would mark
    correct answers wrong.
    """
    if isinstance(golds, str):
        golds = [golds]
    p = _norm(pred)
    best = 0.0
    for g in golds:
        gg = _norm(str(g))
        if not p and not gg:
            best = max(best, 1.0)
            continue
        denom = max(len(p), len(gg))
        if denom == 0:
            continue
        sim = 1.0 - _levenshtein(p, gg) / denom
        best = max(best, sim)
    return best if best >= threshold else 0.0


# ---------------------------------------------------------------------------
# VSI-Bench


def mra(pred: str, gold: str,
        thresholds: tuple[float, ...] = tuple(
            round(x, 2) for x in [0.5 + 0.05 * i for i in range(10)])) -> float:
    """Mean Relative Accuracy, as implemented in lmms-eval's vsibench utils.

    MRA is the mean over thresholds theta of 1[relative error < 1 - theta].
    This is NOT max(0, 1 - relative_error); that substitution is explicitly
    avoided.
    """
    p, g = _to_float(pred), _to_float(gold)
    if p is None or g is None or g == 0:
        return 0.0
    rel_err = abs(p - g) / abs(g)
    hits = sum(1 for t in thresholds if rel_err < 1.0 - t)
    return hits / len(thresholds)


def exact_numeric(pred: str, gold: str) -> float:
    p, g = _to_float(pred), _to_float(gold)
    if p is None or g is None:
        return 0.0
    return 1.0 if abs(p - g) < 1e-9 else 0.0


# ---------------------------------------------------------------------------
# NExT-GQA grounding


def temporal_iou(pred: tuple[float, float],
                 gold_spans: list[tuple[float, float]]) -> float:
    """Best IoU of a predicted span against any gold span.

    A question may have several disjoint evidence segments; taking only the
    first would misscore, so the max over all spans is used.
    """
    from ..schema import TimeSpan
    if not gold_spans:
        return 0.0
    p = TimeSpan(*pred)
    return max(p.iou(TimeSpan(*g)) for g in gold_spans)


def grounded_qa_accuracy(qa_correct: float, iou: float,
                         iou_threshold: float = 0.5) -> float:
    """NExT-GQA's joint metric: correct AND adequately grounded.

    The source defines Acc@GQA as the fraction of questions answered
    correctly whose grounding also passes the IoU threshold. Reported
    alongside, never instead of, the separate QA and grounding numbers.
    """
    return 1.0 if (qa_correct >= 1.0 and iou >= iou_threshold) else 0.0


# ---------------------------------------------------------------------------
# GUI grounding (ScreenSpot family)


def parse_point(raw: str) -> tuple[float, float] | None:
    """Pull a click point out of the executor's answer text.

    ScreenSpot's own metric needs a POINT, so the answer is not a letter or a
    phrase and cannot go through the existing parsers. Accepted forms, in the
    order they are tried:

        (0.42, 0.87)      [0.42, 0.87]      0.42, 0.87
        x=0.42 y=0.87     (411, 830)        411 830

    Two decisions that change scores and so are made explicitly rather than
    left to a regex:

    * COORDINATES ARE NORMALISED [0,1] IN THE CONTRACT, but a model that has
      been told the image is 1920 px wide will sometimes answer in pixels
      anyway. A pair where either value exceeds 1.0 is therefore treated as
      pixels and divided by the image size by the CALLER, which is the only
      place that knows it. Returning "pixels or fraction, you decide" would
      push that guess into the scorer where the image size is not available.
      The flag is the magnitude itself: anything > 1.0 cannot be a fraction.
    * THE LAST PAIR WINS. Models narrate ("the icon near (0.1, 0.2) is not it;
      the target is at (0.8, 0.4)"), and taking the first match scores the
      rejected candidate. `note` is a separate JSON field, so the answer field
      should hold one pair, but when it holds several the final one is the
      model's conclusion.
    """
    import re
    if not raw:
        return None
    # A number, optionally signed, decimal or integer.
    num = r"[-+]?\d+(?:\.\d+)?"
    pairs = re.findall(rf"({num})\s*[,;xy= ]{{1,4}}\s*({num})", str(raw))
    if not pairs:
        pairs = re.findall(rf"({num})\D+({num})", str(raw))
    if not pairs:
        return None
    a, b = pairs[-1]
    try:
        return (float(a), float(b))
    except ValueError:
        return None


def point_in_box(raw: str, boxes, img_w: int, img_h: int) -> float:
    """ScreenSpot's metric: 1.0 if the predicted click lands in a gold box.

    This is the benchmark's own definition (ScreenSpot / ScreenSpot-Pro /
    UI-Vision all report "point-in-box accuracy"), not a variant chosen here:
    a prediction counts iff the point falls inside the ground-truth bounding
    box. No IoU, no distance tolerance, no partial credit -- a click either
    hits the control or misses it, which is also what an agent experiences.

    `boxes` are `Region`s in normalised coordinates. The point may arrive
    normalised or in pixels; a value above 1.0 is unambiguously pixels (a
    fraction cannot exceed 1), so it is converted here where `img_w`/`img_h`
    are in scope.
    """
    pt = parse_point(raw)
    if pt is None or not boxes:
        return 0.0
    x, y = pt
    if x > 1.0 or y > 1.0:
        if not (img_w and img_h):
            return 0.0
        x, y = x / img_w, y / img_h
    # A point outside the frame is a miss, not an error: clamping it to the
    # edge would silently move a wild guess onto a border control.
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return 0.0
    for b in boxes:
        if b.x0 <= x <= b.x1 and b.y0 <= y <= b.y1:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# MiniWoB


def miniwob_outcome(raw_reward: float, terminated: bool, truncated: bool,
                    infra_error: bool, wall_timeout: bool) -> dict:
    """Separate env success from reward and from every failure mode.

    Success requires normal termination with positive reward. Truncation,
    wall-clock timeout and infrastructure faults are distinct outcomes and are
    never folded into 'wrong answer'.
    """
    if infra_error:
        status = "infra_error"
    elif wall_timeout:
        status = "wall_timeout"
    elif truncated:
        status = "truncated"
    elif terminated:
        status = "terminated"
    else:
        status = "unfinished"
    success = 1.0 if (status == "terminated" and raw_reward > 0) else 0.0
    return {"status": status, "success": success, "raw_reward": raw_reward}


# ---------------------------------------------------------------------------
# MMLongBench-Doc


def mmlongbench_match(pred: str, gold: str, answer_format: str) -> float:
    """Format-aware match for MMLongBench-Doc.

    The official protocol uses an LLM to extract a short answer from a verbose
    reply, then compares by answer_format. Here extraction is done by a frozen
    prompt with the executor model (a single call, recorded), and this function
    performs the comparison step only. The difference from the official fixed
    judge is declared in RESULTS.md.
    """
    p, g = _norm(pred), _norm(gold)
    if answer_format == "Int":
        pf, gf = _to_float(p), _to_float(g)
        return 1.0 if (pf is not None and gf is not None
                       and int(round(pf)) == int(round(gf))) else 0.0
    if answer_format == "Float":
        pf, gf = _to_float(p), _to_float(g)
        if pf is None or gf is None:
            return 0.0
        if gf == 0:
            return 1.0 if abs(pf) < 1e-9 else 0.0
        return 1.0 if abs(pf - gf) / abs(gf) <= 0.01 else 0.0
    if answer_format == "List":
        return 1.0 if _list_key(p) == _list_key(g) else 0.0
    if answer_format == "None":
        # unanswerable: the model must abstain
        return 1.0 if _is_abstention(p) else 0.0
    return 1.0 if p == g else 0.0


def _list_key(s: str) -> tuple[str, ...]:
    parts = [x.strip() for x in re.split(r"[,;]|\band\b", s) if x.strip()]
    return tuple(sorted(parts))


# Phrasings that count as declining to answer. Deliberately wide: a model
# asked to abstain almost never replies with the literal benchmark string
# "not answerable", and the original three-marker list scored real
# abstentions ("the document does not state this", "cannot be determined
# from the document") as WRONG ANSWERS. On MMLongBench-Doc that error is
# worth up to 22% of the benchmark, because a fifth of its questions are
# unanswerable by construction and score exactly this predicate.
#
# Checked against every released gold answer: of the 847 answerable golds,
# only one ("N/A", an answer_format=="Str" item whose true answer really is
# the string "N/A") matches any marker. Callers therefore test BOTH sides --
# an abstention-looking prediction against an abstention-looking gold is an
# ordinary string comparison, not an abstention -- so that item still scores
# correctly.
ABSTENTION_MARKERS = (
    "not answerable", "unanswerable", "cannot be answered",
    "not mentioned", "no information", "not stated", "no mention",
    "does not state", "does not mention", "does not contain",
    "does not provide", "does not specify", "not provided",
    "not specified", "not found", "not available", "not discussed",
    "not reported", "not given", "cannot be determined",
    "cannot determine", "can not be determined", "unable to determine",
    "n/a", "none",
)


def is_abstention(s: str) -> bool:
    """True when a reply declines to answer rather than answering.

    Takes an already-normalized string (see `_norm`). Substring matching,
    not equality: the markers appear inside sentences.
    """
    return any(m in (s or "") for m in ABSTENTION_MARKERS)


def _is_abstention(s: str) -> bool:
    return is_abstention(s)
