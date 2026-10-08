"""MP-DocVQA adapter (D12): deciding WHICH page to look at, with the page
count released per question rather than inferred from a PDF.

Why this domain is in the study: like MMLongBench-Doc, and unlike every
single-image domain here, the binding decision is SELECTION -- which of a
document's pages enter the context at all -- not resolution. MP-DocVQA is the
multi-page extension of DocVQA: the same question style, but the evidence
lives on ONE page of a document that ships up to 20, and the model is never
told which. A crop cannot recover evidence on a page that was never
materialised, so a wrong selection is unrecoverable rather than merely blurry.

What it adds over mmlongdoc, i.e. why both are worth having: mmlongdoc's
questions are long-form reasoning over reports and 22% of them are
deliberately unanswerable, so its headline mixes an abstention test with a
retrieval one. MP-DocVQA is pure single-hop RETRIEVAL -- every question has an
answer, on exactly one page, and the released `answer_page_idx` says which.
That makes it the cleanest available read on the selection half of the
mechanism on its own: a score change here is a change in "did the right page
reach the model", with no abstention behaviour mixed in.

PAGE COUNT PER QUESTION -- the criterion this domain was chosen against
----------------------------------------------------------------------
MEASURED over the full 5,187-question val split: median 3 pages, mean 5.6,
max 20. The distribution is strongly bimodal -- 1,633 questions (31.5%) are
single-page, and 481 (9.3%) sit at the 20-page cap.

A single-page item cannot exercise selection AT ALL: there is one unit, so
`phi.select_expr` has exactly one legal answer and the item degenerates to
single-image DocVQA. Loading the whole split would therefore dilute the
domain with items that are, for this study's purpose, a different benchmark.

`min_pages` (default 2) drops the single-page items. At the default the
domain is 3,554 questions with a median of 6 pages. `min_pages=9` narrows to
the 1,155 questions with >8 pages (median 20) for a run that wants the
strong-multi-unit regime specifically. The filter is on the RELEASED page
count, a property of the item, decided before any score was measured; the
unfiltered pool is always recoverable with `min_pages=1`.

Released layout: `data/val-*.parquet`, one row per question:

    {questionId, question, doc_id, page_ids, answers, answer_page_idx,
     data_split, image_1 .. image_20}

`page_ids`, `answers` and `answer_page_idx` are PYTHON LITERALS in string
columns ("['pybv0228_p80']", "['0.28']", "0"), not JSON and not native
lists, so they are parsed with ast.literal_eval -- the single quotes make
json.loads fail on every non-empty value. The `image_N` columns are a FIXED
20-wide schema padded with nulls; the real page count is the number of
non-null ones, which is checked against len(page_ids) per row rather than
assumed equal.

VAL SPLIT ONLY. MP-DocVQA is a live RRC leaderboard task and the test split
ships with its answers withheld, so a test-backed domain would score every
item against an empty gold. Same trap `blink` documents for its own test
parquets.

NO CAP IS APPLIED. The release itself caps at 20 pages, which is below the
128-page pool cap mmlongdoc needs, so every page of every item is
materialised and `pool_is_complete` is True on every item. `meta["n_pages"]`
is still the TRUE released page count and is written from `len(page_ids)`
independently of how many images were materialised, so the field means the
same thing it means in mmlongdoc.

Pages are written VERBATIM by `media.write_media` -- the bytes the release
ships, not a re-encode -- so a crop re-reads the original pixels. Unlike
mmlongdoc there is no PDF and no rasteriser: the pages arrive as images
already, so `meta["render_backend"]` is "released_png" rather than a
library name.

SPLIT UNIT
----------
`group_key` is the DOCUMENT (`doc_id`), never the question: the 5,187
questions sit on 927 documents, so a document carries ~5.6 questions on
average and letting one document appear in both gate and test would let a
strategy tune on pages it is later measured on.

SCORING: `mpdocvqa_anls_v1`
---------------------------
ANLS at threshold 0.5 -- `scorers.anls`, the official DocVQA-family metric,
unmodified -- against the FULL released answer list. The release gives
several accepted surface forms per question ("University of California",
"university of california, san diego") and scoring against only the first
would mark correct answers wrong.

ONE DECLARED DEVIATION: the official RRC evaluation scores a submission file
of already-extracted short answers, because its baselines are extractive
models that emit exactly one span. A generative VLM instead replies in prose
("The value is 0.28"), and raw ANLS against such a reply is near zero even
when the answer is right -- the edit distance is dominated by the carrier
sentence. This adapter therefore applies a DETERMINISTIC short-answer
extraction (`extract_span` below: an "Answer:" declaration anchored to the
start of a line, else the reply's last non-empty line) BEFORE the official
ANLS, and no LLM is used as a judge anywhere. The extractor is strictly
stricter than the official protocol's assumption of a pre-extracted span, it
is applied identically to every arm and condition, and the ANLS computation
itself is untouched.

VERIFIED BY REPLAYING THE GOLD
------------------------------
The scorer is checked by feeding each item its OWN first gold answer as the
prediction: a correct scorer must then return exactly 1.0 on every item.
This caught a real defect -- reusing `mmlongdoc.extract_short_answer` strips
every asterisk as markdown emphasis, and four released golds contain literal
ones ("F*", "1(1)*", a table caption ending "Postmenopausal Women*"), so
those items scored 0.5-0.99 against their own gold while the model would
have been exactly right. `extract_span` therefore removes emphasis only when
it WRAPS the whole span. All 3,554 items now self-score 1.0, and a garbage
reply scores 0.0 on all of them.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "mpdocvqa"

# An explicit answer DECLARATION, anchored to the start of a line. Anchored
# for the reason mmlongdoc documents: "answer:" also occurs inside prose, and
# an unanchored search pulls a fragment out of a quoted sentence.
_ANSWER_DECL = re.compile(
    r"^[\s>*#\-`'\"]*(?:final\s+)?answer\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE)

# The release's own fixed schema width: image_1 .. image_20.
MAX_IMAGE_COLS = 20

# Default floor on released page count. 1-page items are pure single-image
# DocVQA and cannot exercise selection at all; see the module docstring.
MIN_PAGES_DEFAULT = 2

# val only: test answers are withheld for the live leaderboard.
VAL_GLOB = "val-*.parquet"


def _parse_literal_list(raw) -> list:
    """Parse a released `page_ids` / `answers` cell.

    These are PYTHON literals in a string column ("['0.28']"), so json.loads
    fails on every single-quoted value. Identical in intent to the helper of
    the same name in mmlongdoc.py; kept local so neither adapter's parsing
    can be changed by an edit aimed at the other.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    s = str(raw).strip()
    if not s or s in ("[]", "nan", "None"):
        return []
    try:
        out = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return list(out) if isinstance(out, (list, tuple)) else [out]


def load(root: str | Path, seed: int = 20260911,
         min_pages: int = MIN_PAGES_DEFAULT,
         limit: int | None = None) -> list[Sample]:
    """Load the MP-DocVQA val split as public/private Sample pairs.

    `seed` is accepted for interface uniformity and is deliberately UNUSED:
    this adapter makes no random choice. Every page of every kept item is
    materialised, so there is no sampling scheme and two loads are identical.

    `min_pages` filters on the RELEASED page count (default 2, dropping the
    31.5% of questions that are single-page and therefore cannot exercise
    selection). Pass 1 to recover the full official pool.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    pqs = sorted((root / "data").glob(VAL_GLOB)) or sorted(root.rglob(VAL_GLOB))
    if not pqs:
        raise FileNotFoundError(f"{DOMAIN}: no {VAL_GLOB} under {root}")

    img_cols = [f"image_{i}" for i in range(1, MAX_IMAGE_COLS + 1)]
    cols = ["questionId", "question", "doc_id", "page_ids", "answers",
            "answer_page_idx"] + img_cols

    samples: list[Sample] = []
    n_single_page = 0
    n_no_gold = 0
    n_short_media = 0

    for p in pqs:
        for row in pq.read_table(p, columns=cols).to_pylist():
            qid = str(row.get("questionId") or "").strip()
            question = str(row.get("question") or "").strip()
            doc_id = str(row.get("doc_id") or "").strip()
            page_ids = [str(x) for x in _parse_literal_list(row.get("page_ids"))]
            golds = [str(x) for x in _parse_literal_list(row.get("answers"))
                     if str(x).strip()]

            if not qid or not question or not doc_id or not page_ids:
                n_no_gold += 1
                continue
            # val carries gold answers; a row without one is unscoreable and
            # is dropped rather than guessed at.
            if not golds:
                n_no_gold += 1
                continue

            # n_pages is the TRUE released page count, taken from page_ids
            # and never from how many image columns happened to be non-null.
            n_pages = len(page_ids)
            if n_pages < min_pages:
                n_single_page += 1
                continue

            blobs = []
            for c in img_cols[:n_pages]:
                v = row.get(c)
                if isinstance(v, dict):
                    v = v.get("bytes")
                if v:
                    blobs.append(v)
            # The image columns are the only way to see a page, so an item
            # whose media is short of its own page_ids cannot be presented as
            # the document it claims to be. Counted and skipped rather than
            # silently scored against a truncated document.
            if len(blobs) != n_pages:
                n_short_media += 1
                continue

            paths = tuple(write_media(DOMAIN, b, f"{doc_id}_p{i}")
                          for i, b in enumerate(blobs))

            # Released answer page index, 0-based within this item's page
            # list. PRIVATE only -- it is post-hoc binning ("did the strategy
            # that scored well actually look at the evidence page?"), never
            # an input to selection, which would make it an oracle.
            try:
                ev_idx = int(str(row.get("answer_page_idx")).strip())
            except (TypeError, ValueError):
                ev_idx = -1
            ev_pages = (ev_idx,) if 0 <= ev_idx < n_pages else ()

            item_id = f"{DOMAIN}/{_bucket(n_pages)}/{doc_id}_{qid}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN, question=question,
                media_ids=paths,
                meta={
                    "doc_id": doc_id,
                    # TRUE total page count of the document for this
                    # question, independent of what was materialised. A
                    # selection expression reads this as `n_pages`.
                    "n_pages": n_pages,
                    "n_pages_pooled": len(paths),
                    "page_indices": list(range(n_pages)),
                    "page_numbers": [i + 1 for i in range(n_pages)],
                    # Always True here: the release caps at 20 pages, which
                    # is below any pool cap we would impose, so nothing is
                    # dropped and no subsampling scheme is applied.
                    "pool_is_complete": True,
                    # Pages arrive as images, not as a PDF: there is no
                    # rasteriser in this domain and the bytes are the
                    # release's own.
                    "render_backend": "released_png",
                    "sampling_scheme": "none_all_pages",
                },
            )
            private = PrivateItem(
                item_id=item_id,
                # The FULL accepted-answer list, not just the first: ANLS is
                # defined as the max over the released surface forms.
                answer=golds,
                answer_type="string_list",
                evidence_pages=ev_pages,
                evidence_source="official" if ev_pages else "none",
                evidence_verified=bool(ev_pages),
                meta={"doc_id": doc_id, "n_pages": n_pages,
                      "question_id": qid, "answers_raw": list(golds)},
            )
            # Group by the DOCUMENT: 5,187 questions sit on 927 documents, so
            # one document appearing in both gate and test would let a
            # strategy tune on pages it is later measured on.
            samples.append(Sample(public=public, private=private,
                                  group_key=doc_id))
            if limit and len(samples) >= limit:
                break
        if limit and len(samples) >= limit:
            break

    print(f"[{DOMAIN}] {len(samples)} questions "
          f"(dropped {n_single_page} with <{min_pages} pages, "
          f"{n_no_gold} without a gold, {n_short_media} with missing page "
          f"images)")
    return samples


def _bucket(n_pages: int) -> str:
    """Stratum label written into item_id as its second path segment.

    `scripts/three_conditions.py --sample stratified` buckets on exactly this
    segment, so without it a 60-item draw would be an arbitrary slice of the
    page-count distribution and the reported median pages/question would not
    be the domain's. The three buckets separate the regimes the mechanism
    cares about: few pages, many pages, and the 20-page cap.
    """
    if n_pages <= 4:
        return "p02_04"
    if n_pages <= 12:
        return "p05_12"
    return "p13_20"


def extract_span(raw: str) -> str:
    """Reduce a free-form reply to the span ANLS should be measured on.

    Same shape as `mmlongdoc.extract_short_answer` -- prefer an "Answer:"
    declaration anchored to the START OF A LINE (so the word appearing inside
    prose does not outrank a real declaration), else take the last non-empty
    line, because a model that reasons before answering ends on its
    conclusion. The last declaration wins when a model corrects itself.

    It is written here rather than reused because mmlongdoc's version strips
    EVERY asterisk as markdown emphasis, and DocVQA golds contain literal
    ones: the footnote markers "F*" and "1(1)*" and a table caption ending
    "...Postmenopausal Women*" are real released answers, and stripping the
    character cost those four items up to half their ANLS while the model was
    exactly right. Emphasis is therefore removed only when it WRAPS the whole
    span (a matched pair), which is what a model's markdown actually looks
    like, so an asterisk that is part of the answer survives.

    No numeric branch: DocVQA golds are spans ("University of California") as
    often as values, and pulling a trailing number out of one destroys it.
    """
    s = (raw or "").strip()
    if not s:
        return ""

    decls = _ANSWER_DECL.findall(s)
    span = decls[-1].strip() if decls else ""
    if not span:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        span = lines[-1] if lines else s

    # Only MATCHED wrappers, innermost last, so "**F***" -> "F*" but a
    # trailing footnote marker is never eaten on its own.
    for mark in ("**", "*", "`"):
        while (len(span) > 2 * len(mark) and span.startswith(mark)
               and span.endswith(mark)):
            span = span[len(mark):-len(mark)].strip()
    return span.strip()


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official DocVQA-family ANLS (threshold 0.5) over the released answers.

    The reply is passed through `extract_span` first -- see the DEVIATION
    paragraph in the module docstring. No model is used as a judge.
    """
    from ..evaluation.scorers import anls

    golds = private.answer
    if isinstance(golds, str):
        golds = [golds]
    return anls(extract_span(answer_text), list(golds))
