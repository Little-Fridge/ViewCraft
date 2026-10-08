"""DUDE adapter (D14): selecting pages in a LONG, heterogeneous document.

Why this domain is in the study: it is the longest-document domain available
here, and the one where selection is hardest to get right by luck. Its val
documents run to 112 pages (median 3, mean 5.6), and unlike MP-DocVQA the
evidence for one question can sit on SEVERAL pages at once -- the release
annotates `answers_page_bounding_boxes` with a page per box, and a list
question routinely spans two or three. At the domain's 8-image request cap a
112-page document is 7% visible per request, so the binding decision is
which pages enter the context at all, and a crop cannot recover evidence on
a page that was never materialised.

What it adds over the other two document domains, i.e. why three is not one
too many:

  mmlongdoc   long reports, 22% deliberately unanswerable, long-form
              reasoning. Headline mixes abstention with retrieval.
  mpdocvqa    <=20 pages, pure SINGLE-hop retrieval, one evidence page,
              every question answerable. The clean selection read.
  DUDE        heterogeneous real-world PDFs, MULTI-page evidence, and an
              unanswerable share of its own. The hard end of selection.

DUDE's documents are also visually unlike the other two: the corpus is drawn
from many sources and mixes scanned faxes, born-digital forms, slide decks
and typed reports, so a rendering choice that works on a clean report page
does not transfer. That is rendering pressure ON TOP OF selection pressure,
which neither of the others supplies.

Released layout, and the two-source join this domain needs:

  PDFs          `jordyvl/DUDE_loader`, inside the 21 GB
                `data/DUDE_train-val-test_binaries.tar.gz`, as
                `DUDE_train-val-test_binaries/PDF/<split>/<docId>.pdf`.
                Only `PDF/val/*` is extracted (748 files); the archive's
                bulk is an OCR/ directory this adapter does not use, and
                train/test PDFs are outside the protocol.
  annotations   NOT in the HF repo. The loader script points at Zenodo
                record 7763635, `2023-03-23_DUDE_gt_test_PUBLIC.json`
                (14 MB, 41,456 rows over train/val/test), which is fetched
                through the same corporate proxy and saved as
                `data/dude/DUDE_gt.json`. A row is

                    {questionId, question, answers, answer_type, docId,
                     answers_page_bounding_boxes, answers_variants,
                     data_split}

                `answers` and `answers_page_bounding_boxes` are PYTHON
                literals in string columns ("['3']", "[]"), not JSON, so
                they are parsed with ast.literal_eval.

VAL SPLIT ONLY. The 11,402 test rows in the same file have their `answers`
withheld for the live RRC leaderboard, so a test-backed domain would score
every item against an empty gold -- the same trap `blink` and `mpdocvqa`
document. val is 6,318 questions over 744 documents, all with golds.

THE UNANSWERABLE HALF
---------------------
831 of the 6,318 val questions (13.2%) carry `answer_type ==
"not-answerable"` and ship an EMPTY answer list. That is not a missing
value: it is a question the document deliberately cannot answer, and the
model is expected to decline. Those items score by ABSTENTION, using the
shared `scorers.is_abstention` predicate mmlongdoc's unanswerable branch
uses, so a model that always guesses scores 0 on 13% of the benchmark. They
are KEPT rather than dropped -- removing them would change the denominator
of any published comparison -- and `aggregate()`-style reading should split
answerable from unanswerable rather than quoting one mean, for the same
reason mmlongdoc's docstring gives.

PAGE POOL, AND WHAT IS CAPPED
-----------------------------
`PublicItem.media_ids` carries the page PNGs and `meta["n_pages"]` carries
the TRUE total page count read from the PDF, never the number materialised.

ONE CAP, DECLARED: the pool is capped at `MAX_POOL_PAGES` = 64 pages. It is
NOT a truncation to the first 64 -- that would delete the back of a long
document -- but a uniform-midpoint subsample across the WHOLE document
(`mmlongdoc.uniform_page_indices`, byte-identical to the scheme vsi.py uses
for frames), so a capped pool still spans front to back and a selection
expression like `range(n - n // 8, n)` can still reach the final pages.
Reason for the number: every pooled page is rasterised at load, and the
distribution here is extremely skewed -- 745 of 748 val documents are 20
pages or fewer, and exactly 3 exceed it (112, and two in the 21-50 band).
A cap of 64 therefore binds on ONE document in the entire split while
bounding the worst-case render cost, and `pool_is_complete` plus the true
`n_pages` make the single capped item visible to any analysis. mmlongdoc
uses 128 because its documents are genuinely long (median 28, max 468);
using 128 here would change nothing except render time on one file.

`min_pages` (default 2) drops the 214 single-page documents. A single-page
item cannot exercise selection AT ALL -- `phi.select_expr` has exactly one
legal answer -- so it would dilute the domain with what is effectively
single-image DocVQA. The filter is on the document's REAL page count, fixed
before any score was measured, and the full pool is recoverable with
`min_pages=1`.

Pages render at `PAGE_LONG_EDGE` = 2048 px, deliberately ABOVE the domain's
1024 px overview budget, for the reason mmlongdoc documents: phi renders
crops by re-reading the source file, so a page stored at the overview size
could only ever be upscaled and a zoom into a form field would return
interpolation instead of detail.

Rasterising is delegated to `mmlongdoc.render_pages`, which is
content-addressed by the sha256 of the PDF BYTES, so two questions on one
document share one render and a re-run never re-renders.

SPLIT UNIT
----------
`group_key` is the DOCUMENT (`docId`), never the question: a document
carries a median of 9 questions (max 20), so letting one PDF appear in both
gate and test would let a strategy tune on pages it is later measured on.

SCORING: `dude_anls_v1`
-----------------------
ANLS at threshold 0.5 -- `scorers.anls`, the official DocVQA-family metric
the DUDE leaderboard uses, unmodified -- against the full released answer
list, plus an ABSTENTION branch for the not-answerable items. The release
also ships `answers_variants`, additional accepted surface forms, and those
are folded into the gold list because the official evaluation accepts them;
ignoring them would mark correct answers wrong.

ONE DECLARED DEVIATION, the same one mpdocvqa declares: the official RRC
evaluation scores a submission of already-extracted short answers, because
its baselines are extractive models emitting one span. A generative VLM
replies in prose, and raw ANLS against a carrier sentence is near zero even
when the answer is right. The reply is therefore reduced by a DETERMINISTIC
extractor (`mpdocvqa.extract_span`: an "Answer:" declaration anchored to the
start of a line, else the last non-empty line, with markdown emphasis
removed only when it wraps the whole span) before the official ANLS. No LLM
judges anything. The extractor is strictly stricter than assuming a
pre-extracted span, and it is applied identically to every arm and
condition.

VERIFIED BY REPLAYING THE GOLD
------------------------------
The scorer is checked by feeding each item its own first gold answer back as
the prediction; a correct scorer returns 1.0 on every answerable item, and
an abstention string returns 1.0 on every unanswerable one.

This caught a real defect. The answerable branch refuses to give ANLS credit
to a reply that declines, so a model answering "not stated" cannot collect
partial credit for sharing characters with the gold ("none" vs "nine"). But
26 ANSWERABLE golds in this split are themselves abstention-shaped strings
-- "None" (18), "not found" (6), "N/a", "Does not specify Capital Fund" --
because the annotator's true answer is that the field is blank. Guarding on
the prediction alone scored those 25 items 0.0 AGAINST THEIR OWN GOLD, i.e.
unwinnable for every arm and a silent 0.6% cap on the headline. The guard
therefore tests BOTH SIDES, exactly as mmlongdoc's compare() does: when the
gold also looks like an abstention the comparison falls through to ordinary
ANLS. All answerable items now self-score 1.0, all unanswerable items score
1.0 on an abstention and 0.0 on a guess, and a garbage reply scores 0.0
everywhere.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "dude"

from ..paths import project_root

ROOT = project_root()
DATA = ROOT / "data" / "dude"
PDF_DIR = DATA / "DUDE_train-val-test_binaries" / "PDF" / "val"
GT_JSON = DATA / "DUDE_gt.json"

# See "PAGE POOL, AND WHAT IS CAPPED". Binds on exactly 1 of 748 val docs.
MAX_POOL_PAGES = 64
# Above the 1024 px overview budget on purpose: crops re-read this file, so
# storing at the overview size would make every zoom an upscale.
PAGE_LONG_EDGE = 2048
# Single-page documents cannot exercise selection; see the docstring.
MIN_PAGES_DEFAULT = 2

SPLIT = "val"
UNANSWERABLE_TYPE = "not-answerable"


def _parse_literal_list(raw) -> list:
    """Parse a released `answers` / `answers_variants` cell.

    These are PYTHON literals in a string column ("['3']"), so json.loads
    fails on every single-quoted value. A cell that will not parse yields []
    rather than raising.
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


def load(root: str | Path = DATA, seed: int = 20260911,
         min_pages: int = MIN_PAGES_DEFAULT,
         max_pool_pages: int = MAX_POOL_PAGES,
         long_edge: int = PAGE_LONG_EDGE,
         limit: int | None = None,
         render_now: bool = True) -> list[Sample]:
    """Load the DUDE val split as public/private Sample pairs.

    `seed` is accepted for interface uniformity and is deliberately UNUSED:
    page selection is the deterministic uniform-midpoint scheme, so two
    loads with different seeds are identical and a split plan is the only
    place randomness belongs.

    Questions whose PDF is absent are skipped with a printed count. Every
    document is rendered once and shared by all of its questions.
    """
    from .mmlongdoc import (_backend, _file_digest, _page_dir, page_count,
                            render_pages, uniform_page_indices)

    root = Path(root)
    gt = root / GT_JSON.name
    if not gt.is_file():
        raise FileNotFoundError(
            f"{DOMAIN}: no {GT_JSON.name} under {root}. The annotations are "
            "NOT in the HF repo: fetch Zenodo record 7763635 "
            "(2023-03-23_DUDE_gt_test_PUBLIC.json) via the proxy, as "
            "scripts/download_wave4.py does.")
    pdf_dir = root / "DUDE_train-val-test_binaries" / "PDF" / SPLIT
    if not pdf_dir.is_dir():
        raise FileNotFoundError(
            f"{DOMAIN}: no PDF/{SPLIT}/ under {root}; extract "
            f"PDF/{SPLIT}/* from DUDE_train-val-test_binaries.tar.gz")

    raw = json.loads(gt.read_text())
    rows = raw.get("data", raw) if isinstance(raw, dict) else raw
    rows = [r for r in rows if r.get("data_split") == SPLIT]
    # Stable order so item_ids and any limit are reproducible.
    rows.sort(key=lambda r: (str(r.get("docId") or ""),
                             str(r.get("questionId") or "")))

    doc_cache: dict[str, tuple] = {}
    samples: list[Sample] = []
    skipped_missing_pdf = 0
    skipped_short = 0
    skipped_empty = 0
    n_unanswerable = 0

    for r in rows:
        doc_id = str(r.get("docId") or "").strip()
        qid = str(r.get("questionId") or "").strip()
        question = str(r.get("question") or "").strip()
        atype = str(r.get("answer_type") or "").strip()
        if not doc_id or not qid or not question:
            skipped_empty += 1
            continue

        unanswerable = atype == UNANSWERABLE_TYPE
        golds = [str(x) for x in _parse_literal_list(r.get("answers"))
                 if str(x).strip()]
        # The release ships extra accepted surface forms; the official
        # evaluation accepts them, so folding them in is faithful and
        # ignoring them would mark correct answers wrong.
        golds += [str(x) for x in _parse_literal_list(r.get("answers_variants"))
                  if str(x).strip()]
        if not unanswerable and not golds:
            # Answerable but no gold: unscoreable, dropped not guessed.
            skipped_empty += 1
            continue

        pdf = pdf_dir / f"{doc_id}.pdf"
        if not pdf.is_file():
            skipped_missing_pdf += 1
            continue

        if doc_id not in doc_cache:
            try:
                n_pages = page_count(pdf)
            except Exception:                                 # noqa: BLE001
                n_pages = 0
            if n_pages <= 0:
                skipped_missing_pdf += 1
                doc_cache[doc_id] = None
                continue
            if n_pages < min_pages:
                doc_cache[doc_id] = None
                skipped_short += 1
                continue
            pool = uniform_page_indices(n_pages, min(max_pool_pages, n_pages))
            if render_now:
                paths, info = render_pages(pdf, pool, long_edge)
            else:
                digest = _file_digest(pdf)
                d = _page_dir(digest, long_edge)
                paths = tuple(str(d / f"page_{k:04d}.png") for k in pool)
                info = {"render_backend": _backend(), "page_digest": digest,
                        "long_edge": long_edge}
            doc_cache[doc_id] = (n_pages, pool, paths, info)
        cached = doc_cache[doc_id]
        if cached is None:
            # Document already rejected (missing, unreadable or too short).
            # Counted once at rejection, not once per question.
            continue
        n_pages, pool, paths, info = cached

        # Evidence PAGES come from the released bounding boxes, whose `page`
        # field is 0-based. PRIVATE only, for post-hoc binning ("did the
        # strategy that scored well actually look at the evidence page?");
        # never an input to selection, which would make it an oracle.
        ev_pages: list[int] = []
        for box_group in _parse_literal_list(r.get("answers_page_bounding_boxes")):
            group = box_group if isinstance(box_group, list) else [box_group]
            for b in group:
                if isinstance(b, dict) and "page" in b:
                    try:
                        pg = int(b["page"])
                    except (TypeError, ValueError):
                        continue
                    if 0 <= pg < n_pages and pg not in ev_pages:
                        ev_pages.append(pg)
        ev_pages.sort()

        if unanswerable:
            n_unanswerable += 1

        stratum = "unanswerable" if unanswerable else _bucket(n_pages)
        item_id = f"{DOMAIN}/{stratum}/{doc_id}_{qid[-12:]}"
        public = PublicItem(
            item_id=item_id, domain=DOMAIN, question=question,
            media_ids=paths,
            meta={
                "doc_id": doc_id,
                # TRUE total page count of the PDF, independent of how many
                # pages were materialised. A selection expression reads this
                # as `n_pages`.
                "n_pages": n_pages,
                "n_pages_pooled": len(paths),
                "page_indices": list(pool),
                "page_numbers": [p + 1 for p in pool],
                "pool_is_complete": len(pool) == n_pages,
                "page_long_edge_px": info["long_edge"],
                "render_backend": info["render_backend"],
                "sampling_scheme": "uniform_midpoint_v1",
                # The executor cannot guess from the question text that
                # declining is an allowed reply, and 13.2% of the split
                # requires it. This is the FACT that some questions are
                # unanswerable, never which ones.
                "may_be_unanswerable": True,
            },
        )
        private = PrivateItem(
            item_id=item_id,
            answer=golds,
            answer_type="not_answerable" if unanswerable else "string_list",
            evidence_pages=tuple(ev_pages),
            evidence_source="official" if ev_pages else "none",
            evidence_verified=bool(ev_pages),
            unanswerable=unanswerable,
            meta={"doc_id": doc_id, "question_id": qid,
                  "answer_type_released": atype, "n_pages": n_pages,
                  "answers_raw": list(golds)},
        )
        # Group by the DOCUMENT: a document carries a median of 9 questions,
        # so one PDF in both gate and test would let a strategy tune on
        # pages it is later measured on.
        samples.append(Sample(public=public, private=private,
                              group_key=doc_id))
        if limit and len(samples) >= limit:
            break

    print(f"[{DOMAIN}] {len(samples)} questions over "
          f"{len({s.group_key for s in samples})} documents "
          f"({n_unanswerable} unanswerable; dropped {skipped_short} docs with "
          f"<{min_pages} pages, {skipped_missing_pdf} questions with no "
          f"readable PDF, {skipped_empty} unscoreable)")
    return samples


def _bucket(n_pages: int) -> str:
    """Stratum label written into item_id as its second path segment.

    `scripts/three_conditions.py --sample stratified` buckets on exactly
    this segment, so without it a 60-item draw would be an arbitrary slice
    of the page-count distribution and the reported median pages/question
    would not be the domain's. Unanswerable items get their own stratum
    (assigned by the caller) so a draw cannot land 0 or 100% of them.
    """
    if n_pages <= 4:
        return "p02_04"
    if n_pages <= 8:
        return "p05_08"
    if n_pages <= 20:
        return "p09_20"
    return "p21_plus"


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official DUDE metric: ANLS 0.5 on answerable, abstention otherwise.

    The reply is reduced by the deterministic extractor first -- see the
    DEVIATION paragraph in the module docstring. No model acts as judge.
    """
    from ..evaluation.scorers import anls, is_abstention, _norm
    from .mpdocvqa import extract_span

    span = extract_span(answer_text)

    if private.unanswerable:
        # The model must DECLINE. Checked on the whole reply rather than the
        # extracted span: a model that explains at length why the document
        # does not say, then ends on a summary line, still declined, and
        # reducing to the last line can drop the refusal.
        return 1.0 if (is_abstention(_norm(answer_text))
                       or is_abstention(_norm(span))) else 0.0

    golds = private.answer
    if isinstance(golds, str):
        golds = [golds]
    if not golds:
        return 0.0
    # An abstention on an ANSWERABLE question is wrong, and must not be
    # allowed to score partial ANLS credit for happening to share characters
    # with the gold ("none" vs "nine").
    #
    # BUT the guard must test BOTH SIDES, exactly as mmlongdoc's compare()
    # does. 26 answerable golds in this split ARE abstention-shaped strings
    # -- "None" (18), "not found" (6), "N/a", "Does not specify Capital
    # Fund" -- because the annotator's true answer really is that the field
    # is blank. Guarding on the prediction alone scored those 25 items 0.0
    # against their OWN gold, i.e. unwinnable for every arm, which is the
    # "a property of the item mistaken for a defect of the candidate"
    # failure this project has hit four times. When the gold also looks like
    # an abstention the comparison falls through to ordinary ANLS and the
    # item stays winnable.
    if (is_abstention(_norm(span))
            and not any(is_abstention(_norm(str(g))) for g in golds)):
        return 0.0
    return anls(span, list(golds))
