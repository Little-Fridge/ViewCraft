"""SlideVQA adapter: multi-hop numerical reasoning over slide decks, where
the evidence for ONE question is spread over 1-2 pages of a uniform 20-page
deck and the model is never told which.

Why this domain is in the study
-------------------------------
Like mmlongdoc and mpdocvqa the binding decision is SELECTION -- which of a
deck's pages enter the context at all -- but SlideVQA sharpens it in two
ways the other two cannot:

  * EVIDENCE CAN BE MULTI-PAGE. MP-DocVQA's answer lives on exactly ONE
    page (its `answer_page_idx` is a scalar), so a selector that retrieves
    one good page is done. 215 of the 1,652 val questions (13.0%) need TWO
    pages together -- a figure on one slide plus a number on another -- so
    retrieving any strict subset of the evidence caps the score no matter
    how good the reading is. It is the one document domain here that
    measures RECALL of the selection, not just precision.
  * THE POOL IS UNIFORMLY DEEP. Every val question ships exactly 20 pages
    (measured: count_pages == 20 on all 1,652 rows). There is no
    single-page degenerate tail to filter the way mpdocvqa's `min_pages`
    had to, so no page-count filter is applied and `pool_is_complete` is
    True on every kept item.

Slides are also visually distinct from report pages: dense infographics,
charts and diagrams rather than prose, so the read half leans harder on the
visual channel than DocVQA-family scans do.

DATA SOURCE -- two HF repos under one root, joined on document_id
-----------------------------------------------------------------
The openxlab TSV route and the Ahren09/SlideVQA_Image route were both
abandoned (the latter's `val/` ships only 1-2 pages per question, not the
full 20-page deck). The released data now lives in:

    root/qa/      AHS-uni/slidevqa-qa      data/val-00000-of-00001.parquet
    root/corpus/  AHS-uni/slidevqa-corpus  data/train-000{00..12}-of-00013.parquet

The QA parquet has one STRUCTURED row per question (1,652 val rows):

    id        "00mlearn2011glahn-...-9_95-0"  (doc_id + "-" + ordinal)
    question  {id, text, type, tags}
    document  {id, type, count_pages, tags}
    evidence  {pages: list[int32], sources, tags}
    answer    {type, variants: list[string], rationale, format, tags}

The corpus shards carry one row per PAGE: {document_id, page_number,
image} -- measured on the release (2026-09-19): 13 shards, ~4k rows each,
`page_number` is int32 and 0-BASED (first page of a deck is 0), and `image`
is the HF Image-feature struct {bytes, path} holding the page's JPEG bytes
(a raw binary column is also accepted). Pages decode to ~1024px-long-edge
JPEGs. The join is QA `document.id` == corpus `document_id`. Pages are
materialised through `media.write_media` VERBATIM (content-addressed,
idempotent, so a re-run is cheap) and `media_ids` are ordered by
page_number, so page_index == position in the tuple. Only the 300
document_ids that val questions reference are materialised (6,000 pages);
the rest of the corpus (2,619 decks total, incl. train/test) is skipped as
it streams past.

VAL SPLIT ONLY. The test QA parquet (2,215 rows) does ship answers and
evidence pages, and the corpus likely covers its decks too, but the val
split alone already gives 1,652 questions over 300 decks and keeping one
released split per domain matches mpdocvqa's precedent. Nothing in the
loader blocks adding test later.

VERIFIED PROPERTIES OF THE VAL RELEASE (measured on the QA parquet):

  * evidence.pages is 0-BASED: 0 occurs, the max is 19, and no value
    reaches its row's count_pages (20). The load still re-derives the base
    per load (a 0 anywhere in the column means 0-based) so a silent
    re-release changes the answer rather than corrupting it.
  * every row is answerable (answer.type == 0) with exactly ONE variant,
    so `all_answers` is usually a singleton; the multi-gold path is kept
    because the schema allows it.
  * answer.format is 93% "other" (7) and 7% "list" (5); question.type is
    "other" (7) on EVERY row, so the release's own type labels carry no
    signal in val. `reasoning_type` is therefore DERIVED:
    len(evidence.pages) > 1 -> "multi-hop", else "single-hop", and
    meta["reasoning_type"] says "derived_*" so nobody mistakes it for a
    released label.

SCORING: continuous max-ANLS
----------------------------
vlmeval's official path extracts the short answer with a GPT JUDGE and then
applies `anls_compute`, which zeroes anything under 0.5. Neither is used
here: no LLM judge anywhere (the deterministic `extract_span` mpdocvqa
uses -- an "Answer:" declaration anchored to line start, else the last
non-empty line -- replaces it, same declared deviation), and the score is
the RAW max similarity over the golds (scorers.anls with threshold=0.0),
because vse's score is a 0..1 continuous value and the binarisation would
throw away partial credit. Gold replay scores exactly 1.0; a garbage reply
scores 0.0 only if it shares no characters with a gold -- the accepted cost
of continuity, since the 0.5 threshold that would zero such noise is
exactly what this scorer deliberately drops.

SPLIT UNIT
----------
`group_key` is the DECK (`document.id`), never the question: the 1,652 val
questions sit on 300 decks (~5.5 questions per deck), and every question of
a deck shares all 20 of its pages, so a deck appearing in both gate and
test would let a strategy tune on pages it is later measured on.

KNOWN PITFALLS
--------------
  * A deck is kept only when ALL `count_pages` of its pages were found in
    the corpus shards. The corpus downloads as 13 shards; a partial mirror
    silently yields a partial domain unless the loader is strict, so decks
    short of their page count are counted and skipped (the mpdocvqa
    leniency pattern), never scored against a truncated deck.
  * corpus `page_number` is verified 0-based on the current release (first
    page of a deck is 0), but the base is still DETECTED per load rather
    than hardcoded -- if any page_number 0 appears the column is 0-based,
    else 1-based -- and every deck is then checked to EXACTLY cover its
    expected range against `count_pages`. A deck with a gap, a duplicate
    or a partial download is skipped, not guessed at.
  * Do not be tempted by the QA-only columns for type stratification:
    question.type is constant in val. Use the item_id's hop bucket
    (derived from evidence page count) -- `scripts/three_conditions.py
    --sample stratified` reads exactly that segment.
"""
from __future__ import annotations

from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "slidevqa"

VAL_GLOB = "val-*.parquet"
CORPUS_GLOB = "train-*.parquet"

# answer.format class_label names from the QA repo's README, index-ordered.
ANSWER_FORMATS = ["string", "reference", "integer", "float", "boolean",
                  "list", "datetime", "other", "none"]


def _find_layout(root: Path) -> tuple[list[Path], list[Path]]:
    """Locate the QA val parquet(s) and the corpus shards under `root`."""
    pqs = sorted((root / "qa" / "data").glob(VAL_GLOB)) \
        or sorted(root.rglob(f"qa/data/{VAL_GLOB}")) \
        or sorted(root.rglob(VAL_GLOB))
    if not pqs:
        raise FileNotFoundError(f"{DOMAIN}: no QA {VAL_GLOB} under {root}")
    corpus = sorted((root / "corpus" / "data").glob(CORPUS_GLOB)) \
        or sorted(root.rglob(f"corpus/data/{CORPUS_GLOB}"))
    if not corpus:
        raise FileNotFoundError(
            f"{DOMAIN}: no corpus {CORPUS_GLOB} under {root}")
    return pqs, corpus


def _answer_format(raw) -> str:
    try:
        return ANSWER_FORMATS[int(raw)]
    except (TypeError, ValueError, IndexError):
        return str(raw) if raw is not None else "string"


def _image_bytes(cell) -> bytes:
    """Corpus `image` cell: raw bytes, or the HF Image-feature struct."""
    if isinstance(cell, dict):
        cell = cell.get("bytes")
    return bytes(cell) if cell else b""


def _materialise_pages(corpus: list[Path], needed: set[str],
                       n_pages_by_doc: dict[str, int]
                       ) -> dict[str, dict[int, str]]:
    """Stream the corpus shards and write each needed page via write_media.

    Returns {document_id: {page_number_RAW: media_path}}. Raw page numbers
    are kept as released; the 0/1-base question is settled by the caller,
    which has the dataset-wide view. Pages are written as they stream past
    so memory stays flat, and write_media's content addressing makes a
    re-run of an interrupted download cheap.
    """
    import pyarrow.parquet as pq

    pages: dict[str, dict[int, str]] = {}
    cols = ["document_id", "page_number", "image"]
    for shard in corpus:
        try:
            table = pq.read_table(shard, columns=cols)
        except Exception as e:
            # A half-written tail shard of an in-progress download is
            # skipped, not fatal: decks it would have covered simply fail
            # their completeness check and are counted, not guessed at.
            print(f"[{DOMAIN}] skipping unreadable shard "
                  f"{Path(shard).name}: {e}")
            continue
        for doc_id, page_no, img in zip(
                table.column("document_id").to_pylist(),
                table.column("page_number").to_pylist(),
                table.column("image").to_pylist()):
            doc_id = str(doc_id)
            if doc_id not in needed:
                continue
            blob = _image_bytes(img)
            if not blob:
                continue
            path = write_media(DOMAIN, blob, f"{doc_id}_p{page_no}")
            pages.setdefault(doc_id, {})[int(page_no)] = path
    return pages


def load(root: str | Path, seed: int = 20260911,
         limit: int | None = None) -> list[Sample]:
    """Load the SlideVQA val split as public/private Sample pairs.

    `seed` is accepted for interface uniformity and is deliberately UNUSED:
    every page of every kept deck is materialised, so there is no sampling
    scheme and two loads are identical.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    pqs, corpus = _find_layout(root)

    rows: list[dict] = []
    for p in pqs:
        rows.extend(pq.read_table(p).to_pylist())

    # Evidence-page indexing base, re-derived per load rather than assumed:
    # measured on the current release, pages are 0-based (0 occurs, max is
    # count_pages-1), and this check turns a silent re-release into a
    # different-but-explicit answer instead of off-by-one evidence.
    all_ev = [pg for r in rows for pg in ((r.get("evidence") or {})
                                          .get("pages") or [])]
    ev_offset = 0 if 0 in all_ev else -1

    n_pages_by_doc: dict[str, int] = {}
    for r in rows:
        doc = r.get("document") or {}
        doc_id = str(doc.get("id") or "").strip()
        try:
            n_pages = int(doc.get("count_pages") or 0)
        except (TypeError, ValueError):
            n_pages = 0
        if doc_id and n_pages > 0:
            n_pages_by_doc[doc_id] = n_pages

    doc_pages = _materialise_pages(corpus, set(n_pages_by_doc),
                                   n_pages_by_doc)

    # corpus page_number base, detected dataset-wide the same way the
    # evidence base is: a 0 anywhere means 0-based, else 1-based.
    raw_pages = [p for d in doc_pages.values() for p in d]
    page_offset = 0 if 0 in raw_pages else -1

    samples: list[Sample] = []
    n_no_qa = 0
    n_short_media = 0
    n_bad_span = 0

    for r in rows:
        q = r.get("question") or {}
        doc = r.get("document") or {}
        ans = r.get("answer") or {}
        ev = r.get("evidence") or {}

        qid = str(r.get("id") or "").strip()
        question = str(q.get("text") or "").strip()
        deck = str(doc.get("id") or "").strip()
        n_pages = n_pages_by_doc.get(deck, 0)
        if not qid or not question or not deck or n_pages <= 0:
            n_no_qa += 1
            continue

        golds = [str(x).strip() for x in (ans.get("variants") or [])
                 if str(x).strip()]
        if not golds:
            # vlmeval convention for not_answerable rows (none in val, but
            # the schema allows them): score against the literal string.
            golds = ["Not answerable"]

        # A deck is presentable only if its pages EXACTLY cover the
        # expected range: 0..n-1 when 0-based, 1..n when 1-based. Anything
        # else -- a partial download, a duplicated page, a gap -- is
        # counted and skipped, never scored against a truncated deck.
        got = doc_pages.get(deck) or {}
        raw_expected = [i + (0 if page_offset == 0 else 1)
                        for i in range(n_pages)]
        if sorted(got) != raw_expected:
            n_short_media += 1
            continue
        # media_ids ordered by page_index: position in the tuple IS the
        # 0-based page index everything else (evidence, selection) uses.
        paths = tuple(got[p] for p in raw_expected)

        ev_pages = tuple(sorted({pg + ev_offset for pg in (ev.get("pages") or [])
                                 if 0 <= pg + ev_offset < n_pages}))

        # DERIVED, not released: the release's own question.type label is
        # constant in val, so hop-ness is read off the evidence annotation.
        reasoning_type = ("derived_multi_hop" if len(ev_pages) > 1
                          else "derived_single_hop")
        hop_bucket = "hop_multi" if len(ev_pages) > 1 else "hop_single"
        answer_type = _answer_format(ans.get("format"))

        item_id = f"{DOMAIN}/{hop_bucket}/{qid}"
        public = PublicItem(
            item_id=item_id, domain=DOMAIN,
            # The released question text is the full prompt; nothing is
            # appended (see the module docstring).
            question=question,
            media_ids=paths,
            meta={
                "deck": deck,
                # TRUE page count of the deck for this question. A
                # selection expression reads this as `n_pages`.
                "n_pages": n_pages,
                "n_pages_pooled": len(paths),
                "page_indices": list(range(n_pages)),
                "page_numbers": list(range(1, n_pages + 1)),
                # True on every kept item: decks short of their released
                # page count are skipped, not truncated.
                "pool_is_complete": True,
                # Pages arrive as JPEG bytes inside the corpus parquet and
                # are written verbatim by write_media; there is no
                # rasteriser and no re-encode anywhere in the path.
                "render_backend": "released_jpeg",
                "sampling_scheme": "none_all_pages",
            },
        )
        private = PrivateItem(
            item_id=item_id,
            answer=golds[0],
            answer_type=answer_type,
            evidence_pages=ev_pages,
            evidence_source="official" if ev_pages else "none",
            # 0-based indexing is verified on the current release (see the
            # module docstring), not on release documentation.
            evidence_verified=True,
            meta={
                "deck": deck,
                "n_pages": n_pages,
                "question_id": qid,
                "all_answers": list(golds),
                "evidence_pages": list(ev_pages),
                "has_evidence": bool(ev_pages),
                # "derived_*": computed from the evidence annotation, not a
                # released label -- question.type is constant in val.
                "reasoning_type": reasoning_type,
                "answer_type": answer_type,
                "answer_format_raw": ans.get("format"),
                "question_type_raw": q.get("type"),
            },
        )
        # Group by the DECK: every question of a deck shares all of its
        # pages, so a deck in both gate and test would leak the tuning
        # pages into the measured ones.
        samples.append(Sample(public=public, private=private,
                              group_key=deck))
        if limit and len(samples) >= limit:
            break

    n_multi = sum(1 for s in samples
                  if s.private.meta["reasoning_type"] == "derived_multi_hop")
    print(f"[{DOMAIN}] {len(samples)} questions over "
          f"{len({s.group_key for s in samples})} decks "
          f"({n_multi} multi-hop; dropped {n_no_qa} without question/id, "
          f"{n_short_media} with incomplete corpus pages)")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Max ANLS over ALL released golds, continuous in [0, 1].

    Two deliberate departures from vlmeval's `anls_compute`, both declared
    in the module docstring: the reply first goes through the deterministic
    `extract_span` mpdocvqa uses (the official protocol's GPT judge is
    replaced, never added back), and the 0.5 binarisation is dropped --
    `scorers.anls` with threshold=0.0 returns the raw max similarity so a
    near-miss keeps its partial credit.
    """
    from ..evaluation.scorers import anls
    from .mpdocvqa import extract_span

    golds = private.meta.get("all_answers") or [private.answer]
    if isinstance(golds, str):
        golds = [golds]
    s = anls(extract_span(answer_text), [str(g) for g in golds],
             threshold=0.0)
    return max(0.0, min(1.0, s))
