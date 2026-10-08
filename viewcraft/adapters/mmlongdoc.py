"""MMLongBench-Doc adapter (D1): deciding WHICH page to look at.

Why this domain is in the study: every other domain here hands the model all
of its pixels at once. A chart is one image, a walkthrough is eight sampled
frames -- the model's only decision is how hard to look at what it was
already given. A 28-page (median) to 468-page (max) PDF cannot be handed over
at all: at the domain's 8-image request cap, the model sees at most 8 of a
median 28 pages, so 71% of the median document is invisible on any single
request and 98% of the largest one is. The binding decision is therefore not
resolution but SELECTION -- which pages enter the context at all -- and that
decision is made before any zoom can help. A crop cannot recover evidence on
a page that was never materialised.

That makes this the one domain in the set where `phi.select_expr` (see
views/selection.py) has something real to choose over, and where a wrong
choice is unrecoverable rather than merely blurry. The mechanism under test
is "find the right page, then read it", and the failure mode it exposes --
answering confidently from the wrong pages -- is invisible in a single-image
domain.

Released layout: one parquet (`data/train-00000-of-00001.parquet`, 1,091
rows, HF split name "train" although these are the benchmark's evaluation
questions) plus `documents/*.pdf`. A row is

    {doc_id, doc_type, question, answer, evidence_pages,
     evidence_sources, answer_format}

`doc_id` is the PDF FILENAME (e.g. "PH_2016.06.08_Economy-Final.pdf") and is
the join key to documents/. `evidence_pages` and `evidence_sources` are
strings holding PYTHON LITERALS ("[19, 20]", "['Chart', 'Table']"), not JSON
and not native lists, so they are parsed with ast.literal_eval rather than
json.loads -- the single quotes make json.loads fail on every non-empty
`evidence_sources` value.

ANSWER FORMATS AND THE "NOT ANSWERABLE" HALF
--------------------------------------------
`answer_format` partitions the 1,091 rows as Int 290, Str 250, None 244,
Float 160, List 147. `answer_format == "None"` is not a missing value: it
marks a question the document deliberately CANNOT answer, and its gold
answer is the literal string "Not answerable" in all 244 cases (verified: the
correspondence is exact in both directions). Those items score by whether the
model ABSTAINS, so a model that always guesses an answer scores 0 on 22% of
the benchmark. This is the point of the unanswerable half -- a long document
makes it cheap to hallucinate support -- and it is why the scorer's
"not answerable" branch is an abstention test rather than a string match.

PAGE POOL, AND WHAT IS CAPPED
-----------------------------
`PublicItem.media_ids` carries the page PNGs, and `meta` carries the true
document shape: `n_pages` (the real total page count from the PDF, never the
number materialised), `page_numbers` (1-based printed labels, parallel to
media_ids), `page_indices` (0-based), plus `pool_is_complete` and
`n_pages_pooled` so a strategy can tell a whole document from a subsampled
one.

ONE CAP, DECLARED: the page pool is capped at `MAX_POOL_PAGES` = 128 pages
per item. It is NOT a truncation to the first 128 pages -- that would delete
the back of long documents -- but a uniform-midpoint subsample across the
WHOLE document (`uniform_page_indices`, the same scheme vsi.py uses for
frames), so a capped pool still spans front to back and a selection
expression like `range(n - n // 8, n)` can still reach a document's final
pages. Measured reason for the number: runtime.run_static_item renders EVERY
pooled page at step 0 (at 89.8 ms per page, measured), whether or not psi's
8-image cap lets the model see it, so an uncapped 468-page document costs
42 s of rendering per item to show 8 pages. At 128 the cap binds on 5 of 125
documents (29 of 1,010 rows) and 98.8% of evidence pages remain reachable in
the pool; the true `n_pages` is still reported for every item, so nothing is
hidden from analysis. 4,096 is the hard ceiling selection.py accepts, so
this cap is ours and deliberate, not a limit of the expression evaluator.

Pages are rendered at `PAGE_LONG_EDGE` = 2048 px, deliberately ABOVE the
domain's 1024 px overview budget. phi renders crops by re-reading the source
file, so a page stored at the overview size could only ever be upscaled and a
zoom into a table would return interpolation instead of detail. 2048 gives
the zoom something real to recover, at 3.7 GB for all pooled pages.

RENDER BACKEND
--------------
pypdfium2 is preferred and pdf2image/poppler accepted, but NEITHER is
installed in the target env (checked: both ImportError, no `pdftoppm` on
PATH); PyMuPDF 1.28.2 is, so it is the third and active branch. The backend
actually used is recorded in `meta["render_backend"]` rather than assumed, so
a page rendered by a different library cannot be silently compared against
one of these.

MISSING DOCUMENTS
-----------------
10 of the 135 referenced `doc_id`s have no PDF in documents/ (125 files
present). The 81 rows on those documents are SKIPPED with a printed count,
not crashed on and not answered from metadata alone.

SPLIT UNIT
----------
`group_key` is the DOCUMENT, like chartqa groups by image: a document carries
a median of 7 questions, and letting one PDF appear in both gate and test
would let a strategy tune on pages it is later measured on.

SCORING: `mmlongdoc_generalized_accuracy_v1`
--------------------------------------------
Generalized accuracy over the five answer formats -- per-item binary, then
an unweighted mean, with a per-format and answerable/unanswerable breakdown
from `aggregate()` (a single mean silently blends an abstention test with a
numeric one). The comparison step is implemented faithfully per format: Int
by integer equality, Float by equality after unit cleaning and rounding, Str
by exact normalized match, List as an order-insensitive set over cleaned
elements, None by abstention.

FOUR DECLARED DEVIATIONS
------------------------1. NO LLM EXTRACTION STEP. The official protocol sends the model's verbose
   reply to a separate LLM that extracts a short answer, then compares that.
   Here extraction is DETERMINISTIC (`extract_short_answer` below): no model
   acts as judge anywhere in ViewCraft, because a judged score cannot be
   reproduced from the ledger. This is the same deviation already declared
   for `scorers.mmlongbench_match`. Direction of the error is knowable: a
   deterministic extractor is STRICTER than an LLM one on verbose replies, so
   reported scores are a lower bound, and it is applied identically to every
   arm and condition so it cannot favour one.

2. FLOAT TOLERANCE. This adapter compares floats by equality after cleaning
   and rounding to `FLOAT_DECIMALS` = 2, matching the official format-aware
   comparison. It deliberately does NOT call `scorers.mmlongbench_match`,
   whose Float branch applies a 1% RELATIVE tolerance -- on a gold of 18.29%
   that accepts 18.11-18.47, which is a different metric from the official
   one. `mmlongbench_match` is left untouched (it has its own tests); this
   domain's scorer does not route through it.

3. LIST GOLDS ARE CANONICALISED IN THE ADAPTER. The released List gold is a
   Python-literal STRING ("['White', '10%']"), and comparing a model's
   "White, 10%" against those raw characters fails on the brackets and quotes
   -- verified: `scorers.mmlongbench_match("White, 10%", "['White', '10%']",
   "List")` returns 0.0 for a correct answer. The literal is therefore parsed
   and re-joined to "White, 10%" before it is ever stored as gold, so the
   comparison is between answers rather than between an answer of Python
   syntax. `private.meta["answer_raw"]` keeps the released string verbatim.

ABSTENTION DETECTION WAS WIDENED (shared helper)
------------------------------------------------
The unanswerable branch is only as good as its abstention predicate, and
`scorers._is_abstention` recognised three phrasings ("not answerable",
"unanswerable", "not mentioned", "no information", "not stated", "none").
A model told to decline almost never replies with the benchmark's literal
string: "the document does not state this", "not provided in the document"
and "cannot be determined" were all scored as WRONG ANSWERS, which on this
benchmark is worth up to the full 22% unanswerable share. The marker list in
scorers.py was therefore widened and exposed as `scorers.is_abstention`.

The widening was checked for FALSE POSITIVES against every released gold: of
the 847 answerable golds exactly one matches a marker -- an
`answer_format == "Str"` item whose true answer is the literal string "N/A".
`compare` therefore tests BOTH sides, so when the gold also looks like an
abstention the comparison falls through to the ordinary string branch and
that item still scores correctly.

4. THREE MISLABELLED ROWS ARE RE-TYPED. Three released rows carry
   `answer_format == "Int"` with a non-numeric gold ("yes", "GPT-4",
   "14:04 CET"; verified as the only three such rows -- every Float gold
   parses). Under the Int comparison the gold itself cannot be parsed, so
   those items are worth 0 to every arm no matter what the model answers:
   not a hard question but an unwinnable one, and a silent 0.3% cap on the
   headline. They are re-typed to Str -- exact match, which is what those
   golds actually support -- rather than dropped, so the question stays in
   the set. `private.meta` keeps `answer_format_released` and
   `answer_format_retyped` so the change is auditable and the affected items
   can be excluded from a per-format table.

   The check is GENERAL (any Int/Float gold that will not parse), not a
   hardcoded list of three strings, so a re-release with different bad rows
   is handled rather than silently mis-scored.

VERIFIED BY REPLAYING THE GOLD
------------------------------
The scorer is checked by feeding each item its OWN gold answer as the
prediction: a correct scorer must then return exactly 1.0 on every item.
This caught both defects above plus an extraction bug -- one Str gold is a
quoted prompt containing "...just output 'Answer: No entities found.'", and
an unanchored search for "answer:" pulled that fragment out of the gold and
scored it against itself as WRONG. `_ANSWER_DECL` is therefore anchored to
the start of a line, so a model's own declaration is honoured while a
mention inside prose is not. All 1,010 items now self-score 1.0.

EVIDENCE PAGES ARE PRIVATE
--------------------------
`evidence_pages` and `evidence_sources` are surfaced on `PrivateItem` only
(`PublicItem.meta` cannot hold them -- schema.py bans the key outright) and
exist for POST-HOC BINNING: "did the strategy that scored well actually look
at the evidence page?". They are never an input to selection; a selection
expression that could read them would be an oracle, not a searched strategy.

Released evidence pages are 1-BASED and 5 of the 784 rows that have them
carry an out-of-range value (three rows list page 0, which cannot be a
1-based label; one lists 1418 in a 20-page document; one lists 115-116 in a
105-page document). Those rows are KEPT -- the question and its gold answer
are unaffected -- but the bad indices are dropped from the converted 0-based
tuple and the row is flagged with `evidence_pages_out_of_range`, so an
evidence-conditioned analysis can exclude them instead of silently binning
against a page that does not exist.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import statistics
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "mmlongdoc"

from ..paths import project_root

ROOT = project_root()
DATA = ROOT / "data" / "mmlongbench"
MEDIA_ROOT = ROOT / "data" / "_media" / DOMAIN

# See "PAGE POOL, AND WHAT IS CAPPED" in the module docstring. 128 is a
# rendering-cost cap, not a claim about documents.
MAX_POOL_PAGES = 128
# Above the 1024 px overview budget on purpose: crops re-read this file, so
# storing at the overview size would make every zoom an upscale.
PAGE_LONG_EDGE = 2048
# Official Float comparison granularity.
FLOAT_DECIMALS = 2

RENDER_SCHEME = "uniform_midpoint_v1"

ANSWER_FORMATS = ("Str", "Int", "Float", "List", "None")

# answer_format -> PrivateItem.answer_type. "None" is an unanswerable
# question, not a missing answer, so it gets its own type rather than "string".
_ANSWER_TYPE = {
    "Str": "string",
    "Int": "integer",
    "Float": "float",
    "List": "list",
    "None": "not_answerable",
}

# A number in a free-form reply, incl. "1,234", ".5", "-3.2e1", "45%".
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?")

# An explicit answer DECLARATION, anchored to the start of a line.
#
# Anchored rather than free-floating because "answer:" also occurs INSIDE
# prose, and one released gold is a quoted prompt containing
# "...just output 'Answer: No entities found.'" -- an unanchored search
# extracted "No entities found.'" from it and scored the gold against itself
# as WRONG. A model's real declaration is its own line ("Answer: 8",
# "**Final answer:** 8"), so anchoring keeps the declaration and ignores the
# mention. Leading markdown/quote/bullet characters are tolerated.
_ANSWER_DECL = re.compile(
    r"^[\s>*#\-`'\"]*(?:final\s+)?answer\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# render backend


def _backend() -> str:
    """Which PDF rasteriser is available, in order of preference.

    pypdfium2 first and pdf2image second because they are the two the task
    specifies; PyMuPDF third because it is the one actually installed here.
    Resolved once per call site and recorded in meta, never assumed.
    """
    try:
        import pypdfium2  # noqa: F401
        return "pypdfium2"
    except ImportError:
        pass
    try:
        import pymupdf  # noqa: F401
        return "pymupdf"
    except ImportError:
        pass
    try:
        import pdf2image  # noqa: F401
        return "pdf2image"
    except ImportError:
        pass
    raise ImportError(
        f"{DOMAIN}: no PDF rasteriser available (tried pypdfium2, pymupdf, "
        "pdf2image). Install one of them; this domain cannot be loaded "
        "without rendering pages.")


def page_count(pdf_path: str | Path) -> int:
    """True total page count of a PDF. This is the number reported as n_pages."""
    backend = _backend()
    if backend == "pypdfium2":
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            return len(doc)
        finally:
            doc.close()
    if backend == "pymupdf":
        import pymupdf
        with pymupdf.open(str(pdf_path)) as doc:
            return doc.page_count
    from pdf2image import pdfinfo_from_path
    return int(pdfinfo_from_path(str(pdf_path))["Pages"])


# ---------------------------------------------------------------------------
# page pool


def uniform_page_indices(n_pages: int, n_sample: int) -> list[int]:
    """`n_sample` page indices spread uniformly over the WHOLE document.

    Midpoints of `n_sample` equal bins: index_i = floor((i + 0.5) * n / N) --
    byte-identical to `vsi.uniform_frame_indices` and to
    `views.selection._uniform`, so a capped pool is the same shape a
    `uniform(k)` selection expression would produce and remains a legitimate
    control rather than a near-miss.

    Uniform rather than first-N because a document's evidence is not
    front-loaded: truncating to the first N pages makes the back of a long
    report permanently unreachable, while a uniform pool still spans front to
    back (measured: 98.8% of evidence pages stay reachable in the pool at
    N=128, and the uniform pool keeps the document's full span addressable
    for a back-loaded selection expression).
    """
    if n_pages <= 0:
        raise ValueError(f"{DOMAIN}: n_pages={n_pages}")
    if n_sample <= 0:
        raise ValueError(f"{DOMAIN}: n_sample={n_sample}")
    if n_sample >= n_pages:
        return list(range(n_pages))
    idx = [int((i + 0.5) * n_pages / n_sample) for i in range(n_sample)]
    return [min(i, n_pages - 1) for i in idx]


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _page_dir(digest: str, long_edge: int) -> Path:
    return MEDIA_ROOT / digest[:2] / digest / f"le{long_edge}"


def render_pages(pdf_path: str | Path, page_indices: list[int],
                 long_edge: int = PAGE_LONG_EDGE,
                 ) -> tuple[tuple[str, ...], dict]:
    """Rasterise the given 0-based pages to PNG. Returns (paths, info).

    CONTENT-ADDRESSED and idempotent: the cache directory is keyed by the
    sha256 of the PDF BYTES (not its filename), so re-running never
    re-renders, two questions on one document share one render, and a
    re-downloaded or edited PDF lands in a different directory instead of
    silently reusing stale pixels.

    Pages are written as real files because phi renders a crop by re-reading
    the SOURCE file at full resolution -- that is what makes a zoom recover
    detail. An in-memory pixmap could not be cropped that way.

    A `manifest.json` written LAST is the completion marker, so a run
    interrupted mid-render is not mistaken for a finished one. Each page is
    published by atomic rename so a concurrent reader never opens a
    half-written PNG.
    """
    pdf_path = Path(pdf_path)
    digest = _file_digest(pdf_path)
    d = _page_dir(digest, long_edge)
    manifest = d / "manifest.json"
    paths = tuple(str(d / f"page_{i:04d}.png") for i in page_indices)
    backend = _backend()

    if manifest.exists():
        try:
            man = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            man = None
        if (man and set(page_indices).issubset(set(man.get("rendered", [])))
                and all(Path(p).exists() for p in paths)):
            return paths, {"render_backend": man.get("backend", backend),
                           "page_digest": digest, "long_edge": long_edge}

    d.mkdir(parents=True, exist_ok=True)
    already = {int(p.stem.split("_")[1]) for p in d.glob("page_*.png")}
    todo = [i for i in page_indices if i not in already]
    if todo:
        _rasterise(pdf_path, todo, long_edge, d, backend)

    rendered = sorted({int(p.stem.split("_")[1]) for p in d.glob("page_*.png")})
    tmp = manifest.with_name(f".manifest.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({
        "pdf_name": pdf_path.name, "pdf_sha256": digest,
        "long_edge": long_edge, "backend": backend,
        "rendered": rendered, "scheme": RENDER_SCHEME,
    }, indent=2))
    tmp.replace(manifest)

    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise RuntimeError(
            f"{DOMAIN}: {pdf_path.name}: {len(missing)} page(s) failed to "
            f"render, e.g. {missing[0]}")
    return paths, {"render_backend": backend, "page_digest": digest,
                   "long_edge": long_edge}


def _rasterise(pdf_path: Path, page_indices: list[int], long_edge: int,
               dest: Path, backend: str) -> None:
    """Render `page_indices` of one PDF into `dest` at the given long edge.

    Scale is computed PER PAGE from that page's own media box, because a
    document can mix portrait body pages with landscape foldouts and a single
    document-wide scale would render one of the two at the wrong size.
    """
    stage = dest / f".stage.{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        if backend == "pypdfium2":
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(str(pdf_path))
            try:
                for i in page_indices:
                    page = doc[i]
                    w, h = page.get_size()
                    scale = long_edge / max(w, h)
                    page.render(scale=scale).to_pil().save(
                        stage / f"page_{i:04d}.png")
            finally:
                doc.close()
        elif backend == "pymupdf":
            import pymupdf
            # MuPDF logs recoverable syntax complaints (e.g. an unparseable
            # colour space) to stderr on a handful of these PDFs; the page
            # still rasterises, so the noise is silenced rather than treated
            # as a failure. A page that genuinely fails is caught by the
            # missing-file check in render_pages.
            try:
                pymupdf.TOOLS.mupdf_display_errors(False)
            except AttributeError:
                pass
            with pymupdf.open(str(pdf_path)) as doc:
                for i in page_indices:
                    page = doc[i]
                    rect = page.rect
                    scale = long_edge / max(rect.width, rect.height)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
                    pix.save(stage / f"page_{i:04d}.png")
        else:
            from pdf2image import convert_from_path
            for i in page_indices:
                # 1-based, inclusive on both ends in pdf2image's API.
                img = convert_from_path(str(pdf_path), first_page=i + 1,
                                        last_page=i + 1, size=long_edge)[0]
                img.save(stage / f"page_{i:04d}.png")

        for src in sorted(stage.glob("page_*.png")):
            src.replace(dest / src.name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# ---------------------------------------------------------------------------
# parsing released fields


def _parse_literal_list(raw) -> list:
    """Parse a released `evidence_pages` / `evidence_sources` cell.

    These are PYTHON literals in a string column ("[19, 20]",
    "['Chart', 'Table']"), so json.loads fails on every single-quoted value.
    A cell that will not parse yields [] rather than raising: it is metadata,
    and a malformed evidence list must not cost a question its gold answer.
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


def canonical_gold(answer_raw: str, answer_format: str) -> str:
    """The gold string the scorer actually compares against.

    Only the List format is rewritten, and only from Python-literal syntax to
    a plain comma-joined string (see deviation 3 in the module docstring):
    the released gold "['White', '10%']" cannot be matched by a model
    answering "White, 10%" until the brackets and quotes are gone. Every
    other format is passed through verbatim -- no unit stripping, no rounding
    -- so the released value remains the gold.
    """
    s = "" if answer_raw is None else str(answer_raw).strip()
    if answer_format != "List":
        return s
    parts = _parse_literal_list(s)
    if not parts:
        return s
    return ", ".join(str(p).strip() for p in parts)


# ---------------------------------------------------------------------------
# loading


def load(root: str | Path = DATA, seed: int = 20260911,
         max_pool_pages: int = MAX_POOL_PAGES,
         long_edge: int = PAGE_LONG_EDGE,
         answer_formats: tuple[str, ...] | None = None,
         doc_types: tuple[str, ...] | None = None,
         limit: int | None = None,
         render_now: bool = True) -> list[Sample]:
    """Load MMLongBench-Doc as public/private Sample pairs.

    `seed` is accepted for interface uniformity and is deliberately UNUSED:
    this adapter makes no random choice. Page selection is the deterministic
    uniform-midpoint scheme, so two loads with different seeds are identical
    and a split plan is the only place randomness belongs.

    `answer_formats` restricts to a subset of Str/Int/Float/List/None; the
    default keeps all five, which is the official headline configuration.
    Dropping "None" would remove the unanswerable half and make the headline
    number incomparable to a published one.

    Rows whose PDF is missing are skipped with a count; every document is
    rendered once and shared by all of its questions.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    pqs = sorted((root / "data").glob("*.parquet")) or sorted(
        root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"{DOMAIN}: no parquet under {root}")
    docs_dir = root / "documents"
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"{DOMAIN}: no documents/ under {root}")

    rows: list[dict] = []
    for p in pqs:
        rows.extend(pq.read_table(p).to_pylist())

    keep_fmt = set(answer_formats) if answer_formats else None
    keep_doc = set(doc_types) if doc_types else None

    # Stable order so item_ids and any limit are reproducible. The release
    # ships no row id, so position within (doc_id, question) is the identity.
    rows.sort(key=lambda r: (str(r.get("doc_id") or ""),
                             str(r.get("question") or "")))

    missing_docs: set[str] = set()
    skipped_missing = 0
    skipped_empty = 0
    bad_evidence = 0
    retyped = 0

    # One page render per DOCUMENT, shared by its questions.
    doc_cache: dict[str, tuple] = {}

    samples: list[Sample] = []
    for i, r in enumerate(rows):
        doc_id = str(r.get("doc_id") or "").strip()
        question = str(r.get("question") or "").strip()
        answer_raw = r.get("answer")
        fmt = str(r.get("answer_format") or "").strip()
        doc_type = str(r.get("doc_type") or "").strip()

        if keep_fmt and fmt not in keep_fmt:
            continue
        if keep_doc and doc_type not in keep_doc:
            continue
        if not doc_id or not question or fmt not in _ANSWER_TYPE:
            skipped_empty += 1
            continue
        # An unanswerable item's gold is the string "Not answerable"; any
        # other format with an empty gold is unscoreable.
        if fmt != "None" and not str(answer_raw or "").strip():
            skipped_empty += 1
            continue

        pdf = docs_dir / doc_id
        if not pdf.is_file():
            # Reported, not raised: 10 of 135 referenced PDFs are absent from
            # the release directory and the other 125 are still usable.
            missing_docs.add(doc_id)
            skipped_missing += 1
            continue

        if doc_id not in doc_cache:
            n_pages = page_count(pdf)
            if n_pages <= 0:
                missing_docs.add(doc_id)
                skipped_missing += 1
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
        n_pages, pool, paths, info = doc_cache[doc_id]

        # Released evidence pages are 1-BASED printed labels; PrivateItem
        # wants 0-based indices. Out-of-range values (page 0, or a page past
        # the end) are dropped from the tuple and flagged rather than clamped
        # to a real page that the annotator never pointed at.
        ev_raw = _parse_literal_list(r.get("evidence_pages"))
        ev_pages: list[int] = []
        ev_bad: list = []
        for v in ev_raw:
            try:
                n = int(v)
            except (TypeError, ValueError):
                ev_bad.append(v)
                continue
            if 1 <= n <= n_pages:
                ev_pages.append(n - 1)
            else:
                ev_bad.append(v)
        if ev_bad:
            bad_evidence += 1
        ev_sources = [str(x) for x in
                      _parse_literal_list(r.get("evidence_sources"))]

        gold = canonical_gold(answer_raw, fmt)
        unanswerable = fmt == "None"

        # THREE RELEASED ROWS ARE MISLABELLED: answer_format == "Int" with a
        # gold of "yes", "GPT-4" and "14:04 CET". Scored as Int they can
        # never be right at all -- the numeric branch cannot parse the gold,
        # so the item is worth 0 to every arm no matter what it answers, and
        # 3 unwinnable items is a silent 0.3% cap on the headline number.
        # They are RE-TYPED to Str (an exact-match comparison, which is what
        # those golds actually support) rather than dropped, so the question
        # stays in the set and stays winnable. The released label is kept in
        # private meta so the change is auditable.
        fmt_released = fmt
        if fmt in ("Int", "Float") and _clean_numeric(gold) is None:
            fmt = "Str"
            retyped += 1

        item_id = f"{DOMAIN}/{fmt_released}/{Path(doc_id).stem}_{i}"

        public = PublicItem(
            item_id=item_id,
            domain=DOMAIN,
            question=question,
            media_ids=paths,
            meta={
                "doc_id": doc_id,
                "doc_type": doc_type,
                # TRUE total page count of the document, independent of how
                # many pages were materialised. A selection expression reads
                # this as `n_pages`.
                "n_pages": n_pages,
                # Per-page identifiers, parallel to media_ids, so a strategy
                # can address a page by position and an audit can map an
                # emitted view back to a printed page label.
                "n_pages_pooled": len(paths),
                "page_indices": list(pool),
                "page_numbers": [p + 1 for p in pool],
                "pool_is_complete": len(pool) == n_pages,
                "page_long_edge_px": info["long_edge"],
                "render_backend": info["render_backend"],
                "sampling_scheme": RENDER_SCHEME,
                # Answer FORMAT only, never the answer: the executor cannot
                # guess from the question text that a bare integer is wanted,
                # or that abstention is an allowed reply.
                "answer_format": fmt,
                "may_be_unanswerable": True,            },
        )
        private = PrivateItem(
            item_id=item_id,
            answer=gold,
            answer_type=_ANSWER_TYPE[fmt],
            # 0-based, official annotation. POST-HOC BINNING ONLY -- never an
            # input to selection (see the module docstring).
            evidence_pages=tuple(ev_pages),
            evidence_source="official" if ev_pages else "none",
            evidence_verified=bool(ev_pages) and not ev_bad,
            unanswerable=unanswerable,
            meta={
                "doc_id": doc_id,
                "doc_type": doc_type,
                "answer_format": fmt,
                "answer_format_released": fmt_released,
                "answer_format_retyped": fmt != fmt_released,
                "answer_raw": ("" if answer_raw is None
                               else str(answer_raw).strip()),
                "evidence_sources": ev_sources,
                "evidence_pages_1based": [int(v) for v in ev_raw
                                          if str(v).lstrip("-").isdigit()],
                "evidence_pages_out_of_range": ev_bad,
                "n_pages": n_pages,
                "n_pages_pooled": len(paths),
                "pool_is_complete": len(pool) == n_pages,
                # True when the evidence page survived the pool cap, so a
                # failure can be attributed to selection rather than blamed
                # on a page the adapter never materialised.
                "evidence_in_pool": bool(ev_pages) and all(
                    p in set(pool) for p in ev_pages),
                "scorer": "mmlongdoc_generalized_accuracy_v1",
                "metric": "generalized_accuracy",
                "answerable": not unanswerable,
            },
        )
        # Group by the DOCUMENT: a PDF carries a median of 7 questions, and a
        # document spanning gate and test would let a strategy tune on pages
        # it is later measured on.
        samples.append(Sample(public=public, private=private,
                              group_key=doc_id))

    if skipped_missing:
        print(f"[{DOMAIN}] skipped {skipped_missing} rows on "
              f"{len(missing_docs)} documents with no PDF in "
              f"{docs_dir.name}/ (kept {len(samples)} rows on "
              f"{len(doc_cache)} documents)")
    if skipped_empty:
        print(f"[{DOMAIN}] skipped {skipped_empty} rows with no question, "
              "no gold answer or an unknown answer_format")
    if bad_evidence:
        print(f"[{DOMAIN}] {bad_evidence} rows carry an out-of-range "
              "evidence page (dropped from evidence_pages, row kept and "
              "flagged in private meta)")
    if retyped:
        print(f"[{DOMAIN}] re-typed {retyped} rows whose released "
              "answer_format was Int/Float but whose gold is not numeric "
              "(e.g. 'yes', 'GPT-4', '14:04 CET'); scored as Str so they "
              "are winnable instead of worth 0 to every arm")
    capped = [s for s in samples if not s.public.meta["pool_is_complete"]]
    if capped:
        print(f"[{DOMAIN}] page pool capped at {max_pool_pages} on "
              f"{len(capped)} of {len(samples)} rows "
              f"({len({s.group_key for s in capped})} documents); "
              "true n_pages is still reported per item")
    return samples


# ---------------------------------------------------------------------------
# scoring


def extract_short_answer(raw: str, answer_format: str) -> str:
    """Reduce a free-form reply to the span the comparison needs.

    This replaces the official protocol's LLM extraction step with a
    deterministic one (declared deviation 1). It is intentionally
    conservative: it prefers an explicit "Answer:" declaration, and otherwise
    returns the reply's last non-empty line, because a model that reasons
    before answering ends on its conclusion.

    For Int/Float it then pulls the trailing NUMBER out of that span, since
    the numeric branches compare magnitudes and "about 12 tables" must reduce
    to 12 before an integer comparison can mean anything.
    """
    s = (raw or "").strip()
    if not s:
        return ""

    # Last anchored declaration wins: a model that corrects itself ends on
    # its final answer.
    decls = _ANSWER_DECL.findall(s)
    span = decls[-1].strip() if decls else ""
    if not span:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        span = lines[-1] if lines else s
    # Strip markdown emphasis and a trailing period the formats never carry.
    span = span.replace("**", "").replace("*", "").strip()

    if answer_format in ("Int", "Float"):
        # An abstention must survive intact, or a "not answerable" reply to a
        # numeric question would be reduced to "" and lose the information
        # that the model declined rather than failed.
        if _is_abstention_text(span) or _is_abstention_text(s):
            return span or s
        num = _last_number(span) or _last_number(s)
        return num if num is not None else span
    return span


def _last_number(text: str) -> str | None:
    cands = [m.group(0) for m in _NUMBER.finditer(text or "")
             if re.search(r"\d", m.group(0) or "")]
    if not cands:
        return None
    return cands[-1].replace(",", "")


def _is_abstention_text(s: str) -> bool:
    from ..evaluation.scorers import _norm, is_abstention
    return is_abstention(_norm(s))

def _clean_numeric(s: str) -> float | None:
    """Parse a released or predicted magnitude, dropping units.

    '%' and '$' are stripped rather than interpreted: 78 of the 160 Float
    golds carry a percent sign and the model may or may not repeat it, so
    comparing 18.29% against 18.29 must succeed. Nothing is RESCALED -- a
    percent is never divided by 100 -- because the gold is itself written in
    percent units and dividing one side only would make every such
    comparison fail.
    """
    from ..evaluation.scorers import _to_float
    return _to_float((s or "").strip())


def compare(pred: str, gold: str, answer_format: str) -> float:
    """Format-aware comparison: the official generalized-accuracy per-item score.

    Semantics per format, all binary:
      * Int   -- integer equality (a float-valued reply is rounded, so "12.0"
                 matches 12 but "12.6" does not match 12).
      * Float -- equality after unit cleaning and rounding to FLOAT_DECIMALS
                 (2). NOT a relative tolerance; see deviation 2.
      * Str   -- exact normalized match (case/whitespace/trailing period
                 insensitive).
      * List  -- order-insensitive set equality over cleaned elements, so
                 "b, a" matches "a, b" but a missing or extra element fails.
      * None  -- the reply must ABSTAIN; any concrete answer scores 0.

    An abstention on an ANSWERABLE question scores 0 in every branch: the
    numeric parses fail on the words and the string branches do not match, so
    declining to answer is never rewarded outside the unanswerable half.
    """
    from ..evaluation.scorers import _list_key, _norm

    fmt = answer_format
    p_raw, g_raw = (pred or ""), (gold or "")

    if fmt == "None":
        return 1.0 if _is_abstention_text(p_raw) else 0.0

    # A model abstaining on an answerable question is simply wrong -- short
    # circuit so an abstention's stray digits cannot accidentally match.
    #
    # Guarded by the GOLD also not looking like an abstention: exactly one
    # released answerable gold is the literal string "N/A", which matches an
    # abstention marker. For that item both sides look like abstentions and
    # the comparison must fall through to the ordinary string branch, where
    # "N/A" correctly matches "N/A".
    if _is_abstention_text(p_raw) and not _is_abstention_text(g_raw):
        return 0.0

    if fmt == "Int":
        pf, gf = _clean_numeric(p_raw), _clean_numeric(g_raw)
        if pf is None or gf is None:
            return 0.0
        return 1.0 if int(round(pf)) == int(round(gf)) else 0.0

    if fmt == "Float":
        pf, gf = _clean_numeric(p_raw), _clean_numeric(g_raw)
        if pf is None or gf is None:
            return 0.0
        return 1.0 if (round(pf, FLOAT_DECIMALS)
                       == round(gf, FLOAT_DECIMALS)) else 0.0

    if fmt == "List":
        # Gold is already canonicalised out of Python-literal syntax by
        # `canonical_gold`; both sides split on the same separators.
        return 1.0 if _list_key(_norm(p_raw)) == _list_key(_norm(g_raw)) \
            else 0.0

    return 1.0 if _norm(p_raw) == _norm(g_raw) else 0.0


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """MMLongBench-Doc generalized accuracy for one item, in [0, 1] (binary).

    Deterministic extraction then format-aware comparison. The per-item score
    is commensurable across formats (all binary), but the MEAN is not
    interpretable without its composition -- 22% of the benchmark is an
    abstention test -- so use `aggregate` rather than a bare mean.
    """
    fmt = str((private.meta or {}).get("answer_format")
              or public.meta.get("answer_format") or "Str")
    pred = extract_short_answer(answer_text, fmt)
    return compare(pred, str(private.answer), fmt)


def aggregate(scored: list[tuple[float, PrivateItem]]) -> dict:
    """Generalized accuracy plus the breakdown a single mean would hide.

    Reports the headline mean, then per-answer-format means and the
    answerable/unanswerable split. The unanswerable subset measures
    ABSTENTION and the answerable subset measures retrieval-and-reading; a
    model can move the headline by changing only its willingness to guess, so
    the two are always reported side by side.

    `evidence_in_pool_frac` is included as an instrument check, not a result:
    if it is low, low scores may reflect the adapter's page cap rather than
    the strategy, and that must be visible.
    """
    by_fmt: dict[str, list[float]] = {}
    ans: list[float] = []
    unans: list[float] = []
    in_pool = 0
    with_ev = 0
    for s, priv in scored:
        meta = priv.meta or {}
        by_fmt.setdefault(str(meta.get("answer_format") or "?"), []).append(s)
        (unans if meta.get("answerable") is False else ans).append(s)
        if priv.evidence_pages:
            with_ev += 1
            if meta.get("evidence_in_pool"):
                in_pool += 1

    def mean(v: list[float]) -> float | None:
        return (sum(v) / len(v)) if v else None

    allv = ans + unans
    return {
        "generalized_accuracy": mean(allv),
        "n": len(allv),
        "answerable_accuracy": mean(ans),
        "n_answerable": len(ans),
        "unanswerable_abstention_rate": mean(unans),
        "n_unanswerable": len(unans),
        "per_answer_format": {
            k: {"mean": mean(v), "n": len(v)}
            for k, v in sorted(by_fmt.items())
        },
        "evidence_in_pool_frac": (in_pool / with_ev) if with_ev else None,
        "composition_note": (
            f"{len(ans)} answerable + {len(unans)} unanswerable "
            "(the latter score abstention, not retrieval)"
            if ans and unans else "single-subset set"),
    }


# ---------------------------------------------------------------------------
# page statistics, for sizing and reporting


def page_stats(samples: list[Sample]) -> dict:
    """Document-shape summary of a loaded set, for the run manifest.

    Counted per DOCUMENT as well as per row, because a 468-page document with
    3 questions and a 9-page one with 20 pull the per-row median in opposite
    directions and only the per-document view describes the corpus.
    """
    if not samples:
        return {}
    per_row = [s.public.meta["n_pages"] for s in samples]
    per_doc: dict[str, int] = {}
    for s in samples:
        per_doc[s.group_key] = s.public.meta["n_pages"]
    docs = sorted(per_doc.values())
    pooled = [s.public.meta["n_pages_pooled"] for s in samples]
    return {
        "n_samples": len(samples),
        "n_documents": len(per_doc),
        "pages_per_row_median": statistics.median(per_row),
        "pages_per_row_max": max(per_row),
        "pages_per_doc_median": statistics.median(docs),
        "pages_per_doc_mean": round(sum(docs) / len(docs), 1),
        "pages_per_doc_min": min(docs),
        "pages_per_doc_max": max(docs),
        "total_pages_in_corpus": sum(docs),
        "pooled_per_row_median": statistics.median(pooled),
        "rows_with_capped_pool": sum(
            1 for s in samples if not s.public.meta["pool_is_complete"]),
    }
