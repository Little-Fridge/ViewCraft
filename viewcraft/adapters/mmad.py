"""MMAD (ICLR 2025) adapter: INDUSTRIAL ANOMALY DETECTION against a normal reference.

Why this domain is in the study. Every other non-video domain here asks the
model to find something inside ONE picture, and the mechanism's most-exercised
move is the zoom. MMAD asks a different question: here is a product image, here
is a picture of a KNOWN-GOOD unit of the same product -- is this one defective,
where, and what kind of defect? The bottleneck is not "can I resolve the
pixels" but "which region of the query corresponds to which region of the
reference, and at what scale is a difference a defect rather than ordinary
manufacturing variation". That is a phi/psi decision (pairing, scale, ordering,
what stays in the context window), and it is the decision the single-image
domains never pose.

The benchmark's own headline finding is why it is worth searching over rather
than prompting once: the paper reports that simply ADDING the normal reference
image does not reliably help, and for several models hurts. A reference image
is only useful if the comparison is organised. That is exactly the object of
the search here.

Released layout: 39,672 multiple-choice questions over 8,366 query images drawn
from four public industrial datasets (GoodsAD, VisA, DS-MVTec, MVTec-LOCO),
across seven subtasks. Every row carries `template_image`, a normal unit of the
same product class, which is attached as the second view.

`mask` IS NEVER ATTACHED AND NEVER ENTERS META. The release ships a pixel mask
of the defect for 24,105 of the rows; it is the localisation ground truth, and
showing it -- or storing it anywhere the executor could reach -- would answer
the "where is the defect" subtask outright. `PublicItem.__post_init__` already
rejects the obvious answer-bearing meta keys; this one is not on that list, so
it is excluded here by construction.

ORDERING HAZARD, stated rather than hidden. The two attached views are
positional: view 1 is the unit under inspection, view 2 is the known-good
reference, and the stem says so. Nothing in the option text recovers that
mapping. Under a psi that evicts a whole-image view or orders crops ahead of
sources, "the first image" stops naming the query, and the item is then scored
wrong for a presentation reason rather than a perceptual one. This is the same
property BLINK was added to detect (registry note: "the only domain here where
a psi ordering bug shows up as a wrong answer rather than as a silent accuracy
dip"), and it is left in place rather than defended against, because navigating
it is part of what the search is being asked to do.

Options are 4-way on 31,400 rows and 2-way (Yes/No anomaly discrimination) on
8,272. Both are kept, because the official metric is accuracy over all seven
subtasks and dropping the binary half would report a number nobody else's
number is comparable to. The per-item option count rides in `meta["n_options"]`
so `scripts/audit_chance_floor.py` can separate the 50% floor from the 25% one.

Gold letters are near-uniform in the release (A 12073, B 11906, C 8057,
D 7636 -- the skew is the 2-way rows, which have no C or D), so there is
nothing to de-bias, and shuffling would add a mapping layer that could silently
invert.

`group_key` is the query image, not the row: 8,366 images carry 39,672
questions, so splitting by row would ask about the same photograph in train and
in test. Note what this does NOT prevent: a product CLASS (38 of them) appears
on both sides of any split, and so does its reference image. That is inherent
to a 38-class benchmark and is stated so the split is not read as stronger than
it is; what is prevented is the same photograph being seen twice.

Scoring is the official MMAD metric: exact multiple-choice accuracy.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "mmad"

# The seven official subtasks, recovered from the question template. Recorded
# as the item's category so the runner's category-aware gate capping keeps the
# mix, and so a per-subtask breakdown is possible without a second pass.
#
# The release ships no subtask column, only free-text questions, so the split
# is made on the one reliable signal: whether the question is about the DEFECT
# or about the PRODUCT. Everything naming a defect belongs to the four defect
# subtasks; everything else is object classification ("what kind of product")
# or object analysis (shape / colour / material / purpose / layout /
# subcomponents -- 9,138 rows phrased ~40 different ways, which is why this
# side is a fallback rather than a keyword list that would silently drop the
# long tail into "other").
DEFECT_SUBTASKS = (
    ("anomaly_discrimination", ("is there any",)),
    ("defect_localization", ("where",)),
    ("defect_classification", ("type of",)),
    ("defect_description", ("appearance of",)),
    ("defect_analysis", ("effect of",)),
)

PREAMBLE = (
    "Two images are attached. The FIRST is the product unit under inspection. "
    "The SECOND is a known-good unit of the same product, provided for "
    "reference; it is not the subject of the question.\n\n"
)


def subtask_of(question: str) -> str:
    q = question.lower()
    if "defect" in q or "anomal" in q:
        for name, keys in DEFECT_SUBTASKS:
            if any(k in q for k in keys):
                return name
        return "defect_analysis"
    if "kind of product" in q or "what product" in q:
        return "object_classification"
    return "object_analysis"


def parse_options(raw: str) -> tuple[str, ...]:
    """Split the release's "A: text\\nB: text" block into option texts.

    The letter prefix is dropped: the executor is shown the option TEXTS and
    answers with a letter the harness assigns, so carrying the release's own
    letters through would give two independent labellings of the same list.
    """
    out = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if len(line) > 2 and line[0].isalpha() and line[1] == ":":
            out.append(line[2:].strip())
        elif line:
            return ()                      # unexpected shape: drop the row
    return tuple(out)


def load(root: str | Path, seed: int = 20260911,
         n_items: int = 1500) -> list[Sample]:
    import pyarrow.parquet as pq

    root = Path(root)
    metas = sorted(root.rglob("mmad_meta.parquet"))
    if not metas:
        raise FileNotFoundError(f"no mmad_meta.parquet under {root}")
    rows = []
    for p in metas:
        rows.extend(pq.read_table(p).to_pylist())

    # Deterministic subsample that preserves the official subtask and source
    # mix: hash each row's identity to a stable float and keep the lowest
    # fraction WITHIN each (subtask, source) stratum. Seed-independent by
    # design -- the benchmark's composition should not move between runs.
    strata: dict[tuple[str, str], list] = {}
    for r in rows:
        q = str(r.get("question") or "")
        img = str(r.get("query_image") or "")
        if not q or not img:
            continue
        strata.setdefault((subtask_of(q), img.split("/")[0]), []).append(r)

    keep = []
    total = sum(len(v) for v in strata.values())
    for key, group in sorted(strata.items()):
        group.sort(key=lambda r: hashlib.sha256(
            f"{r['query_image']}|{r['question']}|{r['options']}".encode()
        ).hexdigest())
        take = max(1, round(n_items * len(group) / total))
        keep.extend(group[:take])

    samples: list[Sample] = []
    dropped_opts = dropped_media = 0
    for r in keep:
        options = parse_options(str(r.get("options") or ""))
        gold = str(r.get("answer") or "").strip().upper()
        if len(options) < 2 or len(gold) != 1:
            dropped_opts += 1
            continue
        idx = ord(gold) - ord("A")
        if not 0 <= idx < len(options):
            dropped_opts += 1
            continue

        query = root / str(r["query_image"])
        template = root / str(r.get("template_image") or "")
        # Both files must exist on disk: the renderer re-reads the ORIGINAL
        # file for every crop, so a path that resolves only at load time would
        # fail mid-trajectory instead of here.
        if not query.exists() or not template.exists():
            dropped_media += 1
            continue

        stem = str(r["query_image"])
        item_id = f"{DOMAIN}/{stem}/{hashlib.sha256(str(r['question']).encode()).hexdigest()[:8]}"
        sub = subtask_of(str(r["question"]))
        public = PublicItem(
            item_id=item_id, domain=DOMAIN,
            question=PREAMBLE + str(r["question"]).strip(),
            media_ids=(str(query), str(template)), options=options,
            # No mask, no answer: see the module docstring.
            meta={"subtask": sub, "category": sub, "n_images": 2,
                  "n_options": len(options),
                  "source": stem.split("/")[0],
                  "product_class": stem.split("/")[1] if "/" in stem else ""},
        )
        private = PrivateItem(
            item_id=item_id, answer=options[idx],
            answer_type="multiple_choice", evidence_source="none",
            meta={"answer_position": idx, "answer_letter": gold,
                  "subtask": sub},
        )
        # The query PHOTOGRAPH is the unit that must not straddle a split.
        samples.append(Sample(public=public, private=private,
                              group_key=hashlib.sha256(stem.encode()
                                                       ).hexdigest()[:16]))

    print(f"[{DOMAIN}] kept {len(samples)} items "
          f"(dropped {dropped_opts} option-shape, {dropped_media} missing "
          f"media); {len({s.group_key for s in samples})} distinct query "
          f"images")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(
        raw=answer_text, options=list(public.options),
        correct_index=private.meta["answer_position"])
