"""ChartQA adapter (D9): reading values off a plotted axis.

Why this domain is in the study: a chart answer is recoverable only if the tick
labels, legend entries and bar ends survive to the pixels the model actually
sees. Downscaling a chart destroys the axis text long before it destroys the
chart's overall shape, so a model can still describe the figure while losing the
ability to read a number off it. That is a resolution failure with a
deterministic, numeric ground truth, which makes ChartQA a clean discriminator
for phi rather than a general "chart reasoning" benchmark.

Released layout: parquet per split with `image` as RAW BYTES (not a
struct<bytes,path> like most HF vision sets), `query` the question, `label` the
short answer, and `type` marking the two question provenances --- "human"
(written by annotators) and "augmented" (machine-generated from the underlying
table). Both halves are 1250 rows in test and both are kept: the official
headline number is the mean over the two, and reporting them separately is
informative because the augmented half is templated and much easier.

Only the TEST split is loaded. train/val exist in the same directory but are
not part of the protocol, and silently absorbing them would change the
denominator of every reported number.

Scoring is scorers.relaxed_accuracy, the official ChartQA definition: numeric
answers correct within a 5% RELATIVE tolerance, non-numeric answers by exact
normalized match. The metric itself is unmodified.

ONE DECLARED DEVIATION: questions whose gold answer is a bare calendar year
are excluded. A 5% relative tolerance is right for magnitudes and vacuous for
years -- at gold=2020 it accepts 1919 to 2121 -- and a real sample scored
"2012" as CORRECT against a gold of "2020". 143 of 2490 test questions (5.7%)
are year-answered, so leaving them in inflates any reported score by up to
5.7 points and, worse, makes the axis-label questions most likely to be
affected by visual presentation the ones the metric cannot measure. They are
dropped rather than rescored so the official metric is untouched on every
question that remains.
"""
from __future__ import annotations

from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "chartqa"


def _is_year_answer(ans: str) -> bool:
    """True when the gold answer is a bare calendar year.

    The official relaxed-accuracy tolerance is 5% RELATIVE, which is correct
    for magnitudes and vacuous for years: at gold=2020 it accepts anything
    from 1919 to 2121. Observed in a real sample -- the model answered 2012
    against a gold of 2020 and was scored CORRECT.
    """
    a = (ans or "").strip()
    if not a.isdigit() or len(a) != 4:
        return False
    return 1900 <= int(a) <= 2035

# train/val ship alongside test in the same data/ directory and are outside the
# protocol; the split is selected by filename rather than by globbing.
TEST_GLOB = "test-*.parquet"


def load(root: str | Path, seed: int = 20260911,
         types: tuple[str, ...] | None = None) -> list[Sample]:
    """Load the ChartQA test split as public/private pairs.

    `types` restricts to a question provenance ("human" / "augmented"); the
    default keeps both, which is the official headline configuration.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    pqs = sorted(root.rglob(TEST_GLOB))
    if not pqs:
        raise FileNotFoundError(f"no {TEST_GLOB} under {root}")

    keep = set(types) if types else None

    samples: list[Sample] = []
    dropped_year: list[str] = []
    # ChartQA's test split contains a handful of rows that are byte-identical in
    # (image, query, label, type) --- the same question asked twice of the same
    # chart. They are not a second observation of anything: scoring both would
    # weight one chart-question twice in the mean and inflate whichever way the
    # model happens to answer it. First occurrence wins; the count of dropped
    # rows is reported in the manifest.
    seen: set[tuple[str, str, str, str]] = set()

    for p in pqs:
        for i, row in enumerate(pq.read_table(p).to_pylist()):
            qtype = str(row.get("type") or "")
            if keep and qtype not in keep:
                continue
            question = str(row.get("query") or "").strip()
            answer = str(row.get("label") or "").strip()
            if not question or not answer:
                # No question or no gold: unscoreable, dropped not guessed.
                continue

            imgname = str(row.get("imgname") or "")
            dedup = (imgname, question, answer, qtype)
            if dedup in seen:
                continue
            seen.add(dedup)

            if _is_year_answer(answer):
                # ChartQA's official relaxed accuracy allows a 5% RELATIVE
                # tolerance, which is right for magnitudes and meaningless
                # for years: at gold=2020 it accepts 1919-2121, so "2012"
                # scored as correct against "2020" in a real sample. 143 of
                # 2490 questions (5.7%) have a year answer, and on those the
                # metric checks nothing, inflating any score by up to 5.7
                # points.
                #
                # They are EXCLUDED rather than rescored, so the official
                # metric is left untouched on every question that remains.
                # The exclusion is by ANSWER TYPE, decided before any score
                # was measured, and is reported as a protocol deviation.
                dropped_year.append(f"{imgname}:{question[:40]}")
                continue

            # `image` is raw bytes here, but tolerate a struct-shaped column so
            # a re-export of the dataset does not silently yield zero samples.
            blob = row.get("image")
            if isinstance(blob, dict):
                blob = blob.get("bytes")
            if not blob:
                continue
            path = write_media(DOMAIN, blob, imgname)

            item_id = f"{DOMAIN}/{qtype}/{Path(imgname).stem}_{i}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN, question=question,
                media_ids=(path,),
                meta={"type": qtype, "imgname": imgname},
            )
            private = PrivateItem(
                item_id=item_id, answer=answer, answer_type="string",
                evidence_source="none",
                meta={"type": qtype, "imgname": imgname},
            )
            # Group by the IMAGE, not by the row: a chart carries up to five
            # questions in this split, and letting one chart appear in both gate
            # and test would let a model tune on pixels it is later measured on.
            # The media digest is the image identity (content-addressed), so two
            # rows sharing a chart share a group even if the names differ.
            samples.append(Sample(public=public, private=private,
                                  group_key=Path(path).stem))
    if dropped_year:
        print(f"[chartqa] excluded {len(dropped_year)} year-answer questions "
              f"({len(dropped_year) / max(1, len(dropped_year) + len(samples)) * 100:.1f}%): "
              "the official 5% relative tolerance is vacuous on years "
              "(gold 2020 accepts 1919-2121). Reported as a deviation.")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official ChartQA relaxed accuracy (5% relative tolerance on numbers)."""
    from ..evaluation.scorers import relaxed_accuracy
    return relaxed_accuracy(pred=answer_text, gold=str(private.answer))
