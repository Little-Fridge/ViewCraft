"""BLINK adapter (D11): core visual perception that language cannot shortcut.

Why this domain is in the study: BLINK targets perception a human solves "in
a blink" and a VLM largely cannot --- relative depth, reflectance, multi-view
correspondence, jigsaw fit, forensic detail. Published SOTA sits near 0.5
against a 0.38-0.50 chance floor, while humans score 0.96, so the headroom is
real rather than a scoring artefact. Two properties make it the sharpest test
in this study of the mechanism's LAYOUT half:

  * Most tasks are MULTI-IMAGE (2-4 images, 1763 of 1901 items carry more
    than one). Every other static domain here has a single image, so psi's
    choices --- how many images fit in a request, what order they arrive in,
    what is evicted from history --- are only ever exercised nontrivially by
    BLINK. With four images and a 6-image cap, "what do I keep" has real
    answers.
  * The questions refer to images POSITIONALLY ("the first image", "image 2"),
    so a presentation change that reorders or drops an image does not merely
    cost information --- it makes the question refer to the wrong picture. This
    is the only domain here that can detect an ordering bug as a wrong ANSWER
    rather than as a silent accuracy dip, which is exactly the failure mode
    the paper claims to repair.

Released layout: one parquet per task per split under `<Task>/val-*.parquet`,
with `image_1`..`image_4` as struct<bytes,path> (later ones NULL when the
task uses fewer), `choices` a list of option strings, `answer` the gold as a
PARENTHESISED LETTER, `prompt` the full question with a task preamble and the
options inlined, `question` a SHORTER stem with the preamble stripped, and
`sub_task`/`idx` for provenance. `explanation` is present but mostly empty.

The question comes from `prompt` (minus its inlined option list), NOT from
`question`. This matters: in 12 of the 14 sub-tasks the preamble that only
`prompt` carries is what DEFINES the task, and dropping it makes items
unanswerable rather than merely harder. Multi-view_Reasoning is the clearest
case --- `prompt` fixes the convention "clockwise (left) or counter-clockwise
(right)" that the gold label depends on, and without it the adapter measured
0.10 on a TWO-option task, i.e. well below the 0.5 chance floor, which is the
signature of a systematically inverted label and not of a hard question. The
official BLINK harness also evaluates on `prompt`, so this matches the
protocol; see question_text for the per-task detail.

Only the VAL split is loaded, and this is a protocol requirement rather than
a convenience: BLINK withholds the test answers for its leaderboard, so
`test-*.parquet` ships an empty `answer` column. Loading it would silently
produce unscoreable items. The val split (1901 items over 14 tasks) IS the
public evaluation set every reported BLINK number uses.

Scoring is the official BLINK metric: exact multiple-choice accuracy over the
released `choices`, unweighted across tasks. Options are presented in the
order the release ships them and never re-sorted --- BLINK's gold letters are
already spread across the option positions, and shuffling would only add a
mapping layer that could silently invert.

ONE RELEASE QUIRK, handled rather than deviated from: 13 of the 14 tasks
write the gold as "(A)" but Relative_Reflectance writes a bare "A". Both
dialects are parsed to the same option index; an adapter that only understood
"(A)" would score all 134 reflectance items as unmatched and silently drop a
whole task.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "blink"

# Gold is "(A)" in 13 tasks and a bare "A" in Relative_Reflectance.
_GOLD = re.compile(r"^\s*\(?\s*([A-H])\s*\)?\s*$")

IMAGE_COLUMNS = ("image_1", "image_2", "image_3", "image_4")

# BLINK withholds test-split answers for its leaderboard; the split is chosen
# by filename rather than by globbing so an unscoreable file cannot be
# absorbed silently.
VAL_GLOB = "val-*.parquet"

# `prompt` = task preamble + question + an inlined option list, introduced by
# one of two headers ("choices." in 11 tasks, "options." in 3). The option
# list is cut off because the runtime renders `options` itself; presenting it
# twice gave the model two numberings to choose between.
_CHOICE_BLOCK = re.compile(
    r"\n?\s*Select from the following (?:choices|options)\.\s*(?:\n.*)?$",
    re.DOTALL | re.IGNORECASE)


def question_text(row: dict) -> str:
    """The full question, INCLUDING the task preamble the release ships.

    This reads `prompt` rather than `question`, and that choice is
    load-bearing rather than cosmetic. In 12 of the 14 sub-tasks `prompt`
    opens with a preamble that DEFINES the task, and `question` omits it:

      * Multi-view_Reasoning states the convention the answer depends on ---
        "the camera is either moving clockwise (LEFT) or counter-clockwise
        (RIGHT)". Without it the question "is the camera moving left or
        right" is genuinely ambiguous, and the model answers under the
        opposite convention. Measured: 0.10 on a TWO-option task, i.e. far
        below the 0.5 chance floor, which is the signature of a systematically
        inverted label rather than of a hard question.
      * Object_Localization / Visual_Correspondence / Relative_Reflectance
        explain what the drawn boxes, circles and labels MEAN.
      * IQ_Test, Jigsaw, Functional_Correspondence and Semantic_Correspondence
        state what relation to look for at all.

    Using `question` there does not make the benchmark harder, it makes it
    unanswerable, and it would be reported as a model weakness. The official
    BLINK evaluation feeds `prompt`, so this matches the protocol.
    """
    prompt = str(row.get("prompt") or "").strip()
    if prompt:
        stem = _CHOICE_BLOCK.sub("", prompt).strip()
        if stem:
            return stem
    # No prompt column (a re-export could drop it): fall back to the bare
    # question rather than returning nothing.
    return str(row.get("question") or "").strip()


def parse_gold(raw: str, n_options: int) -> int | None:
    """Map a released gold answer to an option index, or None.

    Accepts both "(A)" and "A". Returns None when the letter falls outside
    the released `choices`, so the row is dropped rather than scored against
    an option that does not exist.
    """
    m = _GOLD.match(raw or "")
    if not m:
        return None
    idx = ord(m.group(1).upper()) - ord("A")
    return idx if 0 <= idx < n_options else None


def load(root: str | Path, seed: int = 20260911,
         tasks: tuple[str, ...] | None = None) -> list[Sample]:
    """Load the BLINK val split as public/private pairs.

    `tasks` restricts to a subset of the 14 sub-tasks (directory names, e.g.
    "Relative_Depth"); the default keeps all of them, which is the official
    headline configuration.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    pqs = sorted(p for p in root.rglob(VAL_GLOB) if ".cache" not in p.parts)
    if not pqs:
        raise FileNotFoundError(f"no {VAL_GLOB} under {root}")

    keep = set(tasks) if tasks else None

    samples: list[Sample] = []
    for p in pqs:
        # The task name is the directory, which is also the only place it is
        # spelled consistently (`sub_task` is blank on a few rows).
        task = p.parent.name
        if keep and task not in keep:
            continue
        for row in pq.read_table(p).to_pylist():
            choices = [str(c) for c in (row.get("choices") or [])]
            if len(choices) < 2:
                continue
            gold_idx = parse_gold(str(row.get("answer") or ""), len(choices))
            if gold_idx is None:
                # No parseable gold inside the option set: unscoreable, and
                # this is how a test-split file would announce itself.
                continue

            # `prompt` carries the task preamble that `question` omits, and in
            # 12 of 14 sub-tasks that preamble DEFINES the task. See
            # question_text: using the bare `question` scored
            # Multi-view_Reasoning at 0.10 on a two-option task.
            question = question_text(row)
            if not question:
                continue

            # Order matters: the questions refer to images positionally ("the
            # first image"), so images are emitted in column order and a NULL
            # column ends the list rather than leaving a hole.
            media: list[str] = []
            for col in IMAGE_COLUMNS:
                blob = row.get(col)
                if isinstance(blob, dict):
                    blob = blob.get("bytes")
                if not blob:
                    continue
                media.append(write_media(DOMAIN, blob, f"{row.get('idx')}_{col}"))
            if not media:
                continue

            idx = str(row.get("idx") or "")
            item_id = f"{DOMAIN}/{task}/{idx}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN, question=question,
                media_ids=tuple(media), options=tuple(choices),
                meta={"task": task, "n_images": len(media),
                      "sub_task": row.get("sub_task") or task},
            )
            private = PrivateItem(
                item_id=item_id, answer=choices[gold_idx],
                answer_type="multiple_choice", evidence_source="none",
                meta={"task": task, "answer_letter": chr(ord("A") + gold_idx),
                      "answer_position": gold_idx,
                      # Released free-text rationale, empty on most rows. Kept
                      # private: it frequently names the answer.
                      "explanation": row.get("explanation") or ""},
            )
            # Group by the FIRST image, mirroring ChartQA's by-image grouping.
            # 11 val items share an image with another item in the same task
            # (the same painting asked about twice), and letting one image
            # appear in both gate and test would let a model tune on pixels it
            # is later measured on. The media digest is content-addressed, so
            # two rows sharing a first image share a group automatically.
            samples.append(Sample(public=public, private=private,
                                  group_key=Path(media[0]).stem))
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official BLINK accuracy: exact multiple choice over released choices."""
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(
        raw=answer_text, options=list(public.options),
        correct_index=private.meta["answer_position"])
