"""RealWorldQA adapter (D12): spatial understanding in real driving scenes.

Why this domain is in the study: RealWorldQA is photographs from vehicles ---
crowded roads, traffic furniture, other cars at distance --- asking questions
whose answer sits in a SMALL part of a WIDE scene ("what colour is the fire
hydrant ahead", "how many sedans within 30 metres"). That is the same
resolution failure V*Bench isolates, but on natural cluttered imagery at a
much larger question count, and with a second difficulty V*Bench lacks: many
questions are metric or relational ("within 5 metres", "over the jersey
barrier"), so the model must both resolve the object and place it in the
scene. A downscale that keeps the road recognisable still destroys the
hydrant, which makes this a clean phi (crop-targeting) domain rather than a
general "real world reasoning" set.

Released layout: two parquets (383 + 382 = 765 rows) with `image` as
struct<bytes,path> (real BYTES, mostly webp), `image_path` the original file
name, `question` the full released prompt, and `answer` the gold.

The release is MIXED-FORMAT and the format is announced by the last line of
`question`, which is one of exactly two sentences:

  "Please answer directly with only the letter of the correct option and
   nothing else."          -> 438 rows, multiple choice
  "Please answer directly with a single word or number."
                           -> 327 rows, free-form short answer

Both instruction lines are STRIPPED from the stem, because our own prompt
already specifies the answer shape and shipping both produced contradictory
instructions (the same reason mmstar.py strips its "Hint:" line). The option
block is parsed out of the stem and handed over as `options`, so the runtime
renders the choices rather than the question restating them.

Options appear in two dialects --- "A. Left" (428 rows) and "A: Red" (10
rows) --- and both are parsed. An adapter that understood only "A." would
present 10 items with no choices and score them as unanswerable.

Options are presented in the order the release ships them and never
re-sorted; the gold letter is already spread across A/B/C (150/142/142), so
there is nothing to de-bias and shuffling would only add a mapping layer that
could silently invert.

Scoring follows the format of each item, which is the official RealWorldQA
protocol (it is evaluated as plain accuracy over both halves, reported as one
number):
  * multiple choice -> scorers.match_multiple_choice, exact option match
  * free-form       -> case/whitespace-insensitive exact match, with numeric
                       answers compared as NUMBERS so "2" and "2.0" agree

ONE DECLARED DEVIATION: the free-form half is scored by EXACT normalized
match, not by the substring-containment rule some harnesses use. Containment
is unsafe on this set because the gold vocabulary is tiny and overlapping ---
"Yes"/"No" (163 rows) and bare digits 0-8 (116 rows). A reply of "No, there
are 2" contains both "No" and "2", so containment would mark it correct
against several different golds, including contradictory ones. Exact match is
strictly harsher and cannot be gamed by a verbose reply; it is the only
deviation from the loosest published harness and it is documented here rather
than applied silently.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "realworldqa"

# The two instruction sentences the release appends. Stripped from the stem:
# our own prompt already states the answer shape, and keeping both gave the
# model contradictory instructions.
_INSTR_LETTER = "only the letter of the correct option"
_INSTR_SHORT = "a single word or number"
_INSTR_LINE = re.compile(
    r"\n?\s*Please answer directly with (?:only the letter of the correct "
    r"option and nothing else|a single word or number)\.\s*$",
    re.IGNORECASE)

# Dialect 1: "\nA. Left"    Dialect 2: "\nA: Red"
_OPTION_LINE = re.compile(r"(?:^|\n)\s*([A-H])\s*[.:]\s+(.+?)(?=\n\s*[A-H]\s*[.:]\s|\Z)",
                          re.DOTALL)


def parse_question(raw: str) -> tuple[str, list[str], list[str], bool]:
    """Split a released `question` into (stem, letters, options, is_mc).

    `is_mc` comes from the release's own trailing instruction line, not from
    whether options happened to parse, so a multiple-choice row whose option
    block is malformed is reported as MC with no options and dropped by the
    caller rather than silently rescored as free-form.
    """
    text = raw or ""
    tail = text.strip().rsplit("\n", 1)[-1].lower()
    is_mc = _INSTR_LETTER in tail
    body = _INSTR_LINE.sub("", text.strip()).strip()

    if not is_mc:
        return body, [], [], False

    matches = list(_OPTION_LINE.finditer(body))
    if len(matches) < 2:
        return body, [], [], True
    stem = body[: matches[0].start()].strip()
    letters = [m.group(1).upper() for m in matches]
    texts = [m.group(2).strip() for m in matches]
    # Letters must be a contiguous A,B,C... run; anything else means the
    # regex latched onto prose ("A. M. radio") rather than an option block.
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    if letters != expected or not stem:
        return body, [], [], True
    return stem, letters, texts, True


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower().rstrip(".")


def load(root: str | Path, seed: int = 20260911) -> list[Sample]:
    """Load the RealWorldQA test split as public/private pairs."""
    import pyarrow.parquet as pq

    root = Path(root)
    pqs = sorted(p for p in root.rglob("*.parquet") if ".cache" not in p.parts)
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")

    samples: list[Sample] = []
    dropped_mc = 0
    for p in pqs:
        for row in pq.read_table(p).to_pylist():
            stem, letters, texts, is_mc = parse_question(
                str(row.get("question") or ""))
            gold = str(row.get("answer") or "").strip()
            if not stem or not gold:
                continue

            if is_mc:
                if not letters or gold.upper() not in letters:
                    # Announced as multiple choice but the option block did
                    # not parse, or the gold letter is outside it. Either way
                    # unscoreable; dropped rather than guessed at.
                    dropped_mc += 1
                    continue
                pos = letters.index(gold.upper())
                answer, answer_type = texts[pos], "multiple_choice"
            else:
                pos, answer, answer_type = -1, gold, "string"

            blob = row.get("image")
            if isinstance(blob, dict):
                blob = blob.get("bytes")
            if not blob:
                continue
            path = write_media(DOMAIN, blob, str(row.get("image_path") or ""))

            name = Path(str(row.get("image_path") or "x")).stem
            item_id = f"{DOMAIN}/{'mc' if is_mc else 'open'}/{name}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN, question=stem,
                media_ids=(path,), options=tuple(texts),
                meta={"format": "multiple_choice" if is_mc else "open",
                      "image_path": row.get("image_path")},
            )
            private = PrivateItem(
                item_id=item_id, answer=answer, answer_type=answer_type,
                evidence_source="none",
                meta={"format": "multiple_choice" if is_mc else "open",
                      "answer_position": pos,
                      "answer_letter": gold.upper() if is_mc else "",
                      "image_path": row.get("image_path")},
            )
            # Group by the IMAGE, matching ChartQA: one scene photo can carry
            # several questions, and letting it appear in both gate and test
            # would let a model tune on pixels it is later measured on. The
            # media digest is content-addressed, so two rows sharing a photo
            # share a group even if `image_path` differs.
            samples.append(Sample(public=public, private=private,
                                  group_key=Path(path).stem))

    if dropped_mc:
        print(f"[realworldqa] kept {len(samples)}; dropped {dropped_mc} rows "
              "announced as multiple choice whose option block did not parse "
              "(unscoreable, not rescored as free-form).")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official RealWorldQA accuracy, per the item's released format.

    Multiple choice is an exact option match. Free-form is an exact
    normalized match, with numbers compared numerically so "2" and "2.0"
    agree. Containment is deliberately NOT used on the free-form half --- see
    the declared deviation in the module docstring.
    """
    if private.meta.get("format") == "multiple_choice":
        from ..evaluation.scorers import match_multiple_choice
        return match_multiple_choice(
            raw=answer_text, options=list(public.options),
            correct_index=private.meta["answer_position"])

    from ..evaluation.scorers import _to_float
    pred, gold = _norm(answer_text), _norm(str(private.answer))
    if not pred:
        return 0.0
    pnum, gnum = _to_float(pred), _to_float(gold)
    if pnum is not None and gnum is not None:
        return 1.0 if abs(pnum - gnum) < 1e-9 else 0.0
    return 1.0 if pred == gold else 0.0
