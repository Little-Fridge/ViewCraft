"""CountBench adapter (D10): counting objects that are all plainly visible.

Why this domain is in the study: counting is the one visual task where the
answer depends on resolving EVERY instance, not on recognising the gist. A
model that sees a downscaled photo of a flock can say "birds" with total
confidence and still count seven where there are nine, because adjacent or
small instances merge into one blob at low resolution. Unlike V*Bench, where
one small target must be found in a large scene, CountBench's objects are
usually large and unambiguous --- the failure is in ENUMERATION under a fixed
pixel budget, and it is repaired by looking at parts of the image in turn
rather than by zooming to a single place. That makes it a test of the LAYOUT
half of the mechanism (psi: how many views, in what order, what is kept in
history) rather than of crop targeting, which no other domain in the study
isolates.

The existing benchmarks are saturated (raw accuracy 0.77-0.91), and counting
is the documented weak spot of every current VLM, so this is chosen as a
domain with headroom rather than for variety.

Released layout: `nielsr/countbench` is ONE parquet, 540 rows, with
`image` as struct<bytes,path> (real image BYTES), `number` the integer count
(2-10, exactly 60 rows each), `text` the original web caption, and
`image_url` the source URL. The split is named "train" in the release; there
is no other split, and the whole file is the benchmark.

49 of the 540 rows carry a NULL image and are dropped, leaving 491. Those
rows are the ones whose `image_url` had already 404'd when the parquet was
exported --- CountBench originally shipped URLs only, and this re-export
froze whatever was still reachable. The dropped count is printed at load so
a shrinking dataset cannot be mistaken for an adapter bug. We do NOT refetch
from `image_url`: those links are dead, and a benchmark whose contents depend
on the day it was downloaded is not reproducible.

TWO DECLARED DEVIATIONS:

1. TASK FRAMING. Official CountBench (Paiss et al., "Teaching CLIP to Count
   to Ten", ICCV 2023) is a CLIP-style RETRIEVAL benchmark: the image is
   scored against 9 rewrites of its caption, one per count word two..ten, and
   accuracy is how often the true caption wins the similarity ranking. That
   protocol needs an image-text similarity head and cannot be run on a
   generative VLM at all. We therefore ask for the count directly and score
   an exact integer match. The measured quantity is unchanged --- accuracy of
   identifying the correct count from the pixels, chance 1/9 = 11.1% --- but
   it is strictly HARDER than the official framing, because the model is not
   shown the 2-10 candidate set. It is scored as open-ended, so a model that
   answers "12" is wrong rather than being snapped to the nearest option.

2. CAPTION AS OBJECT NAME. "How many?" is unanswerable without saying what to
   count, and the release has no object-label column --- only the caption. The
   noun phrase FOLLOWING the caption's count word is therefore used to name
   the target: "We review the ten best gaming headsets in the market" becomes
   "best gaming headsets". Masking the count word is mandatory and verified:
   every one of the 491 image-bearing captions contains its count word, so an
   unmasked caption would state the answer in the question and the domain
   would score ~1.0 while measuring nothing. A row is DROPPED, never asked,
   when the count word cannot be found, when nothing usable follows it, or
   when the extracted phrase still contains any number --- 80 of 491 rows
   (16%) fail one of those checks, leaving 406 items. Dropping is by CAPTION
   SHAPE, decided before any score was measured, and is independent of how
   hard the image is to count.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "countbench"

# CountBench covers exactly 2..10; the release has 60 rows per count.
_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

# Captions are raw web alt-text: stock-photo boilerplate, prices, and
# "- Illustration of the four colorful..." tails. Only the noun phrase that
# FOLLOWS the count word is used, so the surrounding marketing copy never
# reaches the question.
_NOISE = re.compile(
    r"(https?://\S+)"
    r"|(\$[\d.,]+(\s*usd)?)"
    r"|\b(stock (photo|image|vector|illustration)s?|royalty[- ]free|"
    r"vector art|clip ?art|isolated (over|on)|white background)\b",
    re.IGNORECASE)

# Determiners/adjectival filler that can sit between the count word and the
# noun ("the ten BEST gaming headsets", "four COLORFUL parrots"). Adjectives
# are kept -- they help identify what to count -- but a leading article is
# dropped so the phrase reads naturally after "How many".
_ARTICLE = re.compile(r"^(the|a|an|these|those|its|his|her|their|our|my)\s+",
                      re.IGNORECASE)

# The noun phrase ends at a clause boundary. Without this, a caption like
# "seven pages. Each page has a large 4th of July theme" drags a whole second
# sentence into the question.
_STOP = re.compile(
    r"[.;:!?()\[\]\"“”|]|(\s[-–—]\s)"
    r"|\b(in|on|at|for|with|from|over|under|isolated|against|and|or|that|"
    r"which|who|while|when|as|by|to|of|wrapped|instead|having|is|are|was|"
    r"were|each|every|set)\b",
    re.IGNORECASE)


def mask_count_word(text: str, number: int) -> str:
    """Return the object phrase that FOLLOWS the count word in a caption.

    CountBench has no object-label column, so the caption must name what to
    count. Taking the words AFTER the count word (rather than deleting the
    count word from the whole caption) is what keeps the question readable:
    "We review the ten best gaming headsets in the market" yields
    "best gaming headsets", not "We review the best gaming headsets in the
    market".

    Returns "" when the count word is absent or nothing usable follows it, so
    the caller drops the row rather than asking an unanswerable question or
    one whose own text states the answer.
    """
    word = _NUMBER_WORDS.get(number)
    if not word:
        return ""
    t = _NOISE.sub(" ", text or "")
    t = re.sub(r"\s{2,}", " ", t)
    # Both spellings: a caption may carry the word or the digit ("set of six",
    # "Set of 6 arrows"), and either one states the answer.
    m = re.search(rf"\b({word}|{number})\b", t, re.IGNORECASE)
    if not m:
        return ""
    tail = t[m.end():]
    cut = _STOP.search(tail)
    if cut:
        tail = tail[: cut.start()]
    tail = _ARTICLE.sub("", tail.strip(" ,.-–—:;|'\"").strip())
    tail = re.sub(r"\s{2,}", " ", tail).strip(" ,.-–—:;|").strip()
    # A bare plural noun is fine ("arrows"); an empty or single-letter tail is
    # not, and neither is one that still carries a digit (another count).
    if len(tail) < 3 or re.search(r"\d", tail):
        return ""
    # A SECOND count word in the tail ("three of the five puppies") would
    # state a number in the question even though the gold count was masked,
    # so the row is dropped rather than asked.
    if re.search(r"\b(" + "|".join(_NUMBER_WORDS.values()) + r")\b",
                 tail, re.IGNORECASE):
        return ""
    # Cap the length: anything longer than a short noun phrase means the stop
    # list missed a boundary, and a rambling question tests reading rather
    # than counting.
    if len(tail.split()) > 5:
        return ""
    return tail


def parse_count(raw: str) -> int | None:
    """Read an integer count out of a free-form reply, or None.

    Accepts a digit or an English number word, and prefers the FIRST number
    in the reply: "7, because I see seven cars" must resolve to 7 and not to
    whichever number happens to appear last.
    """
    t = (raw or "").strip().lower()
    if not t:
        return None
    words = {w: n for n, w in _NUMBER_WORDS.items()}
    words.update({"zero": 0, "one": 1, "eleven": 11, "twelve": 12})
    m = re.search(r"(\d+)|\b(" + "|".join(words) + r")\b", t)
    if not m:
        return None
    if m.group(1) is not None:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return words[m.group(2)]


def load(root: str | Path, seed: int = 20260911) -> list[Sample]:
    """Load CountBench as public/private pairs (the release's only split)."""
    import pyarrow.parquet as pq

    root = Path(root)
    pqs = sorted(p for p in root.rglob("*.parquet") if ".cache" not in p.parts)
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")

    samples: list[Sample] = []
    no_image = 0
    unmaskable: list[str] = []
    # Two rows can share an image (the same stock photo captioned twice). The
    # media digest is the image identity, so a duplicate (image, count) pair
    # would weight one photo twice in the mean.
    seen: set[tuple[str, int]] = set()

    for p in pqs:
        for i, row in enumerate(pq.read_table(p).to_pylist()):
            blob = row.get("image")
            if isinstance(blob, dict):
                blob = blob.get("bytes")
            if not blob:
                # Image URL was already dead when the parquet was exported.
                # Dropped, never refetched: see the module docstring.
                no_image += 1
                continue

            number = row.get("number")
            if number is None:
                continue
            number = int(number)

            phrase = mask_count_word(str(row.get("text") or ""), number)
            if not phrase:
                # The count word could not be removed, so the question would
                # contain its own answer. Dropped rather than asked.
                unmaskable.append(str(row.get("text") or "")[:40])
                continue

            path = write_media(DOMAIN, blob, str(row.get("image_url") or ""))
            dedup = (Path(path).stem, number)
            if dedup in seen:
                continue
            seen.add(dedup)

            question = (f"How many {phrase} are in this image? "
                        "Answer with a number.")
            item_id = f"{DOMAIN}/{number}/{Path(path).stem[:12]}_{i}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN, question=question,
                media_ids=(path,),
                # `count` is deliberately NOT in meta: PublicItem forbids
                # answer-bearing keys and the count IS the answer.
                meta={"object_phrase": phrase},
            )
            private = PrivateItem(
                item_id=item_id, answer=number, answer_type="integer",
                evidence_source="none",
                meta={"count": number, "object_phrase": phrase,
                      "image_url": row.get("image_url")},
            )
            # Group by the IMAGE, matching ChartQA: if one photo ever carries
            # two questions, letting it appear in both gate and test would let
            # a model tune on pixels it is later measured on.
            samples.append(Sample(public=public, private=private,
                                  group_key=Path(path).stem))

    if no_image or unmaskable:
        print(f"[countbench] kept {len(samples)}; dropped {no_image} rows with "
              f"a null image (source URL already dead at export) and "
              f"{len(unmaskable)} whose count word could not be masked out of "
              "the caption (the question would have stated its own answer).")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Exact integer match against the released count.

    Open-ended: the 2-10 candidate set is never shown and a reply outside it
    is wrong, not snapped to the nearest legal count. See deviation 1.
    """
    got = parse_count(answer_text)
    return 1.0 if got is not None and got == int(private.answer) else 0.0
