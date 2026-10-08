"""MMIU adapter (OpenGVLab/MMIU) -- MULTI-IMAGE CORRESPONDENCE and
DIFFERENCE DETECTION across 2-62 images per question.

WHY THIS DOMAIN IS IN THE STUDY
-------------------------------
It is the substitute for the fifth S-tier slot after CGAVCounting's data
turned out to be gated. What it buys is a bottleneck no other domain here
isolates at this scale: the answer lives in the RELATION between N
independent images, not inside any one of them -- which of four images is
the real photograph, what changed between two frames of a clip, where a
point sits in a second view. Three properties make it a natural mechanism
test rather than another accuracy number:

  * LAYOUT GRID + CANVAS HOME GROUND. With up to 62 addressable images per
    question and an 8-image send cap, the question of HOW the images are
    composed (side-by-side grid, marked candidates, jot-as-you-compare
    canvas notes) is the decision that determines whether the comparison
    can happen at all. Single-image domains never pose it; muirbench poses
    it at 2-9 images without per-subtask difficulty spread.
  * TOPK PAGE SELECTION IS APPROXIMATELY INERT. Unlike the document
    domains, where select_expr picking the right page is most of the game,
    here most subtasks need ALL of the compared images co-resident -- a
    top-k over the image set cannot win by picking one. That makes MMIU a
    built-in CONTROL for the selection coordinate: if a gain shows up
    here, it came from composition/rendering/memory, not from retrieval.
  * POSITIONAL OPTIONS. As in blink and scifibench, the option letter is
    often positional ("A: the first image"), so a psi that evicts or
    reorders a whole-image view shows up as a wrong answer rather than a
    silent dip.

RELEASED STRUCTURE
------------------
One parquet (`all.parquet`, 11,698 rows; the card's "11K questions"),
one row per question:

    task                    subtask name (60 observed, card says 52)
    visual_input_component  e.g. "natural image and synthetic image"
    source                  source dataset (blink, ...)
    options                 ONE string, "A: text\\nB: text\\n..."
    question                the question
    context                 task preamble, usually ending in
                            "Select from the following choices.\\nA: ..."
    input_image_path        ["./<Domain-dir>/<task>/<file>", ...], ordered
    output                  gold letter ("A".."J")

Images are NOT embedded in the parquet; they live in per-domain archives
(`Low-level-semantic.zip`, `Continuous-temporal.zip`, `2D-spatial.zip`,
... seven in total) that extract next to the parquet, so `media_ids` are
real file paths the renderer can re-read for crops -- no write_media
detour and no re-encode. ONE LAYOUT QUIRK: the archives extract FLAT as
`<task>/<file>` at the data root, WITHOUT the `<Domain-dir>/` prefix the
parquet's `input_image_path` carries ("./Low-level-semantic/<task>/<f>").
Each referenced path is therefore resolved twice -- first verbatim against
the root, then with the leading domain component dropped -- so both the
flat layout observed here and a nested re-extraction load unchanged.

COVERAGE DECLARATION
--------------------
Only THREE of the seven official domains are on disk here
(low-level-semantic, continuous-temporal, 2d-spatial; the remaining four
archives were not downloaded). As in mvbench, a question whose images are
absent is SKIPPED with a per-domain printed count rather than failing the
load, so the domain is usable on the partial download and grows to full
coverage when the other archives land. All numbers reported for this
domain are over the three downloaded sub-domains only.

CHOICES MADE, EACH DECLARED
---------------------------
* QUESTION TEXT. `context` usually repeats the choice list (9,726 of
  11,698 rows contain the "Select from the following choices." block)
  and occasionally repeats the question itself, while the runtime's
  answer-style machinery already presents the options (mvbench
  DEVIATION paragraph). The trailing choice block is therefore stripped
  from `context`, and what remains (the task preamble, when any) is
  prepended to `question` unless the question is already contained in
  it. This keeps task instructions that only exist in `context` without
  printing every option twice.
* GOLD AS INDEX. The release's `output` is a letter; the scorer needs an
  index into the presented options, so the gold index is recovered by
  matching the letter against the parsed option letters. 11 rows whose
  gold letter is absent from their own option string, and 1 row whose
  letters are not A..N contiguous (letter-position mapping would
  silently invert), are DROPPED with a printed count, not guessed at.
* GROUP KEY IS THE IMAGE SET. The parquet has no group/series field, so
  `group_key` is the content hash of the ordered `input_image_path`
  list: two questions over the same image set share a split unit and
  can never straddle gate/test.
* min_images=2 (default) drops the single 1-image row, on which image
  selection has one legal answer and the item is effectively a
  single-image VQA question -- the mpdocvqa min_pages=2 argument
  verbatim. The full pool is recoverable with min_images=1.

SCORING
-------
Official MMIU accuracy is plain multiple-choice over the released
options; `scorers.match_multiple_choice` (declared letter first, then
exact option text, then unique containment) is that metric under
ViewCraft's own prompting, identical to the mvbench deviation and
applied identically to every arm. No LLM judges anything.
"""
from __future__ import annotations

import hashlib
import re
import string
from collections import Counter
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "mmiu"

# "A: text" / "(A) text" / "A. text" per line of the options string.
_OPT_LINE = re.compile(r"^\s*\(?([A-Z])\)?\s*[:.)]\s*(.*\S)\s*$")
# The choice list the context repeats; the runtime presents options itself.
_SELECT_BLOCK = re.compile(
    r"Select from the following choices\.?\s*\n(?:\s*\(?[A-Z]\)?\s*[:.)].*\n?)*\s*$",
    re.IGNORECASE)


def _parse_options(raw: str) -> tuple[list[str], list[str]] | None:
    """Split the release's single-string options into (letters, texts).

    Returns None when fewer than two option lines parse, and also when the
    letters are not exactly A..N in order: match_multiple_choice resolves a
    letter by position (A=0), so a non-contiguous list would silently map
    the model's letter onto the wrong option.
    """
    pairs = []
    for line in str(raw or "").splitlines():
        m = _OPT_LINE.match(line)
        if m:
            pairs.append((m.group(1), m.group(2)))
    if len(pairs) < 2:
        return None
    letters = [p[0] for p in pairs]
    if letters != list(string.ascii_uppercase[:len(letters)]):
        return None
    return letters, [p[1] for p in pairs]


def _compose_question(context: str, question: str) -> str:
    """Task preamble + question, with the duplicated choice block removed."""
    ctx = _SELECT_BLOCK.sub("", str(context or "")).strip()
    q = str(question or "").strip()
    if ctx and q not in ctx:
        return f"{ctx}\n{q}" if q else ctx
    return q or ctx


def _resolve(root: Path, rel: str) -> Path | None:
    """Locate one referenced image on disk.

    The archives extract flat (`<task>/<file>`) while the parquet path
    carries a domain prefix (`./<Domain>/<task>/<file>`), so the verbatim
    path is tried first and the domain-stripped path second.
    """
    rel = rel.lstrip("./")
    p = root / rel
    if p.is_file():
        return p
    parts = Path(rel).parts
    if len(parts) > 1:
        p = root.joinpath(*parts[1:])
        if p.is_file():
            return p
    return None


def load(root: str | Path, seed: int = 20260911,
         min_images: int = 2, limit: int | None = None) -> list[Sample]:
    """Load MMIU as public/private Sample pairs.

    `seed` is accepted for interface uniformity and is UNUSED: nothing here
    samples. Questions whose image archive is not on disk are skipped with
    a per-domain printed count (mvbench's tolerant-partial-download mode);
    a question with SOME images missing is dropped entirely, because a
    missing frame silently renumbers the positional options.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"{DOMAIN}: no parquet under {root}")

    samples: list[Sample] = []
    n_bad_options = 0
    n_gold_missing = 0
    n_partial_images = 0
    n_too_few_images = 0
    skipped_domains: Counter = Counter()
    n_images_dist: Counter = Counter()

    for p in pqs:
        for i, r in enumerate(pq.read_table(p).to_pylist()):
            rels = [str(x) for x in (r.get("input_image_path") or [])]
            if not rels:
                n_partial_images += 1
                continue
            # "./Low-level-semantic/task/file.jpg"; the archives extract
            # flat, so resolution tolerates the missing domain prefix.
            resolved = [_resolve(root, x) for x in rels]
            domain_dir = rels[0].lstrip("./").split("/")[0]
            present = [q for q in resolved if q is not None]
            if not present:
                skipped_domains[domain_dir] += 1
                continue
            if len(present) != len(resolved):
                n_partial_images += 1
                continue
            if len(present) < min_images:
                n_too_few_images += 1
                continue

            parsed = _parse_options(r.get("options"))
            if parsed is None:
                n_bad_options += 1
                continue
            letters, texts = parsed
            gold = str(r.get("output") or "").strip().upper()
            if gold not in letters:
                n_gold_missing += 1
                continue

            task = str(r.get("task") or "").strip()
            question = _compose_question(r.get("context"), r.get("question"))
            if not task or not question:
                n_bad_options += 1
                continue

            item_id = f"{DOMAIN}/{domain_dir}/{task}/{i}"
            media = tuple(str(q) for q in present)
            public = PublicItem(
                item_id=item_id, domain=DOMAIN, question=question,
                media_ids=media,
                # Release order, never re-sorted.
                options=tuple(texts),
                meta={
                    # Official 7-domain name, lowercased for bucket analysis;
                    # the on-disk directory name is kept for path forensics.
                    "domain": domain_dir.lower(),
                    "domain_dir": domain_dir,
                    # 52-task (60 observed) subtask name: the per-subtask
                    # spread IS the value of this domain, so it rides on
                    # every item.
                    "subtask": task,
                    "source": r.get("source"),
                    "visual_input_component": r.get("visual_input_component"),
                    "n_images": len(media),
                })
            private = PrivateItem(
                item_id=item_id,
                # The INDEX, so the scorer never re-derives it (mvbench).
                answer=letters.index(gold),
                answer_type="option_index",
                # No evidence locations are released, so oracle / probe
                # conditions cannot run here.
                evidence_source="none",
                meta={"subtask": task, "domain": domain_dir.lower(),
                      "answer_letter": gold, "n_options": len(letters)})
            # No group field in the release: the ordered image-set hash is
            # the split unit, so one image set never spans gate and test.
            gkey = hashlib.sha256("\n".join(rels).encode()).hexdigest()[:16]
            samples.append(Sample(public=public, private=private,
                                  group_key=gkey))
            n_images_dist[len(media)] += 1
            if limit and len(samples) >= limit:
                break
        if limit and len(samples) >= limit:
            break

    kept_domains = Counter(s.public.meta["domain"] for s in samples)
    print(f"[{DOMAIN}] kept {len(samples)} questions; "
          f"by domain: {dict(sorted(kept_domains.items()))}; "
          f"n_images dist: {dict(sorted(n_images_dist.items()))}")
    if skipped_domains:
        print(f"[{DOMAIN}] skipped {sum(skipped_domains.values())} questions "
              f"whose image archive is absent: "
              f"{dict(sorted(skipped_domains.items()))}")
    print(f"[{DOMAIN}] dropped: {n_bad_options} unparseable options, "
          f"{n_gold_missing} gold letter not in options, "
          f"{n_partial_images} partially/absent image sets, "
          f"{n_too_few_images} below min_images={min_images}")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """Official MMIU metric: plain multiple-choice accuracy.

    Same resolution as mvbench (declared letter, then exact option text,
    then unique containment). No model acts as judge.
    """
    from ..evaluation.scorers import match_multiple_choice
    return match_multiple_choice(answer_text, list(public.options),
                                 int(private.answer))


if __name__ == "__main__":
    """Smoke: real data if present, else a synthetic parquet in /tmp."""
    import sys

    import os
    candidates = [Path(sys.argv[1]) if len(sys.argv) > 1 else
                  Path(os.environ.get("VIEWCRAFT_DATA_ROOT", "data")) / "mmiu"]
    root = candidates[0]
    if (len(sys.argv) > 1 and sys.argv[1] == "synthetic") or not (
            root / "all.parquet").is_file():
        import tempfile
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image

        root = Path(tempfile.mkdtemp(prefix="mmiu_smoke_"))
        img_dir = root / "Low-level-semantic" / "toy_task"
        img_dir.mkdir(parents=True)
        rows = []
        for k in range(6):
            rels = []
            for j in range(2 + k % 3):
                fp = img_dir / f"toy_task_{k}_{j}.png"
                Image.new("RGB", (64, 48), (k * 40, j * 80, 128)).save(fp)
                rels.append(f"./Low-level-semantic/toy_task/{fp.name}")
            rows.append({
                "task": "toy_task", "visual_input_component": "synthetic",
                "source": "smoke",
                "options": "A: the first image\nB: the second image\n"
                           "C: the third image",
                "question": f"Which image is number {k}?",
                "context": "Toy preamble.\nSelect from the following "
                           "choices.\nA: the first image\nB: the second "
                           "image\nC: the third image",
                "input_image_path": rels,
                "output": "B"})
        # One row from an absent archive: must be skipped, not fatal.
        rows.append({**rows[0],
                     "input_image_path": ["./3D-spatial/toy/x_0_0.png",
                                          "./3D-spatial/toy/x_0_1.png"]})
        pq.write_table(pa.Table.from_pylist(rows), root / "all.parquet")
        print(f"[{DOMAIN}] synthetic smoke root: {root}")

    got = load(root)
    assert got, "no samples loaded"
    # score(gold) must be 1.0 for every item.
    bad = [s.public.item_id for s in got
           if score(chr(ord("A") + int(s.private.answer)),
                    s.private, s.public) != 1.0]
    assert not bad, f"gold scored != 1.0 on {bad[:5]}"
    from collections import Counter as _C
    print(f"[{DOMAIN}] SMOKE OK: {len(got)} items, "
          f"domains={dict(_C(s.public.meta['domain'] for s in got))}, "
          f"subtasks={len({s.public.meta['subtask'] for s in got})}, "
          f"score(gold)==1.0 on all")
