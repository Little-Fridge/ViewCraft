"""BabyVision adapter (UnipatAI/BabyVision) -- visual tasks a 3-year-old solves
but frontier MLLMs fail. Directly on-thesis: PERCEPTION and working-memory
failures, not reasoning, so it isolates the visual strategy the mechanism is
meant to evolve.

Structure: 388 rows, embedded image bytes, columns taskId/type/subtype/question/
ansType/options/choiceAns/blankAns. Four task TYPES (Fine-grained
Discrimination, Spatial Perception, Visual Tracking, Visual Pattern
Recognition) over ~20 subtypes (counting, mazes, line tracing, 3D views, shadow
matching, pattern completion...). Counting/tracking/maze subtypes are exactly
the territory the visual memory canvas is built for.

Per the user's instruction, only subtypes with >= MIN_SAMPLES items are kept;
the tiny ones (~10-12) are dropped because a handful of items cannot carry a
measurement. Answers are grid coordinates "(r,c)", counts, or a choice index;
the scorer normalises all three.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "babyvision"
MIN_SAMPLES = 15


def _norm_numbers(s: str) -> tuple:
    """Extract the ordered integer tuple from an answer, ignoring format."""
    return tuple(int(x) for x in re.findall(r"-?\d+", str(s) or ""))


def load(root: str | Path, seed: int = 20260911) -> list[Sample]:
    import pyarrow.parquet as pq
    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")
    rows = []
    for p in pqs:
        rows.extend(pq.read_table(p).to_pylist())

    # keep only subtypes with enough samples to measure
    from collections import Counter
    counts = Counter(r.get("subtype") for r in rows)
    keep = {st for st, n in counts.items() if n >= MIN_SAMPLES}
    dropped = {st: n for st, n in counts.items() if n < MIN_SAMPLES}
    if dropped:
        print(f"[babyvision] dropped tiny subtypes (<{MIN_SAMPLES} items): "
              f"{dropped}")

    samples: list[Sample] = []
    for i, r in enumerate(rows):
        st = r.get("subtype")
        if st not in keep:
            continue
        blob = r.get("image")
        blob = blob.get("bytes") if isinstance(blob, dict) else blob
        if not blob:
            continue
        path = write_media(DOMAIN, blob, str(r.get("taskId") or i))
        ans_type = str(r.get("ansType") or "blank")
        options: tuple = ()
        gold_letter = ""
        if ans_type == "choice":
            try:
                opts = ast.literal_eval(str(r.get("options") or "[]"))
            except (ValueError, SyntaxError):
                opts = []
            options = tuple(str(o) for o in opts)
            try:
                idx = int(r.get("choiceAns"))
                gold_letter = chr(ord("A") + idx) if 0 <= idx < len(options) else ""
            except (TypeError, ValueError):
                gold_letter = ""
        item_id = f"{DOMAIN}/{r.get('taskId', i)}"
        public = PublicItem(item_id=item_id, domain=DOMAIN,
                            question=str(r.get("question") or ""),
                            media_ids=(path,), options=options,
                            meta={"subtype": st, "type": r.get("type"),
                                  "ans_type": ans_type})
        private = PrivateItem(
            item_id=item_id,
            answer=(gold_letter if ans_type == "choice"
                    else str(r.get("blankAns") or "")),
            answer_type=("letter" if ans_type == "choice" else "string"),
            evidence_source="none",
            meta={"subtype": st, "blank_ans": str(r.get("blankAns") or ""),
                  "choice_letter": gold_letter})
        samples.append(Sample(public=public, private=private,
                              group_key=f"{st}_{r.get('taskId', i)}"))
    print(f"[babyvision] kept {len(samples)} items over "
          f"{len(keep)} subtypes (>= {MIN_SAMPLES} each)")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    """Choice -> letter match; blank -> integer-tuple match (coords or count)."""
    ans = (answer_text or "").strip()
    if (private.meta or {}).get("choice_letter"):
        want = (private.meta["choice_letter"] or "").upper()
        m = re.search(r"\b([A-Z])\b", ans.upper())
        return 1.0 if m and m.group(1) == want else 0.0
    want = _norm_numbers(private.meta.get("blank_ans", ""))
    if not want:
        return 0.0
    return 1.0 if _norm_numbers(ans) == want else 0.0
