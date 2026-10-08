"""MathVision adapter (MathLLMs/MathVision) -- math+visual reasoning, the
"constructive workspace" task type the ViLoMem paper points to and the suite
was missing.

Why it is a better fit than the synthetic symbolic tasks (syn_math, which
saturated). MathVision's geometry subjects (combinatorial / metric / solid /
transformation / descriptive geometry, graph theory) are not solvable by pure
symbolic one-shot reasoning -- the model must read a figure and often reason by
DRAWING on it (auxiliary lines, marked angles, computed values written onto the
diagram), which is exactly what a human does and what the visual canvas
externalises. It also has a difficulty gradient (levels 1-5) and real headroom
(frontier baselines are ~46%, per the source and ViLoMem). Answers are short
text or an option; per-subject/level meta is kept for the "which kinds of
visual failure does the canvas help" analysis.
"""
from __future__ import annotations

import ast
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "mathvision"


def load(root: str | Path, seed: int = 20260911) -> list[Sample]:
    import pyarrow.parquet as pq
    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")
    samples: list[Sample] = []
    for p in pqs:
        for i, r in enumerate(pq.read_table(p).to_pylist()):
            blob = r.get("decoded_image")
            blob = blob.get("bytes") if isinstance(blob, dict) else blob
            if not blob:
                continue
            media = write_media(DOMAIN, blob, str(r.get("id", i)))
            try:
                opts = ast.literal_eval(str(r.get("options") or "[]"))
            except (ValueError, SyntaxError):
                opts = []
            options = tuple(str(o) for o in opts)
            item_id = f"{DOMAIN}/{r.get('id', i)}"
            public = PublicItem(item_id=item_id, domain=DOMAIN,
                                question=str(r.get("question") or ""),
                                media_ids=(media,), options=options,
                                meta={"subject": r.get("subject"),
                                      "level": r.get("level")})
            private = PrivateItem(item_id=item_id,
                                  answer=str(r.get("answer") or "").strip(),
                                  answer_type="string", evidence_source="none",
                                  meta={"subject": r.get("subject"),
                                        "level": r.get("level")})
            samples.append(Sample(public=public, private=private,
                                  group_key=str(r.get("id", i))))
    print(f"[mathvision] kept {len(samples)} items")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    import re
    want = str(private.answer).strip().lower()
    ans = (answer_text or "").strip().lower()
    # option letter first
    m = re.search(r"\b([a-e])\b", ans)
    if m and want in "abcde" and m.group(1) == want:
        return 1.0
    # exact / numeric match
    if ans == want:
        return 1.0
    wa = re.findall(r"-?\d+\.?\d*", want)
    aa = re.findall(r"-?\d+\.?\d*", ans)
    return 1.0 if wa and aa and wa[0] == aa[0] else 0.0
