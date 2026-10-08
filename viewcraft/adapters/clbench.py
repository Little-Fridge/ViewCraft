"""CL-bench adapter (tencent/CL-bench) -- the boldest test: a NON-visual,
long-context task, where the model is given a long novel RULE BOOK / domain
spec and must apply it PRECISELY. This is the real "CL-Bench idea": can a
visual workspace -- a reference sheet the model jots the key rules/state onto
-- act as an external memory that improves fidelity on a task that is not
about images at all?

Why it is a better fit than the synthetic symbolic tasks (syn_state/syn_math,
which saturated): those were short enough to hold in one reasoning pass. Here
the spec is LONG (median ~24K chars, p90 ~100K), and the failure mode is
precise application -- exact strings, priorities, formats -- which slips when
the relevant rule is buried 40K characters in. Offloading the key rules to a
reference sheet is a genuinely plausible external-memory intervention.

Scoring is RUBRIC-JUDGED (the release has no exact answers), so the scorer
calls an independent judge model (the OPTIMIZER, never the executor, so judge
!= answerer) to grade the final answer against the item's rubrics.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "clbench"
MAX_CONTEXT_CHARS = 110_000     # keep the run tractable; covers ~90% of items

_JUDGE_CLIENT = None


def _judge():
    """Lazily build the judge client -- the OPTIMIZER, not the executor, so the
    grader is independent of the model being graded."""
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        from ..models import ModelClient
        _JUDGE_CLIENT = ModelClient("optimizer")
    return _JUDGE_CLIENT


def _workspace_image(root: Path) -> str:
    """A blank reference-sheet canvas the model writes key rules/state onto.

    This IS the visual workspace: a structured empty sheet (sections for RULES /
    STATE / NOTES) the model annotates via the canvas note action.
    """
    from PIL import Image, ImageDraw
    p = root / "_workspace.png"
    if not p.exists():
        w, h = 900, 620
        img = Image.new("RGB", (w, h), (252, 252, 250))
        d = ImageDraw.Draw(img)
        d.rectangle([8, 8, w - 8, h - 8], outline=(50, 50, 50), width=3)
        d.line([8, 60, w - 8, 60], fill=(50, 50, 50), width=2)
        d.text((20, 26), "WORKSPACE - write the key rules / state / notes here",
               fill=(90, 90, 90))
        for i, lab in enumerate(("RULES", "STATE", "NOTES")):
            y = 90 + i * 170
            d.text((20, y), lab, fill=(150, 150, 150))
            d.line([8, y + 150, w - 8, y + 150], fill=(200, 200, 200), width=1)
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
    return str(p)


def load(root: str | Path, seed: int = 20260911,
         categories: tuple = ("Rule System Application",),
         max_chars: int = MAX_CONTEXT_CHARS, n_items: int = 300) -> list[Sample]:
    root = Path(root)
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no jsonl under {root}")
    rows = []
    for f in files:
        for line in f.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    ws = _workspace_image(root)
    import random
    rng = random.Random(seed)
    samples: list[Sample] = []
    for r in rows:
        md = r.get("metadata") or {}
        if md.get("context_category") not in categories:
            continue
        msgs = r.get("messages") or []
        ctx = "\n\n".join(str(m.get("content") or "") for m in msgs)
        if len(ctx) > max_chars or not ctx.strip():
            continue
        rubrics = [str(x) for x in (r.get("rubrics") or [])]
        if not rubrics:
            continue
        tid = md.get("task_id") or f"row{len(samples)}"
        item_id = f"{DOMAIN}/{tid[:16]}"
        question = (
            "You are given a long specification or rule book below, and a task "
            "that requires applying it precisely. You may use the attached "
            "blank WORKSPACE image to jot the key rules, priorities, formats "
            "or state as you read, then apply them.\n\n"
            f"===== SPECIFICATION / CONTEXT =====\n{ctx}")
        public = PublicItem(item_id=item_id, domain=DOMAIN, question=question,
                            media_ids=(ws,),
                            meta={"category": md.get("context_category"),
                                  "sub_category": md.get("sub_category")})
        private = PrivateItem(item_id=item_id, answer=rubrics,
                              answer_type="rubrics", evidence_source="none",
                              meta={"rubrics": rubrics})
        samples.append(Sample(public=public, private=private,
                              group_key=tid[:16]))
        if len(samples) >= n_items:
            break
    rng.shuffle(samples)
    print(f"[clbench] kept {len(samples)} items in categories "
          f"{categories} (context<= {max_chars} chars)")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    """Rubric-judged: an independent judge grades the answer against the
    item's rubrics and returns the fraction satisfied (0..1)."""
    rubrics = (private.meta or {}).get("rubrics") or []
    ans = (answer_text or "").strip()
    if not rubrics or not ans:
        return 0.0
    rub_list = "\n".join(f"  {i + 1}. {x}" for i, x in enumerate(rubrics[:20]))
    prompt = (
        "You are grading an AI assistant's answer against a checklist of "
        "rubrics. For EACH rubric, decide if the answer satisfies it (yes/no). "
        "Be strict: a rubric is satisfied only if the answer clearly meets it.\n\n"
        f"RUBRICS:\n{rub_list}\n\nANSWER:\n{ans[:6000]}\n\n"
        "Reply with a JSON object: {\"satisfied\": [list of rubric numbers "
        "satisfied], \"total\": <number of rubrics>}")
    try:
        from ..models import user_message
        resp = _judge().call([user_message(prompt)], max_tokens=1024,
                             use_cache=False)
        m = re.search(r"\{.*\}", resp.content or "", re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
        sat = obj.get("satisfied") or []
        total = int(obj.get("total") or len(rubrics[:20]))
        return min(1.0, len(sat) / max(1, total))
    except Exception:                                       # noqa: BLE001
        return 0.0
