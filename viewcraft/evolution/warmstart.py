"""Domain-adapted initial skill (warm start).

Every domain used to start from the same hand-written generic coordinates.
Measured consequence: the generic protocol costs points on domains where one
pass suffices or where its answer discipline mismatches the scorer
("protocol debt": direct - baseline > 0), and the first evolution rounds were
spent repaying that debt instead of buying insight.

Here the optimizer authors the STARTING state instead: a few diagnostic-split
items are rolled out twice -- once direct (bare model, one call) and once
under the generic protocol -- and the optimizer reads the side-by-side traces
and writes the initial theta and psi contracts for THIS domain. A mini-gate on
the diagnostic split adopts the authored state only when it does not score
below the generic default, so the initial skill can never be made worse by
construction. The diagnostic split is disjoint from train/gate/test, so
nothing here leaks into any reported number.

Scope is deliberately textual: theta and the three psi contracts are free
text, so cold authoring cannot emit an uncompilable expression. The phi
expression languages stay at their native defaults; the evolution loop can
rewrite them from round 1 with real traces and compile feedback.
"""
from __future__ import annotations

import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..models import image_to_data_url, user_message
from ..runtime import ANSWER_STYLES, run_static_item
from ..views.config import Coordinates

PROBE_K = 5          # diagnostic items shown to the author
GATE_CAP = 16        # diagnostic items the candidate initial is judged on
MAX_FIELD = 4000     # chars per authored text field

AUTHOR_PROMPT = """You are designing the INITIAL skill state of a visual agent for one benchmark domain, from a blank page. Nothing is prescribed: you decide how the agent should work this domain, and your design becomes the starting point that later evolution rounds refine.

The agent is a frozen multimodal model working in observation steps. These are the TRUE interface facts it runs under -- your text must stay consistent with them, but how you teach/prioritise them is entirely yours:
{interface_facts}

Domain: {domain}
Answer style: {answer_style}
Scorer contract: {scorer_note}

DIAGNOSTIC EVIDENCE. A few items from this domain were each run twice: DIRECT (bare model, one call, all sources attached) and HARNESS (a generic multi-step observation protocol, not tailored to anything). Read the failures and the successes. Where DIRECT beats HARNESS, the generic protocol is in the way; where both fail, the procedure you write must supply the missing method; where HARNESS beats DIRECT, keep whatever the protocol did right.

{digest}

Now write the complete initial state. Reply with exactly one JSON object, any subset of these keys (empty string = fall back to the runtime's built-in default for that field):
{{
  "theta": "the reading procedure the agent follows for THIS domain (markdown, <= 3000 chars): what to inspect first, when one pass is enough, when to zoom, how to verify before answering",
  "protocol_contract": "the observation-protocol text shown to the agent EVERY step (<= 2000 chars). It must keep the JSON action syntax and coordinate rules exactly as documented above, but you choose the emphasis, order, and brevity. May use placeholders {{pool_size}} {{max_crops}} {{ans_header}} {{ans_example}}",
  "view_contract": "one sentence describing each attached view to the agent, or empty",
  "answer_contract": "the exact surface form the final answer must take for THIS scorer, or empty"
}}
Design from the evidence, not for generality. If direct wins by answering in one step, say when one step is enough. If the harness lost points to answer formatting, pin the exact surface form. If a task type needs a specific working sequence, write that sequence."""


def _direct_answer(item, media_paths, executor, answer_style: str) -> str:
    """One bare call: question + source images, no protocol. This is the
    diagnostic reference, not a reported condition."""
    images = []
    for p in media_paths[:8]:
        try:
            images.append(image_to_data_url(p))
        except Exception:
            continue
    q = item.question
    if getattr(item, "options", None):
        q += "\nOptions:"
        for i, o in enumerate(item.options):
            q += f"\n  {chr(ord('A')+i)}. {o}"
    if answer_style == "letter":
        suffix = "Answer with the option letter only."
    elif answer_style == "yes_no":
        suffix = "Answer exactly yes or no."
    else:
        suffix = ("Answer with only the value or short phrase asked for. "
                  "No explanation, no units unless asked.")
    resp = executor.call(
        [user_message(f"{q}\n\n{suffix}", images)],
        max_tokens=2048)
    return (resp.content or "").strip()


def _harness_score(item, coords, executor, budget, phash, answer_style, scorer, private, public):
    traj = run_static_item(public, coords, executor, budget, phash,
                           arm="warmstart", condition="normal",
                           split="diagnostic", answer_style=answer_style)
    return traj, scorer(traj.answer_text, private, public)


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _valid_protocol(text: str) -> bool:
    # The runtime substitutes the four placeholders with literal token
    # replacement and otherwise treats the text as opaque, so NOTHING the
    # author writes can break it -- JSON examples with braces included.
    # An earlier str.format() probe rejected exactly those texts; that was
    # stricter than the consumer, i.e. wrong.
    return bool(text.strip())


def warmstart_coordinates(domain: str, samples, spec, executor, optimizer,
                          budget, phash: str, run_dir: Path,
                          default: Coordinates,
                          scorer_note: str = "",
                          concurrency: int = 4) -> Coordinates:
    """Return the domain-adapted initial coordinates, or `default`.

    Never raises: any failure falls back to the generic default so a warm
    start problem can never block a run.
    """
    cache_dir = run_dir / "configs" / "warmstart_initial"
    meta_path = run_dir / "configs" / "warmstart_meta.json"
    if cache_dir.exists():
        print(f"[warmstart] reused cached adapted initial from {cache_dir}")
        return Coordinates.read_snapshot(cache_dir)

    try:
        diag = [s for s in samples if s.split == "diagnostic"]
        if not diag:
            print("[warmstart] no diagnostic split; generic initial kept")
            return default
        probe = diag[:PROBE_K]
        ans_style = spec.answer_style

        # 1. dual rollouts on the probe items
        records = []
        def one_probe(s):
            traj, hs = _harness_score(s, default, executor, budget, phash,
                                      ans_style, spec.scorer, s.private, s.public)
            try:
                da = _direct_answer(s.public, list(s.public.media_ids),
                                    executor, ans_style)
                ds = spec.scorer(da, s.private, s.public)
            except Exception as e:                            # noqa: BLE001
                da, ds = f"<direct call failed: {e}>", None
            return s, traj, hs, da, ds
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for s, traj, hs, da, ds in ex.map(one_probe, probe):
                records.append({
                    "item": s.public.item_id, "question": s.public.question[:400],
                    "n_sources": len(s.public.media_ids),
                    "harness_score": round(hs, 3),
                    "harness_answer": (traj.answer_text or "")[:300],
                    "harness_steps": getattr(traj, "n_steps", None),
                    "direct_score": (round(ds, 3) if ds is not None else None),
                    "direct_answer": da[:300],
                })
        digest = json.dumps(records, indent=1, ensure_ascii=False)

        # 2. author the adapted initial FROM A BLANK PAGE: the prompt shows
        # only the interface facts and the diagnostic evidence, never the
        # incumbent texts -- an anchored author edits the default, a blank
        # author designs for the domain.
        ans_header, ans_example = ANSWER_STYLES.get(
            ans_style, ("To answer:", '{"action": "answer", "answer": "..."}'))
        interface_facts = (
            "- It sees rendered views of the evidence (images/frames/pages) and replies with exactly one JSON action block per step.\n"
            "- zoom: {\"action\": \"zoom\", \"regions\": [{\"x0\":..,\"y0\":..,\"x1\":..,\"y1\":..}], \"image\": N, \"note\": why} -- coordinates plain decimals in [0,1] in the named image's frame, x0<x1, y0<y1; zooming INSIDE a zoom is allowed; a zoom re-reads the original file so it recovers real detail. Up to "
            f"{default.phi.max_crops_per_step} regions per step.\n"
            "- note: {\"action\": \"note\", \"notes\": [{\"at\": [x,y], \"text\": t, \"image\": N}]} -- persistent visual markers that stay visible on later views of the same source, including inside zooms.\n"
            "- view: {\"action\": \"view\", \"units\": [i,...]} -- pages in source units not currently attached (the pool size varies per item; write {pool_size} in your protocol text and the runtime substitutes the number).\n"
            "- compose: {\"action\": \"compose\", \"units\": [...], \"cols\": c} -- tiles several units into one labelled comparison sheet.\n"
            f"- answer: {ans_header} e.g. {ans_example}\n"
            "- Observation steps and images-per-request are budgeted; the budget itself is fixed and not yours to change.")
        prompt = AUTHOR_PROMPT.format(
            interface_facts=interface_facts,
            domain=domain, answer_style=ans_style,
            scorer_note=scorer_note or ans_style,
            digest=digest)
        resp = optimizer.call([user_message(prompt)], max_tokens=8192)
        obj = _extract_json(resp.content)
        if obj is None:
            print("[warmstart] author returned unparseable JSON; default kept")
            return default

        adapted = copy.deepcopy(default)
        touched = []
        theta = str(obj.get("theta") or "").strip()
        if theta and theta != default.theta.text:
            adapted.theta.text = theta[:MAX_FIELD]
            touched.append("theta")
        for field in ("view_contract", "answer_contract", "protocol_contract"):
            val = str(obj.get(field) or "").strip()
            if not val:
                continue
            if field == "protocol_contract" and not _valid_protocol(val):
                print("[warmstart] protocol_contract placeholders invalid; dropped")
                continue
            setattr(adapted.psi, field, val[:MAX_FIELD])
            touched.append(f"psi.{field}")
        if not touched:
            print("[warmstart] author proposed no changes; default kept")
            return default

        # 3. mini-gate on the diagnostic split. The IDEAL ordering is
        #    viewcraft > initial skill > direct: the authored initial is a
        #    domain workflow the optimizer wrote after seeing direct's
        #        results, so it should already clear BOTH bars -- the generic
        #    default it replaces AND the bare model. Adopt only then.
        gate = diag[:GATE_CAP]
        def gate_score(coords):
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = [ex.submit(_harness_score, s, coords, executor, budget,
                                  phash, ans_style, spec.scorer,
                                  s.private, s.public) for s in gate]
                scores = [f.result()[1] for f in futs]
            return sum(scores) / max(1, len(scores))
        def direct_score():
            def one(s):
                da = _direct_answer(s.public, list(s.public.media_ids),
                                    executor, ans_style)
                return spec.scorer(da, s.private, s.public)
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                scores = list(ex.map(one, gate))
            return sum(scores) / max(1, len(scores))
        s_default = gate_score(default)
        s_adapted = gate_score(adapted)
        s_direct = direct_score()
        # The direct bar encodes "initial skill should clear the bare model".
        # When direct SATURATES the diagnostic slice (>= 0.95), that bar is
        # uninformative -- nothing can beat perfection, and an authored
        # initial that beats the default (reasonmap g38w2: 0.812 vs 0.750)
        # would be thrown away for failing an unmeasurable test. The
        # default comparison always applies; the direct comparison applies
        # only where it has headroom to discriminate.
        direct_vacuous = s_direct >= 0.95
        adopted = s_adapted >= s_default and (s_adapted >= s_direct
                                              or direct_vacuous)
        print(f"[warmstart] diagnostic gate: default={s_default:.3f} "
              f"adapted={s_adapted:.3f} direct={s_direct:.3f}"
              f"{' (direct saturated, bar vacuous)' if direct_vacuous else ''} -> "
              f"{'ADOPT' if adopted else 'KEEP DEFAULT'} "
              f"({', '.join(touched)})")
        if not adopted:
            return default

        cache_dir.mkdir(parents=True, exist_ok=True)
        adapted.write_snapshot(cache_dir)
        meta_path.write_text(json.dumps({
            "domain": domain, "probe_k": len(records),
            "gate_n": len(gate), "gate_default": s_default,
            "gate_adapted": s_adapted, "gate_direct": s_direct, "touched": touched,
            "records": records}, indent=1, ensure_ascii=False, default=str))
        return adapted
    except Exception as e:                                    # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[warmstart] failed ({type(e).__name__}: {e}); default kept")
        return default


def load_cached(run_dir: Path) -> Coordinates | None:
    """The adapted initial this run started from, if warm start ran."""
    cache_dir = run_dir / "configs" / "warmstart_initial"
    if cache_dir.exists():
        return Coordinates.read_snapshot(cache_dir)
    return None
