"""Visual strategy induction: a persistent, per-chain memory of WHAT visual
strategies this domain's failures call for, and how each has fared.

WHY THIS EXISTS. Without it the search is myopic: each round the optimizer sees
a handful of failures and re-rolls the same generic prior (a contrast/sharpen
tweak), because nothing accumulates "the failures here decompose into THESE
visual causes, and THAT is what has already been tried". Measured on the old
mechanism: 33 pixel operators available, 9 ever used; every accepted
pixel_expr was a contrast/unsharp variant. The narrowness was in the SEARCH,
not the vocabulary.

The StrategyBook is the visual analogue of the reflective memory in text-side
skill evolution (GEPA-style), but induced over VISUAL failure causes. It is
deliberately NOT a menu: the strategies are written by the optimizer FROM this
domain's failure evidence, are content-open, and grow/accumulate across rounds.

Four steps per round, all driven by the optimizer reading real failure images:
  diagnose  -- tag each failure with an open-vocabulary VISUAL cause
               (target too small / event boundary missed / two instances need
               disambiguating marks / colour cue destroyed / ...), not the
               binary grounding verdict.
  induce    -- cluster causes across rounds into domain-specific strategy
               hypotheses: {failure_mode, prevalence, mechanism sketch,
               text coupling, status}.
  direct    -- condition the proposal on the book: aim THIS round at the
               highest-value UNRESOLVED strategy, and forbid re-proposing ones
               already marked hurt / no-op. This is what forces diversity
               across rounds instead of collapse onto the contrast prior.
  payoff    -- after the gate decides, write back whether the candidate's
               strategy helped or hurt, so the book becomes an empirical bandit
               over this domain's visual strategies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..models import ModelClient, ModelFormatError, user_message


# A strategy's lifecycle. `untried` -> proposed; `helped` / `hurt` are written
# back from the gate; `blocked` means the diagnosis named a failure mode the
# current action space cannot reach (feeds the expressiveness backlog).
STATUS = ("untried", "helped", "hurt", "blocked")


@dataclass
class Strategy:
    """One induced visual-strategy hypothesis for THIS domain."""
    failure_mode: str          # the open-vocab visual cause it addresses
    mechanism: str             # sketch of the visual mechanism to try
    prevalence: float = 0.0    # share of recent failures with this cause
    text_coupling: str = ""    # the theta/contract change the mechanism needs
    status: str = "untried"    # untried | helped | hurt | blocked
    tries: int = 0
    last_round: int = 0

    def key(self) -> str:
        return re.sub(r"\s+", " ", self.failure_mode.strip().lower())[:80]


@dataclass
class StrategyBook:
    """The accumulating set of strategy hypotheses for one (domain, arm)."""
    strategies: list[Strategy] = field(default_factory=list)

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "StrategyBook":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return cls()
        out = cls()
        for d in raw.get("strategies", []):
            try:
                out.strategies.append(Strategy(**{
                    k: d[k] for k in
                    ("failure_mode", "mechanism", "prevalence",
                     "text_coupling", "status", "tries", "last_round")
                    if k in d}))
            except TypeError:
                continue
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"strategies": [asdict(s) for s in self.strategies]},
            indent=2, ensure_ascii=False))

    # -- queries -----------------------------------------------------------
    def by_key(self, failure_mode: str) -> Strategy | None:
        k = re.sub(r"\s+", " ", failure_mode.strip().lower())[:80]
        for s in self.strategies:
            if s.key() == k:
                return s
        return None

    def next_target(self) -> Strategy | None:
        """The highest-value UNRESOLVED strategy to aim this round at.

        Value = prevalence (how much of the failure mass it covers), restricted
        to strategies that have not yet succeeded and are not blocked. This is
        the anti-monotony driver: the round is aimed at the largest UNTAPPED
        failure mode, not at whatever the generic prior re-rolls.
        """
        open_ = [s for s in self.strategies if s.status == "untried"]
        if not open_:
            return None
        return max(open_, key=lambda s: (s.prevalence, -s.tries))


REFLECT_PROMPT = """\
You are building a map of WHY a vision model fails on THIS domain, and what
VISUAL strategy would fix each kind of failure. You are given the failure
traces (with the actual images the model saw) and the current strategy map.

Current strategy map (may be empty):
{book}

New failure traces from this round:
{failures}

Diagnose each failure by its VISUAL cause -- an open-vocabulary label for what
about the PRESENTATION made it hard (e.g. "target too small to read", "decisive
moment falls between sampled frames", "two similar instances need marking to
disambiguate", "colour cue was destroyed", "working memory: lost track while
counting"). Do NOT use the binary evidence-supported verdict; look at the
images and say what visually went wrong.

Then update the strategy map. For each distinct visual cause, propose the
VISUAL mechanism that would address it (a pixel transform, a selection rule, a
layout, a canvas/marking strategy, or a combination), and the instruction-text
change it would need to stay consistent. Merge with the existing map; do not
duplicate a cause already present.

Reply with a JSON object:
{{"strategies": [
   {{"failure_mode": "the visual cause",
     "mechanism": "the visual strategy to try, as a concrete sketch",
     "prevalence": <0..1 share of THESE failures with this cause>,
     "text_coupling": "the instruction change it needs, or empty"}}]}}
"""


def reflect_update(book: StrategyBook, failures: list[dict],
                   client: ModelClient, round_index: int) -> StrategyBook:
    """One reflection step: diagnose this round's failures and fold them into
    the book. Robust to parse failure -- a bad reflection returns the book
    unchanged rather than crashing the chain (the book is advisory)."""
    from .propose import format_failures
    if not failures:
        return book
    imgs = []
    for f in failures[:3]:
        for u in (f.get("images") or [])[:2]:
            if u and len(imgs) < 6:
                imgs.append(u)
    book_txt = ("\n".join(
        f"  - [{s.status}] {s.failure_mode} -> {s.mechanism} "
        f"(prevalence {s.prevalence:.2f}, tried {s.tries}x)"
        for s in book.strategies)) or "(empty)"
    prompt = REFLECT_PROMPT.format(
        book=book_txt, failures=format_failures(failures, max_items=8))
    try:
        resp = client.call([user_message(prompt, imgs or None)],
                           max_tokens=4096, use_cache=False)
    except Exception:                                   # noqa: BLE001
        return book
    m = re.search(r"\{.*\}", resp.content or "", re.DOTALL)
    if not m:
        return book
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return book
    for d in obj.get("strategies", []) or []:
        fm = str(d.get("failure_mode") or "").strip()
        mech = str(d.get("mechanism") or "").strip()
        if not fm or not mech:
            continue
        existing = book.by_key(fm)
        if existing is not None:
            # update prevalence as a running max, refresh the sketch
            existing.prevalence = max(existing.prevalence,
                                      float(d.get("prevalence") or 0.0))
            existing.mechanism = mech
            existing.text_coupling = str(d.get("text_coupling") or "")
            existing.last_round = round_index
        else:
            book.strategies.append(Strategy(
                failure_mode=fm, mechanism=mech,
                prevalence=float(d.get("prevalence") or 0.0),
                text_coupling=str(d.get("text_coupling") or ""),
                status="untried", tries=0, last_round=round_index))
    return book


def directive_block(book: StrategyBook) -> str:
    """The steering context injected into propose_joint.

    Names each induced strategy with its track record, then points THIS round
    at the highest-value unresolved one and bars re-rolling the failures. This
    is the diversity mechanism: successive rounds are aimed at DIFFERENT
    failure modes.
    """
    if not book.strategies:
        return ""
    lines = ["STRATEGY MAP induced from this domain's failures so far:"]
    for s in sorted(book.strategies, key=lambda x: -x.prevalence):
        lines.append(
            f"  [{s.status:8s}] ({s.prevalence:.0%} of failures) "
            f"{s.failure_mode}\n             -> try: {s.mechanism}"
            + (f" | needs text: {s.text_coupling}" if s.text_coupling else ""))
    tgt = book.next_target()
    if tgt is not None:
        lines.append(
            f"\nTHIS ROUND, aim at the highest-value UNRESOLVED failure mode: "
            f"\"{tgt.failure_mode}\" via: {tgt.mechanism}.")
    done = [s for s in book.strategies if s.status in ("helped", "hurt")]
    if done:
        lines.append("Do NOT re-propose strategies already marked "
                     + ", ".join(f"\"{s.failure_mode}\" ({s.status})"
                                 for s in done) + ".")
    return "\n".join(lines) + "\n\n"


def record_outcome(book: StrategyBook, cand_diff: str, rationale: str,
                   accepted: bool, round_index: int) -> None:
    """Write the gate's verdict back onto the strategy a candidate targeted.

    The candidate is matched to a strategy by overlap between its diff/rationale
    and each strategy's failure_mode/mechanism text. Imperfect lexical matching
    is acceptable: the book is a bandit, and a mis-attributed write-back just
    adds noise to one arm of it, it cannot corrupt a measurement.
    """
    text = (cand_diff + " " + rationale).lower()
    best, best_score = None, 0
    for s in book.strategies:
        words = set(re.findall(r"[a-z]{4,}", (s.failure_mode + " "
                                              + s.mechanism).lower()))
        score = sum(1 for w in words if w in text)
        if score > best_score:
            best, best_score = s, score
    if best is None or best_score < 2:
        return
    best.tries += 1
    best.last_round = round_index
    # One accepted candidate marks the strategy as helping; a rejection marks
    # it hurt only after it has actually been tried (tries >= 1), so a single
    # unlucky rejection of a fresh strategy does not permanently ban it.
    if accepted:
        best.status = "helped"
    elif best.tries >= 1 and best.status == "untried":
        best.status = "hurt"
