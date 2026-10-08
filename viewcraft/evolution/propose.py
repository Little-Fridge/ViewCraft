"""Candidate proposal: the optimizer reads failures and proposes ONE change.

What the optimizer sees: the images that were actually on screen, the phi/psi
config that produced them, the executor's own judgement at the time, and the
consistency verification. What it never sees: test samples, gold answers,
evidence page/box/time ground truth, or sample ids to look up.

Its output is confined to a single change at the currently selected
coordinate, must parse, must diff, and must be reproducible. Attempts to embed
a specific question, answer, or id lookup are rejected before the candidate is
ever evaluated.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import cycle

from ..models import ModelClient, ModelFormatError, user_message
from ..schema import Trajectory, sha
from ..views.config import (Coordinates, PhiConfig, PSI_FIELDS, PHI_FIELDS,
                            PsiConfig, SkillConfig)

COORD_ROTATION = ("theta", "phi", "psi")


class CandidateRejected(RuntimeError):
    """The proposal violated a hard constraint and was never evaluated."""


@dataclass
class Candidate:
    coordinate: str                 # "theta" | "phi" | "psi"
    coords: Coordinates
    rationale: str
    diff: str
    proposer_request_id: str = ""

    def to_dict(self) -> dict:
        return {"coordinate": self.coordinate, "rationale": self.rationale,
                "diff": self.diff, "config_hash": self.coords.config_hash,
                "proposer_request_id": self.proposer_request_id}


# ---------------------------------------------------------------------------
# attribution: which coordinate to try


def select_coordinate(failures: list[dict], arm: str,
                      round_index: int) -> str:
    """Pick the coordinate to modify.

    Heuristic (explicitly not claimed to be error-free causal attribution):
      * target missing or locally unreadable -> phi
      * evidence stale, or coverage/retention/order wrong -> psi
      * evidence already supported but processing wrong -> theta
      * undecidable -> fixed rotation

    Ground-truth evidence locations are never an input here.
    """
    allowed = allowed_coordinates(arm)

    if not failures:
        return _rotate(allowed, round_index)

    votes = {"theta": 0, "phi": 0, "psi": 0}
    for f in failures:
        g = f.get("grounding") or {}
        per = g.get("per_requirement") or {}
        labels = list(per.values())
        n = max(1, len(labels))
        unsupported = sum(1 for v in labels if v == "unsupported") / n
        partial = sum(1 for v in labels if v == "partial") / n
        supported = sum(1 for v in labels if v == "supported") / n

        if partial >= 0.34:
            votes["phi"] += 1          # present but unreadable -> resolution
        elif unsupported >= 0.34:
            # absent entirely: either never looked (psi coverage) or cropped
            # away (phi selection). Steps used distinguishes the two.
            if f.get("observation_steps", 0) <= 1:
                votes["psi"] += 1
            else:
                votes["phi"] += 1
        elif supported >= 0.67:
            sp = f.get("spatial") or {}
            if sp.get("answer_inside_own_last_crop") is False:
                # The evidence was on screen AND the model's own answer does
                # not land inside the region it inspected: the failure lives
                # in the ANSWER channel (protocol/contract wording), not in
                # the skill text. Voting theta here sent every such round to
                # instruction edits while the point kept missing -- measured
                # on screenspot_pro (one accepted text round, zero transfer).
                votes["psi"] += 1
            else:
                votes["theta"] += 1        # evidence was there, handling failed
        else:
            votes["psi"] += 1

    ranked = [c for c in sorted(votes, key=lambda k: -votes[k])
              if c in allowed]
    if not ranked or votes[ranked[0]] == 0:
        return _rotate(allowed, round_index)

    # The attribution heuristic is strongly biased towards theta: on these
    # domains 68% of failures come back "evidence present, handling failed",
    # which legitimately points at the skill text. Left alone, that gave a
    # joint search 6 of 7 rounds on theta and ONE on the visual coordinates,
    # so the visual side of the skill got a seventh of the budget.
    #
    # The visual coordinates therefore get a guaranteed share: every other
    # round goes to phi/psi regardless of the vote. This is a fixed,
    # pre-declared allocation, not a response to any measured score. A scope
    # restricted to one side is unaffected, its allowed set already pins it.
    if len(allowed) == 3 and ranked[0] == "theta" and round_index % 2 == 1:
        return "phi" if round_index % 4 == 1 else "psi"
    return ranked[0]


def allowed_coordinates(arm: str) -> tuple[str, ...]:
    """Coordinates a proposal may touch under a given scope.

    "view_only" is how interface projection is built: it reverts theta to the
    incumbent and keeps the candidate's phi/psi (see
    Chain._visual_projection_fallback), so this scope is part of the
    mechanism rather than a selectable condition.
    """
    if arm == "text_only":
        return ("theta",)
    if arm == "view_only":
        return ("phi", "psi")
    return COORD_ROTATION


def _rotate(allowed: tuple[str, ...], round_index: int) -> str:
    return allowed[round_index % len(allowed)]


# ---------------------------------------------------------------------------
# scorer contracts: what the parser on the other end can actually read
#
# The optimizer may rewrite the answer protocol (psi.answer_contract) and
# always writes theta, but it cannot see how the SCORER parses the answer.
# Measured cost of that blindness (bv_maze / bv_count_clusters, 2026-09-18):
# joint candidates taught genuinely better work disciplines -- per-entrance
# verdicts ("B: REACHED"), coordinate-annotated enumerations ("1: flower at
# (0.35,0.29)") -- and every one scored 0.000, because the letter scorer
# takes the FIRST standalone A-Z (the first entrance's label, not the one
# marked REACHED) and the count scorer reads EVERY number in the text (so an
# enumeration with coordinates can never equal the count). text_only kept
# winning those rounds not because its strategy was better but because a
# bare answer survives the parser. The proposer must know the interface its
# text is judged through, the same way it knows the resolution facts.

SCORER_CONTRACTS = {
    "letter": (
        "The scorer extracts the FIRST standalone A-Z letter from the "
        "answer field. Any per-option verdicts, audit trails or reasoning "
        "MUST live in the note field or before the final answer -- if the "
        "answer text mentions several option letters, the first one wins "
        "and it is usually the wrong one."),
    "short_text": (
        "The scorer matches the answer as ONE value (number, phrase or "
        "tuple). An enumeration with coordinates ('1: item at (x, y)') "
        "contains many numbers and can never equal the count -- put the "
        "enumeration in the note field and the bare final value in the "
        "answer field. On ANLS-scored domains a NUMERIC answer with a "
        "trailing unit word ('7 movies', '3.61 miles') scores 0.1-0.4 "
        "while the bare number ('7', '3.61') scores 1.0: the gold is the "
        "bare value, and any answer_contract you write MUST preserve the "
        "no-units discipline of the default template verbatim."),
    "yes_no": "The scorer matches exactly 'yes' or 'no'.",
    "point": (
        "The scorer reads two numbers x, y (in [0,1] or pixels) from the "
        "answer field, plus an optional view reference ('image': k or "
        "'image k' / 'image=k' in text). It ignores everything else."),
    "route": (
        "The scorer parses Route Name / Departure Stop / Arrival Stop "
        "blocks per leg, in travel order. Extra prose between legs breaks "
        "the leg chain."),
}


def scorer_contract_block(answer_style: str) -> str:
    """Tell the optimizer how the scorer on this domain parses answers."""
    text = SCORER_CONTRACTS.get(answer_style)
    if not text:
        return ""
    return ("\nSCORER INTERFACE (hard constraint on any text you write -- "
            "the model's answer is judged by this parser, not by how "
            "careful it sounds):\n  " + text + "\n")

THETA_PROMPT = """\
You are improving the instruction text (a skill) given to a vision model that
answers questions about high-resolution images. The model can request zoomed
views of regions it names in normalized coordinates.

Current skill text:
---
{theta}
---

Below are training failures. For each you see what the model was shown, what it
did, and an independent check of whether the evidence the question needed was
actually present in the images it received.

{failures}

Propose ONE change to the skill text. It must be a strategy or discipline that
applies to any question of this kind — not a fact about any specific image,
question, or answer. Do not include example questions, answers, coordinates
for a particular image, or any lookup table.

Reply with a JSON object:
{{"rationale": "why this change addresses the observed failures",
  "new_text": "the complete new skill text"}}
"""

PHI_PROMPT = """\
You are tuning the view configuration of a system that answers questions about
high-resolution images. The configuration controls which pixels reach the model.

Current configuration:
{phi}

{resolution_facts}{dead_fields}Field meanings:
  overview_long_edge_px: long edge of the whole-image view
  crop_long_edge_px: long edge of a zoomed view
  crop_margin: extra context around a requested region, as a fraction of the frame
  max_crops_per_step: how many regions may be requested at once
  allow_upscale: whether a small crop may be interpolated larger. Interpolation
      adds no new INFORMATION, but it changes how many vision tokens the
      evidence occupies: a crop is otherwise emitted at native source
      resolution, so a 60px control stays 60px inside the emitted image even
      when the budget allows 1024px. Whether that trade is worth it is an
      empirical question about the executor, not a settled one.
  min_crop_frac: floor on crop size, so a crop keeps surrounding context
  union_crop_for_multi_target: put multiple targets in one covering crop, which
      preserves their spatial relation
  zoom_gate_expr: a CONDITION deciding whether a requested zoom is rendered at
      all, written by you over neutral facts about the request. Empty allows
      every zoom, which is the current behaviour.
{gate_help}
  grid: null, or {{"rows": r, "cols": c, "overlap": o}} to tile the image

Training failures, with an independent check of whether the needed evidence was
present in the images the model received:

{failures}

Propose ONE field change. Reply with a JSON object:
{{"rationale": "why", "field": "<field name>", "value": <new value>}}
"""

RESOLUTION_FACTS = """\
Measured facts about THIS domain's source images (read these before choosing a
value - a change that cannot alter a single emitted pixel wastes the round):
  median source long edge: {median_px} px
  90th percentile:         {p90_px} px
  largest:                 {max_px} px
  items whose source exceeds the current overview budget: {headroom}

The whole-image view is NEVER upscaled. So raising overview_long_edge_px above
the source long edge emits byte-identical pixels and changes nothing, while
lowering it discards detail that currently reaches the model.

A crop is not upscaled BY DEFAULT, and that default is `allow_upscale`, which
you control. With it off, a zoom re-reads the source file and so recovers real
detail only where the source has detail left to give: on a region already
delivered at the source's own resolution the crop returns the same pixels the
model has already seen, one round later. With it off the crop is also emitted
at NATIVE size, which can be far below the crop budget -- a 60 px control
stays 60 px inside the emitted image even when 1024 px are available.

"""


def resolution_facts_block(stats: dict | None) -> str:
    """Tell the optimizer the resolution facts it cannot otherwise know.

    Without this the optimizer proposed overview_long_edge_px 1024 -> 1536 on
    a domain whose images are 800 px. The render is a no-op above the source
    size, so task and C_ground came back identical to the digit and the round
    was spent learning nothing. This is a fact about the DATA, not about any
    measured score, so supplying it does not tune the search on results.
    """
    if not stats:
        return ""
    return RESOLUTION_FACTS.format(
        median_px=stats.get("src_long_edge_median", "?"),
        p90_px=stats.get("src_long_edge_p90", "?"),
        max_px=stats.get("src_long_edge_max", "?"),
        headroom=f"{stats.get('headroom_frac', 0) * 100:.0f}%")



DEAD_FIELDS_BLOCK = """\
Fields measured to be INERT on this domain's own data -- changing them emits
byte-identical pixels here, so a proposal naming one is refused before it is
evaluated. Do not propose these; pick a field that can actually alter what the
model sees:
{dead}

"""


def dead_fields_block(dead: list[str] | None) -> str:
    """Name the fields that provably cannot change a pixel on this domain.

    Measured 2026-09-12: 44-61% of the phi/psi moves the optimizer proposes
    are inert on any given dataset, and the optimizer kept picking them -- on
    MMStar `view_only` all four attempts in a round were refused as
    byte-identical, so the round ended with no candidate at all. The optimizer
    had no way to know: inertness depends on the DATA and on what the executor
    actually did, neither of which is visible in the config it is shown.

    This is a structural fact about the action space, not a measured score, so
    supplying it does not tune the search on results.
    """
    if not dead:
        return ""
    # Strip any phi./psi. prefix. The first version listed them prefixed, the
    # optimizer copied that form into its `field` value, and the validator
    # then rejected `psi.order` as unsupported -- so a hint meant to save a
    # round cost one instead. The prompt asks for a bare field name, so the
    # hint must use bare field names too.
    bare = set()
    for d in dead:
        name = d.split("=")[0].strip()
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        if name:
            bare.add(name)
    if not bare:
        return ""
    return DEAD_FIELDS_BLOCK.format(
        dead="\n".join(f"  - {d}" for d in sorted(bare)))


PSI_PROMPT = """\
You are tuning the layout and visual-history configuration of a system that
answers questions about high-resolution images.

Current configuration:
{psi}

{dead_fields}Field meanings:
  keep_overview: keep the whole-image view in context alongside crops
  annotate_overview_boxes: outline on the overview where crops came from
  history_keep_last_n: how many earlier views survive in context
  order: "overview_first" or "crops_first"
  evict_policy: "fifo" or "keep_overview_drop_crops"
  max_images_per_request: cap on images in one request
  answer_contract: rewrite the ANSWER protocol text (how the model must
      express its final answer, e.g. which image's coordinate frame a point
      is given in). The runtime resolves coordinates to the source in code;
      this field only changes the WORDS defining that interface.

Training failures, with an independent check of whether the needed evidence was
present in the images the model received:

{failures}

Propose ONE field change. Reply with a JSON object:
{{"rationale": "why", "field": "<field name>", "value": <new value>}}
"""

REJECTED_BLOCK = """

Changes already tried and REJECTED in earlier rounds (do not repeat them;
propose something different):
{rejected}
"""


def format_rejected(rejected: list[str]) -> str:
    """Tell the optimizer what has already failed, so rounds are not wasted
    re-proposing an identical change that the gate has already refused."""
    if not rejected:
        return ""
    lines = "\n".join(f"  - {r}" for r in rejected[-12:])
    return REJECTED_BLOCK.format(rejected=lines)


def format_failures(failures: list[dict], max_items: int = 8) -> str:
    """Render failure evidence for the optimizer. Contains no gold answers."""
    out = []
    for f in failures[:max_items]:
        g = f.get("grounding") or {}
        per = g.get("per_requirement") or {}
        reqs = f.get("requirements") or {}
        kind = f.get("outcome", "wrong")
        head = ("failure" if kind == "wrong"
                else "CORRECT answer but weak evidence support")
        lines = [f"--- {head} on a {f.get('category', 'question')} question"]
        if f.get("images"):
            lines.append(f"  [{len(f['images'])} image(s) attached above]")
        lines.append(f"  question: {f.get('question', '')[:200]}")
        verdict = ("(incorrect)" if f.get("outcome", "wrong") == "wrong"
                   else "(correct, but the evidence check was weak)")
        lines.append(
            f"  model answered: {str(f.get('answer'))[:80]} {verdict}")
        lines.append(f"  observation steps used: {f.get('observation_steps')}")
        lines.append(f"  images sent: {f.get('n_images_sent')}")
        sp = f.get("spatial") or {}
        if sp:
            lines.append(
                "  spatial check (no gold involved): crops="
                f"{sp.get('n_crops', 0)}, answer lands inside the model's "
                f"own last zoom: {sp.get('answer_inside_own_last_crop')}")
        views = f.get("view_summary") or []
        for v in views[:6]:
            lines.append(f"    view {v}")
        if per:
            lines.append("  evidence check:")
            for rid, label in per.items():
                text = reqs.get(rid, rid)
                lines.append(f"    [{label}] {str(text)[:120]}")
        if f.get("note"):
            lines.append(f"  model's own note: {str(f['note'])[:200]}")
        out.append("\n".join(lines))
    return "\n\n".join(out) if out else "(no failures recorded this round)"


# ---------------------------------------------------------------------------
# leakage guards

# These target SAMPLE-SPECIFIC content: an id, or a rule keyed to one
# question's content. Generic instructional phrasing such as "state which
# answer is correct" is legitimate skill text and must not be blocked -
# over-matching silently disabled the whole text_only arm, which is a worse
# failure than the leak it guards against.
_BANNED_PATTERNS = [
    re.compile(r"\bsa_\d+\b"),                       # V*Bench item stems
    re.compile(r"\bitem_id\b", re.IGNORECASE),
    # "the answer is B" / "correct answer: white" - an answer being asserted
    re.compile(r"\b(?:the\s+)?answer\s+is\s+"
               r"(?:[A-D]\b|['\"])", re.IGNORECASE),
    re.compile(r"\bcorrect answer\s*(?:is|:)\s*"
               r"(?:[A-D]\b|['\"])", re.IGNORECASE),
    # a rule conditioned on one specific question's wording
    re.compile(r"\bif the question (?:is|asks|mentions)\s+"
               r"['\"]", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\s*(?:->|contains)", re.IGNORECASE),
]


def assert_no_leakage(text: str) -> None:
    """Reject a proposal that hard-codes sample-specific knowledge."""
    for pat in _BANNED_PATTERNS:
        m = pat.search(text or "")
        if m:
            raise CandidateRejected(
                f"proposal contains sample-specific content: {m.group(0)!r}")


def enforce_arm_scope(arm: str, before: Coordinates,
                      after: Coordinates) -> Coordinates:
    """REVERT a frozen coordinate instead of discarding the candidate.

    THE ONE PLACE THIS INVARIANT LIVES. All four proposers pass through
    here, which is why the fix belongs here and not in each of them: a
    per-proposer guard is how `view_only` ended up burning every attempt
    of every round. `propose_joint` had been taught the arm's scope, but
    `view_only` has two coordinates so it takes `propose_expression`,
    which had not -- one patched call site out of four.

    Rejecting was also the wrong REMEDY, not just the wrong place. An
    optimizer that proposes a skill rewrite on a view-only arm has still
    proposed a perfectly good visual change; throwing the whole candidate
    away searches nothing. Keeping the in-scope half and reverting the
    rest is what the ablation actually means. Measured before this:
    `[view_only] round 1 attempt 1/2/3: refused (view_only may not change
    theta)` on mmlongdoc_FULL and chartqa_F2 -- three attempts, no search.

    The returned coordinates are guaranteed to differ from `before` in
    ONLY the arm's allowed coordinates; a candidate left identical after
    reverting is refused, because it would search nothing.
    """
    allowed = allowed_coordinates(arm)
    theta = after.theta if "theta" in allowed else before.theta
    phi = after.phi if "phi" in allowed else before.phi
    psi = after.psi if "psi" in allowed else before.psi
    scoped = Coordinates(theta=theta, phi=phi, psi=psi)
    if scoped.config_hash == before.config_hash:
        raise CandidateRejected(
            f"{arm} may only change {allowed}; after reverting the frozen "
            "coordinates this candidate is identical to the current one")
    return scoped


def describe_change(before: Coordinates, after: Coordinates) -> str:
    """Name the fields that ACTUALLY differ, read off the coordinates.

    THE LABEL MUST BE DERIVED, NEVER ACCUMULATED. Each proposer used to
    append to a string as it applied fields (`label += " +skill"`), so the
    label recorded what the optimizer ASKED for rather than what was
    applied. After `enforce_arm_scope` reverts a frozen coordinate the two
    disagree, and the ledger then lies: `view_only` round 1 was recorded as
    `psi.layout_expr +contract +skill` on an arm that cannot change theta
    at all. The coordinates were right; the audit trail was wrong, which is
    worse than a crash because it is silent.

    Deriving from the object makes drift impossible by construction.
    """
    parts = []
    if before.theta.hash != after.theta.hash:
        parts.append(
            f"theta: {len(before.theta.text)} -> {len(after.theta.text)} "
            "chars")
    bp, ap = before.phi.to_dict(), after.phi.to_dict()
    for k in sorted(set(bp) | set(ap)):
        if str(bp.get(k)) != str(ap.get(k)):
            parts.append(f"phi.{k}: {str(ap.get(k))[:60]}")
    bs, as_ = before.psi.to_dict(), after.psi.to_dict()
    for k in sorted(set(bs) | set(as_)):
        if str(bs.get(k)) != str(as_.get(k)):
            parts.append(f"psi.{k}: {str(as_.get(k))[:60]}")
    return " + ".join(parts) if parts else "(no field changed)"


def assert_frozen_respected(arm: str, before: Coordinates,
                            after: Coordinates) -> None:
    """Kept as an assertion for tests; enforcement is enforce_arm_scope."""
    allowed = allowed_coordinates(arm)
    for name, a, b in (("theta", after.theta.hash, before.theta.hash),
                       ("phi", after.phi.hash, before.phi.hash),
                       ("psi", after.psi.hash, before.psi.hash)):
        if name not in allowed and a != b:
            raise CandidateRejected(f"{arm} may not change {name}")


# ---------------------------------------------------------------------------
# proposal


def propose(arm: str, coords: Coordinates, failures: list[dict],
            client: ModelClient, round_index: int,
            coordinate: str | None = None,
            rejected: list[str] | None = None,
            resolution_stats: dict | None = None,
            dead_fields: list[str] | None = None,
            answer_style: str = "") -> Candidate:
    """Ask the optimizer for one change at one coordinate."""
    coord = coordinate or select_coordinate(failures, arm, round_index)
    fail_text = format_failures(failures)
    tried = format_rejected([r for r in (rejected or [])
                             if r.startswith(coord)])

    if coord == "theta":
        prompt = (THETA_PROMPT.format(theta=coords.theta.text,
                                      failures=fail_text)
                  + scorer_contract_block(answer_style) + tried)
    elif coord == "phi":
        from ..views.zoomgate import zoom_gate_help
        prompt = PHI_PROMPT.format(
            phi=json.dumps(coords.phi.to_dict(), indent=2),
            resolution_facts=resolution_facts_block(resolution_stats),
            dead_fields=dead_fields_block(dead_fields),
            gate_help=zoom_gate_help(),
            failures=fail_text) + tried
    else:
        prompt = PSI_PROMPT.format(
            psi=json.dumps(coords.psi.to_dict(), indent=2),
            dead_fields=dead_fields_block(dead_fields),
            failures=fail_text) + tried

    # A repeated round must not reuse a cached reply, or the optimizer can
    # never move past a rejected candidate.
    resp = client.call([user_message(prompt)], max_tokens=8192,
                       use_cache=False)
    obj = _parse_obj(resp.content)
    if obj is None:
        raise ModelFormatError(
            f"optimizer reply unparsable: {resp.content[:200]!r}")

    if coord == "theta":
        new_text = str(obj.get("new_text") or "")
        if not new_text.strip():
            raise ModelFormatError("optimizer returned empty skill text")
        assert_no_leakage(new_text)
        new = Coordinates(theta=SkillConfig(new_text), phi=coords.phi,
                          psi=coords.psi)
        diff = f"theta: {len(coords.theta.text)} -> {len(new_text)} chars"
    else:
        field = str(obj.get("field") or "")
        value = obj.get("value")
        valid = PHI_FIELDS if coord == "phi" else PSI_FIELDS
        if field not in valid:
            # A prefixed name ("psi.order") is a formatting slip, not a real
            # attempt at an unsupported field: accept it rather than burning
            # one of the round's attempts on punctuation. Observed twice in
            # w6 after the dead-field hint listed names with prefixes.
            stripped = field.rsplit(".", 1)[-1] if "." in field else field
            if stripped in valid:
                field = stripped
            else:
                raise CandidateRejected(
                    f"{coord} field {field!r} is not supported by the "
                    f"runtime; supported: {sorted(valid)}")
        assert_no_leakage(json.dumps(obj))
        d = (coords.phi.to_dict() if coord == "phi"
             else coords.psi.to_dict())
        old = d.get(field)
        d[field] = value
        if coord == "phi":
            new = Coordinates(theta=coords.theta,
                              phi=PhiConfig.from_dict(d), psi=coords.psi)
        else:
            new = Coordinates(theta=coords.theta, phi=coords.phi,
                              psi=PsiConfig.from_dict(d))
        diff = f"{coord}.{field}: {old!r} -> {value!r}"

    if new.config_hash == coords.config_hash:
        raise CandidateRejected("proposal is identical to the current config")

    if diff in (rejected or []):
        raise CandidateRejected(f"change already tried and rejected: {diff}")

    new = enforce_arm_scope(arm, coords, new)
    # The label is DERIVED from the scoped coordinates, so it can never
    # claim a field the arm did not actually change.
    diff = describe_change(coords, new)

    return Candidate(coordinate=coord, coords=new,
                     rationale=str(obj.get("rationale", ""))[:1000],
                     diff=diff, proposer_request_id=resp.request_id or "")


def _parse_obj(text: str):
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text or "", re.DOTALL)
    for c in ([fenced.group(1)] if fenced else []) + [text or ""]:
        i, j = c.find("{"), c.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(c[i:j + 1])
            except json.JSONDecodeError:
                # Optimizers (claude-opus-5 especially) pretty-print JSON with
                # LITERAL newlines inside string values, which is invalid
                # strict JSON but unambiguous to a tolerant parser. strict=False
                # admits control characters inside strings; it cannot turn
                # non-JSON into JSON, only keep well-formed-ish replies alive.
                try:
                    return json.loads(c[i:j + 1], strict=False)
                except json.JSONDecodeError:
                    continue
    return None


OPERATOR_PROMPT = """\
You are designing VISUAL OPERATORS for a system that answers questions about
images. You are not choosing from a menu: you WRITE the operator as a program,
and the runtime executes it on the pixels the model will see.

An operator is a pipeline of steps. Each step is an object with an "op" key
plus its parameters. Numeric parameters may be arithmetic expressions over
the neutral context: {context_keys}.

Available steps (deliberately low-level, so that techniques come from how YOU
combine them, not from a fixed list of named effects):
{step_help}

Recognisable techniques are compositions, not steps. For example:
  coordinate grid     : draw_lines with axis="both" and a computed spacing
  adaptive binarise   : autocontrast -> local_threshold with a computed block
  unsharp + contrast  : autocontrast -> sharpen
  set-of-mark         : draw_box on regions -> draw_text with their numbers
  contact sheet       : tile(cols=...) -> label_tiles(times=true)
  temporal strip      : tile(cols=N) -> label_tiles with timestamps
  difference of a pair: blend(mode="difference") -> autocontrast
Invent others. The useful operator for THIS domain may be none of the above.

Two places an operator can live:
  phi.view_ops    - applied to EVERY view separately (one image in, one out).
                    Use for legibility: contrast, sharpening, thresholding,
                    grids, marks.
  psi.context_ops - applied to the ASSEMBLED set of views, and may REDUCE
                    many images to one. Use for organisation: tiling frames
                    into a labelled sheet, differencing a paired figure.

Current phi.view_ops:     {view_ops}
Current psi.context_ops:  {context_ops}

{resolution_facts}Training failures, with an independent check of whether the needed evidence
was present in the images the model received:

{failures}

Design ONE operator program that addresses what you see in these failures.
Think about what the model could not READ or could not RELATE, and what
rearrangement of the pixels would fix it.

Hard constraints:
  * the program must change the emitted pixels; a no-op is rejected unevaluated
  * at most {max_steps} steps
  * it must apply to any question of this kind, never encode a specific answer

Reply with a JSON object:
{{"rationale": "what failure this addresses and why this program fixes it",
  "target": "phi.view_ops" or "psi.context_ops",
  "name": "a short descriptive name",
  "program": [{{"op": "...", ...}}, ...]}}
"""


def step_help_block() -> str:
    """One line per step, generated from the live registry.

    Generated rather than written out, so the prompt can never advertise a
    step the runtime does not implement -- that mismatch is how the optimizer
    previously wasted rounds proposing fields nothing read.
    """
    from ..views.steps import STEPS
    import inspect
    lines = []
    for name in sorted(STEPS):
        fn, kind = STEPS[name]
        doc = (inspect.getdoc(fn) or "").strip().split("\n")[0]
        tag = "REDUCE" if kind == "reduce" else "pixel"
        lines.append(f"  {name:18s} [{tag}] {doc}")
    return "\n".join(lines)


def propose_operator(arm: str, coords: Coordinates, failures: list[dict],
                     client: ModelClient, round_index: int,
                     resolution_stats: dict | None = None,
                     rejected: list[str] | None = None,
                     answer_style: str = "") -> Candidate:
    """Ask the optimizer to AUTHOR a visual operator program.

    This is the model-authored half of phi/psi. The harness supplies a step
    vocabulary and executes what comes back; it does not supply the ideas.
    """
    from ..views.operators import CTX_KEYS, MAX_STEPS, OperatorRejected
    from ..views.steps import step_names

    prompt = OPERATOR_PROMPT.format(
        context_keys=", ".join(sorted(CTX_KEYS)),
        step_help=step_help_block(),
        view_ops=json.dumps(list(coords.phi.view_ops or []))[:800],
        context_ops=json.dumps(list(coords.psi.context_ops or []))[:800],
        resolution_facts=resolution_facts_block(resolution_stats),
        failures=format_failures(failures),
        max_steps=MAX_STEPS,
    ) + format_rejected([r for r in (rejected or []) if "ops" in r])

    resp = client.call([user_message(prompt)], max_tokens=8192,
                       use_cache=False)
    obj = _parse_obj(resp.content)
    if obj is None:
        raise ModelFormatError(
            f"operator reply unparsable: {resp.content[:200]!r}")

    target = str(obj.get("target") or "phi.view_ops")
    program = obj.get("program")
    name = str(obj.get("name") or f"op_r{round_index}")
    if not isinstance(program, list) or not program:
        raise CandidateRejected("operator program is empty")
    for st in program:
        if not isinstance(st, dict) or "op" not in st:
            raise CandidateRejected("each step needs an 'op' key")
        if st["op"] not in step_names():
            raise CandidateRejected(
                f"unknown step {st['op']!r}; available: {step_names()}")
    assert_no_leakage(json.dumps(obj))

    # Arity check BEFORE the candidate costs a gate pass. The first live
    # operator put a REDUCE step (blend) inside a phi.view_ops program, which
    # runs on one view at a time; that is decidable from the program alone.
    from ..views.steps import validate_program
    is_ctx = target.endswith("context_ops")
    validate_program(program, n_inputs=(4 if is_ctx else 1),
                     kind=("compose" if is_ctx else "pixel"))

    from ..views.operators import ViewOperator
    op = ViewOperator(name=name, program=program, kind=(
        "compose" if target.endswith("context_ops") else "pixel"),
        rationale=str(obj.get("rationale", ""))[:1000],
        author_request_id=resp.request_id or "")
    op.validate()

    phi_d, psi_d = coords.phi.to_dict(), coords.psi.to_dict()
    if target.endswith("context_ops"):
        psi_d["context_ops"] = program
        coord = "psi"
    else:
        phi_d["view_ops"] = program
        coord = "phi"
    new = Coordinates(theta=coords.theta,
                      phi=PhiConfig.from_dict(phi_d),
                      psi=PsiConfig.from_dict(psi_d))
    if new.config_hash == coords.config_hash:
        raise CandidateRejected("operator is identical to the current config")

    diff = f"{coord}.{'context_ops' if coord == 'psi' else 'view_ops'}: " \
           f"{name} [{'+'.join(str(s.get('op')) for s in program)}]"
    new = enforce_arm_scope(arm, coords, new)
    # The label is DERIVED from the scoped coordinates, so it can never
    # claim a field the arm did not actually change.
    diff = describe_change(coords, new)
    return Candidate(coordinate=coord, coords=new,
                     rationale=op.rationale, diff=diff,
                     proposer_request_id=resp.request_id or "")


PIXEL_LANGUAGE_NOTE = """\
  UNITS: red/green/blue/lum are 0..255; redn/greenn/bluen/lumn are the
  same planes in 0..1. Pick one convention; mixing them clips to black.

  KEEP IT FLAT. Bind intermediates first -- a three-channel transform is
  far easier to get right that way than as one nested expression, which
  is where parentheses get miscounted:
      base = clip((red - local_mean(red, 9)) * 1.5 + red, 0, 255)
      rgb(base, base, base)
  The last line must be the expression that produces the image. Source
  planes cannot be rebound.

  CHAINING TWO STAGES uses the same bindings -- there is no pipe
  operator and no lambda. To sharpen and THEN stretch contrast, bind the
  first stage and read it in the second:
      r1 = clip(red + 0.9*(red - gaussian(red, 1.2)), 0, 255)
      g1 = clip(green + 0.9*(green - gaussian(green, 1.2)), 0, 255)
      b1 = clip(blue + 0.9*(blue - gaussian(blue, 1.2)), 0, 255)
      rgb(clip(128 + 1.35*(r1-128), 0, 255),
          clip(128 + 1.35*(g1-128), 0, 255),
          clip(128 + 1.35*(b1-128), 0, 255))
"""

EXPR_PROMPT = """\
You are designing the VISUAL PRESENTATION for a model answering questions
about images. You WRITE the transform as an expression; you are not choosing
from a list of named effects.

Attached below are the IMAGES THE MODEL ACTUALLY SAW on failures, in the
exact form it received them. Look at them. Decide what about the
presentation made the question hard to answer.

Two things you can write. Pick whichever the failures call for.

1) pixels -- an array expression producing a new image.
   UNITS MATTER. Two sets of names for the same planes:
     red, green, blue, lum        0..255   (integer-range algebra)
     redn, greenn, bluen, lumn    0..1     (normalised algebra)
   Pick ONE convention and stay in it. `clip(0.5 + 1.35*(red-0.5), 0, 1)`
   mixes them and clips the whole frame to black.
{pixel_note}
   Names: red, green, blue -- the colour channels (USE THESE).
          lum (= luminance). Legacy aliases: g IS LUMINANCE, NOT GREEN;
          gc is the green channel. Prefer red/green/blue/lum so a colour
          expression cannot silently swap channels -- writing
          rgb(gc, g, b) puts green in the red slot and luminance in the
          green slot, which tints the whole image.
          x, y (0..1), cx, cy (-1..1), radius (0 at centre);
          width, height, n_frames, duration_s, long_edge.
   Functions:
{fn_help}
   Examples of what is EXPRESSIBLE (do not just copy these):
     where(local_mean(g,31) - g > 8, 0, 255)      local adaptive threshold
     clip(g + 0.8*(g - gaussian(g,3)), 0, 255)    unsharp mask
     clip((g - local_mean(g,25))*3 + 128, 0, 255) local contrast equalisation
     where(radius < 0.5, g, g*0.4)                de-emphasise the periphery
     clip(abs(sobel_x(g)) + abs(sobel_y(g)),0,255) edge structure

2) selection -- WHICH source units the model sees at all. For a video or a
   multi-page document this is usually the biggest lever: the pool holds 32
   frames/pages and only a few are sent, so "which ones" is a real decision.
   The default is uniform sampling; `uniform(k)` reproduces it exactly.
{selection_help}

3) layout -- a composition over the views the model receives.
   Free names: `views` (all sent views), `crops` (zoom views only).
   Functions:
{layout_help}
   Examples of what is EXPRESSIBLE:
     grid(views, cols=4, label=times)             one labelled sheet, not 8 images
     stack_v(pick(views,0), grid(crops, cols=2))  context above its details
     beside(pick(views,0,1), label=letter)        a paired comparison
     overlay(pick(views,0), boxes=crops)          show where crops came from

LAYOUT IS AS IMPORTANT AS PIXELS. If the model is receiving several images
and failing to relate them, reorganising the visual context is usually worth
more than sharpening any one of them.

3) THE WORDS THAT DESCRIBE THE PIXELS. This is co-evolution, and it is not
   optional bookkeeping -- it is where the last attempt lost its points.

   The model is told, in text, what each attached image IS. By default:
     "Image 1: a FULL frame at t=3.2s (640x480 shown)"   (one line per image)
     "Coordinates are relative to the image you name (default: the FULL frame)."

   A layout that tiles 8 frames into 1 sheet makes BOTH lines false: there is
   no per-frame image any more and no full frame to address. Measured, when
   the description was left frozen: observation steps went 2.2 -> 7.0, images
   piled up to the 24-step cap on 19% of items, and 16 of 18 spiralling units
   ran out of budget without answering. On the items that did NOT spiral the
   layout was worth -0.019, i.e. nothing. The pixels were fine; the stale
   description is what cost the points.

   So if your change alters what an image MEANS, ship the matching text:
     "contract" -- how the attached images are described. Optional
                   placeholders {{n_images}} and {{image_lines}}.
     "skill"    -- the full replacement task instructions, if the task
                   framing itself should change.
   Send either, both, or neither. Omit them if the default text is still
   TRUE of what you are sending.

Current pixels expression: {cur_pixel}
Current layout expression: {cur_layout}
Current view contract: {cur_contract}
Current skill text:
{cur_skill}

{resolution_facts}Failure traces (the attached images correspond to these, in order):

{failures}

Hard constraints:
  * the expression must change what is emitted; a no-op is rejected
  * it must generalise to any question of this kind, never encode an answer

Reply with a JSON object:
{{"rationale": "what you SAW in the images and why this fixes it",
  "target": "pixels" or "layout",
  "expr": "the expression",
  "select": "selection expression over source units, or omit",
  "contract": "replacement image description, or omit",
  "skill": "replacement task instructions, or omit"}}
"""


def propose_expression(arm: str, coords: Coordinates, failures: list[dict],
                       client: ModelClient, round_index: int,
                       resolution_stats: dict | None = None,
                       rejected: list[str] | None = None,
                       answer_style: str = "") -> Candidate:
    """Ask the optimizer to WRITE a pixel or layout expression.

    The failure images are ATTACHED. Before this, the optimizer designed
    visual operators from a text summary and had never seen a failing image,
    which is why its proposals were generic legibility boilerplate.
    """
    from ..views.exprs import function_help
    from ..views.layout import layout_help
    from ..views.selection import selection_help

    imgs: list[str] = []
    for f in failures[:3]:
        for u in (f.get("images") or [])[:2]:
            if u and len(imgs) < 6:
                imgs.append(u)

    prompt = EXPR_PROMPT.format(
        pixel_note=PIXEL_LANGUAGE_NOTE,
        fn_help=function_help(),
        layout_help=layout_help(),
        selection_help=selection_help(),
        cur_pixel=(coords.phi.pixel_expr or "(none)"),
        cur_layout=(coords.psi.layout_expr or "(none)"),
        cur_contract=(getattr(coords.psi, "view_contract", "")
                      or "(default: per-image FULL frame lines + "
                         "'coordinates are relative to the image you name')"),
        cur_skill=coords.theta.text,
        resolution_facts=resolution_facts_block(resolution_stats),
        failures=format_failures(failures),
    ) + format_rejected([r for r in (rejected or []) if "expr" in r])

    resp = client.call([user_message(prompt, imgs or None)],
                       max_tokens=8192, use_cache=False)
    obj = _parse_obj(resp.content)
    if obj is None:
        raise ModelFormatError(
            f"expression reply unparsable: {resp.content[:200]!r}")

    target = str(obj.get("target") or "pixels").lower()
    expr = str(obj.get("expr") or "").strip()
    if not expr:
        raise CandidateRejected("empty expression")
    assert_no_leakage(json.dumps(obj))

    phi_d, psi_d = coords.phi.to_dict(), coords.psi.to_dict()
    if target.startswith("layout"):
        psi_d["layout_expr"] = expr
        coord, label = "psi", "psi.layout_expr"
    else:
        phi_d["pixel_expr"] = expr
        coord, label = "phi", "phi.pixel_expr"

    # CO-EVOLUTION. A change to the pixels can make the TEXT that describes
    # them false, and until now that text was frozen Python the optimizer
    # could not reach. Collapsing 8 frames into 1 sheet left the prompt
    # still saying "Image i: a FULL frame" and "coordinates must refer to
    # the FULL frame"; the executor answered by zooming until it ran out of
    # budget (steps 2.2 -> 7.0, 16 of 18 spiralling units truncated), and
    # the layout was blamed for what the stale description cost.
    #
    # The optimizer may now ship a matching `contract` (how the images are
    # described) and `skill` (the task instructions) in the SAME proposal,
    # so the visual change and its explanation move together.
    theta = coords.theta
    contract = str(obj.get("contract") or "").strip()
    if contract:
        assert_no_leakage(contract)
        psi_d["view_contract"] = contract
        label += " +contract"
    skill = str(obj.get("skill") or "").strip()
    if skill:
        assert_no_leakage(skill)
        theta = SkillConfig(skill)
        label += " +skill"

    new = Coordinates(theta=theta,
                      phi=PhiConfig.from_dict(phi_d),
                      psi=PsiConfig.from_dict(psi_d))
    if new.config_hash == coords.config_hash:
        raise CandidateRejected("expression is identical to the current one")
    diff = f"{label}: {expr[:70]}"
    if diff in (rejected or []):
        raise CandidateRejected(f"already tried: {diff}")
    new = enforce_arm_scope(arm, coords, new)
    # The label is DERIVED from the scoped coordinates, so it can never
    # claim a field the arm did not actually change.
    diff = describe_change(coords, new)
    return Candidate(coordinate=coord, coords=new,
                     rationale=str(obj.get("rationale", ""))[:1000],
                     diff=diff, proposer_request_id=resp.request_id or "")

# ---------------------------------------------------------------------------
# JOINT PROPOSAL: theta + phi + psi in one move
#
# WHY THIS REPLACES COORDINATE ROTATION
# -------------------------------------
# `select_coordinate` returns ONE of "theta"/"phi"/"psi" and the round edits
# only that one. That is coordinate rotation, not co-evolution, and in a
# multimodal agent the three coordinates are not independent:
#
#   * a layout that tiles 8 frames into 1 sheet makes the skill text's
#     "track objects across frames" and the contract's "Image i: a FULL
#     frame" both FALSE
#   * a pixel transform that drops colour makes a skill instruction to
#     "identify the red object" unanswerable
#   * a skill that says "zoom into the axis labels" is empty if phi's crop
#     budget cannot resolve them
#
# Measured cost of moving one without the others: tiling the frames while
# leaving the description frozen sent observation steps from 2.2 to 7.0 and
# truncated 16 of 18 spiralling units, while the same layout on units that
# did not spiral was worth -0.019. The pixels were fine. The coupling was
# what failed, and rotation cannot express a fix to a coupling.
#
# A joint proposal is ONE candidate: one config_hash, one gate pass, accepted
# or rejected as a whole. Attribution is recovered by ablation (the
# single-coordinate arms), not by restricting what a proposal may touch.

JOINT_PROMPT = """\
You are improving a multimodal agent by rewriting THREE coupled things at
once. Attached are the IMAGES THE MODEL ACTUALLY SAW on failures, in the
exact form it received them. Look at them first.

The three coordinates, and why they are one object:

  theta  the task instructions (SKILL.md)
  phi    a PIXEL EXPRESSION -- an array expression producing a new image
  psi    a LAYOUT EXPRESSION over the views, plus the VIEW CONTRACT: the
         text that tells the model what each attached image IS

They are coupled. Change the pixels and the words describing the pixels can
become false. That is not hypothetical -- it is the largest measured effect
in this study so far: tiling 8 video frames into one labelled sheet while
leaving the description saying "Image i: a FULL frame" and "coordinates must
refer to the FULL frame" drove observation steps from 2.2 to 7.0 and left 16
of 18 affected items answering nothing at all, purely because the text no
longer matched the pixels. On the items where that protocol did not break,
the same layout was worth -0.019, i.e. the layout itself was neutral.

So: propose a change, and make the OTHER TWO consistent with it.

PIXEL EXPRESSION vocabulary
  UNITS: red/green/blue/lum are 0..255; redn/greenn/bluen/lumn are the
  same planes in 0..1. Pick one convention; mixing them clips to black.

  KEEP IT FLAT. You may bind intermediates first, and a three-channel
  transform is far easier to get right that way than as one nested
  expression -- deeply nested clip() calls are where parentheses get
  miscounted:
      base  = clip((red - local_mean(red, 9)) * 1.5 + red, 0, 255)
      detail = red - gaussian(red, 2)
      rgb(base, base, base)
  The last line must be the expression that produces the image. Source
  planes cannot be rebound.

  CHAINING TWO STAGES uses the same bindings -- there is no pipe operator.
  To sharpen and THEN stretch contrast, bind the first stage and read it
  in the second:
      r1 = clip(red + 0.9*(red - gaussian(red, 1.2)), 0, 255)
      g1 = clip(green + 0.9*(green - gaussian(green, 1.2)), 0, 255)
      b1 = clip(blue + 0.9*(blue - gaussian(blue, 1.2)), 0, 255)
      rgb(clip(128 + 1.35*(r1-128), 0, 255),
          clip(128 + 1.35*(g1-128), 0, 255),
          clip(128 + 1.35*(b1-128), 0, 255))
  Names: red, green, blue (colour channels -- USE THESE), lum (luminance).
         Legacy: g IS LUMINANCE not green; gc is green. rgb(gc, g, b) is
         the classic mistake: it green-casts the image.
         x, y (0..1), cx, cy (-1..1), radius; width, height, n_frames,
         duration_s, long_edge.
  Functions:
{fn_help}
  NOTE: an expression returning ONE plane is rendered as GREYSCALE. Use
  rgb(r_plane, g_plane, b_plane) to keep hue. Measured: a luminance-only
  contrast transform cost 0.958 -> 0.621 on a benchmark where 54% of
  questions ask about colour.

SELECTION EXPRESSION vocabulary -- WHICH source units are shown at all.
  For a video or multi-page document this is usually the biggest lever. The
  pool holds many more units than the send budget, so choosing which ones
  survive is a real decision; uniform sampling is merely the DEFAULT, not
  the law. `uniform(k)` reproduces the default exactly.
{selection_help}

LAYOUT EXPRESSION vocabulary
  Free names: `views` (all sent views), `crops` (zoom views only).
  Functions:
{layout_help}

VISUAL MEMORY LAYER (canvas) -- the model can WRITE on the image, not only read it.
  The executor may drop a `note` (a normalised point + optional text) on the
  full frame; notes accumulate across steps and are burned back into the view,
  so the image becomes an external working memory (mark each counted object,
  cross off ruled-out options, label a region to remember). You do NOT choose
  the gestures; you write `canvas_expr`, an array expression over the base
  planes PLUS these note arrays, deciding how the marks are rendered:
    dist        per-pixel distance to the nearest note (normalised by long edge)
    nearest_idx 1-based ordinal of that note, in the order the model wrote them
    note_count  scalar; how many notes are on the image
  The notes' ordinals and text are always drawn on top; your expression sets
  the MARK. Examples of what is EXPRESSIBLE (do not just copy):
    rgb(where(dist<0.04,255,red), where(dist<0.04,80,green), blue)  bright ring per mark
    where(dist<0.05, red*0.4, red)                                 dim each marked spot
    rgb(red, green, where(nearest_idx>note_count/2, 255, blue))    colour later marks
  A canvas_expr that ignores the note arrays is a no-op and is rejected.

ZOOM GATE -- WHETHER a requested zoom is performed at all.
  The executor asks for a region; `phi.zoom_gate_expr` is a condition you write
  that decides whether that crop is actually rendered. A refused request costs
  the model one prompt line telling it to ask elsewhere or answer now.
  This exists because a crop is NEVER upscaled: over 1808 recorded zooms the
  median magnification was 1.000x, i.e. the model looked again and received the
  pixels it already had, one round later. Looking twice is not free -- it spends
  an observation step and a re-read of the same evidence.
{gate_help}

Current state
  theta:
{cur_skill}
  phi.select_expr:  {cur_select}
  phi.pixel_expr:   {cur_pixel}
  phi.canvas_expr:  {cur_canvas}
  phi.zoom_gate_expr: {cur_gate}
  psi.layout_expr:  {cur_layout}
  psi.view_contract: {cur_contract}
  psi.answer_contract: {cur_answer_contract}
  psi.protocol_contract: {cur_protocol_contract}

{resolution_facts}Failure traces (the attached images correspond to these, in order):

{failures}

Hard constraints
  * at least one of skill / expr / layout / canvas / gate must actually change
  * an expression that emits identical pixels is rejected
  * never encode an answer; it must generalise to any question of this kind
  * if you change what an image MEANS, ship the contract that matches it

{arm_scope}Reply with a JSON object. Omit any field you do not want to change:
{{"rationale": "what you SAW, and why these change together",
  "skill": "full replacement task instructions, or omit",
  "select": "selection expression over source units, or omit",
  "expr": "pixel expression, or omit",
  "canvas": "canvas expression over the note arrays, or omit",
  "gate": "zoom-gate condition, or omit",
  "layout": "layout expression, or omit",
  "contract": "how attached images are described, or omit",
  "answer_contract": "how the final answer must be expressed, or omit",
  "coupling": "one sentence: which change forced which other change",
  "missing_primitive": "if you wanted an operator that does not exist, "
                       "name it and say what it would compute; omit "
                       "otherwise"}}
"""


def propose_joint(arm: str, coords: Coordinates, failures: list[dict],
                  client: ModelClient, round_index: int,
                  resolution_stats: dict | None = None,
                  rejected: list[str] | None = None,
                  directive: str = "",
                  answer_style: str = "") -> Candidate:
    """One proposal that may move theta, phi and psi together.

    This is the co-evolution unit. It is deliberately NOT restricted to one
    coordinate: the defect it exists to fix is a mismatch BETWEEN
    coordinates, which no single-coordinate edit can express.

    `directive` is the strategy-book steering context (see strategy_book.py):
    it points the round at the highest-value unresolved visual failure mode and
    bars re-rolling strategies that already failed, which is what forces
    diversity across rounds instead of collapse onto the contrast prior.
    """
    from ..views.exprs import function_help
    from ..views.layout import layout_help
    from ..views.selection import selection_help
    from ..views.zoomgate import zoom_gate_help

    imgs: list[str] = []
    for f in failures[:3]:
        for u in (f.get("images") or [])[:2]:
            if u and len(imgs) < 6:
                imgs.append(u)

    # THE ARM'S SCOPE MUST BE IN THE PROMPT.
    #
    # `assert_frozen_respected` rejects a candidate that moves a frozen
    # coordinate, but nothing told the optimizer which ones are frozen, so
    # on `view_only` every attempt proposed a skill rewrite and was refused
    # ("view_only may not change theta"). The arm burned all four attempts
    # per round and searched nothing -- measured on mmlongdoc_FULL round 1.
    #
    # The fix belongs before the proposal, not in a post-hoc filter: an
    # optimizer that is told the constraint can satisfy it.
    _allowed = allowed_coordinates(arm)
    _fields = []
    if "theta" in _allowed:
        _fields.append("skill")
    if "phi" in _allowed:
        _fields.append("expr (pixel transform), select (which units), "
                       "gate (whether a zoom is performed)")
    if "psi" in _allowed:
        _fields.append("layout, contract, answer_contract, protocol_contract")
    scope = ("SCOPE FOR THIS ARM -- you may change ONLY: "
             + "; ".join(_fields) + ".\nEvery other field is frozen here; "
             "a proposal that changes one is discarded and the round is "
             "wasted. Do not emit it.\n\n") if len(_allowed) < 3 else ""

    prompt = directive + scorer_contract_block(answer_style) + JOINT_PROMPT.format(
        arm_scope=scope,
        fn_help=function_help(),
        layout_help=layout_help(),
        selection_help=selection_help(),
        gate_help=zoom_gate_help(),
        cur_select=(getattr(coords.phi, "select_expr", "")
                    or "(default: uniform sampling over the pool)"),
        cur_skill=coords.theta.text,
        cur_pixel=(coords.phi.pixel_expr or "(none)"),
        cur_canvas=(getattr(coords.phi, "canvas_expr", "")
                    or "(default: a plain disc + the note's auto-number)"),
        cur_gate=(getattr(coords.phi, "zoom_gate_expr", "")
                  or "(default: every requested zoom is performed)"),
        cur_layout=(coords.psi.layout_expr or "(none)"),
        cur_contract=(getattr(coords.psi, "view_contract", "")
                      or "(default: per-image 'FULL frame' lines + "
                         "'coordinates are relative to the image you name')"),
        cur_answer_contract=(getattr(coords.psi, "answer_contract", "")
                             or "(default: the domain's ANSWER_STYLES entry)"),
        cur_protocol_contract=(getattr(coords.psi, "protocol_contract", "")
                               or "(default: the built-in observation-protocol "
                                  "HOWTO: zoom/note/view/compose usage)"),
        resolution_facts=resolution_facts_block(resolution_stats),
        failures=format_failures(failures),
    ) + format_rejected(rejected or [])

    resp = client.call([user_message(prompt, imgs or None)],
                       max_tokens=12288, use_cache=False)
    obj = _parse_obj(resp.content)
    if obj is None:
        raise ModelFormatError(
            f"joint reply unparsable: {resp.content[:200]!r}")
    assert_no_leakage(json.dumps(obj))

    allowed = allowed_coordinates(arm)
    phi_d, psi_d = coords.phi.to_dict(), coords.psi.to_dict()
    theta = coords.theta
    touched: list[str] = []

    skill = str(obj.get("skill") or "").strip()
    if skill and "theta" not in allowed:
        skill = ""          # frozen for this arm: ignore, do not fail
    if skill and "theta" in allowed and skill != coords.theta.text:
        theta = SkillConfig(skill)
        touched.append("theta")

    expr = str(obj.get("expr") or "").strip()
    if expr and "phi" in allowed and expr != coords.phi.pixel_expr:
        phi_d["pixel_expr"] = expr
        touched.append("phi.pixel_expr")

    sel = str(obj.get("select") or "").strip()
    if sel and "phi" in allowed and sel != getattr(
            coords.phi, "select_expr", ""):
        phi_d["select_expr"] = sel
        touched.append("phi.select_expr")

    canvas = str(obj.get("canvas") or "").strip()
    if canvas and "phi" in allowed and canvas != getattr(
            coords.phi, "canvas_expr", ""):
        phi_d["canvas_expr"] = canvas
        touched.append("phi.canvas_expr")

    gate = str(obj.get("gate") or "").strip()
    if gate and "phi" in allowed and gate != getattr(
            coords.phi, "zoom_gate_expr", ""):
        phi_d["zoom_gate_expr"] = gate
        touched.append("phi.zoom_gate_expr")

    layout = str(obj.get("layout") or "").strip()
    if layout and "psi" in allowed and layout != coords.psi.layout_expr:
        psi_d["layout_expr"] = layout
        touched.append("psi.layout_expr")

    contract = str(obj.get("contract") or "").strip()
    if contract and "psi" in allowed and contract != getattr(
            coords.psi, "view_contract", ""):
        psi_d["view_contract"] = contract
        touched.append("psi.view_contract")

    ans_contract = str(obj.get("answer_contract") or "").strip()
    if ans_contract and "psi" in allowed and ans_contract != getattr(
            coords.psi, "answer_contract", ""):
        psi_d["answer_contract"] = ans_contract
        touched.append("psi.answer_contract")

    proto_contract = str(obj.get("protocol_contract") or "").strip()
    if proto_contract and "psi" in allowed and proto_contract != getattr(
            coords.psi, "protocol_contract", ""):
        psi_d["protocol_contract"] = proto_contract
        touched.append("psi.protocol_contract")

    if not touched:
        raise CandidateRejected(
            "joint proposal changed nothing inside this arm's allowed "
            f"coordinates {allowed}")

    new = Coordinates(theta=theta, phi=PhiConfig.from_dict(phi_d),
                      psi=PsiConfig.from_dict(psi_d))
    # Same single chokepoint as every other proposer. `propose_joint` runs
    # only on the full arm today, but routing is a property of the chain,
    # not of this function, and the last bug was exactly one call site out
    # of four being patched.
    new = enforce_arm_scope(arm, coords, new)
    # The label is DERIVED from the scoped coordinates, so it can never
    # claim a field the arm did not actually change.
    diff = describe_change(coords, new)
    if new.config_hash == coords.config_hash:
        raise CandidateRejected("joint proposal is identical to current")

    # The recorded coordinate is "joint" when more than one moved, so the
    # ledger never claims a multi-coordinate move was a single-coordinate
    # one. Attribution comes from the ablation arms.
    coordinate = ("joint" if len(set(t.split(".")[0] for t in touched)) > 1
                  else touched[0].split(".")[0])
    # The diff must identify the CONTENT, not just which fields moved.
    # "theta + phi.pixel_expr + psi.layout_expr + psi.view_contract" is the
    # label of almost every joint proposal, so a label-only dedup declared
    # each new candidate "already tried" and burned the round: blink_p11
    # round 3 lost all attempts that way. Appending a short content hash
    # keeps genuine repeats out while letting different programs through.
    # `canvas` and `gate` belong in the hash for the same reason as the rest:
    # two candidates that differ only in the gate carry the identical label
    # "phi.zoom_gate_expr", so a hash omitting it would call the second one
    # "already tried" and burn the attempt.
    _sig = sha({"skill": skill, "expr": expr, "layout": layout,
                "select": sel, "contract": contract, "canvas": canvas,
                "gate": gate, "answer_contract": ans_contract})[:8]
    diff = (" + ".join(touched))[:88] + f" #{_sig}"
    if diff in (rejected or []):
        raise CandidateRejected(f"already tried: {diff}")

    rationale = str(obj.get("rationale") or "")
    # WHETHER THE OPERATOR TABLE BINDS IS A MEASUREMENT, NOT A GUESS.
    # The vocabulary is author-chosen, so "is it expressive enough" can
    # only be settled by asking the one entity that hits the wall. If this
    # field keeps naming the same operator, that is the signal to add it.
    missing = str(obj.get("missing_primitive") or "").strip()
    if missing:
        rationale = f"{rationale}  [MISSING PRIMITIVE: {missing[:200]}]"
    coupling = str(obj.get("coupling") or "").strip()
    if coupling:
        rationale = f"{rationale}  [coupling: {coupling}]"
    return Candidate(coordinate=coordinate, coords=new, rationale=rationale,
                     diff=diff, proposer_request_id=resp.request_id or "")
