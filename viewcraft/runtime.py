"""Executor runtime: run one sample once under one (arm, condition).

The executor receives only PublicItem material plus views it requested. It
never receives an answer, an evidence location, or a ground-truth box. Views
are produced by phi/psi executing in code; the model asks for regions in
normalized coordinates and the runtime performs the crop.

An observation budget bounds how many view-producing steps may occur. The
budget is shared by every non-oracle arm in a domain.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from .models import (CFG, ModelClient, ModelFormatError, ModelUnavailable,
                     system_message, user_message)
from .schema import (ExecResult, PublicItem, Region, Trajectory, ViewRecord)
from .views.config import Coordinates, clamp_region
from .views.operators import OperatorRejected
from .views.render import (RenderedView, apply_context_ops, apply_view_ops,
                           render_crop, render_full,
                           render_overview_with_boxes)
from .views.zoomgate import eval_zoom_gate, zoom_facts

# A reasoning executor spends its completion allowance on hidden reasoning
# tokens before emitting any text; 512-1024 left nothing for the answer and
# produced empty replies. Sized from the config so a non-reasoning model does
# not pay for headroom it never uses.
ANSWER_TOKENS = int(
    (CFG.get("models", {}).get("executor", {}) or {}).get(
        "answer_max_tokens", 1024))


def answer_tokens_for(domain: str) -> int:
    """Completion allowance for a domain.

    The global value applies unless a domain was MEASURED to be pressed
    against it. On VSI-Bench 8.2% of replies reached the 2048 cap (p99 =
    2044) and 0.57% emitted no text at all, so part of every VSI score was
    the token budget rather than the method; no other domain exceeded 0.5%.
    The allowance is identical for every arm and condition within a domain,
    so this removes an instrument limit rather than tuning a result.
    """
    per = ((CFG.get("models", {}).get("executor", {}) or {})
           .get("answer_max_tokens_by_domain") or {})
    return int(per.get(domain, ANSWER_TOKENS))

FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def _json_objects(text: str) -> list[str]:
    """Extract balanced top-level JSON objects from free-form text.

    A regex cannot do this: the action block contains nested region objects,
    and a character-class pattern like `[^{}]*` silently fails to match them,
    which made every unfenced zoom request invisible to the parser.
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text or ""):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
    return out


def _candidate_blocks(text: str) -> list[str]:
    """Fenced blocks first, then any balanced object found in the raw text."""
    blocks: list[str] = []
    for m in FENCE_RE.finditer(text or ""):
        blocks.extend(_json_objects(m.group(1)) or [m.group(1)])
    blocks.extend(_json_objects(text or ""))
    seen = set()
    uniq = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ViewRequest:
    """A parsed model request for more pixels."""
    action: str                 # "zoom" | "answer" | "note" | "view" | "compose"
    region: Region | None = None
    regions: list[Region] = field(default_factory=list)
    # Which attached image each region refers to, 0-based, parallel to
    # `regions`. Single-image domains always emit 0. A video domain must carry
    # this: a region alone is ambiguous across 8 frames, and defaulting to
    # frame 0 would crop the wrong moment while looking like a valid request.
    # Coordinates are interpreted IN THE FRAME OF THE REFERENCED VIEW: naming
    # a full frame reproduces the old global-coordinate behaviour exactly,
    # naming a crop means crop-local coords, which the runtime maps back to
    # the source with the recorded transform (never by the model's mental
    # arithmetic). This is what makes coarse-to-fine zoom compose: a crop of
    # a crop is just a region named against the crop's own image.
    targets: list[int] = field(default_factory=list)
    note: str = ""
    answer: str = ""
    # For action == "view" / "compose": 0-based indices into the FULL source
    # pool (not the initially-shown subset). This is the generic mid-episode
    # source-access primitive: which units to look at, and when, is the
    # model's strategy -- the runtime only materialises what is named.
    units: list[int] = field(default_factory=list)
    # For action == "compose": tile the named units into ONE labelled sheet
    # with this many columns (0 = runtime picks a near-square layout).
    cols: int = 0
    # For action == "answer": which attached image the answer's coordinates
    # refer to (0-based), or None for the default whole-source frame. The
    # runtime resolves (view, u, v) -> source coordinates in code, so the
    # scorer only ever sees source-frame answers.
    answer_view: int | None = None
    # VISUAL MEMORY LAYER: for action == "note", the points the model wants to
    # jot on the canvas. Each entry is {"x","y","text","target"}: a normalised
    # point, the model's own free-text label, and the 0-based image it refers
    # to. The model supplies only WHERE and WHAT the note says; HOW it is
    # rendered is the optimizer's canvas_expr, never the model's -- that is
    # what keeps the write interface a neutral channel rather than a strategy.
    canvas_notes: list = field(default_factory=list)


def _coerce_coord(v) -> float:
    """Coerce one coordinate to a normalized float.

    The executor sometimes emits a coordinate the way a human would sketch it:
    a fraction like "470/1000", a percentage, or a raw pixel offset. Silently
    discarding those turned an intended zoom into a scored wrong answer, so
    they are normalized here and anything still out of range is rejected.
    """
    if isinstance(v, str):
        s = v.strip().rstrip("%")
        if "/" in s:
            num, _, den = s.partition("/")
            return float(num) / float(den)
        v = float(s)
    return float(v)


def _region_from(obj) -> Region | None:
    """Build a Region from a dict or 4-sequence, or None if unusable.

    Accepts the dialects executors actually emit:
      * {"x0":..,"y0":..,"x1":..,"y1":..}  the requested schema
      * {"box_2d": [ymin, xmin, ymax, xmax]}  Gemini's native detection format,
        on a 0-1000 integer grid with the axis order TRANSPOSED
      * [x0, y0, x1, y1]  a bare 4-sequence
    Rejecting box_2d silently threw away accurate requests: measured against
    ground truth, two such boxes scored IoU 0.554 and 0.471 while the parser
    recorded "no region".
    """
    if isinstance(obj, dict) and "box_2d" in obj:
        b = obj["box_2d"]
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            return None
        try:
            ymin, xmin, ymax, xmax = [_coerce_coord(v) for v in b]
        except (ValueError, TypeError, ZeroDivisionError):
            return None
        vals = [xmin, ymin, xmax, ymax]
    elif isinstance(obj, dict):
        try:
            vals = [_coerce_coord(obj[k]) for k in ("x0", "y0", "x1", "y1")]
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return None
    elif isinstance(obj, (list, tuple)) and len(obj) == 4:
        try:
            vals = [_coerce_coord(v) for v in obj]
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    else:
        return None

    # Out-of-range values are only recoverable when the WHOLE box shares one
    # interpretation: 0-1000 is Gemini's grid, 0-100 a percentage. A box mixing
    # scales (0.7 alongside 7.0) is a typo - rescaling all of them would
    # corrupt the coordinates that were already correct, and guessing which one
    # is wrong would fabricate a region the model never asked for.
    if any(v > 1.0 for v in vals):
        if not all(v > 1.0 for v in vals):
            return None
        if max(vals) <= 100.0:
            vals = [v / 100.0 for v in vals]
        elif max(vals) <= 1000.0:
            vals = [v / 1000.0 for v in vals]
        else:
            return None

    x0, y0, x1, y1 = vals
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    vals = [min(max(v, 0.0), 1.0) for v in (x0, y0, x1, y1)]
    try:
        return Region(*vals)
    except ValueError:
        return None


def _explicit_view_index(obj) -> int | None:
    """The view an answer/region names, or None when nothing was named.

    Unlike `_target_index` (which defaults to 0 because a zoom must refer to
    SOME image), an answer that names no image keeps the legacy meaning:
    coordinates on the whole source. The distinction is load-bearing for
    backward compatibility with every recorded trajectory.
    """
    if not isinstance(obj, dict):
        return None
    for key in ("image", "frame", "image_index", "frame_index", "target",
                "view"):
        if key not in obj:
            continue
        try:
            v = int(float(obj[key]))
        except (TypeError, ValueError):
            continue
        return max(0, v - 1) if v >= 1 else 0
    return None


def _map_point_to_source(pt: tuple[float, float], rec) -> tuple[float, float]:
    """Map a normalised point in a view's own frame to source-frame [0,1].

    Exact affine composition through the recorded region -- the transform the
    model previously had to do in its head from a two-decimal text line. For
    a full/overview view (region = 0..1) this is the identity, so naming a
    full frame reproduces the old global-coordinate behaviour bit-for-bit.
    """
    base = rec.region or Region(0.0, 0.0, 1.0, 1.0)
    return (base.x0 + pt[0] * base.width, base.y0 + pt[1] * base.height)


def _map_region_to_source(reg: Region, rec) -> Region:
    """Map a region expressed in a view's own frame to source-frame coords."""
    base = rec.region or Region(0.0, 0.0, 1.0, 1.0)
    return Region(
        min(1.0, max(0.0, base.x0 + reg.x0 * base.width)),
        min(1.0, max(0.0, base.y0 + reg.y0 * base.height)),
        min(1.0, max(0.0, base.x0 + reg.x1 * base.width)),
        min(1.0, max(0.0, base.y0 + reg.y1 * base.height)))


def _map_point_to_view(pt: tuple[float, float], rec
                       ) -> tuple[float, float] | None:
    """Inverse: a source-frame point into a view's frame, None if outside."""
    base = rec.region or Region(0.0, 0.0, 1.0, 1.0)
    if base.width <= 0 or base.height <= 0:
        return None
    u = (pt[0] - base.x0) / base.width
    v = (pt[1] - base.y0) / base.height
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    return (u, v)


def _target_index(region_obj, action_obj) -> int:
    """Which attached image a zoom region refers to, as a 0-based index.

    The executor is told to label frames "Image 1..N", so a 1-based `image`
    or `frame` key is the expected dialect and is converted here. The value is
    read off the region first and the action block second, so a per-region
    label wins over a whole-request one. An absent or unusable value means
    image 0, which is correct for every single-image domain.
    """
    for src in (region_obj, action_obj):
        if not isinstance(src, dict):
            continue
        for key in ("image", "frame", "image_index", "frame_index", "target"):
            if key in src:
                try:
                    v = int(float(src[key]))
                except (TypeError, ValueError):
                    continue
                # 1-based in the prompt; a literal 0 is taken at face value
                # rather than silently shifted to -1.
                return max(0, v - 1) if v >= 1 else 0
    return 0


def _point_from(obj) -> tuple[float, float] | None:
    """Build a normalised (x, y) point from the dialects executors emit.

    Accepts [x, y], {"x":..,"y":..}, {"at":[x,y]}, and Gemini's 0-1000 grid.
    Same normalisation philosophy as `_region_from`: a coordinate pair that
    shares one out-of-range scale (0-100 or 0-1000) is rescaled; a mixed-scale
    pair is a typo and is rejected rather than silently fabricated.
    """
    vals = None
    if isinstance(obj, dict):
        if "at" in obj:
            return _point_from(obj["at"])
        if "point" in obj:
            return _point_from(obj["point"])
        try:
            vals = [_coerce_coord(obj["x"]), _coerce_coord(obj["y"])]
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return None
    elif isinstance(obj, (list, tuple)) and len(obj) == 2:
        try:
            vals = [_coerce_coord(obj[0]), _coerce_coord(obj[1])]
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    if vals is None:
        return None
    if any(v > 1.0 for v in vals):
        if not all(v > 1.0 for v in vals):
            return None
        if max(vals) <= 100.0:
            vals = [v / 100.0 for v in vals]
        elif max(vals) <= 1000.0:
            vals = [v / 1000.0 for v in vals]
        else:
            return None
    x, y = (min(max(v, 0.0), 1.0) for v in vals)
    return (x, y)


def _notes_from(obj) -> list[dict]:
    """Parse a note action's payload into [{"x","y","text","target"}].

    Accepts "notes": [...], or a single "note"/"at" with an optional "text".
    Each note's target image follows the same 1-based `image`/`frame` dialect
    as a zoom region, so which frame a note lands on is never ambiguous.
    """
    raw = obj.get("notes")
    if raw is None:
        # single-note shorthand: {"action":"note","at":[x,y],"text":"3"}
        if "at" in obj or "point" in obj or ("x" in obj and "y" in obj):
            raw = [obj]
        else:
            raw = []
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for n in raw:
        if not isinstance(n, dict):
            continue
        pt = _point_from(n)
        if pt is None:
            continue
        out.append({"x": pt[0], "y": pt[1],
                    "text": str(n.get("text", n.get("label", "")) or "")[:24],
                    "target": _target_index(n, obj),
                    "id": str(n.get("id", "") or "")[:24]})
    return out


def parse_model_turn(text: str, n_options: int) -> ViewRequest:
    """Parse the executor's reply into a zoom request, an answer, or an error.

    A reply that declares action "zoom" but carries no usable region is a
    MALFORMED ZOOM, not an answer. Treating it as an answer would score a
    protocol failure as a wrong answer and silently depress every arm.
    Format errors get one structured reminder from the caller, per the
    pre-registered protocol - never an open-ended re-ask loop.
    """
    blocks = _candidate_blocks(text)
    saw_zoom_intent = False
    saw_note_intent = False
    saw_view_intent = False

    for b in reversed(blocks):
        try:
            obj = json.loads(b)
        except json.JSONDecodeError:
            continue
        # A bare JSON ARRAY is a legitimate ANSWER, not an action object: on
        # grid-answer domains (ARC) the model emits the output grid as
        # [[...]]. Calling .get("action") on a list raised AttributeError and
        # turned a CORRECT grid answer into a needs_review failure -- 19% of
        # ARC items were being lost this way, which then broke the paired gate
        # ("gate sets differ by N items"). Treat a top-level list as the
        # answer text itself.
        if isinstance(obj, list):
            return ViewRequest(action="answer", answer=b,
                               note="raw grid/matrix answer")
        act = str(obj.get("action", "")).lower()
        if act == "zoom":
            saw_zoom_intent = True
            raw = obj.get("regions") or ([obj.get("region")]
                                         if obj.get("region") else [])
            regs, tgts = [], []
            for x in raw:
                r = _region_from(x)
                if r is None:
                    continue
                regs.append(r)
                tgts.append(_target_index(x, obj))
            if regs:
                return ViewRequest(action="zoom", region=regs[0], regions=regs,
                                   targets=tgts,
                                   note=str(obj.get("note", "")))
        if act in ("note", "mark", "annotate"):
            saw_note_intent = True
            notes = _notes_from(obj)
            if notes:
                return ViewRequest(action="note", canvas_notes=notes,
                                   note=str(obj.get("note", "")))
        if act in ("view", "show", "compose", "grid", "sheet"):
            saw_view_intent = True
            raw_u = obj.get("units") or obj.get("pages") or obj.get("frames")
            if raw_u is None and obj.get("unit") is not None:
                raw_u = [obj.get("unit")]
            units = []
            for u in (raw_u if isinstance(raw_u, list) else []):
                try:
                    units.append(int(u))
                except (TypeError, ValueError):
                    continue
            if units:
                try:
                    cols = int(obj.get("cols", 0) or 0)
                except (TypeError, ValueError):
                    cols = 0
                return ViewRequest(
                    action=("compose" if act in ("compose", "grid", "sheet")
                            else "view"),
                    units=units, cols=cols,
                    note=str(obj.get("note", "")))
        if act in ("answer", "point", "click") or (act == ""
                                                   and "answer" in obj):
            # An object carrying an "answer" field is an answer even when the
            # "action" key is missing: evolved thetas keep teaching a
            # structured {"note": ..., "answer": ...} reply, and the model
            # drops the action field. Measured across 8 runs (2026-09-18):
            # 217 such answers, ~2/3 scored near zero because the whole JSON
            # blob was scored as free text -- a systematic, viewcraft-only
            # suppression (only its theta rewrites induce the dialect).
            return ViewRequest(action="answer",
                               answer=str(obj.get("answer", "")
                                          or obj.get("point", "")
                                          or obj.get("click", "")),
                               note=str(obj.get("note", "")),
                               answer_view=_explicit_view_index(obj))

    # An unclosed / truncated JSON block that mentions zoom is also intent.
    if not saw_zoom_intent and re.search(r'"action"\s*:\s*"zoom"', text or ""):
        saw_zoom_intent = True
    if not saw_note_intent and re.search(
            r'"action"\s*:\s*"(?:note|mark|annotate)"', text or ""):
        saw_note_intent = True
    if re.search(r'"action"\s*:\s*"(?:view|show|compose|grid|sheet)"',
                 text or ""):
        saw_view_intent = True

    if saw_zoom_intent or saw_note_intent or saw_view_intent:
        return ViewRequest(action="malformed_zoom", note=(text or "")[:300])

    # No action block at all: treat the reply as a final answer attempt, since
    # a bare option letter is a legitimate response shape.
    return ViewRequest(action="answer", answer=text or "")


def describe_visible(views: list["RenderedView"],
                     contract: str = "", sources: tuple = (),
                     labels: tuple = ()) -> str:
    """Label each attached image with what it shows and where it came from.

    Without this the model receives an unlabelled image stack on later turns
    and cannot tell the full view from its own crop. Coordinates are relative
    to the image they name; the runtime resolves them to the source frame in
    code. For a video sample the label also carries the frame's timestamp, so
    "the moment the dog jumps" can be named by index rather than guessed at.
    """
    if not views:
        return ""
    multi = len({v.record.source_media_id for v in views}) > 1
    lines = ["\nImages attached, in order:"]
    for i, v in enumerate(views, 1):
        r = v.record
        stamp = (f" at t={r.seconds:.3f}s" if r.seconds is not None else "")
        if r.source_media_id in sources:
            unit = sources.index(r.source_media_id)
            label = labels[unit] if unit < len(labels) else Path(r.source_media_id).name
            stamp += f" [source unit {unit}; {label}]"
        if r.source_units:
            members = [f"tile {j+1}: source unit {u}; " +
                       (labels[u] if u < len(labels) else
                        Path(sources[u]).name if u < len(sources) else "unlabeled")
                       for j, u in enumerate(r.source_units)]
            lines.append(f"  Image {i}: COMPOSITE, row-major order; " + "; ".join(members))
            continue
        if r.kind in ("full", "overview"):
            tag = (" with YOUR accumulated notes marked on it"
                   if "phi.canvas" in (r.created_by or "") else "")
            lines.append(f"  Image {i}: a FULL frame{stamp}{tag} "
                         f"({r.out_w}x{r.out_h} shown)")
        else:
            reg = r.region
            lines.append(
                f"  Image {i}: a ZOOM you already requested{stamp}, covering "
                f"x {reg.x0:.4f}-{reg.x1:.4f}, y {reg.y0:.4f}-{reg.y1:.4f} "
                f"of that full frame ({r.out_w}x{r.out_h} shown)")
    # A COMPOSITE view stands for several sources at once. It is pixels
    # rearranged by a layout expression, so it has no single source
    # transform: a coordinate named against it cannot be resolved by code.
    # Telling the model to "zoom relative to the sheet" while the cropper
    # interpreted the numbers as full-frame coordinates was a measured
    # protocol break -- on point-answer domains every zoom after a composite
    # landed somewhere the model never intended, and ScreenSpot-Pro
    # candidates collapsed to ~0.1 against a 0.49 baseline.
    composite = any(v.record.created_by in ("psi.layout_expr",
                                            "psi.context_ops", "action.compose")
                    for v in views)
    if composite:
        lines.append(
            "Images produced by tiling several frames are COMPOSITE sheets. "
            "You cannot zoom or mark notes into a composite: name a "
            "single-source image (a FULL frame or a ZOOM) for any region or "
            "point you give.")
    # THE COORDINATE CONTRACT. Coordinates are relative to the image they
    # name, and the runtime maps them back to the source in code. Naming a
    # FULL frame is the old whole-image frame; naming a ZOOM lets the model
    # point inside the close-up it is actually looking at -- which is the
    # whole reason the mapping must not be the model's mental arithmetic
    # over a two-decimal text line.
    lines.append("Coordinates you give are always relative to the image you "
                 'name with "image": <number>. Naming a FULL frame means the '
                 "whole original image; naming a ZOOM means that close-up's "
                 "own frame. If you name nothing, coordinates refer to the "
                 "FULL frame.")
    if multi and not composite:
        lines.append('When you zoom, add "image": <the Image number '
                     'above> to each region so it is clear which frame '
                     'to magnify.')
    body = "\n".join(lines)

    # psi.view_contract lets the optimizer REWRITE this description, so a
    # proposal can change the composition and the words describing it in one
    # move. This is the co-evolution the study was missing: the text that
    # explains the pixels is part of the searched coordinate, not frozen
    # Python underneath it.
    #
    # A contract that omits {image_lines} does not get to delete the
    # manifest. Measured on screenspot_pro: a joint proposal paired a theta
    # saying "convert crop-local points through the stated bounds" with a
    # contract containing no placeholder -- the per-image bounds the theta
    # referred to were never sent, and the held-out score fell 0.4845 ->
    # 0.1056. The manifest is the only channel carrying which image is
    # which, so an omission APPENDS it instead of amputating it.
    if contract.strip():
        try:
            rendered = contract.format(n_images=len(views), image_lines=body)
        except (KeyError, IndexError, ValueError):
            # A malformed template must not silently send the model nothing.
            return body
        if "{image_lines" not in contract and "{n_images" not in contract:
            rendered = rendered + "\n" + body
        return rendered
    return body


# How the final answer must be shaped, per domain. A multiple-choice domain
# wants a letter; ChartQA and OCRBench want the value itself, and telling those
# models to "give the option letter" made them invent an A-D that the scorer
# then marked wrong - a harness artefact indistinguishable from a real error.
ANSWER_STYLES = {
    "free_text": (
        "Answer in the user's requested format. Include relevant units, source "
        "references, table rows and uncertainty. The answer field is a string; "
        "it may contain Markdown, CSV or serialized JSON as requested.",
        '{"action": "answer", "answer": "The requested result with supporting evidence."}'),
    "letter": ("To answer (give the option letter):",
               '{"action": "answer", "answer": "A", "note": "what you saw"}'),
    "short_text": (
        "To answer, give ONLY the value or short phrase asked for. No "
        "explanation, no sentence, and no unit or currency word unless the "
        "question explicitly asks for one - write 509, not \"509 billion "
        "U.S. dollars\"; write 14, not \"14 items\". If the chart labels a "
        "value in thousands or millions, report the number as the axis "
        "labels it.",
        '{"action": "answer", "answer": "509", "note": "what you read"}'),
    "yes_no": ("To answer (exactly 'yes' or 'no'):",
               '{"action": "answer", "answer": "yes", "note": "what you saw"}'),
    # Route-planning on a map (ReasonMap): the answer is a MULTI-LINE route in
    # the benchmark's own template, so "short_text" (which forbids everything
    # but a bare value) would truncate it to one phrase and score every valid
    # route as a format failure.
    "route": (
        "To answer, give the route in the exact template the question shows: "
        "one Route Name / Departure Stop / Arrival Stop block per leg, legs "
        "in travel order, separated by lines containing only --.",
        '{"action": "answer", "answer": "Route Name: Line 1\\nDeparture '
        'Stop: A\\nArrival Stop: B\\n--\\nRoute Name: Line 5\\nDeparture '
        'Stop: B\\nArrival Stop: C", "note": "how you traced it"}'),
    # GUI grounding wants a CLICK POINT, which no existing style can carry.
    #
    # The point is given WITHIN ANY attached image, named by its number, and
    # the runtime maps it back to the full screenshot with the recorded
    # transform -- the model never does the crop->global affine in its head.
    # Measured why this matters: on ScreenSpot-Pro 47% of baseline errors are
    # near-misses (median 104 px off against a 63 px median target), the
    # signature of a mental coordinate transform, not of a missing crop.
    "point": (
        "To answer, give the CLICK POINT as two numbers x, y in [0, 1] "
        "within ONE of the attached images, and say which image: x is the "
        "fraction from that image's left edge, y from its top edge. Within "
        "a FULL frame that is the whole screenshot; within a ZOOM it is "
        "that close-up's own frame -- pointing inside the close-up you "
        "actually inspected is usually the most precise choice. (0, 0) is "
        "the top-left corner and (1, 1) the bottom-right of the image you "
        "name. Give the centre of the target control.",
        '{"action": "answer", "answer": "0.412, 0.867", "image": 2, '
        '"note": "what you clicked"}'),
}


def build_task_prompt(item: PublicItem, coords: Coordinates,
                      steps_left: int,
                      visible: list["RenderedView"] | None = None,
                      answer_style: str = "letter") -> str:
    """The question, the options as presented, and the action protocol."""
    _meta = item.meta or {}
    pool_size = int(_meta.get("full_pool_size") or _meta.get("n_frames")
                    or _meta.get("n_pages") or len(item.media_ids) or 1)
    source_pool = tuple(_meta.get("source_pool", item.media_ids))
    attached_sources = {v.record.source_media_id for v in (visible or [])}
    unseen_units = [i for i, source in enumerate(source_pool)
                    if source not in attached_sources]
    view_example = json.dumps((unseen_units or list(range(pool_size)))[:3])
    compose_example = json.dumps(list(range(min(pool_size, 4))))
    marker_image = next((i for i, v in enumerate(visible or [], 1)
                         if v.record.source_media_id in source_pool), 1)
    lines = [f"Question: {item.question}"]
    if item.options:
        lines.append("\nOptions:")
        for i, o in enumerate(item.options):
            lines.append(f"  {chr(ord('A') + i)}. {o}")
    if visible:
        lines.append(describe_visible(
            visible, str(getattr(coords.psi, "view_contract", "") or ""),
            tuple(_meta.get("source_pool", item.media_ids)),
            tuple(_meta.get("source_labels", ()))))
    lines.append('Use the top-level "note" string on any action to save concise '
                 'observations, checked regions and unresolved items for later steps. '
                 'These notes are your own working record, not verified evidence. '
                 'Record corrections explicitly; the latest correction supersedes the old entry.')
    if _meta.get("working_notes"):
        lines.append("Previous working notes (chronological):\n" +
                     "\n".join(_meta["working_notes"]))
    ans_header, ans_example = ANSWER_STYLES[answer_style]
    # psi.answer_contract lets the optimizer rewrite the ANSWER protocol the
    # same way psi.view_contract rewrites the image descriptions: the text
    # that defines the interface is a searched coordinate, not frozen Python.
    custom_ans = str(getattr(coords.psi, "answer_contract", "") or "").strip()
    if custom_ans:
        # FORMAT-REGRESSION GUARD (2026-09-19, from the slidevqa_g55 autopsy:
        # a candidate answer_contract dropped the no-units discipline and 13/96
        # correct numeric answers came back as '7 movies' / '3.61 miles',
        # scoring 0.1-0.4 under ANLS while the bare number scored 1.0 -- the
        # STRATEGY was killed by a format regression it never intended). A
        # custom contract may REWORD the answer protocol, but on scored
        # short_text domains it may not drop the bare-value discipline: if the
        # rewrite omits it, the default no-units sentence is appended. This
        # guards the MEASUREMENT, not the strategy -- the strategy space
        # (theta, phi, psi) is untouched.
        try:
            ans_block = custom_ans.format(ans_header=ans_header,
                                          ans_example=ans_example)
        except (KeyError, IndexError, ValueError):
            ans_block = f"{ans_header}\n```json\n{ans_example}\n```"
        # FORMAT-REGRESSION GUARD v2 (2026-09-19, after the keyword-based v1
        # missed a contract that merely MENTIONED "unit"): the bare-value
        # format rule is part of the MEASUREMENT protocol on ANLS-scored
        # domains, so it is appended UNCONDITIONALLY -- the strategy may
        # reword everything else, but the scorer's parsing contract (bare
        # value, no trailing unit word, no quoting, no trailing qualifier
        # phrase) is not negotiable and not searchable. Measured tonight:
        # '33 USERS' vs gold '33' scored 0.25, '"Sweezey"' scored 0.78 vs
        # 'Sweezey' 1.0, '3.61 miles' scored 0.40 vs '3.61' 1.0.
        if answer_style == "short_text":
            ans_block += ("\n(Hard format rule: the answer field carries the "
                          "bare value only -- write 33, not \"33 USERS\"; "
                          "write 3.61, not \"3.61 miles\"; write Sweezey, "
                          "not quoted \"Sweezey\"; if the question asks for "
                          "a phrase, give the exact phrase and nothing after "
                          "it.)")
    else:
        ans_block = f"{ans_header}\n```json\n{ans_example}\n```"

    # A customizable answer description must not remove the executable answer
    # action. Otherwise the remaining examples teach only observation actions,
    # and an executor may repeatedly emit notes followed by an unparsed answer.
    ans_block += (
        '\nWhen ready, finish with exactly one JSON object whose action is '
        '"answer". Put your final answer in its "answer" field; do not append '
        'an answer after a zoom, note, view, or compose action.\n'
        f'Answer action format:\n```json\n{ans_example}\n```')

    # THE SECOND HARDCODED CONTRACT. `describe_visible` is not the only text
    # that describes the visual interface: this block also tells the model
    # what it is looking at and what a zoom does. Fixing only the first one
    # left this one still promising that "a zoom re-reads the original file",
    # which on a composite sheet re-reads ONE SOURCE FRAME and throws the
    # sheet away. Measured on VSI with the first contract already corrected:
    # 62 of 95 units truncated and accuracy fell 0.253 -> 0.053, because the
    # model kept zooming to recover a detail the sheet already contained.
    #
    # Co-evolution is only sound if EVERY text that describes the pixels
    # moves with them. Two blocks existed; both are now conditioned on what
    # was actually sent.
    composite = any(v.record.created_by in ("psi.layout_expr",
                                            "psi.context_ops",
                                            "action.compose")
                    for v in (visible or []))
    if composite:
        preamble = (
            "Some attached images are COMPOSITE SHEETS: several source "
            "frames tiled into one labelled image. A sheet has no single "
            "source frame, so you cannot zoom or mark notes into it -- name "
            "a single-source image (a FULL frame or one of your ZOOMS) for "
            "any region or point, and the system will resolve it exactly.")
    else:
        preamble = (
            "Images may be downscaled to fit the view budget. You may "
            "request a zoomed view of any region of any "
            "attached image -- including zooming INSIDE a zoom you already "
            "have, to go progressively finer. Coordinates are normalised "
            "[0,1] in the frame of the image you name: (0,0) is its "
            "top-left and (1,1) its bottom-right. A zoom re-reads the "
            "original file and can recover detail lost to downscaling; it "
            "cannot add detail beyond the source resolution.")
    custom_protocol = str(getattr(coords.psi, "protocol_contract", "") or "").strip()
    if custom_protocol:
        # EVOLVED PROTOCOL (psi.protocol_contract): the optimizer rewrote the
        # observation-protocol HOWTO. It substitutes the runtime-computed
        # pieces; the preamble (what the pixels are) and the scoring rule
        # still come from runtime because those are not searchable.
        ans_header, ans_example = ANSWER_STYLES[answer_style]
        # Literal token substitution, NOT str.format: protocol texts carry
        # JSON action examples whose braces crash .format(), silently
        # dropping every placeholder substitution (measured 2026-09-20:
        # both warmstart-authored protocols hit it).
        proto_block = (custom_protocol
                       .replace("{pool_size}", str(pool_size))
                       .replace("{max_crops}",
                                str(coords.phi.max_crops_per_step))
                       .replace("{ans_header}", ans_header)
                       .replace("{ans_example}", ans_example))
        lines.append(f"\n{proto_block}\n\n{ans_block}")
    else:
        lines.append(f"""
{preamble}

Observation steps remaining: {steps_left}

Reply with exactly one JSON block.

To zoom (you may pass up to {coords.phi.max_crops_per_step} regions):
```json
{{"action": "zoom", "regions": [{{"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4}}],
  "image": 1, "note": "why this region"}}
```
Coordinate rules: give each of x0, y0, x1, y1 exactly once, as a plain decimal
between 0 and 1 (write 0.47, not 470/1000, not 47%, not a pixel count), with
x0 < x1 and y0 < y1, measured in the frame of the image named by "image"
(default: the FULL frame). Close every brace.

To leave a NOTE on the image (a persistent marker, e.g. to mark items you have
already counted, cross off options you have ruled out, or label a region you
want to remember). Notes accumulate across steps and stay visible on later
views of the same source -- including inside your zooms. Give points in the
frame of the image you name:
```json
{{"action": "note", "notes": [{{"at": [0.52, 0.31], "id": "3", "text": "counted", "image": {marker_image}}}],
  "note": "what you are marking"}}
```

Use up to 32 markers per action. Reuse a marker ID on the same source to
correct its position or text; different IDs remain distinct even when close.

To see source units NOT currently attached (the pool has {pool_size} units,
0-based indices; the first images you were shown may be a subset of it):
```json
{{"action": "view", "units": {view_example}, "note": "why these units"}}
```
To tile several units into ONE labelled comparison sheet (useful when the
evidence is spread across more units than you can hold at once):
```json
{{"action": "compose", "units": {compose_example}, "cols": 2, "note": "compare"}}
```
Each view/compose call shows at most the per-request image budget of units;
across steps you can page through the whole pool.

{ans_block}""")
    return "\n".join(lines)


REMINDER = (
    "Your previous zoom request could not be used: every coordinate must be a "
    "plain decimal between 0 and 1 in the frame of the image you name "
    "(default: the FULL frame), with x0 < x1 and y0 < y1 and each key given "
    'exactly once. Either send one corrected zoom block, or answer now with '
    '{"action": "answer", "answer": "..."}.')

FINAL_CALL = (
    "This is your last step. Answer now using what you have already seen, "
    "with exactly one action block in the answer form shown above.")

# What the model is told when `phi.zoom_gate_expr` refused its request. Every
# clause here has to be TRUE of what happened, which is the whole lesson of the
# two hardcoded contracts in `build_task_prompt`. The gate is MODEL-AUTHORED, so
# the reason is not fixed: "already at full source resolution" is true when the
# refused region could not have magnified, and false when the gate refused for
# some other condition it wrote (a size bound, a crop cap). Two texts, chosen
# from the region's measured `zoom_gain` rather than assumed -- a refusal
# explained by the wrong reason is the same defect as a contract describing
# pixels that were never sent.
ZOOM_REFUSED_NO_DETAIL = (
    "Your zoom was not performed. The region you asked for is already shown "
    "at the full resolution of the source file, so magnifying it would return "
    "the same pixels you can already see -- there is no further detail there "
    "to recover. Either ask for a DIFFERENT region that might still be "
    "unresolved, or answer now from what you have.")

ZOOM_REFUSED_POLICY = (
    "Your zoom was not performed: the view configuration does not permit a "
    "zoom of that region. Either ask for a DIFFERENT region, or answer now "
    "from what you have already seen.")

ZOOM_REFUSED_COMPOSITE = (
    "Your zoom was not performed: the image you named is a COMPOSITE sheet "
    "tiled from several sources, so a region on it cannot be resolved to "
    "one source image. Name a single-source image instead -- a FULL frame "
    "or one of your earlier ZOOMS -- or answer now from what you have.")


_IMG_REF_RE = re.compile(r"\bimage\b\s*[=:]?\s*(\d+)", re.IGNORECASE)


def _resolve_answer_point(answer: str, answer_view: int | None,
                          visible: list["RenderedView"]) -> tuple[str, str]:
    """Resolve a view-referenced point answer to source-frame "x, y" text.

    Returns (answer_text, note). The scorer's contract is unchanged: answers
    are normalised coordinates on the full source. Only when the model named
    a ZOOM (a crop) AND the text parses as a point does the runtime apply
    the recorded transform; naming a full frame or naming nothing is the
    identity, so every non-point and every legacy answer is byte-identical.

    The view reference is honoured in BOTH dialects the executor emits: the
    JSON key ("image": 2) and free text ("... | image 2"), the latter being
    what an evolved theta produces when it asks for a labelled verification
    trace. Measured on sspro_v2 viewcraft r2-r4: three candidates taught a
    "label | x, y | image k" answer, the crop-frame point was read as
    full-frame, and every zoomed answer scored 0 -- a format intolerance
    masquerading as a bad candidate (gate 0.708 -> 0.01).
    """
    if answer_view is None:
        m = None
        for m in _IMG_REF_RE.finditer(answer or ""):
            pass                                # the LAST reference wins
        if m is not None:
            answer_view = int(m.group(1)) - 1
            # Remove the reference from the text before point parsing: its
            # own number ("image=2, x=0.49") pairs with the coordinate and
            # wins parse_point's last-pair rule (measured: x resolved as the
            # image number, 0.5 -> 0.6 after mapping).
            answer = (answer[:m.start()] + " " + answer[m.end():])
    if answer_view is None or not (0 <= answer_view < len(visible)):
        return answer, ""
    rec = visible[answer_view].record
    if rec.kind in ("full", "overview") or "," in (rec.source_media_id or ""):
        # Full frame: identity. Composite: no transform exists -- interpret
        # as the full frame rather than refuse the answer.
        return answer, ""
    from .evaluation.scorers import parse_point
    pt = parse_point(answer)
    if pt is None:
        return answer, ""
    x, y = pt
    if x > 1.0 or y > 1.0:
        # pixels within the named view: normalise by its emitted size first
        x, y = x / rec.out_w, y / rec.out_h
    gx, gy = _map_point_to_source((min(max(x, 0.0), 1.0),
                                   min(max(y, 0.0), 1.0)), rec)
    return (f"{gx:.4f}, {gy:.4f}",
            f"answer resolved from Image {answer_view + 1} "
            f"(a zoom covering x {rec.region.x0:.4f}-{rec.region.x1:.4f}, "
            f"y {rec.region.y0:.4f}-{rec.region.y1:.4f})")


def self_answer(item: PublicItem, coords: Coordinates, client: ModelClient,
                visible: list["RenderedView"], traj: Trajectory,
                answer_style: str = "letter") -> str:
    """Force one final answer from the views already collected.

    Used when the budget runs out or a zoom stays malformed. This is a single
    dependent request in the same trajectory, not a second sample of the same
    decision.
    """
    prompt = (build_task_prompt(item, coords, 0, visible, answer_style)
              + "\n\n" + FINAL_CALL)
    max_tok = answer_tokens_for(item.domain)
    messages = []
    if coords.theta.text.strip():
        messages.append(system_message(coords.theta.text))
    messages.append(user_message(prompt, [v.data_url for v in visible]))
    try:
        resp = client.call(messages, max_tokens=max_tok)
    except (ModelFormatError, ModelUnavailable):
        return ""
    traj.request_ids.append(resp.request_id or "")
    traj.input_tokens += resp.usage.input_tokens
    traj.output_tokens += resp.usage.output_tokens
    req = parse_model_turn(resp.content, len(item.options))
    if req.action == "answer":
        resolved, note = _resolve_answer_point(req.answer, req.answer_view,
                                               visible)
        if note:
            traj.notes += " | " + note
        return resolved
    return resp.content


def run_static_item(item: PublicItem, coords: Coordinates,
                    client: ModelClient, budget: dict,
                    protocol_hash: str, arm: str, condition: str,
                    split: str, forced_regions: list[Region] | None = None,
                    answer_style: str = "letter") -> Trajectory:
    """Execute one sample once.

    Handles both a single image and a fixed set of media (sampled video
    frames): every entry of `item.media_ids` gets its own whole-frame view at
    step 0, and a zoom names which of them to magnify. A video sample is
    therefore the same mechanism with a wider step-0 context, not a separate
    code path that could drift from the image one.

    `forced_regions` is used only by oracle / probe conditions, which are
    driven from a separate entry point; normal arms pass None and the model
    chooses its own regions.
    """
    t0 = time.monotonic()
    traj = Trajectory(
        item_id=item.item_id, domain=item.domain, split=split, arm=arm,
        condition=condition, config_hash=coords.config_hash,
        protocol_hash=protocol_hash,
    )

    max_tok = answer_tokens_for(item.domain)
    media_list = list(item.media_ids)
    if not media_list:
        traj.status = "error"
        traj.error = "sample carries no media"
        traj.exec_result = ExecResult(False, ("no media",))
        traj.wall_time_s = time.monotonic() - t0
        return traj
    max_steps = int(budget["observations"])
    max_imgs = min(int(coords.psi.max_images_per_request),
                   int(budget["max_images_per_request"]))

    # phi.select_expr decides WHICH source units are shown. For a video or a
    # multi-page document this is the dominant visual decision, and it was
    # outside the action space for the first half of this study: the
    # adapter's uniform sampling was hardcoded and no coordinate could reach
    # it. The search could rearrange eight frames but never ask whether
    # those were the right eight.
    #
    # The budget cap is the SAME one every other condition gets, so a
    # selection strategy cannot win by buying more context than its control.
    sel_expr = str(getattr(coords.phi, "select_expr", "") or "").strip()
    # The FULL source pool, kept regardless of what the initial selection
    # shows: the generic "view"/"compose" actions let the model materialise
    # any unit of it mid-episode. The step-0 trim decides what is shown
    # FIRST, not what is reachable.
    full_pool = list(media_list)
    item = replace(item, meta={**(item.meta or {}), "source_pool": full_pool,
                               "working_notes": []})
    full_times = list((item.meta or {}).get("frame_times_s") or [])
    if sel_expr and len(media_list) > 1:
        from .views.selection import eval_selection_expr
        from .views.operators import OperatorRejected as _OpRej
        meta = item.meta or {}
        sel_ctx = {k: meta[k] for k in
                   ("duration_s", "n_pages", "n_frames", "fps")
                   if k in meta}
        # CONTENT-AWARE selection. Compute the neutral per-unit features
        # (motion / edges / colorful / bright) so the expression can choose
        # WHICH units by what is in them, not only by index arithmetic. This
        # was the closed half of the action space: select_expr stayed empty in
        # every accepted config because it could not see a pixel.
        from .views.content import content_features
        sel_ctx["content_features"] = content_features(media_list)
        try:
            keep = eval_selection_expr(sel_expr, len(media_list), sel_ctx,
                                       max_imgs)
        except _OpRej as e:
            # A bad selection must fail the CANDIDATE, not silently fall back
            # to uniform: a silent fallback would score the default while the
            # ledger recorded the candidate's config hash, which is the
            # "config changed, pixels identical" defect in another costume.
            traj.status = "error"
            traj.error = f"select_expr rejected: {e}"
            traj.exec_result = ExecResult(False, (str(e)[:120],))
            traj.wall_time_s = time.monotonic() - t0
            return traj
        traj.selected_units = list(keep)
        all_times = list((item.meta or {}).get("frame_times_s") or [])
        timestamps_sel = [all_times[i] for i in keep if i < len(all_times)]
        media_list = [media_list[i] for i in keep]
        item = replace(item, media_ids=tuple(media_list),
                       meta={**(item.meta or {}),
                             "frame_times_s": timestamps_sel,
                             "full_pool_size": len(full_pool),
                             "selected_unit_indices": list(keep)})
    elif not sel_expr and len(media_list) > max_imgs:
        # THE DOCUMENTED DEFAULT, IMPLEMENTED AT LAST. chain.py tells the
        # optimizer that an empty select_expr means "the adapter's own
        # default, which is uniform sampling", but the runtime rendered the
        # WHOLE pool at step 0 and `select_visible` then kept the LAST
        # max_imgs heads -- "most recent" is meaningful for evicting crops,
        # not for a pool the model has never seen. On a ~100-page report the
        # model was sent the assurance statements and the GRI index and
        # answered "Data not available" for data on an unreachable page
        # (measured: docvqa2026 business_report_3_q2). Every document domain
        # (dude/mmlongdoc/mpdocvqa/docvqa2026) and every 32-frame video pool
        # ran this way. Trim to a uniform spread instead: unshown pages were
        # never addressable for a zoom, so nothing reachable is lost, and
        # the render cost drops from N pages to max_imgs.
        from .views.selection import _uniform
        keep = _uniform(len(media_list), max_imgs)
        all_times = list((item.meta or {}).get("frame_times_s") or [])
        timestamps_sel = [all_times[i] for i in keep if i < len(all_times)]
        media_list = [media_list[i] for i in keep]
        item = replace(item, media_ids=tuple(media_list),
                       meta={**(item.meta or {}),
                             "frame_times_s": timestamps_sel,
                             "full_pool_size": len(full_pool),
                             "default_uniform_units": list(keep)})

    history: list[RenderedView] = []
    messages = []
    if coords.theta.text.strip():
        messages.append(system_message(coords.theta.text))

    # Step 0: the ordinary whole-media view(s) every arm starts from. For a
    # video sample these are the uniformly sampled frames, which is the
    # ordinary native behaviour we are comparing against.
    timestamps = list((item.meta or {}).get("frame_times_s") or [])
    overviews: list[RenderedView] = []
    for i, m in enumerate(media_list):
        ov = render_full(m, m, coords.phi.overview_long_edge_px,
                         created_by="phi.overview",
                         seconds=(timestamps[i]
                                  if i < len(timestamps) else None))
        ov = _with_view_ops(ov, coords, item, len(media_list), i)
        overviews.append(ov)
        history.append(ov)
        traj.views.append(ov.record)

    # Clean pixels let later marker corrections replace, rather than bake over,
    # earlier marks on both full views and crops.
    raw_views = {v.record.view_id: v for v in overviews}
    canvas_notes: dict[str, list] = {}
    composed_sheets: dict = {}

    reminded = False
    # THE GATE'S REFUSAL STATE. `refusal_notice` is prepended to the next
    # prompt so the model learns why nothing came back; `gate_refusals` bounds
    # how many times that can happen before the answer is forced.
    refusal_notice = ""
    gate_refusals = 0
    gate_expr = str(getattr(coords.phi, "zoom_gate_expr", "") or "").strip()

    for step in range(max_steps):
        steps_left = max_steps - step
        visible = select_visible(history, coords, max_imgs)
        visible = _with_context_ops(visible, coords, item, len(media_list),
                                     max_imgs=max_imgs)
        urls = [v.data_url for v in visible]
        for v in visible:
            v.record.actually_sent = True
            if v.record not in traj.views:
                traj.views.append(v.record)

        prompt = build_task_prompt(item, coords, steps_left, visible,
                                   answer_style)
        if reminded:
            prompt = REMINDER + "\n\n" + prompt
        if refusal_notice:
            prompt = refusal_notice + "\n\n" + prompt
            refusal_notice = ""
        messages_now = messages + [user_message(prompt, urls)]

        try:
            resp = client.call(messages_now, max_tokens=max_tok)
        except ModelFormatError as e:
            traj.status = "error"
            traj.error = str(e)[:400]
            break

        traj.request_ids.append(resp.request_id or "")
        traj.input_tokens += resp.usage.input_tokens
        traj.output_tokens += resp.usage.output_tokens
        traj.observation_steps = step + 1
        for v in visible:
            v.record.request_id = resp.request_id

        # A reply that emitted no text at all has hit the completion cap
        # (a reasoning model can spend the entire allowance on hidden
        # reasoning). That is a LIMIT event, not a wrong answer: recording it
        # as status "ok" with an empty answer let the scorer mark it 0, which
        # measures the token budget rather than the method.
        if not (resp.content or "").strip():
            traj.status = "truncated"
            traj.error = (f"empty reply at step {step + 1}: "
                          f"{resp.usage.output_tokens} completion tokens "
                          f"produced no text (cap {max_tok})")

        req = parse_model_turn(resp.content, len(item.options))
        if req.note.strip() and req.action in ("zoom", "note", "view", "compose"):
            item.meta["working_notes"].append(f"Step {step + 1}: {req.note.strip()}")

        if req.action == "answer":
            resolved, res_note = _resolve_answer_point(req.answer,
                                                       req.answer_view,
                                                       visible)
            traj.answer_text = resolved
            traj.notes = req.note + ((" | " + res_note) if res_note else "")
            if (req.answer or "").strip():
                # a real answer arrived later in the trajectory; the earlier
                # empty reply is no longer the outcome
                if traj.status == "truncated" and not traj.error.startswith(
                        "budget of"):
                    traj.status = "ok"
                    traj.error = ""
            break

        if req.action in ("view", "compose"):
            # GENERIC SOURCE ACCESS (added 2026-09-19 after the hardcoded-
            # action-space defect was called out): the model may materialise
            # ANY unit of the full source pool mid-episode, not only the
            # units the initial selection showed. `view` attaches the units
            # as ordinary full views; `compose` tiles them into ONE labelled
            # sheet, which is how a >budget evidence set (a 16-image
            # comparison, a 20-page deck) becomes co-present at all. Which
            # units, in what order, at what moment -- that is the model's
            # strategy, exactly like WHERE to zoom. The per-call unit count
            # is capped by the same image budget step 0 obeys, so this
            # buys access, never extra context per request.
            units = [u for u in req.units
                     if isinstance(u, int) and 0 <= u < len(full_pool)]
            if not units:
                if reminded:
                    traj.status = "format_error"
                    traj.error = ("view/compose malformed after one reminder; "
                                  "forced final answer from views seen")
                    traj.answer_text = self_answer(item, coords, client,
                                                   visible, traj, answer_style)
                    break
                reminded = True
                continue
            units = units[: max_imgs]
            traj.n_actions += 1
            reminded = False
            if req.action == "view":
                # A cached page can have left the attachment window.
                attached = {v.record.source_media_id for v in visible
                            if v.record.kind in ("full", "overview")}
                fresh = [u for u in units if full_pool[u] not in attached]
                if not fresh:
                    refusal_notice = (
                        f"Source units {units} are already attached above "
                        f"-- you can read them now; do not request them "
                        f"again. Continue with zoom/note/compose or answer.")
                    continue
                for u in fresh:
                    m = full_pool[u]
                    ov = render_full(
                        m, m, coords.phi.overview_long_edge_px,
                        created_by="action.view",
                        seconds=(full_times[u]
                                 if u < len(full_times) else None))
                    ov = _with_view_ops(ov, coords, item, len(full_pool), u)
                    raw = ov
                    ov = _with_notes_on_view(ov, canvas_notes, coords, item, len(full_pool), u)
                    raw_views[ov.record.view_id] = raw
                    history = [v for v in history if not (
                        v.record.source_media_id == m and v.record.kind in ("full", "overview"))]
                    overviews = [v for v in overviews if v.record.source_media_id != m] + [ov]
                    history.append(ov)
                    traj.views.append(ov.record)
                if len(fresh) < len(units):
                    refusal_notice = (
                        f"Units {sorted(set(units) - set(fresh))} were "
                        f"already attached; only the new ones were added. "
                        f"Do not re-request units you already have.")
                continue
            # compose: tile the named units into ONE composite sheet. A
            # sheet has no single source transform, so (like psi.layout_expr
            # composites) it is display-only: the model must name a
            # single-source view for any zoom/note, and the prompt says so.
            key = (tuple(units), req.cols)
            if key in composed_sheets:
                sheet = composed_sheets[key]
                history = [v for v in history if v.record.view_id != sheet.record.view_id]
                history.append(sheet)
                continue
            import io as _io
            from PIL import Image as _Image
            from .views.layout import eval_layout_expr as _eval_layout
            from .views.render import _encode as _enc
            from .views.operators import safe_context as _safe_ctx
            tiles = []
            for u in units:
                ov = render_full(full_pool[u], full_pool[u],
                                 coords.phi.overview_long_edge_px,
                                 created_by="action.compose_src",
                                 seconds=(full_times[u]
                                          if u < len(full_times) else None))
                tiles.append(_Image.open(_io.BytesIO(ov.png_bytes))
                             .convert("RGB"))
            cols = req.cols if req.cols > 0 else max(
                1, round(len(tiles) ** 0.5))
            ctx = _safe_ctx(item, len(full_pool),
                            (tiles[0].size[0], tiles[0].size[1]),
                            long_edge=int(coords.phi.overview_long_edge_px))
            try:
                out = _eval_layout(f"grid(views, cols={cols}, label=index)",
                                   tiles, ctx)
            except Exception as e:                            # noqa: BLE001
                if reminded:
                    traj.status = "format_error"
                    traj.error = (f"compose render failed ({e}); forced "
                                  "final answer from views seen")
                    traj.answer_text = self_answer(item, coords, client,
                                                   visible, traj, answer_style)
                    break
                reminded = True
                continue
            im = out[0]
            from .schema import ViewKind as _VK, ViewRecord as _VR, \
                new_view_id as _nid, Region as _Region
            srcs = ",".join(full_pool[u] for u in units)[:400]
            sheet = RenderedView(_VR(
                view_id=_nid("cmp"), source_media_id=srcs,
                kind=_VK.OVERVIEW.value, long_edge_px=max(im.size),
                out_w=im.size[0], out_h=im.size[1],
                # C_exec requires every view to carry a source region; a
                # composite sheet spans the whole (synthetic) frame, so the
                # unit region is the identity. Found 2026-09-19 on the g55
                # line: every compose-using trajectory was failing C_exec
                # ("view cmp_* has no source region") and dragging whole
                # candidates into rejection.
                region=_Region(0.0, 0.0, 1.0, 1.0),
                src_w=im.size[0], src_h=im.size[1],
                created_by="action.compose", source_units=tuple(units)), _enc(im))
            history.append(sheet)
            traj.views.append(sheet.record)
            composed_sheets[key] = sheet
            continue

        if req.action == "note":
            # VISUAL MEMORY LAYER: the model jots notes on the canvas. Only
            # WHERE and the note's own text come from the model; the rendering
            # is the optimizer's canvas_expr. Notes accumulate per source media
            # for the whole trajectory -- that accumulation is the memory.
            # The point is given in the frame of the view the model names
            # (a zoom included) and mapped to source coords here, so a note
            # dropped inside a close-up lands at the exact source pixel.
            applied = 0
            changed_sources = set()
            for note in req.canvas_notes[:32]:
                tgt = note.get("target", 0)
                ref = visible[tgt] if 0 <= tgt < len(visible) else None
                if ref is None or "," in (ref.record.source_media_id or ""):
                    continue
                sid = ref.record.source_media_id
                sx, sy = _map_point_to_source((note["x"], note["y"]),
                                              ref.record)
                existing = canvas_notes.setdefault(sid, [])
                marker_id = note.get("id", "")
                match = next((e for e in existing if
                    (marker_id and e.get("id") == marker_id) or
                    (not marker_id and not e.get("id") and
                     abs(sx - e["x"]) < 1e-6 and abs(sy - e["y"]) < 1e-6 and
                     e.get("text") == note.get("text", ""))), None)
                kept = {"x": sx, "y": sy, "text": note.get("text", ""), "id": marker_id}
                if match == kept:
                    continue
                if match is None:
                    existing.append(kept)
                else:
                    match.update(kept)
                changed_sources.add(sid)
                applied += 1
            traj.canvas_notes = [{"source_media_id": sid, **n}
                                 for sid, ns in canvas_notes.items() for n in ns]
            traj.n_actions += applied
            # All notes were duplicates of marks already on the canvas: the
            # model is re-sweeping finished work. That is not a malformed
            # request; leave the canvas unchanged without spending a reminder.
            if not applied and req.canvas_notes:
                continue
            if not applied:
                # Declared a note but every point was unusable: same protocol
                # as a malformed zoom -- one reminder, then a forced answer.
                if reminded:
                    traj.status = "format_error"
                    traj.error = ("note malformed after one reminder; forced "
                                  "final answer from views already seen")
                    traj.answer_text = self_answer(item, coords, client,
                                                   visible, traj, answer_style)
                    break
                reminded = True
                continue
            reminded = False
            for i, old in enumerate(history):
                sid = old.record.source_media_id
                if sid not in changed_sources:
                    continue
                raw = raw_views.get(old.record.view_id, old)
                ann = _with_notes_on_view(raw, canvas_notes, coords, item,
                                          len(full_pool), full_pool.index(sid))
                raw_views[ann.record.view_id] = raw
                history[i] = ann
                traj.views.append(ann.record)
            overviews = [v for v in history if v.record.kind in ("overview", "full")
                         and v.record.source_media_id in full_pool]
            continue

        if req.action == "malformed_zoom":
            # The model tried to zoom but the coordinates were unusable. This
            # is a protocol failure, not a wrong answer: give the one
            # registered reminder, then require an answer from what it has
            # already seen rather than discarding the trajectory. Scoring an
            # unattempted answer as wrong would depress every arm equally and
            # measure the harness, not the method.
            traj.malformed_zooms += 1
            if reminded:
                traj.status = "format_error"
                traj.error = ("zoom malformed after one reminder; forced final "
                              "answer from views already seen")
                answer = self_answer(item, coords, client, visible, traj,
                                     answer_style)
                traj.answer_text = answer
                break
            reminded = True
            continue

        # zoom: execute phi in code. Each region is expressed in the frame of
        # the view the model named; map it to source coords HERE, so a zoom
        # inside a zoom is an exact affine composition, never mental
        # arithmetic by the model over a text description.
        regions = req.regions[: coords.phi.max_crops_per_step]
        targets = (req.targets + [0] * len(regions))[: len(regions)]
        resolved: list[tuple[Region, RenderedView, int]] = []
        composite_refs = 0
        for reg, tgt in zip(regions, targets):
            ref = visible[tgt] if 0 <= tgt < len(visible) else None
            if ref is None:
                continue
            if "," in (ref.record.source_media_id or ""):
                composite_refs += 1
                continue
            sid = ref.record.source_media_id
            mi = media_list.index(sid) if sid in media_list else 0
            resolved.append((_map_region_to_source(reg, ref.record), ref, mi))
        # A zoom names an index into the images ATTACHED this step, which is
        # what the prompt numbered; resolving it against media order instead
        # would silently crop a different frame once psi starts evicting.
        if coords.phi.union_crop_for_multi_target and len(resolved) > 1 \
                and len({r.record.source_media_id for _, r, _ in resolved}) == 1:
            merged = resolved[0][0]
            for r, _, _ in resolved[1:]:
                merged = merged.union(r)
            # a union that covers most of the frame gains nothing; keep both
            if merged.width * merged.height < 0.6:
                resolved = [(merged, resolved[0][1], resolved[0][2])]

        made_any = False
        refused_here = 0
        refused_no_detail = 0
        gate_broke = False
        for reg, ref, mi in resolved:
            src_id = ref.record.source_media_id
            reg = clamp_region(reg, coords.phi)
            # phi.zoom_gate_expr decides WHETHER this crop is rendered. A crop
            # is never upscaled, so on a source that already fits the overview
            # budget it returns the same pixels one round later: measured
            # median magnification 1.000x over 1808 crops, 73.5% at <=1.05x,
            # and the futile ones mark failure (Fisher p=6.3e-05). Empty
            # expression = render every region, i.e. the incumbent behaviour
            # reproduced exactly.
            if gate_expr:
                # Measure the region `render_crop` will ACTUALLY cut, margin
                # included. The margin makes the crop larger and therefore less
                # magnified, so gating on the unexpanded box would let through
                # crops the gate meant to refuse.
                boxed = (reg.expand(coords.phi.crop_margin)
                         if coords.phi.crop_margin else reg)
                facts = zoom_facts(
                    src_w=ref.record.src_w, src_h=ref.record.src_h,
                    overview_px=coords.phi.overview_long_edge_px,
                    crop_px=coords.phi.crop_long_edge_px,
                    region=boxed, step=step, steps_left=steps_left - 1,
                    crops_so_far=sum(1 for v in traj.views
                                     if v.kind == "crop"),
                    ctx=(item.meta or {}),
                    out_w=ref.record.out_w, out_h=ref.record.out_h)
                try:
                    allow = eval_zoom_gate(gate_expr, facts)
                except OperatorRejected as e:
                    # A gate that fails on a real item must not silently fall
                    # back to allowing the zoom: that would score the DEFAULT
                    # under the candidate's config hash, which is the exact
                    # class of silent confound this codebase treats as a bug.
                    # `chain._assert_expressions_parse` is what keeps a broken
                    # gate from ever reaching here, so reaching here is a
                    # harness defect and is recorded as one.
                    traj.status = "error"
                    traj.error = f"zoom_gate_expr failed: {str(e)[:200]}"
                    gate_broke = True
                    break
                if not allow:
                    refused_here += 1
                    traj.refused_zooms += 1
                    # Remember WHY, so the notice below can be true of this
                    # refusal rather than of the common case.
                    if facts.get("zoom_gain", 0.0) <= 1.05:
                        refused_no_detail += 1
                    continue
            view = render_crop(src_id, src_id, reg,
                               coords.phi.crop_long_edge_px,
                               parent_view_id=ref.record.view_id,
                               margin=coords.phi.crop_margin,
                               allow_upscale=coords.phi.allow_upscale,
                               created_by="phi.crop",
                               seconds=ref.record.seconds)
            view = _with_view_ops(view, coords, item, len(media_list), mi)
            # The canvas follows the model into its close-ups: notes are
            # stored in source coords, so any later crop of the same source
            # shows the marks that fall inside it, exactly placed by the
            # recorded transform. A note the model dropped to remember "this
            # button" is still there when it zooms in to click it.
            raw = view
            view = _with_notes_on_view(view, canvas_notes, coords, item,
                                       len(media_list), mi)
            raw_views[view.record.view_id] = raw
            history.append(view)
            traj.views.append(view.record)
            made_any = True
        if gate_broke:
            break

        if not made_any and composite_refs:
            # The model named a COMPOSITE sheet as the thing to zoom into.
            # A composite has no single source transform, so the request is
            # unresolvable -- not malformed, not gate-refused: tell it which
            # images ARE addressable and let it ask again, on the same
            # two-strikes budget as a gate refusal.
            gate_refusals += 1
            if gate_refusals >= 2:
                traj.answer_text = self_answer(item, coords, client,
                                               visible, traj, answer_style)
                break
            refusal_notice = ZOOM_REFUSED_COMPOSITE
            continue

        if not made_any and refused_here:
            # THE GATE REFUSED EVERY REGION. This is not a malformed request
            # and must not consume the malformed-zoom reminder -- same
            # precedent as an all-duplicate note above: the model asked for
            # something the harness will not render, so tell it the truth and
            # let it choose again. Two refusals in one trajectory is where it
            # stops: a model re-asking the same region would otherwise spend
            # the whole observation budget on refusals, which measures the
            # protocol rather than the method.
            gate_refusals += 1
            if gate_refusals >= 2:
                traj.answer_text = self_answer(item, coords, client,
                                               visible, traj, answer_style)
                break
            refusal_notice = (ZOOM_REFUSED_NO_DETAIL
                              if refused_no_detail == refused_here
                              else ZOOM_REFUSED_POLICY)
            continue

        if not made_any:
            if reminded:
                traj.answer_text = resp.content
                traj.status = "error"
                traj.error = "no parsable action after one reminder"
                break
            reminded = True
            continue
        reminded = False

        if coords.psi.annotate_overview_boxes:
            # Provenance boxes are drawn per SOURCE media, so a crop taken
            # from frame 5 is outlined on frame 5 rather than on frame 1.
            crops = [v for v in history if v.record.kind == "crop"]
            by_src: dict[str, list] = {}
            for v in crops:
                if v.record.region:
                    by_src.setdefault(v.record.source_media_id,
                                      []).append(v.record.region)
            if by_src:
                new_overviews = []
                for ov in overviews:
                    regs = by_src.get(ov.record.source_media_id)
                    if not regs:
                        new_overviews.append(ov)
                        continue
                    ann = render_overview_with_boxes(
                        ov.record.source_media_id, ov.record.source_media_id,
                        regs, coords.phi.overview_long_edge_px,
                        created_by="psi.annotated_overview")
                    ann.record.seconds = ov.record.seconds
                    new_overviews.append(ann)
                    traj.views.append(ann.record)
                overviews = new_overviews
                history = overviews + [v for v in history
                                       if v.record.kind != "overview"
                                       and v.record.kind != "full"]
    else:
        # Budget exhausted without a declared answer. Ask once for a final
        # answer from what was collected, so an out-of-budget trajectory is
        # not silently identical to a wrong answer.
        traj.status = "truncated"
        traj.error = f"budget of {max_steps} observation steps exhausted"
        traj.answer_text = self_answer(
            item, coords, client,
            select_visible(history, coords, max_imgs), traj, answer_style)

    traj.wall_time_s = time.monotonic() - t0
    traj.exec_result = check_exec_contract(traj, coords, budget)
    return traj


def _with_view_ops(view, coords, item, n_media: int, media_index: int):
    """Apply phi's model-authored per-view transform, if any.

    Two forms, both written by the optimizer: a `pixel_expr` array
    expression (the open-ended one) and the older `view_ops` step list.
    """
    from .views.operators import safe_context
    expr = str(getattr(coords.phi, "pixel_expr", "") or "").strip()
    if expr:
        import io as _io
        from PIL import Image as _Image
        from .views.exprs import render_pixel_expr
        from .views.render import _encode
        from .views.operators import OperatorRejected
        before = _Image.open(_io.BytesIO(view.png_bytes)).convert("RGB")
        ctx = safe_context(item, n_media,
                           (view.record.out_w, view.record.out_h),
                           media_index=media_index,
                           long_edge=int(coords.phi.overview_long_edge_px))
        after = render_pixel_expr(before, expr, ctx)
        if after.size == before.size and after.tobytes() == before.tobytes():
            # A PER-ITEM no-op is not a reason to fail the unit.
            #
            # The inertness guard exists to refuse a CANDIDATE that cannot
            # change any emitted pixel anywhere -- it is checked against
            # recorded traces before the candidate costs an evaluation
            # (`_assert_candidate_changes_emitted_pixels`). Raising here
            # instead kills the single unit, and a unit that never completes
            # silently shrinks the gate set: measured on mmlongdoc_ev1 round
            # 1, three blank pages were inert under a contrast expression, so
            # the candidate was scored on 96 items against a 99-item baseline
            # and the round was discarded as incomparable after 2752s.
            #
            # A blank page IS legitimately unchanged by a contrast curve.
            # That is a property of the page, not a defect in the candidate,
            # and the honest handling is to pass the original through and
            # keep the unit comparable.
            pass
        else:
            png = _encode(after)
            rec = view.record
            rec.out_w, rec.out_h = after.size
            rec.bytes_len = len(png)
            rec.created_by = f"{rec.created_by}|phi.expr"
            view = RenderedView(rec, png)

    ops = list(getattr(coords.phi, "view_ops", []) or [])
    if not ops:
        return view
    from .views.operators import safe_context
    ctx = safe_context(item, n_media,
                       (view.record.out_w, view.record.out_h),
                       media_index=media_index,
                       long_edge=int(coords.phi.overview_long_edge_px))
    return apply_view_ops(view, ops, ctx)


def _with_context_ops(views, coords, item, n_media: int,
                      max_imgs: int | None = None):
    """Apply psi's model-authored layout to the assembled visual context.

    `layout_expr` is the open-ended form: a composition over the sent views,
    e.g. `grid(views, cols=4, label=times)`. This is the half of the
    mechanism the old step menu starved.
    """
    from .views.operators import safe_context
    expr = str(getattr(coords.psi, "layout_expr", "") or "").strip()
    if expr and views:
        import io as _io
        from PIL import Image as _Image
        from .views.layout import eval_layout_expr
        from .views.render import _encode
        from .views.operators import OperatorRejected
        from .schema import Region as _Region, ViewKind as _VK, \
            ViewRecord as _VR, new_view_id as _nid

        imgs = [_Image.open(_io.BytesIO(v.png_bytes)).convert("RGB")
                for v in views]
        ctx = safe_context(item, n_media,
                           (views[0].record.out_w, views[0].record.out_h),
                           long_edge=int(coords.phi.overview_long_edge_px))
        ctx["crop_views"] = [im for im, v in zip(imgs, views)
                             if v.record.kind == "crop"]
        ctx["crop_regions"] = [
            [v.record.region.x0, v.record.region.y0,
             v.record.region.x1, v.record.region.y1]
            for v in views if v.record.kind == "crop" and v.record.region]
        out = eval_layout_expr(expr, imgs, ctx)
        inert_here = (len(out) == len(imgs) and all(
            a.size == b.size and a.tobytes() == b.tobytes()
            for a, b in zip(out, imgs)))
        if inert_here:
            # Same reasoning as the pixel path above: a per-item no-op is a
            # property of THIS item (e.g. a single-image sample, where a
            # grid of one is the identity), not evidence that the candidate
            # is inert. Failing the unit here would shrink the gate set and
            # make the round incomparable. Candidate-level inertness is
            # checked separately, against recorded traces, before the
            # candidate is ever evaluated. Fall through with the views
            # unchanged so context_ops below still runs.
            out = None
        if out is not None:
            srcs = ",".join(sorted({v.record.source_media_id
                                    for v in views}))[:400]
            made = []
            for im in out:
                png = _encode(im)
                made.append(RenderedView(_VR(
                    view_id=_nid("lay"), source_media_id=srcs,
                    kind=(_VK.OVERVIEW.value if len(out) < len(imgs)
                          else views[0].record.kind),
                    long_edge_px=max(im.size), out_w=im.size[0],
                    out_h=im.size[1], src_w=views[0].record.src_w,
                    src_h=views[0].record.src_h,
                    region=_Region(0.0, 0.0, 1.0, 1.0), scale=1.0,
                    bytes_len=len(png), created_by="psi.layout_expr"), png))
            # ADDITIVE, NOT REPLACING (bug fixed 2026-09-19, found on the
            # videoholmes_g55 line): the composed sheet used to REPLACE the
            # source views, so the model received ONLY the tiled sheet and
            # the original frames became un-zoomable (a sheet has no single
            # source transform). The strategy asked for BOTH -- sheet for
            # the arc, frames for detail -- and the runtime silently dropped
            # the second half. Keep the source views (they stay
            # zoom/note-addressable) and append the sheet(s).
            #
            # BUDGET INVARIANT (same day, second bug): every REQUEST is
            # capped at max_imgs images, and the sheet counts as one. When
            # 8 source views + 1 sheet would be 9, the OLDEST source views
            # are thinned from the front so the sheet the strategy asked
            # for always ships. This is the same retention rule
            # select_visible applies, enforced after composition so the cap
            # holds on the FINAL attachment list -- a general invariant of
            # "derived views consume budget", not a per-domain patch.
            views = views + made
            if max_imgs and len(views) > max_imgs:
                n_sheet = len(made)
                keep_src = max(0, max_imgs - n_sheet)
                if keep_src < len(views) - n_sheet:
                    views = views[-(keep_src + n_sheet):]

    ops = list(getattr(coords.psi, "context_ops", []) or [])
    if not ops or not views:
        return views
    from .views.operators import safe_context
    ctx = safe_context(item, n_media,
                       (views[0].record.out_w, views[0].record.out_h),
                       long_edge=int(coords.phi.overview_long_edge_px))
    return apply_context_ops(views, ops, ctx)


def _with_notes_on_view(view: RenderedView, canvas_notes: dict,
                        coords: Coordinates, item: PublicItem, n_media: int,
                        media_index: int) -> RenderedView:
    """Burn the source's accumulated notes onto ANY view of it (crop or full).

    Notes live in source coordinates; this maps the ones inside the view's
    region into the view's own frame and renders them with the same
    canvas_expr the overview uses. The overview path (`_with_canvas`) is the
    special case where the view's frame IS the source frame.
    """
    notes_all = canvas_notes.get(view.record.source_media_id)
    if not notes_all or "," in (view.record.source_media_id or ""):
        return view
    local = []
    for n in notes_all:
        uv = _map_point_to_view((n["x"], n["y"]), view.record)
        if uv is not None:
            local.append({"x": uv[0], "y": uv[1],
                          "text": n.get("text", ""), "id": n.get("id", "")})
    if not local:
        return view
    return _with_canvas(view, local, coords, item, n_media, media_index)


def _with_canvas(base_ov: RenderedView, notes: list, coords: Coordinates,
                 item: PublicItem, n_media: int,
                 media_index: int) -> RenderedView:
    """Burn the model's accumulated notes onto one source's overview.

    The overlay is derived from the RAW overview every time (never from an
    already-annotated copy), so re-noting replaces rather than double-bakes
    the previous marks. The mark style is the optimizer's `canvas_expr`; empty
    falls back to a simple disc + the note's text, so the affordance works
    before evolution improves it.
    """
    import io as _io
    from PIL import Image as _Image
    from .views.exprs import render_canvas
    from .views.render import _encode
    from .views.operators import safe_context
    from .schema import ViewRecord as _VR, new_view_id as _nid

    base = _Image.open(_io.BytesIO(base_ov.png_bytes)).convert("RGB")
    ctx = safe_context(item, n_media, base.size, media_index=media_index,
                       long_edge=int(coords.phi.overview_long_edge_px))
    expr = str(getattr(coords.phi, "canvas_expr", "") or "").strip()
    out = render_canvas(base, expr, notes, ctx)
    png = _encode(out)
    rec = base_ov.record
    new = _VR(view_id=_nid("canvas"),
              source_media_id=base_ov.record.source_media_id,
              kind=base_ov.record.kind,
              long_edge_px=base_ov.record.long_edge_px,
              out_w=out.size[0], out_h=out.size[1],
              src_w=base_ov.record.src_w, src_h=base_ov.record.src_h,
              region=base_ov.record.region,
              page_index=base_ov.record.page_index,
              seconds=base_ov.record.seconds, scale=base_ov.record.scale,
              bytes_len=len(png),
              created_by=f"{base_ov.record.created_by}|phi.canvas")
    return RenderedView(new, png)


def _resolve_source(visible: list[RenderedView],
                    overviews: list[RenderedView],
                    target: int) -> RenderedView | None:
    """The media a zoom refers to, given its index into the attached images.

    Preference order: the image actually at that position this step, then the
    whole-media view of the same source, then media 0. Cropping is always
    performed against the ORIGINAL source file even when the named image was
    itself a crop, so coordinates stay in one frame of reference - the prompt
    tells the model its coordinates refer to the full image.

    A COMPOSITE view (a layout_expr grid, a context_ops sheet) has no single
    source file: its `source_media_id` is a comma-joined list of every
    contributor, kept for provenance. Passing that string to render_crop as
    a path raised FileNotFoundError on every zoom taken after a composite --
    observed on nextgqa and vsi, where a sample IS eight frames and the
    layout composes them. A composite is therefore never returned as a crop
    source; the zoom falls back to a real single-file view.
    """
    if not visible and not overviews:
        return None

    def real(v) -> bool:
        return "," not in (v.record.source_media_id or "")

    pick = None
    if 0 <= target < len(visible) and real(visible[target]):
        pick = visible[target]
    if pick is None:
        pick = next((v for v in visible if real(v)), None)
    if pick is None:
        pick = next((v for v in overviews if real(v)), None)
    if pick is None:
        return None
    for ov in overviews:
        if ov.record.source_media_id == pick.record.source_media_id:
            return ov
    return pick


def select_visible(history: list[RenderedView], coords: Coordinates,
                   max_imgs: int) -> list[RenderedView]:
    """Apply psi: retention, eviction and ordering of the visual history.

    "Overview" here means every whole-media view, which for a video sample is
    the whole sampled frame set. Treating only one of them as the head would
    throw away the temporal context the domain is built on.
    """
    # A layout SHEET (psi.layout_expr / action.compose) is kind "overview"
    # for display purposes, but for retention it behaves like a crop: it is
    # a DERIVED view, and letting it into `heads` meant a recent sheet could
    # evict real frames from the capped per-request window -- silently
    # undoing the additive-not-replacing fix in _with_context_ops.
    # Symmetric classification: sheets are crops-class ONLY (never heads).
    def _is_sheet(v) -> bool:
        return v.record.created_by in ("psi.layout_expr", "action.compose")
    heads = [v for v in history
             if v.record.kind in ("overview", "full") and not _is_sheet(v)]
    crops = [v for v in history
             if v.record.kind not in ("overview", "full") or _is_sheet(v)]

    if max_imgs <= 0:
        return []
    keep_n = max(0, int(coords.psi.history_keep_last_n))
    kept_crops = crops[-min(keep_n, max_imgs):] if keep_n else []
    head = heads if coords.psi.keep_overview else []
    if head and kept_crops:
        last_derived = next(i for i, v in reversed(list(enumerate(history)))
                            if v is kept_crops[-1])
        recent_ids = {v.record.view_id for v in history[last_derived + 1:]}
        recent = [v for v in head if v.record.view_id in recent_ids][-max_imgs:]
        # Explicit page requests made after the last crop take priority.
        # Otherwise reserve the source overview for the latest crop.
        source = kept_crops[-1].record.source_media_id
        anchor = next((v for v in reversed(head) if v.record.source_media_id == source), head[-1])
        reserved = list(recent)
        if max_imgs - len(reserved) > 1 and all(v is not anchor for v in reserved):
            reserved.insert(0, anchor)
        room = max_imgs - len(reserved)
        kept_crops = kept_crops[-room:] if room else []
        spare = max_imgs - len(reserved) - len(kept_crops)
        reserved_ids = {v.record.view_id for v in reserved}
        others = [v for v in head if v.record.view_id not in reserved_ids]
        head = (others[-spare:] if spare else []) + reserved
    else:
        room = max_imgs - len(kept_crops)
        head = head[-room:] if room > 0 else []
    return kept_crops + head if coords.psi.order == "crops_first" else head + kept_crops


def check_exec_contract(traj: Trajectory, coords: Coordinates,
                        budget: dict) -> ExecResult:
    """C_exec: deterministic contract checks on what actually happened."""
    v: list[str] = []

    sent = traj.sent_views
    if not sent:
        v.append("no image was actually sent to the model")

    if traj.observation_steps > int(budget["observations"]):
        v.append(f"observation steps {traj.observation_steps} exceed budget "
                 f"{budget['observations']}")

    for rec in traj.views:
        if rec.actually_sent and not rec.request_id:
            v.append(f"view {rec.view_id} marked sent without a request id")
        if rec.region is None:
            v.append(f"view {rec.view_id} has no source region")
        if rec.out_w <= 0 or rec.out_h <= 0:
            v.append(f"view {rec.view_id} has empty dimensions")
        if rec.parent_view_id and rec.parent_view_id not in {
                x.view_id for x in traj.views}:
            v.append(f"view {rec.view_id} references unknown parent")

    per_req = max((sum(1 for x in traj.views if x.request_id == rid)
                   for rid in set(r for r in traj.request_ids if r)),
                  default=0)
    if per_req > int(coords.psi.max_images_per_request):
        v.append(f"{per_req} images in one request exceeds psi limit "
                 f"{coords.psi.max_images_per_request}")

    return ExecResult(passed=not v, violations=tuple(v))
