"""Pixel-identity check: did a candidate config actually change what was sent?

The bug class this closes is the most dangerous one in this codebase, because
it produces a perfectly plausible number. A candidate changes the config hash,
the gate runs a full pass, and task plus C_ground come back identical to the
digit -- not because the change was neutral, but because it emitted the SAME
PIXELS. Measured on 2026-09-12: chartqa `view_only` reached round 7 with
49/49 gate items byte-identical across five different config hashes, and
ocrbench reached round 6 with 51/51 identical. Those rounds searched nothing.

The guard used before this was arithmetic and narrow: it rejected only
`overview_long_edge_px` raised above the source long edge. That misses every
other way a field can be inert on a given dataset --- `crop_long_edge_px`
above the source, `max_images_per_request` raised when the history never
holds that many views, `history_keep_last_n` raised past the number of views
that ever exist, `grid` set when the model never uses a tile, and so on. All
of those depend on the DATA and on the model's actual behaviour, so they
cannot be predicted from the config alone.

So the check here is empirical and general: render what the candidate WOULD
send for a handful of real items, replaying each item's recorded view
requests, and compare the resulting geometry against the current config. If
nothing differs on any probed item, the candidate cannot change any measured
quantity and the round is better spent elsewhere.

Replaying recorded requests rather than re-querying the model is deliberate:
it costs no inference, it is deterministic, and it isolates the effect of phi
and psi from the model's own variability. It is a NECESSARY-condition test --
if the pixels are identical the result is inert for certain; if they differ,
the candidate is merely worth evaluating, not guaranteed useful.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from ..schema import Region, Trajectory
from ..views.config import Coordinates
from PIL import Image

from ..views.render import (render_crop, render_full,
                            render_overview_with_boxes)
from ..views.operators import OperatorRejected
from ..views.zoomgate import eval_zoom_gate, zoom_facts


@dataclass
class PixelDiff:
    """Whether a candidate changes emitted pixels, and on how many items."""
    n_probed: int
    n_changed: int
    detail: str = ""

    @property
    def changes_pixels(self) -> bool:
        return self.n_changed > 0


def _geometry(coords: Coordinates, traj: Trajectory,
              media_ids: tuple[str, ...],
              ctx_extra: dict | None = None) -> tuple:
    """The geometry a config WOULD emit, replaying this trajectory's requests.

    Returns a hashable signature of every view that would actually be SENT:
    its kind, output size, and source region. Two configs with equal
    signatures on an item send that item byte-identical pixels, because
    rendering is deterministic given (source, region, long_edge, upscale).

    The signature contains ONLY rendered geometry. An earlier version also
    folded in psi's scalar config values, which made every psi change look
    like a pixel change by construction and defeated the whole check -- the
    guard would then have waved through a retention change that altered
    nothing. What psi does is decide WHICH views survive, so psi is measured
    by the selected set, never by its own settings.
    """
    from ..runtime import select_visible
    from ..views.config import clamp_region

    history = []
    # Step 0: the whole-media view(s), exactly as run_static_item builds them.
    for m in media_ids:
        history.append(render_full(m, m, coords.phi.overview_long_edge_px,
                                   created_by="probe.overview"))

    # Then the crops this trajectory actually requested, re-rendered under the
    # candidate's phi. The REGIONS come from the record, so the model's choice
    # is held fixed and only phi's treatment of it varies.
    #
    # phi.zoom_gate_expr can REFUSE a request, so the replay has to drop the
    # refused crop rather than render it. Omitting this would report every gate
    # as inert -- the same false positive `annotate_overview_boxes` produced
    # below, and with the same consequence: the guard built to stop inert
    # candidates would permanently block a live coordinate.
    gate_expr = str(getattr(coords.phi, "zoom_gate_expr", "") or "").strip()
    crop_recs = [v for v in traj.views if v.kind == "crop" and v.region]
    for i, rec in enumerate(crop_recs):
        reg = clamp_region(rec.region, coords.phi)
        src = rec.source_media_id
        if gate_expr:
            # The GEOMETRIC facts are exact in a replay: the source's own size
            # and the requested region are both recorded. The trajectory-
            # position facts are proxies -- a record does not say which
            # observation step it came from -- so a gate written purely over
            # `step`/`steps_left` is judged approximately here. That is the
            # conservative direction for those two: the proxy still varies
            # across crops, so a position gate that refuses anything on a
            # multi-crop item still shows up as a difference.
            # `rec.region` is what render_crop STORED, which is already the
            # margin-expanded box -- so unlike the live path in runtime.py this
            # must not expand again, or the margin is counted twice and the
            # replayed gain is lower than the one the gate really saw.
            facts = zoom_facts(
                src_w=rec.src_w, src_h=rec.src_h,
                overview_px=coords.phi.overview_long_edge_px,
                crop_px=coords.phi.crop_long_edge_px,
                region=reg, step=i,
                steps_left=max(0, int(traj.observation_steps) - i - 1),
                crops_so_far=sum(1 for h in history
                                 if h.record.kind == "crop"),
                ctx=(ctx_extra or {}),
                out_w=rec.out_w, out_h=rec.out_h)
            try:
                if not eval_zoom_gate(gate_expr, facts):
                    continue
            except OperatorRejected:
                # A gate that cannot be evaluated is not "inert"; it is broken,
                # and `chain._assert_expressions_parse` is what rejects it. The
                # probe must not swallow it into a false identical/differs
                # verdict, so the item is dropped from the probe instead.
                raise ValueError("zoom_gate_expr failed during pixel probe")
        try:
            history.append(render_crop(
                src, src, reg, coords.phi.crop_long_edge_px,
                margin=coords.phi.crop_margin,
                allow_upscale=coords.phi.allow_upscale,
                created_by="probe.crop"))
        except (OSError, ValueError):
            continue

    # psi may replace the plain overview with one that has the crop
    # provenance drawn on it. That IS a pixel change, so the replay has to
    # model it -- omitting it made the guard report
    # `annotate_overview_boxes: False -> True` as inert, which is a false
    # positive: the guard would then block a live field permanently.
    crops_made = [v for v in history if v.record.kind == "crop"]
    if coords.psi.annotate_overview_boxes and crops_made:
        by_src: dict[str, list] = {}
        for v in crops_made:
            if v.record.region:
                by_src.setdefault(v.record.source_media_id,
                                  []).append(v.record.region)
        annotated = []
        for v in history:
            if v.record.kind not in ("full", "overview"):
                continue
            regs = by_src.get(v.record.source_media_id)
            if not regs:
                continue
            try:
                annotated.append((v, render_overview_with_boxes(
                    v.record.source_media_id, v.record.source_media_id,
                    regs, coords.phi.overview_long_edge_px,
                    created_by="probe.annotated")))
            except (OSError, ValueError):
                continue
        for old, new in annotated:
            history = [new if h is old else h for h in history]

    # psi then decides which of that history is actually sent.
    sent = select_visible(history, coords,
                          int(coords.psi.max_images_per_request))

    # Model-authored operators change PIXEL VALUES, often at identical
    # geometry: a contrast curve or a local threshold leaves every dimension
    # untouched. A geometry-only signature therefore called every authored
    # operator "byte-identical" and the guard refused all of them -- the
    # guard built to stop inert candidates was deleting the only genuinely
    # open part of the search space. So the operators are actually RUN here
    # and the signature carries a content digest.
    from ..views.steps import run_program

    view_ops = list(getattr(coords.phi, "view_ops", []) or [])
    ctx_ops = list(getattr(coords.psi, "context_ops", []) or [])
    pixel_expr = str(getattr(coords.phi, "pixel_expr", "") or "").strip()
    layout_expr = str(getattr(coords.psi, "layout_expr", "") or "").strip()
    canvas_expr = str(getattr(coords.phi, "canvas_expr", "") or "").strip()
    ectx = dict(ctx_extra or {})
    ectx.setdefault("long_edge", int(coords.phi.overview_long_edge_px))

    # The canvas is CONDITIONALLY active: it changes pixels only where the
    # model has jotted a note. This trajectory's recorded notes are grouped by
    # source. If none were recorded but the candidate sets a canvas_expr, a
    # synthetic probe note is injected, because "identical pixels only because
    # the model had not noted yet" is NOT inertness -- it is a conditional
    # mechanism the historical trace happens not to exercise. Refusing it
    # there would be the "item's nature fails the candidate" defect again, and
    # would permanently block the visual-memory coordinate from the search.
    notes_by_src: dict[str, list] = {}
    for n in (getattr(traj, "canvas_notes", []) or []):
        if isinstance(n, dict) and "source_media_id" in n:
            notes_by_src.setdefault(n["source_media_id"], []).append(n)
    _probe_note = [{"x": 0.5, "y": 0.5, "text": "1"}]

    imgs = []
    for v in sent:
        im = Image.open(io.BytesIO(v.png_bytes)).convert("RGB")
        c = dict(ectx, width=im.width, height=im.height,
                 n_media=len(media_ids), media_index=0)
        if pixel_expr:
            from ..views.exprs import render_pixel_expr
            im = render_pixel_expr(im, pixel_expr, c)
        if view_ops:
            im = run_program(view_ops, [im], c)[0]
        if canvas_expr and v.record.kind in ("overview", "full"):
            from ..views.exprs import render_canvas
            notes = notes_by_src.get(v.record.source_media_id) or _probe_note
            im = render_canvas(im, canvas_expr, notes, c)
        imgs.append(im)

    if layout_expr and imgs:
        from ..views.layout import eval_layout_expr
        c = dict(ectx, width=imgs[0].width, height=imgs[0].height,
                 n_media=len(media_ids), media_index=0)
        c["crop_views"] = [im for im, v in zip(imgs, sent)
                           if v.record.kind == "crop"]
        c["crop_regions"] = [
            [v.record.region.x0, v.record.region.y0,
             v.record.region.x1, v.record.region.y1]
            for v in sent if v.record.kind == "crop" and v.record.region]
        imgs = eval_layout_expr(layout_expr, imgs, c)

    if ctx_ops and imgs:
        c = dict(ectx, width=imgs[0].width, height=imgs[0].height,
                 n_media=len(media_ids), media_index=0)
        imgs = run_program(ctx_ops, imgs, c)

    sig = []
    for im in imgs:
        # a content digest, because equal dimensions no longer imply equal
        # pixels once operators are in play
        digest = hashlib.sha256(im.tobytes()).hexdigest()[:20]
        sig.append((im.width, im.height, digest))
    return tuple(sig)


def compare_emitted_pixels(current: Coordinates, candidate: Coordinates,
                           pairs, cap: int = 8) -> PixelDiff:
    """Does `candidate` emit different pixels from `current` on real items?

    `pairs` is a list of (Sample, Trajectory) already executed under
    `current`, so each item contributes the view requests the model really
    made.

    Items whose trajectory contains CROPS are probed first. Without that
    ordering the probe filled up with items the model never zoomed on, where
    a crop-sizing field legitimately does nothing, and `crop_long_edge_px`
    looked inert on a domain where it is not. A field that only acts once a
    crop exists can only be judged on an item that has one.
    """
    ordered = sorted(
        list(pairs),
        key=lambda st: (0 if any(v.kind == "crop" and v.region
                                 for v in st[1].views) else 1,
                        st[0].public.item_id))

    probed = 0
    changed = 0
    examples: list[str] = []

    for s, t in ordered[:cap]:
        media = tuple(s.public.media_ids)
        if not media:
            continue
        try:
            extra = {"n_options": len(s.public.options or ()),
                     "duration_s": float((s.public.meta or {})
                                         .get("duration_s") or 0.0),
                     "frame_times_s": list((s.public.meta or {})
                                           .get("frame_times_s") or []),
                     "n_frames": len((s.public.meta or {})
                                     .get("frame_times_s") or []),
                     "media_index": 0,
                     "question": str(s.public.question or "")[:500],
                     "question_type": ""}
            a = _geometry(current, t, media, extra)
            b = _geometry(candidate, t, media, extra)
        except (OSError, ValueError):
            continue
        probed += 1
        if a != b:
            changed += 1
            if len(examples) < 2:
                examples.append(s.public.item_id)

    detail = ("identical on every probed item"
              if probed and not changed
              else f"differs on {changed}/{probed} probed items "
                   f"(e.g. {', '.join(examples)})" if changed
              else "could not probe any item")
    return PixelDiff(n_probed=probed, n_changed=changed, detail=detail)
