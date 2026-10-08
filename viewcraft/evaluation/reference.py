"""Reference-evidence conditions: oracle O and the P_select / P_read probes.

These live behind a SEPARATE entry point because they are the only code paths
allowed to read PrivateItem evidence locations. They are diagnostic
instruments, not methods: O may exceed the ordinary observation budget, and
when it does the trajectory is tagged so it can never be reported as a
cost-matched result.

O changes evidence coverage, presentation and retention only. It does not
change the executor model, raise the reasoning setting, or inject the answer,
a solution chain, or an action script.
"""
from __future__ import annotations

import time

from ..models import CFG, ModelClient, user_message, system_message
from ..schema import (ExecResult, PrivateItem, PublicItem, Region, Trajectory)
from ..views.config import Coordinates
from ..views.render import (render_crop, render_full,
                            render_overview_with_boxes)


# A reasoning executor spends its completion allowance on hidden reasoning
# tokens before emitting any text; 512-1024 left nothing for the answer and
# produced empty replies. Sized from the config so a non-reasoning model does
# not pay for headroom it never uses.
ANSWER_TOKENS = int(
    (CFG.get("models", {}).get("executor", {}) or {}).get(
        "answer_max_tokens", 1024))


class OracleAccessError(RuntimeError):
    """Raised if an oracle path is invoked from a normal evolution context."""


def _answer_prompt(item: PublicItem) -> str:
    lines = [f"Question: {item.question}"]
    if item.options:
        lines.append("\nOptions:")
        for i, o in enumerate(item.options):
            lines.append(f"  {chr(ord('A') + i)}. {o}")
    lines.append(
        "\nYou are given the full image plus close-up views of the relevant "
        "regions. Answer with the option letter.\n"
        'Reply with exactly one JSON block: {"action": "answer", '
        '"answer": "A", "note": "what you saw"}')
    return "\n".join(lines)


def build_reference_views(item: PublicItem, private: PrivateItem,
                         coords: Coordinates, annotate: bool = True):
    """Render the reference-evidence view package for a static image.

    Design choices that matter for validity:
      * a multi-target question gets ONE crop covering all targets, so the
        spatial relation between them survives; cropping each target to its
        own center would destroy exactly what such a question asks about
      * an annotated overview is included so the crop's location in the frame
        is recoverable
    """
    media = item.media_ids[0]
    regions = list(private.evidence_regions)
    if not regions:
        return [], "no_evidence_regions"

    views = []
    overview = (render_overview_with_boxes(
                    media, media, regions,
                    coords.phi.overview_long_edge_px,
                    created_by="oracle.overview_boxed")
                if annotate else
                render_full(media, media, coords.phi.overview_long_edge_px,
                            created_by="oracle.overview"))
    views.append(overview)

    if len(regions) > 1:
        merged = regions[0]
        for r in regions[1:]:
            merged = merged.union(r)
        views.append(render_crop(
            media, media, merged, coords.phi.crop_long_edge_px,
            parent_view_id=overview.record.view_id,
            margin=coords.phi.crop_margin, allow_upscale=False,
            created_by="oracle.union_crop"))
        # plus each target individually, for readability of small objects
        for r in regions:
            views.append(render_crop(
                media, media, r, coords.phi.crop_long_edge_px,
                parent_view_id=overview.record.view_id,
                margin=max(coords.phi.crop_margin, 0.04),
                allow_upscale=False, created_by="oracle.target_crop"))
    else:
        views.append(render_crop(
            media, media, regions[0], coords.phi.crop_long_edge_px,
            parent_view_id=overview.record.view_id,
            margin=max(coords.phi.crop_margin, 0.04), allow_upscale=False,
            created_by="oracle.target_crop"))

    return views, "ok"


def run_oracle_static(item: PublicItem, private: PrivateItem,
                      coords: Coordinates, client: ModelClient,
                      budget: dict, protocol_hash: str, split: str,
                      condition: str = "oracle") -> Trajectory:
    """One O run on a static-image sample. Single call, single answer."""
    t0 = time.monotonic()
    traj = Trajectory(
        item_id=item.item_id, domain=item.domain, split=split,
        arm="oracle", condition=condition,
        config_hash=coords.config_hash, protocol_hash=protocol_hash,
    )

    views, status = build_reference_views(item, private, coords)
    if status != "ok":
        traj.status = "error"
        traj.error = status
        traj.wall_time_s = time.monotonic() - t0
        traj.exec_result = ExecResult(False, (status,))
        return traj

    # O may exceed the ordinary per-request image cap; record that it did.
    n_imgs = len(views)
    over_budget = n_imgs > int(coords.psi.max_images_per_request)
    traj.notes = (f"oracle_images={n_imgs}"
                  + (" exceeds_normal_image_cap" if over_budget else ""))

    messages = []
    if coords.theta.text.strip():
        messages.append(system_message(coords.theta.text))
    messages.append(user_message(_answer_prompt(item),
                                 [v.data_url for v in views]))

    resp = client.call(messages, max_tokens=ANSWER_TOKENS)

    for v in views:
        v.record.actually_sent = True
        v.record.request_id = resp.request_id
        traj.views.append(v.record)

    traj.request_ids.append(resp.request_id or "")
    traj.input_tokens = resp.usage.input_tokens
    traj.output_tokens = resp.usage.output_tokens
    traj.observation_steps = 1

    from ..runtime import parse_model_turn
    req = parse_model_turn(resp.content, len(item.options))
    traj.answer_text = req.answer if req.action == "answer" else resp.content
    traj.notes += f" | {req.note}"[:400]
    traj.wall_time_s = time.monotonic() - t0
    traj.exec_result = ExecResult.ok()
    return traj


def run_probe_select(item: PublicItem, private: PrivateItem,
                     coords: Coordinates, client: ModelClient,
                     budget: dict, protocol_hash: str,
                     split: str) -> Trajectory:
    """P_select: reference POSITION, ordinary presentation.

    Only the choice of where to look is replaced. Crop size, image count cap
    and ordering stay at the normal protocol, so this isolates selection from
    readability. If the evidence needs more views than the normal cap allows,
    the sample is marked not-constructible rather than quietly given extra
    images.
    """
    t0 = time.monotonic()
    traj = Trajectory(
        item_id=item.item_id, domain=item.domain, split=split,
        arm="probe", condition="p_select", config_hash=coords.config_hash,
        protocol_hash=protocol_hash,
    )
    regions = list(private.evidence_regions)
    if not regions:
        traj.status = "error"
        traj.error = "no_evidence_regions"
        traj.exec_result = ExecResult(False, ("no_evidence_regions",))
        return traj

    media = item.media_ids[0]
    # normal protocol: one overview + up to the normal crop allowance
    allowance = max(1, int(coords.psi.max_images_per_request) - 1)
    if len(regions) > allowance:
        traj.status = "not_constructible"
        traj.error = (f"{len(regions)} evidence regions exceed normal "
                      f"allowance of {allowance} crops")
        traj.exec_result = ExecResult(False, (traj.error,))
        return traj

    views = [render_full(media, media, coords.phi.overview_long_edge_px,
                         created_by="probe.overview")]
    if len(regions) > 1 and coords.phi.union_crop_for_multi_target:
        merged = regions[0]
        for r in regions[1:]:
            merged = merged.union(r)
        use = [merged]
    else:
        use = regions
    for r in use[:allowance]:
        views.append(render_crop(media, media, r,
                                 coords.phi.crop_long_edge_px,
                                 parent_view_id=views[0].record.view_id,
                                 margin=coords.phi.crop_margin,
                                 allow_upscale=coords.phi.allow_upscale,
                                 created_by="probe.select_crop"))

    messages = []
    if coords.theta.text.strip():
        messages.append(system_message(coords.theta.text))
    messages.append(user_message(_answer_prompt(item),
                                 [v.data_url for v in views]))
    resp = client.call(messages, max_tokens=ANSWER_TOKENS)

    for v in views:
        v.record.actually_sent = True
        v.record.request_id = resp.request_id
        traj.views.append(v.record)
    traj.request_ids.append(resp.request_id or "")
    traj.input_tokens = resp.usage.input_tokens
    traj.output_tokens = resp.usage.output_tokens
    traj.observation_steps = 1

    from ..runtime import parse_model_turn
    req = parse_model_turn(resp.content, len(item.options))
    traj.answer_text = req.answer if req.action == "answer" else resp.content
    traj.wall_time_s = time.monotonic() - t0
    traj.exec_result = ExecResult.ok()
    return traj


def run_probe_read(item: PublicItem, private: PrivateItem,
                   coords: Coordinates, client: ModelClient,
                   budget: dict, protocol_hash: str,
                   split: str) -> Trajectory:
    """P_read: same selected positions, but rendered readably.

    Keeps the reasoning setting fixed and the positions identical to P_select;
    what changes is that each region is delivered at native detail together
    with its global context.
    """
    t0 = time.monotonic()
    traj = Trajectory(
        item_id=item.item_id, domain=item.domain, split=split,
        arm="probe", condition="p_read", config_hash=coords.config_hash,
        protocol_hash=protocol_hash,
    )
    views, status = build_reference_views(item, private, coords, annotate=True)
    if status != "ok":
        traj.status = "error"
        traj.error = status
        traj.exec_result = ExecResult(False, (status,))
        return traj

    messages = []
    if coords.theta.text.strip():
        messages.append(system_message(coords.theta.text))
    messages.append(user_message(_answer_prompt(item),
                                 [v.data_url for v in views]))
    resp = client.call(messages, max_tokens=ANSWER_TOKENS)

    for v in views:
        v.record.actually_sent = True
        v.record.request_id = resp.request_id
        traj.views.append(v.record)
    traj.request_ids.append(resp.request_id or "")
    traj.input_tokens = resp.usage.input_tokens
    traj.output_tokens = resp.usage.output_tokens
    traj.observation_steps = 1

    from ..runtime import parse_model_turn
    req = parse_model_turn(resp.content, len(item.options))
    traj.answer_text = req.answer if req.action == "answer" else resp.content
    traj.wall_time_s = time.monotonic() - t0
    traj.exec_result = ExecResult.ok()
    return traj


def audit_evidence_completeness(item: PublicItem, private: PrivateItem
                                ) -> dict:
    """Deterministic audit of whether reference evidence is usable.

    Checks the released annotation itself rather than asking a model to bless
    its own labels. A region that is degenerate, out of frame, or absent means
    the sample cannot support a strong mechanism claim.
    """
    issues = []
    regs = list(private.evidence_regions)
    if not regs:
        issues.append("no_regions")
    for i, r in enumerate(regs):
        if r.width <= 0 or r.height <= 0:
            issues.append(f"region{i}_degenerate")
        if r.width * r.height > 0.9:
            issues.append(f"region{i}_covers_whole_frame")
        if r.width * r.height < 1e-6:
            issues.append(f"region{i}_below_one_millionth")
    return {
        "item_id": item.item_id,
        "n_regions": len(regs),
        "source": private.evidence_source,
        "complete": not issues,
        "issues": issues,
    }
