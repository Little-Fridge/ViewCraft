"""phi / psi configuration objects and their execution.

phi (VIEW.yaml)  decides WHICH pixels get requested and at what size.
psi (LAYOUT.yaml) decides HOW the emitted views are arranged, retained and
                  evicted across the visual history.

Both are executed by code here. A field that the runtime does not implement
raises ViewConfigError rather than being ignored — a config that changes no
pixels while claiming to is the specific bug class this guards against.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from ..schema import Region, sha
from .render import ViewConfigError

# `view_ops` is the model-authored part of phi: a list of operator programs
# applied to every view this config emits. It exists because the original
# 9 resolution/crop scalars made the search space inert on any benchmark whose
# images already fit the view budget, and because the set of useful visual
# operators is unbounded -- it must be written by the optimizer from failure
# traces, not enumerated here.
PHI_FIELDS = {
    "select_expr",               # str: WHICH source units the model sees
    "view_ops",
    "pixel_expr",
    # `canvas_expr` is the model-authored part of the VISUAL MEMORY LAYER.
    # The executor can drop a `note` (a point + free text) on the image it is
    # looking at; this expression decides HOW the accumulated notes are burned
    # back into the view (a spot, a numbered marker, a dimmed region, a path).
    # It is deliberately a general array expression over the base planes plus
    # note-derived arrays (dist / nearest_idx / note_count), so the *strategy*
    # (mark-as-you-count, elimination, tracing) is discovered by evolution,
    # not enumerated here as named gestures.
    "canvas_expr",
    "overview_long_edge_px",     # int: whole-image view size
    "crop_long_edge_px",         # int: detail view size
    "crop_margin",               # float: context padding, fraction of frame
    "max_crops_per_step",        # int
    "allow_upscale",             # bool: interpolate a tiny crop up
    "grid",                      # None | {rows, cols, overlap}
    "union_crop_for_multi_target",  # bool: keep 2 targets in one crop
    "min_crop_frac",             # float: floor on crop size to keep context
    # `zoom_gate_expr` is the model-authored part of the ZOOM DECISION: a
    # condition over neutral facts about a requested region (how much
    # magnification it would actually deliver, how large it is, where in the
    # trajectory it came) deciding whether that crop is rendered at all.
    #
    # It exists because a crop is NEVER upscaled, so on a source that already
    # fits the overview budget a zoom returns the same pixels one round later:
    # measured median magnification 1.000x over 1808 crops, 73.5% at <=1.05x,
    # and a futile zoom marks failure (Fisher p=6.3e-05). Nothing in the action
    # space could express "do not look again when looking again cannot help".
    # Empty means allow every zoom, which reproduces the incumbent behaviour
    # exactly rather than approximately.
    "zoom_gate_expr",
}

# `context_ops` is the model-authored part of psi: one program applied to the
# assembled set of views, so the optimizer can reorganise the visual CONTEXT
# (tile 8 frames into one labelled sheet, difference a paired figure) rather
# than only choosing how many images to retain.
PSI_FIELDS = {
    "context_ops",
    "layout_expr",
    "view_contract",             # str: how the attached images are DESCRIBED
    "answer_contract",           # str: how the ANSWER must be expressed
    "protocol_contract",         # str: the OBSERVATION PROTOCOL text itself
    "keep_overview",             # bool: overview stays in context
    "annotate_overview_boxes",   # bool: outline crop provenance on overview
    "history_keep_last_n",       # int: how many prior views survive
    "order",                     # "overview_first" | "crops_first"
    "evict_policy",              # "fifo" | "keep_overview_drop_crops"
    "max_images_per_request",    # int
}

# `route_by_question_kind` was offered to the optimizer for eight rounds and
# read by nothing: the runtime has no branch on question kind, so flipping it
# changed the config hash and not one emitted pixel. Rounds spent on it were
# structurally wasted, and the resulting "task and C_ground both unchanged to
# the digit" rows are the exact signature this codebase treats as a bug.
# It is retired rather than implemented, because a per-kind branch is a
# different mechanism from the one under test. Snapshots written before the
# retirement still carry the key, so it is accepted and dropped on read
# instead of raising - an old snapshot must stay loadable for audit.
RETIRED_PSI_FIELDS = {"route_by_question_kind"}

# `second_pass_zoom` ("allow zooming inside an existing crop") was in
# PHI_FIELDS, in PhiConfig, and written out in PHI_PROMPT's field-meanings
# block for the whole study -- and read by no code. Three references existed
# and all three were declarations. So a round that proposed it changed the
# config hash and not one emitted pixel, which is the same structurally wasted
# round `route_by_question_kind` produced above.
#
# It is retired rather than implemented because the mechanism it names is not
# the one the data asks for. Recursive cropping deepens a zoom that already
# magnified; what 1808 recorded crops show is the opposite problem -- 73.5% of
# FIRST zooms could not magnify at all, because a crop is never upscaled. The
# decision worth searching is therefore WHETHER to render a requested crop,
# which is `zoom_gate_expr` above, and it is a different mechanism from a
# second pass.
#
# Snapshots written before the retirement carry the key (every VIEW.yaml in
# `runs/` has `second_pass_zoom: false`), so it is accepted and dropped on read
# instead of raising - an old snapshot must stay loadable for audit.
RETIRED_PHI_FIELDS = {"second_pass_zoom"}


@dataclass
class PhiConfig:
    view_ops: list = field(default_factory=list)
    # A model-WRITTEN array expression over the image, e.g.
    #   "where(local_mean(g,31) - g > 8, 0, 255)"
    # This replaces choosing from a menu of named effects: the 18-step menu
    # was 15 pixel ops to 3 layout ops, which is why the optimizer proposed
    # contrast and sharpening nearly every round.
    pixel_expr: str = ""
    # WHICH SOURCE UNITS ARE SHOWN AT ALL.
    #
    # For a video or a multi-page document this is the dominant decision,
    # and for the first half of this study it was not in the action space:
    # `uniform_frame_indices` was hardcoded in the adapter and no phi/psi
    # field could reach it. So the search could reorganise eight frames but
    # never ask whether those were the right eight.
    #
    # Empty means the adapter's own default, which is uniform sampling.
    # `uniform(k)` reproduces that default EXACTLY, so it is a legitimate
    # control rather than a near-miss that would confound comparisons.
    select_expr: str = ""
    # Model-WRITTEN overlay for the visual memory layer. Empty means the
    # DEFAULT note marker (a small disc + the note's own text label). When set,
    # it is an array expression over the base planes plus the note-derived
    # arrays `dist` (per-pixel distance to the nearest note) and `nearest_idx`
    # (1-based ordinal of that note), so the rendering of the model's notes is
    # itself an evolved coordinate rather than a fixed icon.
    canvas_expr: str = ""
    overview_long_edge_px: int = 1024
    crop_long_edge_px: int = 1024
    crop_margin: float = 0.02
    max_crops_per_step: int = 2
    allow_upscale: bool = False
    grid: dict | None = None
    union_crop_for_multi_target: bool = True
    min_crop_frac: float = 0.0
    # Model-WRITTEN condition deciding whether a requested crop is rendered at
    # all. Empty means allow every zoom, which is the incumbent behaviour
    # reproduced exactly. See `views/zoomgate.py` for the vocabulary and for
    # why this is an expression rather than a boolean.
    zoom_gate_expr: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "PhiConfig":
        d = {k: v for k, v in d.items() if k not in RETIRED_PHI_FIELDS}
        unknown = set(d) - PHI_FIELDS
        if unknown:
            raise ViewConfigError(f"unsupported phi field(s): {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def hash(self) -> str:
        return sha(self.to_dict())


@dataclass
class PsiConfig:
    context_ops: list = field(default_factory=list)
    # A model-WRITTEN composition over the sent views, e.g.
    #   "grid(views, cols=4, label=times)"
    #   "stack_v(pick(views,0), grid(crops, cols=2))"
    # This is the half of the mechanism the step menu starved.
    layout_expr: str = ""
    # HOW THE IMAGES ARE DESCRIBED TO THE MODEL.
    #
    # This was hardcoded Python (`describe_visible`) for the whole study, and
    # that is a co-evolution defect with a measured cost. The frozen text
    # says "Image i: a FULL frame" per source and "Any new coordinates you
    # give must refer to the FULL frame". Both become FALSE the moment a
    # layout expression collapses 8 frames into 1 composite: there is no
    # per-source image any more and no full frame to address.
    #
    # Measured consequence on VSI / NExT-GQA with
    # `grid(views, cols=4, label=index)`: observation steps 2.2 -> 7.0,
    # images accumulating to the 24-step cap on 19% of items, 16 of 18
    # spiralling NExT-GQA units ending `truncated`. On the units that did
    # NOT spiral the delta was -0.019 -- i.e. the layout itself was ~neutral
    # and the loss came from a TEXT description that no longer matched the
    # PIXELS it described.
    #
    # Empty means "use the default describe_visible()". Otherwise this is a
    # template the optimizer may rewrite, with {n_images} and {image_lines}
    # substituted. Making it a psi field is what lets one proposal change
    # the composition and its description together.
    view_contract: str = ""
    # HOW THE ANSWER MUST BE EXPRESSED.
    #
    # `view_contract` covers how the attached images are described; the
    # ANSWER protocol text (`runtime.ANSWER_STYLES`) was still frozen Python
    # underneath the search -- the third instance of the "text that describes
    # the interface is not co-evolvable" defect. On ScreenSpot-Pro the
    # optimizer needed the answer frame to move with the views it proposed
    # and could only smuggle that into theta, where it collided with the
    # frozen protocol and cost 0.4845 -> 0.1056 on held-out items.
    #
    # Empty means the domain's default ANSWER_STYLES entry. Otherwise this
    # is a template replacing the whole answer block, with optional
    # placeholders {ans_header} and {ans_example} for the default pieces.
    answer_contract: str = ""
    # THE OBSERVATION PROTOCOL TEXT (the "harness" itself).
    #
    # Added 2026-09-19 after the g55 line showed direct > baseline on 6/8
    # domains: the hardcoded protocol template in `build_task_prompt` was a
    # measured NEGATIVE asset on this executor family (mmiu direct 0.967 vs
    # harness-start 0.667), so the protocol is now an evolvable coordinate
    # like every other text that describes the interface. theta teaches "how
    # to solve", this teaches "how to use the tools" (what zoom/note/view/
    # compose do and when to use them).
    #
    # Empty means the DEFAULT protocol template in build_task_prompt (the
    # incumbent behaviour; the baseline arm is unchanged by this field).
    # Otherwise this is a template replacing the protocol HOWTO block, with
    # optional placeholders {pool_size}, {max_crops}, {ans_header} and
    # {ans_example} for the runtime-computed pieces.
    #
    # UNTOUCHABLE invariants (locked in runtime, not searchable): the scorer
    # parsing rule (bare value / letter extraction), the budget caps (steps
    # per item, images per request), and C_exec. Those are the measurement,
    # not the strategy -- the optimizer may reword how the tools are USED,
    # it may not change what counts as an answer.
    protocol_contract: str = ""
    keep_overview: bool = True
    annotate_overview_boxes: bool = False
    history_keep_last_n: int = 4
    order: str = "overview_first"
    evict_policy: str = "fifo"
    max_images_per_request: int = 6

    def __post_init__(self):
        if self.order not in ("overview_first", "crops_first"):
            raise ViewConfigError(f"unsupported psi.order {self.order!r}")
        if self.evict_policy not in ("fifo", "keep_overview_drop_crops"):
            raise ViewConfigError(
                f"unsupported psi.evict_policy {self.evict_policy!r}")
        if int(self.max_images_per_request) < 1:
            raise ViewConfigError(
                "psi.max_images_per_request must be >= 1; a value of "
                f"{self.max_images_per_request} sends the model no image at "
                "all, which scores as a wrong answer rather than as the "
                "protocol failure it is")
        if int(self.history_keep_last_n) < 0:
            raise ViewConfigError(
                f"psi.history_keep_last_n must be >= 0, got "
                f"{self.history_keep_last_n}")

    @classmethod
    def from_dict(cls, d: dict) -> "PsiConfig":
        d = {k: v for k, v in d.items() if k not in RETIRED_PSI_FIELDS}
        unknown = set(d) - PSI_FIELDS
        if unknown:
            raise ViewConfigError(f"unsupported psi field(s): {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def hash(self) -> str:
        return sha(self.to_dict())


@dataclass
class SkillConfig:
    """theta: the executor's skill text (SKILL.md)."""
    text: str = ""

    @property
    def hash(self) -> str:
        return sha(self.text)


@dataclass
class Coordinates:
    """The full (theta, phi, psi) triple defining one arm's current state."""
    theta: SkillConfig = field(default_factory=SkillConfig)
    phi: PhiConfig = field(default_factory=PhiConfig)
    psi: PsiConfig = field(default_factory=PsiConfig)

    @property
    def config_hash(self) -> str:
        return sha({"theta": self.theta.hash, "phi": self.phi.hash,
                    "psi": self.psi.hash})

    def to_dict(self) -> dict:
        return {"theta": self.theta.text, "phi": self.phi.to_dict(),
                "psi": self.psi.to_dict(), "config_hash": self.config_hash}

    @classmethod
    def from_dict(cls, d: dict) -> "Coordinates":
        return cls(theta=SkillConfig(d.get("theta", "")),
                   phi=PhiConfig.from_dict(d.get("phi", {})),
                   psi=PsiConfig.from_dict(d.get("psi", {})))

    def write_snapshot(self, dirpath: Path) -> str:
        """Write an immutable SKILL.md / VIEW.yaml / LAYOUT.yaml triple."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        (dirpath / "SKILL.md").write_text(self.theta.text)
        (dirpath / "VIEW.yaml").write_text(
            yaml.safe_dump(self.phi.to_dict(), sort_keys=True))
        (dirpath / "LAYOUT.yaml").write_text(
            yaml.safe_dump(self.psi.to_dict(), sort_keys=True))
        h = self.config_hash
        (dirpath / "CONFIG_HASH").write_text(h)
        return h

    @classmethod
    def read_snapshot(cls, dirpath: Path) -> "Coordinates":
        dirpath = Path(dirpath)
        theta = SkillConfig((dirpath / "SKILL.md").read_text())
        phi = PhiConfig.from_dict(
            yaml.safe_load((dirpath / "VIEW.yaml").read_text()) or {})
        psi = PsiConfig.from_dict(
            yaml.safe_load((dirpath / "LAYOUT.yaml").read_text()) or {})
        return cls(theta=theta, phi=phi, psi=psi)


def clamp_region(reg: Region, phi: PhiConfig) -> Region:
    """Apply phi.min_crop_frac so a crop keeps some surrounding context."""
    if phi.min_crop_frac <= 0:
        return reg
    cx, cy = (reg.x0 + reg.x1) / 2, (reg.y0 + reg.y1) / 2
    half = phi.min_crop_frac / 2
    w2 = max(reg.width / 2, half)
    h2 = max(reg.height / 2, half)
    x0, x1 = cx - w2, cx + w2
    y0, y1 = cy - h2, cy + h2
    # shift back inside bounds instead of squashing the box
    if x0 < 0:
        x1, x0 = x1 - x0, 0.0
    if y0 < 0:
        y1, y0 = y1 - y0, 0.0
    if x1 > 1:
        x0, x1 = x0 - (x1 - 1), 1.0
    if y1 > 1:
        y0, y1 = y0 - (y1 - 1), 1.0
    return Region(max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))
