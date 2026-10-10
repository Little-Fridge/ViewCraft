"""Core data types for ViewCraft.

The central invariant: what the executor may see and what the scorer may see
are *different types*, not two views of one dict. A PublicItem physically
cannot carry an answer or an evidence location, so leaking one is a type
error caught by tests rather than a string scan over field names.

Geometry conventions (uniform across domains):
  * regions are normalized (x0, y0, x1, y1) in [0, 1], origin top-left
  * `long_edge_px` is the rendered long edge of an emitted view
  * document pages use `page_index` (0-based) with `page_number` reserved for
    the 1-based label printed on the page
  * video time uses float `seconds`
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Literal

Domain = Literal["mmlongdoc", "chartqa", "vstar", "nextgqa", "vsi", "miniwob"]
Split = Literal["diagnostic", "train", "gate", "test"]
# Run-directory label. "view_only" is retained because interface
# projection builds its projected candidate through
# enforce_arm_scope("view_only", ...); it is a coordinate scope, not a
# selectable condition.
Arm = Literal["viewcraft", "view_only"]
Condition = Literal["normal", "oracle", "p_select", "p_read",
                    "motiv_a", "motiv_b", "motiv_c", "motiv_d"]


def sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# geometry


@dataclass(frozen=True)
class Region:
    """Normalized box in [0, 1], origin top-left."""
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self):
        for name in ("x0", "y0", "x1", "y1"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"Region.{name}={v} outside [0,1]")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"Region has non-positive area: {self}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def to_pixels(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        return (round(self.x0 * img_w), round(self.y0 * img_h),
                round(self.x1 * img_w), round(self.y1 * img_h))

    @classmethod
    def from_pixels(cls, box: tuple[float, float, float, float],
                    img_w: int, img_h: int) -> "Region":
        x0, y0, x1, y1 = box
        return cls(x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h)

    def expand(self, margin: float) -> "Region":
        """Grow by `margin` (fraction of the full image), clipped to bounds."""
        return Region(max(0.0, self.x0 - margin), max(0.0, self.y0 - margin),
                      min(1.0, self.x1 + margin), min(1.0, self.y1 + margin))

    def union(self, other: "Region") -> "Region":
        """Smallest box covering both - used to keep two objects' spatial
        relation intact rather than cropping each to its own center."""
        return Region(min(self.x0, other.x0), min(self.y0, other.y0),
                      max(self.x1, other.x1), max(self.y1, other.y1))


@dataclass(frozen=True)
class TimeSpan:
    start_s: float
    end_s: float

    def __post_init__(self):
        if self.end_s < self.start_s:
            raise ValueError(f"TimeSpan end {self.end_s} < start {self.start_s}")

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s

    def iou(self, other: "TimeSpan") -> float:
        inter = max(0.0, min(self.end_s, other.end_s)
                    - max(self.start_s, other.start_s))
        union = self.duration + other.duration - inter
        return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# the public / private split


@dataclass(frozen=True)
class PublicItem:
    """Everything the executor is allowed to see.

    Carries no answer, no evidence page, no ground-truth box or time span.
    """
    item_id: str
    domain: str
    question: str
    media_ids: tuple[str, ...] = ()
    options: tuple[str, ...] = ()          # multiple choice, original order
    instruction: str = ""                  # e.g. MiniWoB utterance
    meta: dict = field(default_factory=dict)   # neutral: page count, duration

    def __post_init__(self):
        banned = {"answer", "gold", "label", "evidence_pages", "evidence",
                  "bbox", "gt", "target", "span", "solution"}
        bad = banned & set(self.meta)
        if bad:
            raise ValueError(
                f"PublicItem.meta may not contain answer-bearing keys: {sorted(bad)}")


@dataclass(frozen=True)
class PrivateItem:
    """Everything only the scorer and the oracle entry point may see."""
    item_id: str
    answer: Any
    answer_type: str = "string"
    evidence_pages: tuple[int, ...] = ()       # 0-based page_index
    evidence_regions: tuple[Region, ...] = ()
    evidence_spans: tuple[TimeSpan, ...] = ()
    evidence_source: str = ""                  # "official" | "derived" | "none"
    evidence_verified: bool = False            # human/auditable check passed
    unanswerable: bool = False
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    """A public/private pair. Only `.public` may cross into execution."""
    public: PublicItem
    private: PrivateItem
    split: str = ""
    group_key: str = ""      # doc_id / image_id / scene / video - split unit

    def __post_init__(self):
        if self.public.item_id != self.private.item_id:
            raise ValueError("public/private item_id mismatch")


# ---------------------------------------------------------------------------
# views


class ViewKind(str, Enum):
    FULL = "full"               # whole page/image/frame, downscaled to budget
    CROP = "crop"               # region of a source
    OVERVIEW = "overview"       # contact sheet / global map
    SCREENSHOT = "screenshot"   # live GUI capture


@dataclass
class ViewRecord:
    """Audit record for one emitted image.

    Every field is required to reconstruct exactly what the model saw and
    where it came from. `actually_sent` distinguishes a rendered-but-dropped
    view from one that reached the model.
    """
    view_id: str
    source_media_id: str
    kind: str
    long_edge_px: int
    out_w: int
    out_h: int
    src_w: int
    src_h: int
    parent_view_id: str | None = None
    region: Region | None = None            # normalized, in source coords
    page_index: int | None = None
    seconds: float | None = None
    scale: float = 1.0                      # out/src linear factor
    upscaled: bool = False                  # interpolation adds no detail
    ui_state_version: int | None = None     # MiniWoB screenshot freshness
    actually_sent: bool = False
    request_id: str | None = None
    bytes_len: int = 0
    created_by: str = ""                    # which phi/psi rule produced it

    source_units: tuple[int, ...] = ()      # ordered members of a composite

    @property
    def pixels(self) -> int:
        return self.out_w * self.out_h

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.region is not None:
            d["region"] = asdict(self.region)
        return d


def new_view_id(prefix: str = "v") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# requirements and consistency


@dataclass(frozen=True)
class Requirement:
    """One entry of A0(q): a piece of evidence the question needs."""
    req_id: str
    text: str
    kind: str = "visual"        # "visual" | "textual" | "derived"


SupportLabel = Literal["supported", "partial", "unsupported", "unknown"]
SUPPORT_WEIGHT: dict[str, float] = {
    "supported": 1.0, "partial": 0.5, "unsupported": 0.0, "unknown": 0.0,
}


@dataclass
class GroundingResult:
    """C_ground: per-requirement evidence support, plus the unknown rate."""
    per_requirement: dict[str, str]
    score: float
    unknown_rate: float
    n_requirements: int
    verifier_request_ids: tuple[str, ...] = ()

    @classmethod
    def from_labels(cls, labels: dict[str, str],
                    request_ids: tuple[str, ...] = ()) -> "GroundingResult":
        n = len(labels)
        if n == 0:
            raise ValueError("C_ground needs at least one requirement")
        total = sum(SUPPORT_WEIGHT[v] for v in labels.values())
        unknown = sum(1 for v in labels.values() if v == "unknown")
        return cls(per_requirement=dict(labels), score=total / n,
                   unknown_rate=unknown / n, n_requirements=n,
                   verifier_request_ids=request_ids)


@dataclass
class ExecResult:
    """C_exec: deterministic execution-contract check."""
    passed: bool
    violations: tuple[str, ...] = ()

    @classmethod
    def ok(cls) -> "ExecResult":
        return cls(True, ())


# ---------------------------------------------------------------------------
# trajectories


@dataclass
class Trajectory:
    """One sample executed once under one (arm, condition)."""
    item_id: str
    domain: str
    split: str
    arm: str
    condition: str
    config_hash: str
    protocol_hash: str
    environment_seed: int | None = None
    answer_text: str = ""
    parsed_answer: Any = None
    score: float | None = None
    scored_by: str = ""
    views: list[ViewRecord] = field(default_factory=list)
    request_ids: list[str] = field(default_factory=list)
    observation_steps: int = 0
    n_actions: int = 0
    malformed_zooms: int = 0     # protocol failures, distinct from wrong answers
    # Zooms `phi.zoom_gate_expr` refused. Recorded because a refusal is the
    # mechanism WORKING, and it is otherwise invisible: the trajectory simply
    # has fewer crops, which is indistinguishable from a model that chose not
    # to ask. A gate whose refusal count is 0 on a domain was inert there.
    refused_zooms: int = 0
    # Which source units phi.select_expr chose, when it was set. Empty means
    # the adapter's own default sampling was used. Recorded so a run can be
    # audited for whether the SELECTION coordinate actually moved, rather
    # than inferred from a config hash that says only that it was set.
    selected_units: list = field(default_factory=list)
    # VISUAL MEMORY LAYER: every note the model jotted, as
    # {"source_media_id","x","y","text"}. Persisted so C_ground replays the
    # exact annotated pixels the executor saw, and so a case study can show the
    # canvas the model built up. Empty when the canvas was never used.
    canvas_notes: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0
    exec_result: ExecResult | None = None
    grounding: GroundingResult | None = None
    status: str = "ok"     # ok | error | timeout | truncated | invalid | uncertain
    error: str = ""
    notes: str = ""

    @property
    def sent_views(self) -> list[ViewRecord]:
        return [v for v in self.views if v.actually_sent]

    @property
    def sent_pixels(self) -> int:
        return sum(v.pixels for v in self.sent_views)

    @property
    def n_images_sent(self) -> int:
        return len(self.sent_views)

    def unit_key(self) -> str:
        """The experiment-unit identity. One valid completion per key.

        `arm` is deliberately NOT part of this key. An arm is provenance -
        which chain asked for the work - not part of what is measured. A
        measurement is fully determined by the item, the split, the
        environment seed, the condition, the config and the protocol.

        Including arm made every arm re-execute the identical starting
        config as its own unit. With residual nondeterminism at temperature
        0 those four executions returned four different answers, so four
        arms holding the SAME config_hash reported round-0 gate scores of
        0.694 / 0.735 / 0.735 / 0.755 on the same 49 items. That is a
        spread of 6 points of pure sampling noise sitting underneath every
        later comparison, and it also violates the one-run rule: the same
        condition was sampled four times. Keying on the measurement instead
        means identical configs share one recorded result by construction,
        so arms provably start from equal footing.
        """
        return sha({
            "domain": self.domain, "split": self.split, "item_id": self.item_id,
            "environment_seed": self.environment_seed,
            "condition": self.condition, "config_hash": self.config_hash,
            "protocol_hash": self.protocol_hash,
        })

    def to_dict(self) -> dict:
        d = asdict(self)
        d["views"] = [v.to_dict() for v in self.views]
        d["sent_pixels"] = self.sent_pixels
        d["n_images_sent"] = self.n_images_sent
        d["unit_key"] = self.unit_key()
        return d
