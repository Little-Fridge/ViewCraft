"""Model-authored view operators, expressed as declarative programs.

WHY THIS EXISTS
---------------
The first operationalisation of phi/psi was 15 config fields, 9 of them
resolution/crop scalars, over a renderer that could emit exactly four things:
whole image, crop, grid tile, overview-with-boxes. Every "view evolution"
round was therefore a search over numbers like `crop_long_edge_px`. On
benchmarks whose images already fit the view budget that space is provably
inert -- and that was wrongly written up as the benchmarks being unsuitable.
It was the action space being hardcoded to a single idea: magnification.

Visual presentation is not one idea. Contact sheets, coordinate overlays,
axis extension, adaptive binarisation, set-of-mark numbering, orientation
compasses, difference maps, channel isolation, temporal strips -- the list has
no natural end, and any list written here would substitute the harness
author's guess for the optimizer's reading of real failure traces.

So the optimizer does not pick from a menu. It WRITES OPERATOR PROGRAMS: a
pipeline of typed steps whose parameters may be arithmetic expressions over
neutral context (image size, frame count, timestamps). The space of programs
is combinatorially open, and new behaviour comes from composition rather than
from the harness shipping another special case.

WHY A PROGRAM DSL RATHER THAN RAW PYTHON
----------------------------------------
Executing model-written Python would be the most expressive option and is the
obvious first instinct. It is not used here, deliberately: arbitrary code
execution in the experiment loop is a real hazard, and the safety review
blocked it. The DSL keeps what actually matters for this study -- the
optimizer authors unbounded compositions rather than choosing from a fixed
set -- while every primitive step is harness-implemented, bounded, and
auditable. The honest cost is that an operator needing a genuinely novel
pixel-level algorithm cannot be expressed until that step type exists; the
step vocabulary is therefore deliberately low-level (kernels, per-pixel
curves, local statistics) so that most novelty lands in composition.

AUDIT
-----
An operator's identity is the hash of its full program plus parameters, so an
emitted pixel is always attributable to an exact program. Programs are stored
verbatim in the run manifest.
"""
from __future__ import annotations

import ast
import hashlib
import operator as _op
from dataclasses import dataclass, field
from typing import Any

from ..schema import sha


class OperatorRejected(RuntimeError):
    """The program violated a hard constraint and was never executed."""


# ---------------------------------------------------------------------------
# parameter expressions: arithmetic over neutral context, nothing else

_BINOPS = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
           ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv,
           ast.Mod: _op.mod, ast.Pow: _op.pow}
_UNARY = {ast.UAdd: _op.pos, ast.USub: _op.neg}
_FUNCS = {"min": min, "max": max, "abs": abs, "round": round,
          "int": int, "float": float}

# Context keys an expression may read. Whitelisted, so an expression cannot
# reach an answer or an evidence location even if it names one.
CTX_KEYS = {"width", "height", "n_media", "media_index", "n_options",
            "duration_s", "n_frames", "long_edge"}


def eval_param(expr: Any, ctx: dict) -> Any:
    """Evaluate a parameter that may be a literal or an arithmetic string.

    `"min(width, height) / 16"` becomes a number. This is arithmetic over a
    whitelisted numeric context -- no attribute access, no calls other than
    min/max/abs/round/int/float, no names beyond CTX_KEYS. It is how a
    program says "grid spacing proportional to the frame" without the
    harness having to anticipate that intent.
    """
    if not isinstance(expr, str):
        return expr
    s = expr.strip()
    if not s:
        return expr
    # a bare non-numeric string is a literal (e.g. a colour name or mode)
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return expr

    def ev(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise OperatorRejected(f"non-numeric constant in expr: {s!r}")
        if isinstance(node, ast.Name):
            if node.id not in CTX_KEYS:
                raise OperatorRejected(
                    f"expression may not read {node.id!r}; "
                    f"available: {sorted(CTX_KEYS)}")
            return float(ctx.get(node.id, 0) or 0)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or \
                    node.func.id not in _FUNCS:
                raise OperatorRejected(
                    f"only {sorted(_FUNCS)} may be called in an expression")
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise OperatorRejected(f"unsupported expression: {s!r}")

    try:
        return ev(tree.body)
    except OperatorRejected:
        raise
    except Exception:                                        # noqa: BLE001
        # not an arithmetic expression: treat as a literal string
        return expr


# ---------------------------------------------------------------------------
# the operator


MAX_STEPS = 12


@dataclass
class ViewOperator:
    """One model-authored visual operator: a named program plus parameters.

    `program` is a list of steps, each `{"op": <step name>, ...params}`.
    Step names and their meanings are declared in `steps.py`; validation
    happens there so this module stays free of rendering concerns.
    """
    name: str
    program: list = field(default_factory=list)
    rationale: str = ""
    # "pixel"   : one image in, one image out
    # "compose" : several images in, one image out
    kind: str = "pixel"
    author_request_id: str = ""

    @property
    def program_hash(self) -> str:
        blob = sha(self.program)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def identity(self) -> str:
        return sha({"name": self.name, "program": self.program,
                    "kind": self.kind})

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "program": self.program,
                "rationale": self.rationale[:600],
                "program_hash": self.program_hash,
                "identity": self.identity,
                "author_request_id": self.author_request_id}

    @classmethod
    def from_dict(cls, d: dict) -> "ViewOperator":
        return cls(name=str(d["name"]),
                   program=list(d.get("program") or []),
                   rationale=d.get("rationale", ""),
                   kind=d.get("kind", "pixel"),
                   author_request_id=d.get("author_request_id", ""))

    def validate(self) -> None:
        """Structural checks that do not need a renderer."""
        if not self.name or not str(self.name).strip():
            raise OperatorRejected("operator needs a name")
        if self.kind not in ("pixel", "compose"):
            raise OperatorRejected(f"unknown operator kind {self.kind!r}")
        if not isinstance(self.program, list) or not self.program:
            raise OperatorRejected("operator program is empty")
        if len(self.program) > MAX_STEPS:
            raise OperatorRejected(
                f"program has {len(self.program)} steps; max {MAX_STEPS}")
        for i, st in enumerate(self.program):
            if not isinstance(st, dict) or "op" not in st:
                raise OperatorRejected(
                    f"step {i} must be an object with an 'op' key")


def safe_context(item, n_media: int, img_size: tuple[int, int],
                 media_index: int = 0, long_edge: int = 1024) -> dict:
    """Neutral facts a program may branch on.

    Built by WHITELIST. An operator cannot read an answer, a gold label or an
    evidence location, because none of them is ever placed here. Everything
    present is already visible to the executor in its own prompt.
    """
    meta = item.meta or {}
    times = [float(t) for t in (meta.get("frame_times_s") or [])][:64]
    return {
        "width": int(img_size[0]),
        "height": int(img_size[1]),
        "n_media": int(n_media),
        "media_index": int(media_index),
        "n_options": len(item.options or ()),
        "duration_s": float(meta.get("duration_s") or 0.0),
        "n_frames": len(times),
        "long_edge": int(long_edge),
        # non-numeric extras usable by steps that take them directly
        "frame_times_s": times,
        "question": str(item.question or "")[:500],
        "question_type": str(meta.get("question_type")
                             or meta.get("category")
                             or meta.get("type") or ""),
    }

# SHARED SCALAR HELPERS -- one definition for all three expression
# languages. The optimizer writes `minimum(1, len(x))` in one language and
# `min(...)` in another because it does not track which of OUR languages
# has which helper. An asymmetry here is not a mistake by it: `maximum`
# existed in layout and pixels but not in selection, and 93 of 149 dude
# units died on `unknown selection function 'minimum'`, scoring the
# candidate 0.000 -- a collapse that looked like a bad strategy and was a
# missing alias.
SCALAR_HELPERS = {
    "int": int, "float": float, "round": round, "abs": abs,
    "min": min, "max": max, "minimum": min, "maximum": max,
    "len": lambda x: len(x) if hasattr(x, "__len__") else 1,
}
