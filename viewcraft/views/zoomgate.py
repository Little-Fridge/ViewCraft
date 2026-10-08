"""Zoom-gate expressions: WHETHER a requested crop is worth rendering.

WHY THIS EXISTS
---------------
The renderer never upscales a crop. `render._fit` returns the image untouched
when the fit factor exceeds 1.0 and `allow_upscale` is off, so on a source that
already fits the overview budget a crop CANNOT add one pixel of detail: the
model asks for a second look and receives the same pixels it already had, one
round later.

Measured over 1808 recorded crops (paper/EXPLORATION_FINDINGS.md section 11):
the area-based median magnification is 1.000x and 73.5% of crops deliver
<=1.05x. Those futile zooms are not free. They mark failure -- 217/388 wrong
where the zoom was futile versus 90/230 where it magnified, Fisher p=6.3e-05 --
and a single threshold on the source long edge separates the two populations at
98.9% accuracy, confirmed against the real files on disk.

The action space could not express the decision. The only field that gestured
at it, `phi.second_pass_zoom`, was declared, advertised in the optimizer's
prompt, and READ BY NO CODE (now retired; see `views/config.py`). Across 312
snapshots not one evolved conditional anywhere was predicated on resolution,
because none could be written.

WHY AN EXPRESSION AND NOT A BOOLEAN
-----------------------------------
A boolean `skip_futile_zooms` would be the 18-step-menu mistake again (see
`operators.py` and `selection.py`): the optimizer would be picking the one
condition the harness author already imagined. "Refuse a zoom that cannot
magnify" is one policy; "refuse a zoom at a region larger than a tenth of the
frame", "refuse a third crop in the same step", "allow anything while steps
remain and tighten near the budget" are others, and the list has no natural end.

So the optimizer WRITES the condition over neutral facts about the request. The
measured rule is `zoom_gain > 1.05`. AdaZoom-GUI's conditional second stage is
`region_frac < 0.1`. Neither is enumerated here.

An EMPTY expression allows every zoom, which reproduces the incumbent behaviour
exactly rather than approximately -- the same reason `uniform(k)` exists in the
selection language. A control has to be reproducible to the pixel or every
comparison against it is confounded.

WHAT IT DELIBERATELY CANNOT DO
------------------------------
The facts are a whitelist of geometry and position, all of them already visible
to the executor in its own prompt. No answer, no gold label, no evidence
region. A gate that could read ground truth would be an oracle deciding when to
look, and every number produced under it would be meaningless.
"""
from __future__ import annotations

import ast

from .operators import SCALAR_HELPERS, OperatorRejected
from .scalars import SHARED_SCALAR_NAMES, shared_scalars

MAX_EXPR_LEN = 600

# Facts about the REQUEST, on top of the shared scalars every language binds.
# Names only; the values are built by `zoom_facts` below.
ZOOM_FACT_NAMES = (
    "src_long_edge",       # the source file's long edge, in pixels
    "overview_px",         # phi.overview_long_edge_px
    "crop_px",             # phi.crop_long_edge_px
    "region_frac",         # requested region's AREA, fraction of the frame
    "region_long_edge",    # requested region's long edge, fraction of the frame
    "zoom_gain",           # linear magnification this crop would DELIVER
    "step",                # 0-based observation step this request came from
    "steps_left",          # observation steps remaining after this one
    "crops_so_far",        # crops already rendered in this trajectory
)


def zoom_gain(src_long_edge: float, overview_px: float, crop_px: float,
              region_long_edge: float) -> float:
    """Linear magnification a crop of this region would actually deliver.

    DERIVED BY THE HARNESS, not left to the optimizer, because it is a fact
    about what `render._fit` does rather than a score. Two ceilings apply and
    both are ours, not the model's:

      * the crop is fitted to `crop_px`, so a region occupying `f` of the long
        edge is delivered at `crop_px / f` source-pixels-per-long-edge at most;
      * the source has only `src_long_edge` pixels and a crop is NEVER
        upscaled, so nothing beyond the source's own resolution is available.

    The overview already gave the model `overview_px` across the whole frame,
    so the gain is the ratio of the two. Expecting the optimizer to rederive
    the upscale cap from first principles is the same defect as advertising
    `n_frames` in the prompt and not binding it: the prompt would be telling it
    to reason about a rule it cannot see.

    The OVERVIEW is not upscaled either, so its effective long edge is
    `min(src_long_edge, overview_px)`. Using the budget alone in the
    denominator made a small source report a gain BELOW 1.0 (0.488x on a 500 px
    file), which is unreachable -- a crop of the same pixels cannot be worse
    than the overview. The verdict at the measured 1.05 threshold was unchanged,
    but a gate written as `zoom_gain < 0.9` would have been reading an artefact
    of ours rather than a fact about the file.
    """
    f = max(1e-6, float(region_long_edge))
    ov = max(1e-6, min(float(src_long_edge), float(overview_px)))
    # what the region has natively in the source file, per long edge
    native = float(src_long_edge) * f
    # what the crop render can carry
    carried = min(native, float(crop_px))
    # what the overview already delivered for that same region
    already = ov * f
    return max(0.0, carried / max(1e-6, already))


def zoom_facts(*, src_w: int, src_h: int, overview_px: int, crop_px: int,
               region, step: int = 0, steps_left: int = 0,
               crops_so_far: int = 0, ctx: dict | None = None,
               out_w: int = 0, out_h: int = 0) -> dict:
    """Bind every name a zoom-gate expression may read, always.

    `region` is a `schema.Region` (normalised). Every shared scalar is bound
    too, whatever the domain reported -- see `views/scalars.py`. A name that is
    advertised in one language and absent in another makes a candidate's
    validity depend on which item it happens to meet, which is the recurring
    defect that module exists to prevent.
    """
    rw = max(0.0, float(region.x1) - float(region.x0))
    rh = max(0.0, float(region.y1) - float(region.y0))
    src_long = float(max(int(src_w), int(src_h)))
    r_long = max(rw, rh)
    env = shared_scalars(1, 1, ctx or {},
                         width=float(out_w or src_w),
                         height=float(out_h or src_h))
    env.update({
        "src_long_edge": src_long,
        "overview_px": float(overview_px),
        "crop_px": float(crop_px),
        "region_frac": rw * rh,
        "region_long_edge": r_long,
        "zoom_gain": zoom_gain(src_long, overview_px, crop_px, r_long),
        "step": float(step),
        "steps_left": float(steps_left),
        "crops_so_far": float(crops_so_far),
    })
    return env


def eval_zoom_gate(expr: str, facts: dict) -> bool:
    """Evaluate a gate expression to allow (True) or refuse (False) one crop.

    An empty expression is not evaluated at all by callers; if it reaches here
    it allows the zoom, because the incumbent behaviour is to render every
    requested region.

    The accepted node set matches the selection language's (`selection.py`)
    rather than `operators.eval_param`'s. `eval_param` has no `ast.Compare` and
    no `ast.IfExp`, so a gate -- which IS a comparison -- is inexpressible in
    it. Recurring class: one capability present in one of our languages and
    absent in another, our defect rather than the optimizer's.
    """
    if not isinstance(expr, str) or not expr.strip():
        return True
    s = expr.strip()
    if len(s) > MAX_EXPR_LEN:
        raise OperatorRejected(f"zoom gate over {MAX_EXPR_LEN} chars")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise OperatorRejected(f"zoom gate does not parse: {e}") from e

    def ev(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise OperatorRejected("only numeric constants allowed")
        if isinstance(node, ast.Name):
            if node.id in facts:
                return facts[node.id]
            raise OperatorRejected(
                f"unknown name {node.id!r}; available: {sorted(facts)}")
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            op = type(node.op)
            if op is ast.Add:
                return a + b
            if op is ast.Sub:
                return a - b
            if op is ast.Mult:
                return a * b
            if op is ast.Div:
                return a / (b or 1e-9)
            if op is ast.FloorDiv:
                return a // (b or 1e-9)
            if op is ast.Mod:
                return a % (b or 1)
            if op is ast.Pow:
                if abs(float(b)) > 64:
                    raise OperatorRejected("exponent out of range")
                return a ** b
            raise OperatorRejected("unsupported operator in zoom gate")
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -ev(node.operand)
            if isinstance(node.op, ast.UAdd):
                return ev(node.operand)
            if isinstance(node.op, ast.Not):
                return not ev(node.operand)
            raise OperatorRejected("unsupported unary operator")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise OperatorRejected("only direct function calls allowed")
            fn = SCALAR_HELPERS.get(node.func.id)
            if fn is None:
                raise OperatorRejected(
                    f"unknown function {node.func.id!r}; available: "
                    f"{sorted(SCALAR_HELPERS)}")
            try:
                return fn(*[ev(a) for a in node.args])
            except OperatorRejected:
                raise
            except (TypeError, ValueError, ZeroDivisionError) as e:
                raise OperatorRejected(
                    f"{node.func.id}() failed: {e}") from e
        if isinstance(node, ast.Compare):
            # CHAINED comparisons are supported (`1.05 < zoom_gain < 4`), which
            # is how a person writes a band. Refusing the chain would be
            # another one-language-has-it gap.
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                t = type(op)
                if t is ast.Gt:
                    ok = left > right
                elif t is ast.GtE:
                    ok = left >= right
                elif t is ast.Lt:
                    ok = left < right
                elif t is ast.LtE:
                    ok = left <= right
                elif t is ast.Eq:
                    ok = left == right
                elif t is ast.NotEq:
                    ok = left != right
                else:
                    raise OperatorRejected("unsupported comparison")
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            vals = [ev(v) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.IfExp):
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        raise OperatorRejected(
            f"unsupported syntax in zoom gate: {type(node).__name__}")

    try:
        return bool(ev(tree.body))
    except OperatorRejected:
        raise
    except Exception as e:                                   # noqa: BLE001
        raise OperatorRejected(f"zoom gate failed: {e}") from e


def mentions_a_zoom_fact(expr: str) -> bool:
    """Does the expression read any fact about the REQUEST?

    A gate written purely over shared scalars (`n > 0`) is a constant on every
    item of a domain: it either blocks all zooming or allows all of it, and
    neither is a conditional mechanism. Same rule as "a canvas_expr that
    ignores the note arrays is a no-op and is rejected".
    """
    if not isinstance(expr, str) or not expr.strip():
        return False
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(nd, ast.Name) and nd.id in ZOOM_FACT_NAMES
               for nd in ast.walk(tree))


def zoom_gate_help() -> str:
    """The gate vocabulary for the optimizer's prompt.

    Generated from the name tuples, never hand-written, so the prompt cannot
    advertise a name the evaluator does not bind -- the defect that cost three
    round-1 proposals to `unknown name 'n_frames'`.
    """
    return (
        "    Facts about the REQUESTED region (all neutral geometry):\n"
        "      src_long_edge     the source FILE's long edge, in pixels\n"
        "      overview_px       long edge of the whole-image view\n"
        "      crop_px           long edge a crop is rendered at\n"
        "      region_frac       the region's AREA, fraction of the frame\n"
        "      region_long_edge  the region's long edge, fraction of frame\n"
        "      zoom_gain         linear magnification this crop would ACTUALLY\n"
        "                        deliver versus the overview. A crop is never\n"
        "                        upscaled, so on a source that already fits\n"
        "                        the overview budget this is ~1.0 and the zoom\n"
        "                        returns the SAME pixels one round later.\n"
        "      step, steps_left, crops_so_far   position in the trajectory\n"
        f"      shared names: {', '.join(SHARED_SCALAR_NAMES)}\n"
        "    Operators: + - * / // % **, comparisons (chained allowed),\n"
        "    and/or/not, `a if c else b`, "
        f"{', '.join(sorted(SCALAR_HELPERS))}\n"
        "    EMPTY means allow every zoom -- the incumbent default, provided\n"
        "    so it can be reproduced exactly. There is NO menu of policies;\n"
        "    you WRITE the condition:\n"
        "      zoom_gain > 1.05             refuse a zoom that cannot magnify\n"
        "      region_frac < 0.1            only zoom on a small target\n"
        "      zoom_gain > 1.2 and crops_so_far < 3   both, with a cap\n"
        "      steps_left > 2 or zoom_gain > 2         tighten near the budget\n"
        "    A gate reading none of the request facts is constant on every\n"
        "    item and is rejected.")
