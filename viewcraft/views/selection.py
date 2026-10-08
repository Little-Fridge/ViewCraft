"""Selection expressions: WHICH source units the model gets to see.

WHY THIS EXISTS
---------------
The study spent its whole first half evolving how eight video frames were
laid out, and never once evolved WHICH eight frames. `uniform_frame_indices`
was hardcoded in the adapter, and `PHI_FIELDS`/`PSI_FIELDS` contained no
sampling field at all. So for a video or a multi-page document -- the two
input types where "what do I even look at" is the dominant decision -- the
action space excluded the decision that matters most.

That is also why ChartQA was a poor venue: one image, 620 px median, already
delivered whole. There is no selection to make, so the only thing left is the
model's own reading and arithmetic. A domain with 1 unit per item cannot
exercise a selection mechanism, and a flat result there says nothing about
the mechanism.

WHAT A SELECTION EXPRESSION IS
------------------------------
An expression over a pool of N addressable source units (video frames, PDF
pages) returning the indices to materialise, under a budget K. The
vocabulary is index ARITHMETIC -- `range`, list addition, comprehension --
not a menu of named schemes:

    range(n // 16, n, n // 8)               evenly spread
    range(0, n // 2, n // 16)               front-loaded
    range(0, 3) + range(n - 3, n)           ends only
    [int(n * 0.7 ** i) for i in range(6)]   log-spaced toward the start

`uniform(k)` exists for exactly one reason: it is the incumbent default and
a control must be reproducible to the index, or every comparison against it
is confounded. It is not a recommendation.

The first version of this file shipped `dense`, `stratified`, `around`,
`first`, `last` as named functions. That was the 18-step-menu mistake
again: the optimizer picks from what the author already imagined, and
sampling strategies are not enumerable. Dense sampling, hierarchical
sampling and retrieval-ordered sampling are things you WRITE here.

WHAT IT DELIBERATELY CANNOT DO
------------------------------
A selection expression sees only NEUTRAL metadata: how many units exist, how
long the document/video is, and per-unit position. It never sees the answer,
the evidence pages, or any per-item label. Selection that could read ground
truth would be an oracle, not a searched strategy, and every number produced
under it would be meaningless. `safe_context` enforces this.
"""
from __future__ import annotations

import ast
from typing import Any

from .operators import SCALAR_HELPERS, OperatorRejected
from .scalars import SHARED_SCALAR_NAMES, shared_scalars

MAX_EXPR_LEN = 1000
MAX_UNITS = 4096


def _clamp_idx(idx, n: int) -> list[int]:
    """Dedup, sort, drop anything outside the pool."""
    out = sorted({int(i) for i in idx if 0 <= int(i) < n})
    return out


def _uniform(n: int, k: int) -> list[int]:
    """Midpoints of k equal bins -- the scheme vsi.py has always used.

    Kept identical so `uniform(k)` reproduces the previous hardcoded
    behaviour exactly, which makes it a legitimate control rather than a
    near-miss that would confound every comparison against it.
    """
    k = max(1, int(k))
    if k >= n:
        return list(range(n))
    return _clamp_idx([int((i + 0.5) * n / k) for i in range(k)], n)






def _union(*groups) -> list[int]:
    """Flatten any nesting. `union(*[range(0,2)])` passes a list of lists."""
    out: list[int] = []

    def walk(g):
        if isinstance(g, (list, tuple, range)):
            for x in g:
                walk(x)
        else:
            out.append(g)

    for g in groups:
        walk(g)
    return sorted({int(i) for i in out})


def build_funcs(n: int) -> dict[str, Any]:
    """Index primitives bound to a pool size.

    DELIBERATELY NOT A MENU OF SAMPLING SCHEMES. The first version of this
    file shipped `uniform`, `dense`, `stratified`, `around`, `first`,
    `last` -- which is the 18-step-menu mistake in a new costume: the
    optimizer would be picking a strategy the author already thought of,
    and sampling strategies are not enumerable.

    What is here instead is index ARITHMETIC. A named scheme is something
    you WRITE:

        evenly spread     range(n // 16, n, n // 8)
        front-loaded      range(0, n // 2, n // 16)
        ends only         range(0, 3) + range(n - 3, n)
        every 4th         range(0, n, 4)
        two clusters      range(a, a + 3) + range(b, b + 3)
        log-spaced        [int(n * 0.7 ** i) for i in range(6)]

    `range`, list arithmetic and comprehension over a bounded index domain
    span these and strategies nobody listed.
    """
    def _range(*a):
        try:
            r = range(*[int(x) for x in a])
        except (TypeError, ValueError) as e:
            raise OperatorRejected(f"range{a}: {e}") from e
        if len(r) > MAX_UNITS:
            raise OperatorRejected(f"range{a} yields {len(r)} indices")
        return list(r)

    def _as_list(feat):
        if not isinstance(feat, (list, tuple)):
            raise OperatorRejected("content feature must be a per-unit list "
                                   "(motion / edges / colorful / bright)")
        return [float(v) for v in feat]

    def _topk(feat, k):
        f = _as_list(feat)
        k = max(0, min(int(k), len(f)))
        return sorted(sorted(range(len(f)), key=lambda i: -f[i])[:k])

    def _bottomk(feat, k):
        f = _as_list(feat)
        k = max(0, min(int(k), len(f)))
        return sorted(sorted(range(len(f)), key=lambda i: f[i])[:k])

    def _above(feat, t):
        f = _as_list(feat)
        return [i for i, v in enumerate(f) if v >= float(t)]

    def _below(feat, t):
        f = _as_list(feat)
        return [i for i, v in enumerate(f) if v <= float(t)]

    def _order_by(feat):
        f = _as_list(feat)
        return sorted(range(len(f)), key=lambda i: -f[i])

    return {
        "range": _range,
        "union": _union,
        # scalar helpers, so a strategy can be written IN TERMS OF n
        **SCALAR_HELPERS,
        # `uniform` is kept for ONE reason: it is the incumbent default and
        # a control has to be reproducible exactly, or every comparison
        # against it is confounded. It is not offered as a strategy to pick.
        "uniform": lambda k: _uniform(n, k),
        # CONTENT-AWARE primitives. These are NOT named sampling schemes: they
        # turn a neutral per-unit content signal (motion / edges / colorful /
        # bright, bound in the env) into an index set, and the optimizer writes
        # the strategy -- which signal, what k/threshold, and how to union it
        # with uniform coverage. e.g. `union(topk(motion, budget//2),
        # uniform(budget//2))` = "the busiest frames, plus even coverage".
        "topk": _topk,
        "bottomk": _bottomk,
        "above": _above,
        "below": _below,
        "order_by": _order_by,
    }


def eval_selection_expr(expr: str, n_units: int, ctx: dict,
                        budget: int) -> list[int]:
    """Evaluate a selection expression to a list of source-unit indices.

    `budget` is a HARD cap applied after evaluation: a selection expression
    may not buy itself more context than any other condition gets, or the
    comparison would be measuring spend rather than strategy. Over-budget
    selections are truncated by uniform subsampling of what was chosen, so
    the strategy's SHAPE survives the cap.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise OperatorRejected("empty selection expression")
    if len(expr) > MAX_EXPR_LEN:
        raise OperatorRejected(f"selection expression over {MAX_EXPR_LEN}")
    n = int(n_units)
    if n <= 0 or n > MAX_UNITS:
        raise OperatorRejected(f"selection: bad pool size {n}")

    funcs = build_funcs(n)
    # Every shared scalar is bound on every item, whether or not this domain
    # reports it -- see views/scalars.py. Binding only `if k in ctx` made a
    # candidate valid on MVBench (which reports n_frames) and invalid on
    # NExT-GQA (which does not), and invalid in the pre-flight validator,
    # which has no item at all.
    env = shared_scalars(n, budget, ctx)
    # Content-aware signals: neutral per-unit features computed from the media
    # (see views/content.py), bound as lists so the primitives above can use
    # them. Always bound (empty list when absent), never `if in ctx` -- a name
    # present on one domain and missing on another makes validity depend on
    # which item the candidate meets, the recurring scope defect.
    feats = (ctx or {}).get("content_features") or {}
    for name in ("motion", "edges", "colorful", "bright"):
        lst = feats.get(name)
        env[name] = ([float(v) for v in lst]
                     if isinstance(lst, (list, tuple)) and len(lst) == n
                     else [0.0] * n)

    # LET-BINDINGS, as the pixel and layout languages already allow.
    #
    # This parsed with mode="eval", so `k = minimum(budget, n)` on its own
    # line was a hard syntax error. On mvbench round 4 the optimizer wrote
    #
    #     k = minimum(budget, n)
    #     base = [int((n-1) * i / maximum(k-1, 1)) for i in range(k)]
    #     ends = range(max(n-3,0), n, 1)
    #     union(base, ends)
    #
    # which is how a person writes even coverage plus explicit end coverage.
    # Naming a subexpression adds no capability -- it inlines away -- so
    # refusing it only cost a round. Fifth instance of one capability being
    # present in one of the three expression languages and absent in
    # another; the others were n_frames, height, the size guards, and
    # comprehension filters.
    let_stmts: list = []
    try:
        mod = ast.parse(expr.strip(), mode="exec")
    except SyntaxError as e:
        raise OperatorRejected(
            f"selection expression does not parse: {e}") from e
    if not mod.body or not isinstance(mod.body[-1], ast.Expr):
        raise OperatorRejected(
            "a selection expression must END in an expression")
    for st in mod.body[:-1]:
        if not isinstance(st, ast.Assign) or len(st.targets) != 1 \
                or not isinstance(st.targets[0], ast.Name):
            raise OperatorRejected(
                "only simple `name = expression` bindings may precede the "
                "final selection expression")
        if st.targets[0].id in env or st.targets[0].id in funcs:
            raise OperatorRejected(
                f"{st.targets[0].id!r} is a provided name and may not be "
                "rebound")
    let_stmts = mod.body[:-1]
    tree = ast.Expression(body=mod.body[-1].value)

    def ev(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise OperatorRejected("only numeric constants allowed")
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise OperatorRejected(
                f"unknown name {node.id!r}; available: {sorted(env)}")
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            op = type(node.op)
            if op is ast.Add:
                if isinstance(a, list) or isinstance(b, list):
                    return _union(a, b)
                return a + b
            if op is ast.Sub:
                return a - b
            if op is ast.Mult:
                return a * b
            if op is ast.Div:
                return a / (b or 1e-9)
            if op is ast.FloorDiv:
                return a // (b or 1e-9)
            if op is ast.Pow:
                if abs(float(b)) > 64:
                    raise OperatorRejected("exponent out of range")
                return a ** b
            if op is ast.Mod:
                return a % (b or 1)
            raise OperatorRejected("unsupported operator")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise OperatorRejected("only direct function calls allowed")
            fn = funcs.get(node.func.id)
            if fn is None:
                raise OperatorRejected(
                    f"unknown selection function {node.func.id!r}; "
                    f"available: {sorted(k for k in funcs if k.islower())}")
            kw = {k.arg: ev(k.value) for k in node.keywords if k.arg}
            try:
                return fn(*[ev(a) for a in node.args], **kw)
            except OperatorRejected:
                raise
            except (TypeError, ValueError, ZeroDivisionError) as e:
                raise OperatorRejected(
                    f"{node.func.id}() failed: {e}") from e
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a, b = ev(node.left), ev(node.comparators[0])
            for t, f in ((ast.Gt, lambda x, y: x > y),
                         (ast.GtE, lambda x, y: x >= y),
                         (ast.Lt, lambda x, y: x < y),
                         (ast.LtE, lambda x, y: x <= y),
                         (ast.Eq, lambda x, y: x == y),
                         (ast.NotEq, lambda x, y: x != y)):
                if type(node.ops[0]) is t:
                    return f(a, b)
        if isinstance(node, ast.BoolOp):
            vals = [ev(v) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.IfExp):
            # `range(0, n//2 if n > 8 else n)` -- a selection that adapts to
            # pool size is exactly what this coordinate is for, and refusing
            # conditionals made it unexpressible.
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        if isinstance(node, ast.Starred):
            val = ev(node.value)
            return val if isinstance(val, (list, tuple, range)) else [val]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [ev(e) for e in node.elts]
        if isinstance(node, ast.ListComp) and len(node.generators) == 1:
            # A comprehension is what makes this primitives rather than a
            # menu: log-spaced, quadratic, or any other index law the
            # optimizer invents is writable without us having named it.
            #
            # `if` FILTERS ARE ALLOWED, and refusing them was a real gap.
            # "Take these indices, but drop the ones outside the window I
            # care about" is the most natural thing to write in index
            # arithmetic, and on mvbench round 3 the optimizer wrote exactly
            # that:
            #
            #   [int((n-1)*t) for t in [.2,.35,.5,.65,.8]
            #    if int((n-1)*t) >= 3 and int((n-1)*t) <= max(n-10, 3)]
            #
            # A filter adds no capability the language lacks -- the same set
            # is expressible by arithmetic on the bounds, just unreadably --
            # so rejecting it only converted a sound visual proposal into a
            # lost round. Same class as `n_frames` being advertised but
            # unbound: our restriction, not the optimizer's error.
            gen = node.generators[0]
            if not isinstance(gen.target, ast.Name):
                raise OperatorRejected(
                    "comprehension: single plain loop variable")
            seq = ev(gen.iter)
            if not isinstance(seq, (list, tuple, range)):
                raise OperatorRejected("comprehension must iterate indices")
            if len(seq) > MAX_UNITS:
                raise OperatorRejected("comprehension too large")
            out_c, name = [], gen.target.id
            saved = env.get(name)
            for v in seq:
                env[name] = v
                # every `if` must hold, matching Python's own semantics
                if gen.ifs and not all(bool(ev(c)) for c in gen.ifs):
                    continue
                out_c.append(ev(node.elt))
            if saved is None:
                env.pop(name, None)
            else:
                env[name] = saved
            return out_c
        if isinstance(node, ast.Subscript):
            # order_by(motion)[:k] / feat[i]: indexing and slicing a list value.
            # Without this the natural "rank then take a prefix" idiom was a
            # hard error -- one of the recurring "capability present in one
            # language, absent in another" gaps.
            base = ev(node.value)
            if not isinstance(base, (list, tuple)):
                raise OperatorRejected("can only subscript a list value")
            sl = node.slice
            if isinstance(sl, ast.Slice):
                lo = ev(sl.lower) if sl.lower else None
                hi = ev(sl.upper) if sl.upper else None
                return list(base[int(lo) if lo is not None else None:
                                 int(hi) if hi is not None else None])
            idx = ev(sl)
            return base[int(idx)]
        raise OperatorRejected(
            f"unsupported syntax in selection: {type(node).__name__}")

    # Run the let-bindings, in order, into the same environment the final
    # expression reads.
    for st in let_stmts:
        env[st.targets[0].id] = ev(st.value)

    out = ev(tree.body)
    if not isinstance(out, (list, tuple, range)):
        raise OperatorRejected(
            "selection expression must evaluate to indices, got "
            f"{type(out).__name__}")
    idx = _clamp_idx(out, n)
    if not idx:
        raise OperatorRejected(
            "selection expression selected NO units; the model would "
            "receive no image at all, which scores as a wrong answer "
            "rather than as the protocol failure it is")
    if len(idx) > budget:
        keep = _uniform(len(idx), budget)
        idx = [idx[i] for i in keep]
    return idx


def selection_help() -> str:
    """Index primitives for the optimizer prompt.

    States plainly that named schemes are compositions, not menu entries,
    because a menu is exactly what biased the pixel search before.
    """
    return (
        "    range(start, stop, step)    indices, like Python's range\n"
        "    a + b  or  union(a, b, ...) combine index lists\n"
        "    [expr for i in range(k)]    comprehension over indices\n"
        "        (ONE generator only; a filter clause is allowed: [... for\n"
        "         i in range(k) if cond]. For a nested/composite law, bind\n"
        "         intermediate lists first, e.g.\n"
        "         xs = range(0, n, 4)\n"
        "         [x for x in xs if x >= n//2]  -- a second 'for' inside\n"
        "         the brackets is NOT valid in this language)\n"
        # Generated from SHARED_SCALAR_NAMES, never hand-written. This line
        # used to say "n, budget, duration_s, n_pages, n_frames" while the
        # evaluator bound only n and budget, so nextgqa round 1 wrote
        # `n_frames` -- because the prompt told it to -- and lost the whole
        # joint proposal to `unknown name 'n_frames'`.
        f"    names: {', '.join(SHARED_SCALAR_NAMES)}\n"
        "    uniform(k)                  the INCUMBENT DEFAULT, provided so\n"
        "                                it can be reproduced exactly; a\n"
        "                                control, not a recommendation\n"
        "    There is NO menu of sampling schemes. You WRITE the strategy:\n"
        "      range(n//16, n, n//8)            evenly spread\n"
        "      range(0, n//2, n//16)            front-loaded\n"
        "      range(0,3) + range(n-3, n)       ends only\n"
        "      range(0, n, 4)                   every 4th\n"
        "      range(a, a+3) + range(b, b+3)    two clusters\n"
        "      [int(n * 0.7 ** i) for i in range(6)]  log-spaced\n"
        "    CONTENT-AWARE selection. Each source unit also carries neutral,\n"
        "    measured features you may read as per-unit lists (length n):\n"
        "      motion   temporal change vs the previous unit (video; 0 at p0)\n"
        "      edges    detail/text density        colorful   saturation\n"
        "      bright   mean luminance\n"
        "    Turn them into index sets with:\n"
        "      topk(feat, k)   indices of the k largest      bottomk(feat, k)\n"
        "      above(feat, t)  indices where feat >= t       below(feat, t)\n"
        "      order_by(feat)  all indices, ranked by feat descending\n"
        "    e.g. union(topk(motion, budget//2), uniform(budget//2)) picks the\n"
        "    busiest frames AND keeps even coverage. These are primitives for\n"
        "    USING a content signal, not named schemes; which signal, what k,\n"
        "    and the blend are yours to write.")
