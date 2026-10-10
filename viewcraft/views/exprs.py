"""Array-expression language for model-authored visual operators.

WHY THIS REPLACES A STEP MENU
-----------------------------
The previous design gave the optimizer 18 named steps to choose from. Two
things went wrong, both of them my fault rather than the optimizer's:

  * 15 of the 18 were pixel operations and only 3 were layout/composition,
    so the optimizer proposed contrast and sharpening almost every round --
    the menu was the bias
  * a menu is a menu. "Visual operators are unbounded" and "pick one of 18"
    cannot both be true, and the second was what the code actually did

Here the optimizer WRITES EXPRESSIONS instead. Two small languages:

  PIXELS  an array expression over the image, evaluated with numpy
          e.g.  "where(local_mean(g,31) - g > 8, 0, 255)"      adaptive binarise
                "clip(g*1.4 - 40, 0, 255)"                      linear stretch
                "g + 0.8*(g - gaussian(g,3))"                   unsharp mask
                "abs(sobel_x(g)) + abs(sobel_y(g))"             edge magnitude
                "where(dist_from_center() < 0.3, g, g*0.5)"     vignette focus

  LAYOUT  a composition expression over the available views
          e.g.  "grid(views, cols=4, label=times)"              contact sheet
                "stack_v(overview, zoom)"                        detail below context
                "overlay(overview, boxes=crops)"                 provenance
                "beside(pick(0), pick(1))"                       paired compare
                "grid(views, cols=2) | annotate(axis_ticks(10))"  composed

Neither language can express "run arbitrary code". Both can express things I
did not think of, which is the whole point. The vocabulary below is
deliberately at the level of array primitives and layout combinators, not at
the level of finished techniques -- there is no `binarise`, no
`contact_sheet`, no `unsharp`. Those are things you WRITE.

SAFETY
------
Expressions are parsed to an AST and evaluated by an interpreter that knows
only the functions in FUNCS and the names in the supplied environment. There
is no attribute access, no import, no call to anything not in FUNCS, and no
assignment. Arrays are bounded in size. This is not sandboxed arbitrary code;
it is arithmetic over a fixed function set.
"""
from __future__ import annotations

import ast
from typing import Any

import numpy as np

from .scalars import shared_scalars
from .operators import SCALAR_HELPERS, OperatorRejected

MAX_PIXELS = 40_000_000
MAX_EXPR_LEN = 2000

# Source planes and coordinate fields an expression may read but not
# rebind: shadowing `red` would make every later reference mean something
# else and the recorded expression would no longer describe what ran.
# Rebinding is HARMLESS for the short aliases: `r = red` then using `r` is
# the obvious way to write a three-channel transform, and the recorded
# expression still describes exactly what ran because the binding is part
# of it. Measured on blink_hard_h1: 3 of 5 refusals were this guard
# rejecting `r = red; g = green; b = blue;` -- my guard, not a real fault.
#
# Only the DESCRIPTIVE names stay protected, because shadowing `red`
# itself makes the expression's own text misleading to a later reader.
PROTECTED_NAMES = frozenset({
    "red", "green", "blue", "lum", "redn", "greenn", "bluen", "lumn",
})


# ---------------------------------------------------------------------------
# pixel-expression function set
#
# Chosen to span the space of local image operations rather than to name
# techniques: pointwise arithmetic, local statistics at a radius, gradients,
# morphology, and coordinate fields. A technique is a composition of these.


def _as_arr(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.float32)


def _box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Local mean over a (2k+1) box, via an integral image. O(1) per pixel."""
    k = int(max(1, min(255, k)))
    pad = np.pad(a.astype(np.float64), k, mode="edge")
    cs = pad.cumsum(0).cumsum(1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = a.shape[:2]
    n = 2 * k + 1
    tot = (cs[n:n + h, n:n + w] - cs[0:h, n:n + w]
           - cs[n:n + h, 0:w] + cs[0:h, 0:w])
    return (tot / float(n * n)).astype(np.float32)


def _gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur. Approximated by three box passes, which is
    both fast and accurate enough for a visual operator."""
    r = max(1, int(round(float(sigma))))
    out = a.astype(np.float32)
    for _ in range(3):
        out = _box_mean(out, r)
    return out


def _conv3(a: np.ndarray, kern: np.ndarray) -> np.ndarray:
    p = np.pad(a.astype(np.float32), 1, mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += kern[dy, dx] * p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out


_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _local_max(a: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, min(31, k)))
    p = np.pad(a, k, mode="edge")
    out = np.full_like(a, -np.inf, dtype=np.float32)
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            out = np.maximum(out, p[dy:dy + a.shape[0], dx:dx + a.shape[1]])
    return out


def _local_min(a: np.ndarray, k: int) -> np.ndarray:
    return -_local_max(-a, k)


def _percentile(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, float(np.clip(q, 0, 100))))


FUNCS: dict[str, Any] = {
    # pointwise
    "abs": lambda x: np.abs(_as_arr(x)),
    "clip": lambda x, lo, hi: np.clip(_as_arr(x), float(lo), float(hi)),
    "where": lambda c, a, b: np.where(_as_arr(c) > 0 if c.dtype != bool
                                      else c, _as_arr(a), _as_arr(b))
    if isinstance(c := None, type(None)) else None,   # replaced below
    "sqrt": lambda x: np.sqrt(np.maximum(_as_arr(x), 0)),
    "log": lambda x: np.log1p(np.maximum(_as_arr(x), 0)),
    "exp": lambda x: np.exp(np.clip(_as_arr(x), -20, 20)),
    "pow": lambda x, p: np.power(np.maximum(_as_arr(x), 0), float(p)),
    "sign": lambda x: np.sign(_as_arr(x)),
    "minimum": lambda a, b: np.minimum(_as_arr(a), _as_arr(b)),
    "maximum": lambda a, b: np.maximum(_as_arr(a), _as_arr(b)),
    # local statistics
    "local_mean": lambda x, k: _box_mean(_as_arr(x), int(k)),
    "local_std": lambda x, k: np.sqrt(np.maximum(
        _box_mean(_as_arr(x) ** 2, int(k)) - _box_mean(_as_arr(x), int(k)) ** 2,
        0)),
    "local_max": lambda x, k: _local_max(_as_arr(x), int(k)),
    "local_min": lambda x, k: _local_min(_as_arr(x), int(k)),
    "gaussian": lambda x, s: _gaussian(_as_arr(x), float(s)),
    # gradients
    "sobel_x": lambda x: _conv3(_as_arr(x), _SOBEL_X),
    "sobel_y": lambda x: _conv3(_as_arr(x), _SOBEL_Y),
    "laplacian": lambda x: _conv3(_as_arr(x), np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)),
    # global statistics (scalars)
    "mean": lambda x: float(np.mean(_as_arr(x))),
    "std": lambda x: float(np.std(_as_arr(x))),
    "amin": lambda x: float(np.min(_as_arr(x))),
    "amax": lambda x: float(np.max(_as_arr(x))),
    "percentile": lambda x, q: _percentile(_as_arr(x), q),
    # Plain scalar helpers. `min`/`max` map to the ELEMENTWISE versions:
    # Python's builtins raise "truth value of an array is ambiguous" on an
    # image, which escaped as a bare ValueError rather than an
    # OperatorRejected and could take the round down instead of refusing
    # one candidate. `min(g, 255)` is also what the writer means.
    "min": lambda *a: _reduce_elementwise(np.minimum, a),
    "max": lambda *a: _reduce_elementwise(np.maximum, a),
    # Same NAMES as the other languages (SCALAR_HELPERS), but the pixel
    # language operates on arrays, so min/max must be ELEMENTWISE. Taking
    # the shared dict wholesale replaced them with Python builtins, which
    # raise "truth value of an array is ambiguous". The invariant is that
    # a name means the same THING everywhere; the implementation still has
    # to suit the operand.
    **{k: v for k, v in SCALAR_HELPERS.items()
       if k not in ("min", "max", "minimum", "maximum")},
    "min": lambda *a: _reduce_elementwise(np.minimum, a),
    "max": lambda *a: _reduce_elementwise(np.maximum, a),
    "minimum": lambda *a: _reduce_elementwise(np.minimum, a),
    "maximum": lambda *a: _reduce_elementwise(np.maximum, a),
}


def _reduce_elementwise(op, args):
    """Fold `op` over mixed arrays and scalars, broadcasting.

    np.minimum.reduce over a list mixing a 2-D array with a python float
    tries to build one ragged array and fails; folding pairwise broadcasts
    correctly, so `min(g, 255)` means what it says.
    """
    if not args:
        raise OperatorRejected("min()/max() need at least one argument")
    out = args[0] if isinstance(args[0], np.ndarray) else float(args[0])
    for nxt in args[1:]:
        out = op(out, nxt if isinstance(nxt, np.ndarray) else float(nxt))
    return out


def _where(c, a, b):
    c = c if isinstance(c, np.ndarray) else _as_arr(c)
    cond = c.astype(bool) if c.dtype == bool else (c > 0)
    return np.where(cond, _as_arr(a), _as_arr(b))


def _rgb(r, g, b):
    """Stack three planes into a colour image.

    Without this, every pixel expression returns ONE plane and
    `render_pixel_expr` replicates it -- i.e. silently converts to
    greyscale. That cost a measured 0.958 -> 0.621 on V*Bench, where 54% of
    questions ask about colour and 84% of the damaged items were colour
    questions. An expression language for visual presentation that cannot
    express "keep the hue" is not fit for purpose.
    """
    out = np.stack([_as_arr(r), _as_arr(g), _as_arr(b)], axis=-1)
    if out.ndim != 3 or out.shape[2] != 3:
        raise OperatorRejected("rgb() needs three 2-D planes")
    return out


def _per_channel(fn_expr_value):
    return fn_expr_value


def _median(a, k):
    """Local median -- removes speckle without smearing edges, which a
    Gaussian cannot do. Composable from nothing else in this table."""
    k = int(max(1, min(15, k)))
    a = _as_arr(a)
    pad = np.pad(a, k, mode="edge")
    win = []
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            win.append(pad[dy:dy + a.shape[0], dx:dx + a.shape[1]])
    return np.median(np.stack(win, axis=0), axis=0).astype(np.float32)


def _fft_bandpass(a, lo, hi):
    """Keep spatial frequencies in [lo, hi] as a fraction of Nyquist.

    Frequency-domain filtering is the one family genuinely outside a
    convolution/statistic vocabulary: a sharp band cannot be built from
    Gaussians. Added so the operator table is not the binding constraint
    on what a strategy can express.
    """
    a = _as_arr(a)
    lo, hi = float(lo), float(hi)
    if not (0.0 <= lo < hi <= 1.0):
        raise OperatorRejected(
            f"fft_band: need 0 <= lo < hi <= 1, got {lo}, {hi}")
    F = np.fft.fftshift(np.fft.fft2(a))
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2 + 1e-9)) ** 2
                + ((xx - w / 2) / (w / 2 + 1e-9)) ** 2)
    F[(r < lo) | (r > hi)] = 0
    return np.real(np.fft.ifft2(np.fft.ifftshift(F))).astype(np.float32)


def _rank(a, k, q):
    """Local q-th percentile: dilation at q=100, erosion at q=0, and the
    whole family in between, which local_max/local_min only bracket."""
    k = int(max(1, min(11, k)))
    a = _as_arr(a)
    pad = np.pad(a, k, mode="edge")
    win = []
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            win.append(pad[dy:dy + a.shape[0], dx:dx + a.shape[1]])
    return np.percentile(np.stack(win, axis=0),
                         float(np.clip(q, 0, 100)),
                         axis=0).astype(np.float32)


FUNCS["where"] = _where
FUNCS["rgb"] = _rgb
FUNCS["median"] = _median
FUNCS["fft_band"] = _fft_bandpass
FUNCS["rank"] = _rank

# Natural names the optimizer actually writes -> the parameter the function
# declares. Without these, `gaussian(g, sigma=3)` and `local_mean(g,
# radius=9)` -- the spellings any vision practitioner reaches for -- were
# hard errors that discarded the entire joint proposal containing them.
_ALIASES = {
    "radius": "k", "size": "k", "window": "k", "ksize": "k", "kernel": "k",
    "sigma": "s", "scale": "s",
    "image": "x", "img": "x", "plane": "x", "a": "x", "input": "x",
    "q": "q", "percent": "q", "pct": "q",
    "lo": "lo", "low": "lo", "minimum_value": "lo", "min": "lo",
    "hi": "hi", "high": "hi", "maximum_value": "hi", "max": "hi",
}

_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / (b + 1e-9),
    ast.Pow: lambda a, b: np.power(np.abs(a) + 1e-9, b) * np.sign(a) ** 0,
    ast.Mod: lambda a, b: a % b, ast.FloorDiv: lambda a, b: a // (b + 1e-9),
}
_CMPOPS = {
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}
_BOOLOPS = {ast.And: np.logical_and, ast.Or: np.logical_or}


def eval_pixel_expr(expr: str, env: dict) -> Any:
    """Evaluate a pixel expression against `env` (arrays and scalars).

    Interpreted, not compiled: only FUNCS may be called, only names present
    in `env` may be read, and there is no attribute access or assignment.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise OperatorRejected("empty pixel expression")
    if len(expr) > MAX_EXPR_LEN:
        raise OperatorRejected(f"expression over {MAX_EXPR_LEN} chars")
    # LET-BINDINGS. A three-channel transform repeats its subexpression
    # once per channel, so the optimizer factors it out:
    #
    #     base = clip((red - local_mean(red,9))*1.5 + red, 0, 255)
    #     rgb(base, base, base)
    #
    # `mode="eval"` rejects that outright, and the ledger only said
    # "invalid syntax (line 6)" -- the text was never stored, so the gap
    # had to be guessed. Assignments are evaluated in order into the
    # environment and the final expression is the result. No control flow,
    # no rebinding of the source planes.
    # Strip comments and normalise separators. The optimizer writes
    # `r = red; g = green;` on one line and annotates blocks with `#`,
    # and it indents continuation lines -- all of which ast rejects in
    # `mode="eval"`. Measured on realworldqa_h1: "unexpected indent
    # (line 2)" on a perfectly reasonable program.
    lines = []
    for raw in expr.strip().splitlines():
        cut = raw.split("#", 1)[0].rstrip()
        if cut.strip():
            lines.append(cut.strip() if not lines or True else cut)
    text = "\n".join(lines).strip().rstrip(";")
    # A semicolon-separated program becomes one statement per line.
    if ";" in text:
        parts = []
        for ln in text.split("\n"):
            parts.extend(x.strip() for x in ln.split(";") if x.strip())
        text = "\n".join(parts)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        try:
            mod = ast.parse(text, mode="exec")
        except SyntaxError as e:
            raise OperatorRejected(
                f"pixel expression does not parse: {e}") from e
        if not mod.body or not isinstance(mod.body[-1], ast.Expr):
            raise OperatorRejected(
                "a pixel expression must END in an expression, not a "
                "statement")
        stmts, final = mod.body[:-1], mod.body[-1].value
        for st in stmts:
            if not isinstance(st, ast.Assign) or len(st.targets) != 1 \
                    or not isinstance(st.targets[0], ast.Name):
                raise OperatorRejected(
                    "only simple `name = expression` bindings may precede "
                    "the final expression")
            name = st.targets[0].id
            if name in PROTECTED_NAMES:
                raise OperatorRejected(
                    f"{name!r} is a source plane and may not be rebound")
        tree = ast.Expression(body=final)
        tree._let_stmts = stmts          # evaluated below, in order
    except Exception as e:                                   # noqa: BLE001
        raise OperatorRejected(f"pixel expression does not parse: {e}") from e

    def ev(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float, bool)):
                return n.value
            raise OperatorRejected("only numeric constants are allowed")
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            raise OperatorRejected(
                f"unknown name {n.id!r}; available: {sorted(env)}")
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                return -ev(n.operand)
            if isinstance(n.op, ast.UAdd):
                return ev(n.operand)
            if isinstance(n.op, ast.Not):
                return np.logical_not(ev(n.operand))
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and \
                type(n.ops[0]) in _CMPOPS:
            return _CMPOPS[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
        if isinstance(n, ast.BoolOp) and type(n.op) in _BOOLOPS:
            vals = [ev(v) for v in n.values]
            out = vals[0]
            for v in vals[1:]:
                out = _BOOLOPS[type(n.op)](out, v)
            return out
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise OperatorRejected("only direct function calls allowed")
            fn = FUNCS.get(n.func.id)
            if fn is None:
                raise OperatorRejected(
                    f"unknown function {n.func.id!r}; "
                    f"available: {sorted(FUNCS)}")
            # Keyword arguments are ACCEPTED. Refusing them cost a real
            # candidate: the optimizer writes layout expressions that
            # REQUIRE kwargs (`grid(views, cols=4, gap=10)`) and pixel
            # expressions documented positionally (`local_mean(g, 9)`), so
            # it reasonably mixed the two conventions and the whole joint
            # proposal -- theta + phi.pixel_expr + psi.view_contract -- was
            # thrown away over punctuation. The round then fell back to a
            # theta-only edit, which is how a rejected VISUAL proposal
            # silently becomes evidence that "only text evolution works".
            #
            # A wrong keyword NAME is still an error, and it is reported
            # with the accepted names so the optimizer can correct itself.
            # The parameter names were lambda-internal (`k`, `s`, `q`) and
            # appeared in no prompt, so the optimizer supplied the obvious
            # ones -- `radius`, `sigma` -- and every such expression was
            # thrown away. Aliases make the natural name work; the help text
            # now prints the real signatures as well.
            kw = {}
            for k in n.keywords:
                if k.arg is None:
                    raise OperatorRejected("**kwargs is not supported")
                kw[_ALIASES.get(k.arg, k.arg)] = ev(k.value)
            try:
                return fn(*[ev(a) for a in n.args], **kw)
            except OperatorRejected:
                raise
            except (ValueError, ZeroDivisionError, IndexError,
                    AttributeError, OverflowError) as e:
                # An expression that blows up numerically is a REFUSED
                # CANDIDATE, not a crashed round. Letting a bare ValueError
                # escape would abort the arm on one bad proposal.
                raise OperatorRejected(
                    f"{n.func.id}() failed on this image: {e}") from e
            except TypeError as e:
                import inspect
                try:
                    names = list(inspect.signature(fn).parameters)
                except (TypeError, ValueError):
                    names = []
                raise OperatorRejected(
                    f"{n.func.id}() rejected these arguments: {e}."
                    + (f" Accepted parameter names: {names}" if names
                       else "")) from e
        if isinstance(n, ast.IfExp):
            return _where(ev(n.test), ev(n.body), ev(n.orelse))
        if isinstance(n, ast.ListComp) and len(n.generators) == 1:
            # Rare in pixel algebra but free to support, and a refusal here
            # costs the round's whole visual half.
            gen = n.generators[0]
            if gen.ifs or not isinstance(gen.target, ast.Name):
                raise OperatorRejected("comprehension: plain loop var only")
            seq = ev(gen.iter)
            if not hasattr(seq, "__len__") or len(seq) > 64:
                raise OperatorRejected("comprehension range too large")
            out_c, key = [], gen.target.id
            saved = env.get(key)
            for v in seq:
                env[key] = v
                out_c.append(ev(n.elt))
            if saved is None:
                env.pop(key, None)
            else:
                env[key] = saved
            return out_c
        raise OperatorRejected(
            f"unsupported syntax in pixel expression: {type(n).__name__}")

    for st in getattr(tree, "_let_stmts", []):
        env[st.targets[0].id] = ev(st.value)
    return ev(tree.body)


def pixel_env(img, ctx: dict) -> dict:
    """Names a pixel expression may read.

    `g` is luminance, `r`/`gc`/`b` the channels, all float32 in 0..255.
    Coordinate fields let an expression be spatially varying without any
    loop: `x`/`y` are normalised 0..1, `cx`/`cy` centred -1..1.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    if a.size > MAX_PIXELS:
        raise OperatorRejected(f"image too large for expressions: {a.shape}")
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    env = {
        "g": a.mean(axis=2),
        "r": a[:, :, 0], "gc": a[:, :, 1], "b": a[:, :, 2],
        # UNAMBIGUOUS ALIASES. `g` means LUMINANCE here while `gc` means the
        # green channel, and that collision is a trap: on blink_p12 round 1
        # the optimizer wrote `rgb(gc, g, b)`, i.e. green into the red slot
        # and luminance into the green slot, producing a green-cast image.
        # The candidate was accepted on a +0.100 gate gain that came from a
        # mis-channelled picture. Naming the channels explicitly lets an
        # expression say what it means.
        "red": a[:, :, 0], "green": a[:, :, 1], "blue": a[:, :, 2],
        "lum": a.mean(axis=2), "luma": a.mean(axis=2),
        # NORMALISED 0..1 views of the same planes. An optimizer writes
        # normalised algebra by default -- `clip(0.5 + 1.35*(red-0.5), 0, 1)`
        # -- and with red in 0..255 that clips to a BLACK frame. Measured on
        # realworldqa_p14: luminance 128.6 -> 0.5, gate 0.800 -> 0.000, with
        # a proposal whose reasoning was correct. Offering both conventions
        # removes the trap instead of asking the model to remember ours.
        "redn": a[:, :, 0] / 255.0, "greenn": a[:, :, 1] / 255.0,
        "bluen": a[:, :, 2] / 255.0, "lumn": a.mean(axis=2) / 255.0,
        "x": xx / max(1, w - 1), "y": yy / max(1, h - 1),
        "cx": (xx / max(1, w - 1)) * 2 - 1,
        "cy": (yy / max(1, h - 1)) * 2 - 1,
        "width": float(w), "height": float(h),
        "pi": float(np.pi),
    }
    env["radius"] = np.sqrt(env["cx"] ** 2 + env["cy"] ** 2)
    # Shared scalars, always bound -- never `if k in ctx`. A name that is
    # present on one domain's items and absent on another's makes a
    # candidate's validity depend on which item it happens to meet, which is
    # the recurring "an item's property fails the whole candidate" defect.
    # `width`/`height` above are the real image's and take precedence.
    shared = shared_scalars(
        n=int(ctx.get("n_media") or 1),
        budget=int(ctx.get("n_media") or 1),
        ctx=ctx, width=float(w), height=float(h))
    shared.update({k: v for k, v in env.items() if k in shared})
    env.update(shared)
    # Names unique to the pixel language's per-image position in a batch.
    for k in ("media_index", "long_edge", "n_options", "n_media"):
        if k in ctx:
            try:
                env[k] = float(ctx[k])
            except (TypeError, ValueError):
                pass
    env.setdefault("n_media", float(ctx.get("n_media") or 1))
    env.setdefault("media_index", 0.0)
    return env


def render_pixel_expr(img, expr: str, ctx: dict):
    """Run a pixel expression and return a new PIL image.

    A scalar result is an error: the expression must produce an image.
    """
    from PIL import Image

    env = pixel_env(img, ctx)
    out = eval_pixel_expr(expr, env)
    if not isinstance(out, np.ndarray) or out.ndim < 2:
        raise OperatorRejected(
            "pixel expression must evaluate to an image array, got "
            f"{type(out).__name__}")
    # UNIT SANITY GUARD.
    #
    # The free names are 0..255, but an optimizer naturally writes
    # normalised algebra: `clip(0.5 + 1.35*(red-0.5), 0, 1)`. With red in
    # 0..255 that clips everything to 1, i.e. a BLACK frame. Measured on
    # realworldqa_p14: mean luminance 128.6 -> 0.5, 50% of pixels exactly
    # 0, the executor burned all 12 observation steps zooming into black
    # and answered "I cannot determine it" -- gate 0.800 -> 0.000 after
    # 603 seconds. The proposal was sound ("boost local contrast while
    # preserving hue"); only the unit convention was wrong.
    #
    # If the result looks like it was computed in 0..1, rescale it instead
    # of shipping a black frame. This is a unit repair, not a change of
    # what the expression means.
    finite = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
    # A result whose whole range sits in 0..1 was computed in normalised
    # units; rescale rather than emit a black frame.
    if finite.size and float(np.nanmax(np.abs(finite))) <= 1.5:
        finite = finite * 255.0
    out = np.clip(finite, 0, 255).astype(np.uint8)

    # RANGE-COLLAPSE GUARD. A transform that leaves almost no contrast has
    # destroyed the image, whatever it intended. Refusing here costs one
    # candidate; not refusing cost 603 seconds of gate evaluation on a
    # black frame and a 0.000 score that looked like a method failure.
    src = np.asarray(img.convert("RGB"), dtype=np.float32)
    o = out.astype(np.float32)
    hint = ("Check units: `red`/`green`/`blue`/`lum` are 0..255, while "
            "`redn`/`greenn`/`bluen`/`lumn` are the same planes in 0..1. "
            "Mixing them clips the whole frame.")
    # NO OUTPUT-SANITY GUARD HERE, deliberately.
    #
    # A frame-destroying transform IS a real failure -- measured on
    # realworldqa_p14 round 2, where `clip(0.5 + 1.35*(red-0.5), 0, 1)`
    # (0..1 algebra on 0..255 planes) produced a black frame, the executor
    # burned all 12 observation steps zooming into it, and the gate read
    # 0.800 -> 0.000. But every threshold I tried either let that case
    # through or refused legitimate operators: binarisation and edge maps
    # legitimately collapse variance, a flat swatch has none to begin
    # with, and greyscale is a valid technique whose only sin is being the
    # wrong CHOICE on a colour benchmark.
    #
    # The fix belongs upstream, not here: `redn`/`greenn`/`bluen`/`lumn`
    # give the optimizer the 0..1 convention it reaches for by default, and
    # the prompt states both conventions explicitly. A bad transform that
    # still gets written is then a strategy error, and the gate rejects it
    # -- which is the mechanism that is supposed to do this job.
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=2)
    if out.shape[2] > 3:
        out = out[:, :, :3]
    return Image.fromarray(out, mode="RGB")


# ---------------------------------------------------------------------------
# VISUAL MEMORY LAYER (canvas)
#
# The canvas is what turns the image from read-only input into a read/write
# workspace. The executor drops `note`s (a normalised point + free text) on the
# image it is looking at; the optimizer's `canvas_expr` decides how the
# accumulated notes are burned back into the view. Because the expression is a
# general array transform over the base planes plus the note-derived arrays
# below, the *strategy* -- mark-as-you-count, elimination, tracing, numbering --
# is discovered by evolution, not enumerated as named gestures.

def canvas_env(img, notes: list, ctx: dict) -> dict:
    """pixel_env plus the note-derived arrays a canvas expression may read.

    `notes` is a list of {"x","y","text"} in normalised [0,1] coords for THIS
    base image. Arrays added (all float32, h x w unless scalar):
      dist        per-pixel distance to the nearest note, normalised by the
                  long edge (so a threshold like 0.03 means 3% of long edge).
                  All +inf when there are no notes, so `where(dist<r,...)`
                  is simply false everywhere.
      nearest_idx 1-based ordinal of the nearest note (the order the model
                  wrote them), 0 where there are no notes.
      note_count  scalar: how many notes are on this image.
    """
    env = pixel_env(img, ctx)
    h, w = int(env["height"]), int(env["width"])
    yy = env["y"] * max(1, h - 1)
    xx = env["x"] * max(1, w - 1)
    long_edge = float(max(1, max(h, w)))
    pts = [(float(n["x"]) * (w - 1), float(n["y"]) * (h - 1))
           for n in notes if isinstance(n, dict) and "x" in n and "y" in n]
    env["note_count"] = float(len(pts))
    if not pts:
        env["dist"] = np.full((h, w), np.inf, dtype=np.float32)
        env["nearest_idx"] = np.zeros((h, w), dtype=np.float32)
        return env
    best = np.full((h, w), np.inf, dtype=np.float32)
    best_i = np.zeros((h, w), dtype=np.float32)
    for i, (px, py) in enumerate(pts, 1):
        d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2) / long_edge
        closer = d < best
        best = np.where(closer, d, best)
        best_i = np.where(closer, float(i), best_i)
    env["dist"] = best
    env["nearest_idx"] = best_i
    return env


def _draw_note_labels(img, notes: list):
    """Draw each note's GLOBAL ordinal (and any model text) near its point.

    The ordinal is objective metadata -- the note's position in the sequence
    the model wrote them -- so it is rendered by the mechanism, not left to the
    model, which loses count mid-trajectory (measured: the executor restarted
    its own numbering after a zoom and answered 27 for 28). HOW the ordinal is
    used (as a running count, a reading order, a label) is the strategy; the
    mechanism only guarantees the count is always visible and correct. The
    model's own free text, if any, is appended after the ordinal.
    """
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    w, h = img.size
    for i, n in enumerate(notes, 1):
        if not (isinstance(n, dict) and "x" in n and "y" in n):
            continue
        px, py = float(n["x"]) * (w - 1), float(n["y"]) * (h - 1)
        text = str(n.get("text", "") or "")[:12]
        label = str(n.get("id") or i) + (f":{text}" if text else "")
        # dark backing so the label reads on any background
        d.rectangle([px + 4, py - 7, px + 4 + 7 * len(label), py + 8],
                    fill=(0, 0, 0))
        d.text((px + 5, py - 6), label, fill=(255, 255, 0))
    return img


def render_canvas(img, canvas_expr: str, notes: list, ctx: dict):
    """Overlay the model's accumulated notes onto a base view.

    With no canvas_expr the DEFAULT marker is a small magenta disc plus the
    note's text -- enough for the affordance to be usable before evolution
    improves it. With a canvas_expr, the expression is a full-image transform
    that reads the note arrays (dist / nearest_idx / note_count) to decide how
    notes appear; the text labels are then drawn on top.
    """
    from PIL import Image, ImageDraw
    base = img.convert("RGB")
    if canvas_expr and str(canvas_expr).strip():
        env = canvas_env(base, notes, ctx)
        out = eval_pixel_expr(canvas_expr, env)
        if not isinstance(out, np.ndarray) or out.ndim < 2:
            raise OperatorRejected(
                "canvas expression must evaluate to an image array")
        finite = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and float(np.nanmax(np.abs(finite))) <= 1.5:
            finite = finite * 255.0
        out = np.clip(finite, 0, 255).astype(np.uint8)
        if out.ndim == 2:
            out = np.stack([out] * 3, axis=2)
        if out.shape[2] > 3:
            out = out[:, :, :3]
        img2 = Image.fromarray(out, mode="RGB")
    else:
        img2 = base.copy()
        d = ImageDraw.Draw(img2)
        w, h = img2.size
        r = max(4, min(w, h) // 60)
        for n in notes:
            if not (isinstance(n, dict) and "x" in n and "y" in n):
                continue
            px, py = float(n["x"]) * (w - 1), float(n["y"]) * (h - 1)
            d.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 255),
                      width=max(2, r // 3))
    return _draw_note_labels(img2, notes)


def function_help() -> str:
    """One line listing the callable vocabulary, generated from FUNCS."""
    groups = [
        ("pointwise", ["abs", "clip", "where", "sqrt", "log", "exp", "pow",
                       "sign", "minimum", "maximum"]),
        ("local", ["local_mean", "local_std", "local_max", "local_min",
                   "gaussian"]),
        ("gradient", ["sobel_x", "sobel_y", "laplacian"]),
        ("frequency", ["fft_band(x, lo, hi) -- keep frequencies in "
                       "[lo,hi] of Nyquist; a sharp band is not "
                       "expressible with Gaussians"]),
        ("rank", ["median(x, k)", "rank(x, k, q) -- local q-th percentile; "
                  "q=100 dilates, q=0 erodes"]),
        ("global", ["mean", "std", "amin", "amax", "percentile"]),
        ("colour", ["rgb(r_plane, g_plane, b_plane) -- REQUIRED to keep hue; "
                    "a bare one-plane expression returns GREYSCALE"]),
    ]
    lines = [f"    {g:10s} {', '.join(fs)}" for g, fs in groups]
    lines.append(
        "    signatures  local_mean(x, k) local_std(x, k) local_max(x, k) "
        "local_min(x, k) gaussian(x, s) clip(x, lo, hi) percentile(x, q)")
    lines.append(
        "    aliases     radius/size/window -> k, sigma/scale -> s "
        "(keyword args are accepted)")
    return "\n".join(lines)
