"""Step vocabulary for model-authored operator programs.

DESIGN INTENT
-------------
These steps are deliberately LOW-LEVEL and general. The point is not to ship
a menu of finished ideas ("contact_sheet", "coordinate_grid") -- that is the
mistake this replaces. It is to provide primitives at roughly the level of
"convolve", "map a per-pixel curve", "tile N images", "draw a line", so that
recognisable techniques arise from COMPOSITION written by the optimizer:

  * a coordinate grid      = repeated `draw_lines` at computed spacing
  * set-of-mark numbering  = `draw_box` + `draw_text` per requested region
  * an orientation compass = `draw_lines` + `draw_text` in a margin
  * adaptive binarisation  = `local_mean` then `threshold_against`
  * a contact sheet        = `tile` with `label_tiles`
  * a temporal strip       = `tile(cols=N)` + `label_tiles(times)`
  * unsharp masking        = `blur` then `blend(mode="subtract")` then `blend`
  * a difference map       = `blend(mode="difference")` of two media

Every step is bounded, deterministic, and implemented here in ordinary code,
so no model-authored source is ever executed. The cost of that safety is
stated plainly in operators.py: a genuinely novel pixel algorithm that cannot
be decomposed into these steps is not expressible until the vocabulary grows.

CONTRACT
--------
Each step takes (images, params, ctx) and returns a list of images.
  * a "pixel" step maps N images to N images (applied elementwise)
  * a "reduce" step maps N images to 1 image
Params may be arithmetic expressions over neutral context; they are resolved
through `operators.eval_param` before the step runs.
"""
from __future__ import annotations

import numpy as np
from PIL import (Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter,
                 ImageOps)

from .operators import OperatorRejected, eval_param

# Bounds that keep a program from producing something unusable or enormous.
MAX_SIDE = 4096
MAX_DRAW_ITEMS = 400


def _f(params, key, ctx, default=None):
    v = params.get(key, default)
    return eval_param(v, ctx)


def _i(params, key, ctx, default=0):
    v = _f(params, key, ctx, default)
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        raise OperatorRejected(f"param {key!r} must be numeric, got {v!r}")


def _colour(params, key, default=(255, 0, 0)):
    v = params.get(key, default)
    if isinstance(v, (list, tuple)) and len(v) in (3, 4):
        return tuple(int(max(0, min(255, c))) for c in v[:3])
    if isinstance(v, str):
        named = {"red": (255, 0, 0), "green": (0, 200, 0),
                 "blue": (0, 80, 255), "yellow": (255, 210, 0),
                 "white": (255, 255, 255), "black": (0, 0, 0),
                 "magenta": (255, 0, 255), "cyan": (0, 220, 220),
                 "orange": (255, 140, 0)}
        if v.lower() in named:
            return named[v.lower()]
    return default


# ---------------------------------------------------------------------------
# tone / contrast


def st_autocontrast(imgs, p, ctx):
    """Per-channel histogram stretch. `cutoff` clips that % of each tail."""
    cut = max(0.0, min(40.0, float(_f(p, "cutoff", ctx, 1.0))))
    return [ImageOps.autocontrast(im.convert("RGB"), cutoff=cut)
            for im in imgs]


def st_curve(imgs, p, ctx):
    """Per-pixel tone curve: out = 255 * (in/255)^gamma, then gain/bias.

    One step covers gamma correction, brightening, darkening and linear
    stretch, which is why there is no separate `gamma` step.
    """
    g = float(_f(p, "gamma", ctx, 1.0))
    gain = float(_f(p, "gain", ctx, 1.0))
    bias = float(_f(p, "bias", ctx, 0.0))
    if not (0.05 <= g <= 20.0):
        raise OperatorRejected(f"gamma {g} outside [0.05, 20]")
    lut = [int(max(0, min(255, ((i / 255.0) ** g) * 255.0 * gain + bias)))
           for i in range(256)]
    out = []
    for im in imgs:
        im = im.convert("RGB")
        out.append(Image.merge("RGB", [c.point(lut) for c in im.split()]))
    return out


def st_saturate(imgs, p, ctx):
    """Colour saturation. 0 = greyscale, 1 = unchanged, >1 = boosted."""
    f = float(_f(p, "factor", ctx, 1.5))
    return [ImageEnhance.Color(im.convert("RGB")).enhance(f) for im in imgs]


def st_channel(imgs, p, ctx):
    """Isolate or drop a colour channel. `which` in R/G/B, `mode` keep/drop."""
    which = str(p.get("which", "R")).upper()[:1]
    mode = str(p.get("mode", "keep")).lower()
    idx = {"R": 0, "G": 1, "B": 2}.get(which)
    if idx is None:
        raise OperatorRejected(f"channel {which!r} must be R, G or B")
    out = []
    for im in imgs:
        ch = list(im.convert("RGB").split())
        if mode == "keep":
            g = ch[idx]
            out.append(Image.merge("RGB", [g, g, g]))
        else:
            ch[idx] = ch[idx].point(lambda _: 0)
            out.append(Image.merge("RGB", ch))
    return out


# ---------------------------------------------------------------------------
# spatial filters


def st_blur(imgs, p, ctx):
    r = float(_f(p, "radius", ctx, 2.0))
    if not (0.0 < r <= 64.0):
        raise OperatorRejected(f"blur radius {r} outside (0, 64]")
    return [im.convert("RGB").filter(ImageFilter.GaussianBlur(r))
            for im in imgs]


def st_sharpen(imgs, p, ctx):
    """Unsharp mask. Kept as one step because the PIL primitive is exactly
    this and decomposing it would only add rounding error."""
    r = float(_f(p, "radius", ctx, 2.0))
    pct = int(_i(p, "percent", ctx, 150))
    thr = int(_i(p, "threshold", ctx, 3))
    if not (0.0 < r <= 32.0) or not (0 < pct <= 500):
        raise OperatorRejected(f"sharpen radius/percent out of range: {r}/{pct}")
    return [im.convert("RGB").filter(
        ImageFilter.UnsharpMask(radius=r, percent=pct, threshold=thr))
        for im in imgs]


def st_edges(imgs, p, ctx):
    """Edge magnitude, optionally blended back over the original."""
    keep = float(_f(p, "blend", ctx, 0.0))
    out = []
    for im in imgs:
        im = im.convert("RGB")
        e = im.filter(ImageFilter.FIND_EDGES)
        out.append(Image.blend(e, im, max(0.0, min(1.0, keep))) if keep
                   else e)
    return out


def st_local_threshold(imgs, p, ctx):
    """Adaptive binarisation against a local mean.

    A GLOBAL threshold destroys documents with uneven lighting, which is the
    common case on OCRBench scans, so the comparison is against a box-filter
    local mean with an offset. `block` is the neighbourhood size in pixels
    and may be an expression such as "min(width, height) / 32".
    """
    block = max(3, _i(p, "block", ctx, 31))
    offset = float(_f(p, "offset", ctx, 8.0))
    out = []
    for im in imgs:
        g = np.asarray(im.convert("L"), dtype=np.float32)
        # box-filter local mean via a separable cumulative sum
        r = max(1, block // 2)
        pad = np.pad(g, r, mode="edge")
        cs = pad.cumsum(0).cumsum(1)
        cs = np.pad(cs, ((1, 0), (1, 0)))
        h, w = g.shape
        tot = (cs[block:block + h, block:block + w] - cs[0:h, block:block + w]
               - cs[block:block + h, 0:w] + cs[0:h, 0:w])
        mean = tot / float(block * block)
        bw = np.where(g > mean - offset, 255, 0).astype(np.uint8)
        out.append(Image.fromarray(bw).convert("RGB"))
    return out


# ---------------------------------------------------------------------------
# geometry


def st_resize(imgs, p, ctx):
    le = _i(p, "long_edge", ctx, ctx.get("long_edge", 1024))
    if not (16 <= le <= MAX_SIDE):
        raise OperatorRejected(f"long_edge {le} outside [16, {MAX_SIDE}]")
    out = []
    for im in imgs:
        w, h = im.size
        s = le / float(max(w, h))
        out.append(im.resize((max(1, round(w * s)), max(1, round(h * s))),
                             Image.LANCZOS))
    return out


def st_pad(imgs, p, ctx):
    """Add a margin. Gives label steps somewhere to draw without occluding."""
    for side in ("top", "bottom", "left", "right"):
        pass
    t, b = _i(p, "top", ctx, 0), _i(p, "bottom", ctx, 0)
    l, r = _i(p, "left", ctx, 0), _i(p, "right", ctx, 0)
    col = _colour(p, "colour", (255, 255, 255))
    out = []
    for im in imgs:
        im = im.convert("RGB")
        w, h = im.size
        nw, nh = w + l + r, h + t + b
        if max(nw, nh) > MAX_SIDE:
            raise OperatorRejected(f"padded size {nw}x{nh} exceeds {MAX_SIDE}")
        canvas = Image.new("RGB", (nw, nh), col)
        canvas.paste(im, (l, t))
        out.append(canvas)
    return out


def st_rotate(imgs, p, ctx):
    deg = float(_f(p, "degrees", ctx, 0.0))
    return [im.convert("RGB").rotate(deg, expand=True, fillcolor=(255,) * 3)
            for im in imgs]


# ---------------------------------------------------------------------------
# annotation. Coordinates are FRACTIONS of the frame, so a program is
# resolution independent.


def st_draw_lines(imgs, p, ctx):
    """Draw lines given as fractional coordinates, or a repeating ruling.

    `lines`: [[x0,y0,x1,y1], ...] in 0..1
    or `axis`: "x"|"y"|"both" with `spacing` (fraction) to rule the frame --
    which is how a program builds a coordinate grid without a grid step.
    """
    width = max(1, _i(p, "width", ctx, 2))
    col = _colour(p, "colour", (255, 0, 0))
    lines = p.get("lines")
    axis = str(p.get("axis", "")).lower()
    spacing = float(_f(p, "spacing", ctx, 0.1) or 0.1)
    if spacing <= 0 or spacing > 1:
        raise OperatorRejected(f"spacing {spacing} must be in (0, 1]")

    out = []
    for im in imgs:
        im = im.convert("RGB").copy()
        d = ImageDraw.Draw(im)
        w, h = im.size
        drawn = 0
        if isinstance(lines, (list, tuple)):
            for seg in list(lines)[:MAX_DRAW_ITEMS]:
                if not (isinstance(seg, (list, tuple)) and len(seg) == 4):
                    continue
                x0, y0, x1, y1 = [float(v) for v in seg]
                d.line([x0 * w, y0 * h, x1 * w, y1 * h], fill=col,
                       width=width)
                drawn += 1
        if axis in ("x", "both"):
            n = int(1.0 / spacing)
            for k in range(1, n):
                x = k * spacing * w
                d.line([x, 0, x, h], fill=col, width=width)
                drawn += 1
        if axis in ("y", "both"):
            n = int(1.0 / spacing)
            for k in range(1, n):
                y = k * spacing * h
                d.line([0, y, w, y], fill=col, width=width)
                drawn += 1
        if not drawn:
            raise OperatorRejected(
                "draw_lines drew nothing: give `lines` or `axis`")
        out.append(im)
    return out


def st_draw_box(imgs, p, ctx):
    """Outline fractional boxes. `boxes`: [[x0,y0,x1,y1], ...] in 0..1."""
    boxes = p.get("boxes") or []
    if not isinstance(boxes, (list, tuple)) or not boxes:
        raise OperatorRejected("draw_box needs a non-empty `boxes` list")
    width = max(1, _i(p, "width", ctx, 3))
    col = _colour(p, "colour", (255, 0, 0))
    out = []
    for im in imgs:
        im = im.convert("RGB").copy()
        d = ImageDraw.Draw(im)
        w, h = im.size
        for bx in list(boxes)[:MAX_DRAW_ITEMS]:
            if not (isinstance(bx, (list, tuple)) and len(bx) == 4):
                continue
            x0, y0, x1, y1 = [float(v) for v in bx]
            d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=col,
                        width=width)
        out.append(im)
    return out


def st_draw_text(imgs, p, ctx):
    """Burn text at fractional positions.

    `items`: [{"text": "...", "x": 0.02, "y": 0.02}, ...]
    Drawn on an opaque plate so it stays legible on any background, and a
    program can `pad` first to keep text out of the content area entirely.
    """
    items = p.get("items") or []
    if not isinstance(items, (list, tuple)) or not items:
        raise OperatorRejected("draw_text needs a non-empty `items` list")
    col = _colour(p, "colour", (0, 0, 0))
    plate = _colour(p, "plate", (255, 255, 255))
    scale = float(_f(p, "scale", ctx, 1.0))
    out = []
    for im in imgs:
        im = im.convert("RGB").copy()
        d = ImageDraw.Draw(im)
        w, h = im.size
        fs = max(10, int(min(w, h) * 0.035 * max(0.2, min(4.0, scale))))
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=fs)
        except Exception:                                    # noqa: BLE001
            font = None
        for it in list(items)[:MAX_DRAW_ITEMS]:
            if not isinstance(it, dict):
                continue
            txt = str(it.get("text", ""))[:120]
            if not txt:
                continue
            x = float(it.get("x", 0.02)) * w
            y = float(it.get("y", 0.02)) * h
            box = d.textbbox((x, y), txt, font=font)
            d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2],
                        fill=plate)
            d.text((x, y), txt, fill=col, font=font)
        out.append(im)
    return out


# ---------------------------------------------------------------------------
# reductions: several images into one


def st_tile(imgs, p, ctx):
    """Tile images into one frame. Reduce step: N images in, 1 out.

    This is the primitive behind contact sheets and temporal strips; the
    difference between them is `cols` and what `label_tiles` writes.
    """
    if not imgs:
        raise OperatorRejected("tile received no images")
    n = len(imgs)
    cols = _i(p, "cols", ctx, 0) or max(1, int(round(n ** 0.5)))
    cols = max(1, min(n, cols))
    rows = (n + cols - 1) // cols
    gap = max(0, _i(p, "gap", ctx, 6))
    bg = _colour(p, "background", (245, 245, 245))
    le = _i(p, "long_edge", ctx, ctx.get("long_edge", 1024))

    # size each tile so the finished sheet respects the budget
    tw = max(32, (le - gap * (cols + 1)) // cols)
    th = max(32, int(tw * 0.75))
    tiles = []
    for im in imgs:
        im = im.convert("RGB")
        w, h = im.size
        s = min(tw / w, th / h)
        tiles.append(im.resize((max(1, round(w * s)), max(1, round(h * s))),
                               Image.LANCZOS))
    cell_w = max(t.width for t in tiles)
    cell_h = max(t.height for t in tiles)
    W = gap + cols * (cell_w + gap)
    H = gap + rows * (cell_h + gap)
    if max(W, H) > MAX_SIDE:
        raise OperatorRejected(f"tiled size {W}x{H} exceeds {MAX_SIDE}")
    sheet = Image.new("RGB", (W, H), bg)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        x = gap + c * (cell_w + gap) + (cell_w - t.width) // 2
        y = gap + r * (cell_h + gap) + (cell_h - t.height) // 2
        sheet.paste(t, (x, y))
    return [sheet]


def st_label_tiles(imgs, p, ctx):
    """Number the tiles of a sheet produced by `tile`.

    Labels go in the gap between tiles, never over tile content, so a
    question about the imagery is never occluded. `times` draws the sample's
    own frame timestamps when the program asks for them.
    """
    if len(imgs) != 1:
        raise OperatorRejected("label_tiles expects one tiled image")
    cols = max(1, _i(p, "cols", ctx, 0) or 1)
    n = max(1, _i(p, "n", ctx, cols))
    use_times = bool(p.get("times", False))
    times = ctx.get("frame_times_s") or []
    col = _colour(p, "colour", (0, 0, 0))
    plate = _colour(p, "plate", (255, 235, 140))

    im = imgs[0].convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    rows = (n + cols - 1) // cols
    cw, ch = W / cols, H / rows
    fs = max(10, int(min(cw, ch) * 0.11))
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=fs)
    except Exception:                                        # noqa: BLE001
        font = None
    for k in range(n):
        r, c = divmod(k, cols)
        txt = str(k + 1)
        if use_times and k < len(times):
            txt = f"{k + 1}  t={times[k]:.1f}s"
        x, y = c * cw + 4, r * ch + 4
        box = d.textbbox((x, y), txt, font=font)
        d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2],
                    fill=plate)
        d.text((x, y), txt, fill=col, font=font)
    return [im]


def st_blend(imgs, p, ctx):
    """Combine images pairwise. Reduce step.

    `mode`: "difference" | "multiply" | "screen" | "add" | "subtract" |
            "alpha" (with `alpha`)
    A difference blend over a paired figure is how a program can make an
    altered detail visible; it is not hardcoded as a "difference map" step.
    """
    if len(imgs) < 2:
        raise OperatorRejected("blend needs at least two images")
    mode = str(p.get("mode", "difference")).lower()
    alpha = float(_f(p, "alpha", ctx, 0.5))
    base = imgs[0].convert("RGB")
    for nxt in imgs[1:]:
        nxt = nxt.convert("RGB").resize(base.size, Image.LANCZOS)
        if mode == "difference":
            base = ImageChops.difference(base, nxt)
        elif mode == "multiply":
            base = ImageChops.multiply(base, nxt)
        elif mode == "screen":
            base = ImageChops.screen(base, nxt)
        elif mode == "add":
            base = ImageChops.add(base, nxt)
        elif mode == "subtract":
            base = ImageChops.subtract(base, nxt)
        elif mode == "alpha":
            base = Image.blend(base, nxt, max(0.0, min(1.0, alpha)))
        else:
            raise OperatorRejected(f"unknown blend mode {mode!r}")
    return [base]


def st_pick(imgs, p, ctx):
    """Select a subset by index, so a program can choose which media to use."""
    idx = p.get("index")
    if isinstance(idx, (int, float)):
        idx = [int(idx)]
    if not isinstance(idx, (list, tuple)) or not idx:
        raise OperatorRejected("pick needs `index`: an int or list of ints")
    out = [imgs[int(i)] for i in idx if 0 <= int(i) < len(imgs)]
    if not out:
        raise OperatorRejected(f"pick selected nothing from {len(imgs)} images")
    return out


# ---------------------------------------------------------------------------
# registry


STEPS = {
    # tone
    "autocontrast": (st_autocontrast, "pixel"),
    "curve": (st_curve, "pixel"),
    "saturate": (st_saturate, "pixel"),
    "channel": (st_channel, "pixel"),
    # filters
    "blur": (st_blur, "pixel"),
    "sharpen": (st_sharpen, "pixel"),
    "edges": (st_edges, "pixel"),
    "local_threshold": (st_local_threshold, "pixel"),
    # geometry
    "resize": (st_resize, "pixel"),
    "pad": (st_pad, "pixel"),
    "rotate": (st_rotate, "pixel"),
    # annotation
    "draw_lines": (st_draw_lines, "pixel"),
    "draw_box": (st_draw_box, "pixel"),
    "draw_text": (st_draw_text, "pixel"),
    # reductions
    "tile": (st_tile, "reduce"),
    "label_tiles": (st_label_tiles, "pixel"),
    "blend": (st_blend, "reduce"),
    "pick": (st_pick, "reduce"),
}


def step_names() -> list[str]:
    return sorted(STEPS)


def validate_program(program: list, n_inputs: int, kind: str) -> None:
    """Type-check a program against its input arity BEFORE any gate pass.

    Caught a real authoring error on the first live operator: the optimizer
    put `blend` (a REDUCE step, needing >=2 images) inside a phi.view_ops
    program, which runs on one view at a time. Discovering that at render
    time wastes a full gate evaluation and reads like a harness fault; the
    arity is decidable from the program alone, so it is decided here and the
    error message tells the optimizer exactly what to fix.
    """
    from .operators import OperatorRejected
    n = int(n_inputs)
    for i, st in enumerate(program):
        name = str(st.get("op"))
        if name not in STEPS:
            raise OperatorRejected(
                f"step {i}: unknown op {name!r}; available: {step_names()}")
        _fn, skind = STEPS[name]
        if skind == "reduce":
            if n < 2 and name in ("blend",):
                raise OperatorRejected(
                    f"step {i} ({name}) is a REDUCE step needing >= 2 images, "
                    f"but this program receives {n}. phi.view_ops runs on ONE "
                    "view at a time; put reduce steps in psi.context_ops, "
                    "which receives the whole assembled set.")
            n = 1
    if kind == "pixel" and n != int(n_inputs):
        raise OperatorRejected(
            "a phi.view_ops program must return one image per input view; "
            f"this one reduces {n_inputs} to {n}. Use psi.context_ops for "
            "operators that combine views.")


def run_program(program: list, imgs: list, ctx: dict) -> list:
    """Execute a validated program over a list of PIL images."""
    from .operators import MAX_STEPS
    if len(program) > MAX_STEPS:
        raise OperatorRejected(f"program exceeds {MAX_STEPS} steps")
    cur = [im.convert("RGB") for im in imgs]
    for i, st in enumerate(program):
        if not isinstance(st, dict) or "op" not in st:
            raise OperatorRejected(f"step {i} has no 'op'")
        name = str(st["op"])
        if name not in STEPS:
            raise OperatorRejected(
                f"unknown step {name!r}; available: {step_names()}")
        fn, _kind = STEPS[name]
        params = {k: v for k, v in st.items() if k != "op"}
        try:
            cur = fn(cur, params, ctx)
        except OperatorRejected:
            raise
        except Exception as e:                               # noqa: BLE001
            raise OperatorRejected(
                f"step {i} ({name}) failed: {type(e).__name__}: {e}") from e
        if not cur:
            raise OperatorRejected(f"step {i} ({name}) produced no images")
        for im in cur:
            if max(im.size) > MAX_SIDE:
                raise OperatorRejected(
                    f"step {i} ({name}) produced {im.size}, over {MAX_SIDE}")
    return cur
