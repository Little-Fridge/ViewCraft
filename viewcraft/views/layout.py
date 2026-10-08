"""Layout-expression language: how the visual CONTEXT is organised.

WHY THIS EXISTS AS ITS OWN LANGUAGE
-----------------------------------
The step menu it replaces had 15 pixel operations and 3 composition steps,
so the optimizer proposed contrast and sharpening nearly every round. That
imbalance was mine, not the optimizer's, and it starved exactly the half of
the mechanism the study is about: a video sample was eight separate
unlabelled images, and the only "layout" knobs were how many to keep and in
what order. That is a retention policy, not a visual-context organisation.

Here layout is WRITTEN, as a composition expression over the available views:

    grid(views, cols=4, label=times)          a labelled contact sheet
    stack_v(pick(0), grid(crops, cols=2))     context above its details
    beside(pick(0), pick(1))                  a paired comparison
    overlay(pick(0), boxes=crops)             provenance drawn on context
    grid(views, cols=2, gap=24, label=index)  wider gutters, numbered

Combinators, not finished layouts. `contact_sheet` is not a function here --
it is what `grid(views, cols=4, label=times)` means. Anything expressible by
composing these is available whether or not I anticipated it.
"""
from __future__ import annotations

import ast
from typing import Any

from PIL import Image, ImageDraw

from .operators import OperatorRejected
from .scalars import SHARED_SCALAR_NAMES, shared_scalars

MAX_SIDE = 4096
MAX_EXPR_LEN = 1200


def _fit_into(im: Image.Image, w: int, h: int) -> Image.Image:
    s = min(w / max(1, im.width), h / max(1, im.height))
    return im.resize((max(1, round(im.width * s)),
                      max(1, round(im.height * s))), Image.LANCZOS)


def _label_text(kind, i, ctx):
    """What a tile's caption says. `kind` is chosen by the expression."""
    if kind == "times":
        ts = ctx.get("frame_times_s") or []
        return f"{i + 1}  t={ts[i]:.1f}s" if i < len(ts) else str(i + 1)
    if kind == "index":
        return str(i + 1)
    if kind == "letter":
        return chr(ord("A") + i)
    return ""


def _draw_caption(canvas, xy, txt, fs):
    if not txt:
        return
    d = ImageDraw.Draw(canvas)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=fs)
    except Exception:                                        # noqa: BLE001
        font = None
    box = d.textbbox(xy, txt, font=font)
    d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2],
                fill=(255, 235, 140))
    d.text(xy, txt, fill=(0, 0, 0), font=font)


# ---------------------------------------------------------------------------
# combinators. Each takes and returns a list of PIL images (usually length 1).


def L_grid(imgs, ctx, cols=None, gap=8, label="", long_edge=None, bg=None):
    """Tile images into one frame, captions drawn in the GUTTER.

    Captions go in the gap, never over tile content, so a question about the
    imagery is never occluded by its own label.

    `long_edge` is the SHEET's long edge, and the per-tile resolution is what
    actually decides legibility. Writing `grid(views, cols=4,
    long_edge=640)` on a domain whose frames are already 640x480 shrinks each
    tile to ~160x140 -- a tenth of the source area. That is what happened on
    VSI-Bench: gate 0.700 -> 0.406, seven items broken. The optimizer had no
    way to know the frames were already 640 wide, so the default now SCALES
    WITH the input rather than defaulting to a single view's budget, and a
    sheet that would render tiles below `min_tile_px` is refused with a
    message naming the arithmetic.
    """
    if not imgs:
        raise OperatorRejected("grid received no images")
    n = len(imgs)
    cols = int(cols or max(1, round(n ** 0.5)))
    cols = max(1, min(n, cols))
    rows = (n + cols - 1) // cols
    gap = max(0, int(gap))

    # Default the sheet budget to "keep each tile near its source size",
    # not to one view's long edge.
    src_long = max(max(im.size) for im in imgs)
    if long_edge is None:
        le = int(min(MAX_SIDE, cols * src_long + gap * (cols + 1)))
    else:
        le = int(long_edge)

    bg = tuple(bg) if isinstance(bg, (list, tuple)) else (245, 245, 245)
    cap = max(0, int(le * 0.022)) if label else 0
    tw = max(32, (le - gap * (cols + 1)) // cols)
    th = max(32, int(tw * 0.78))

    # A tile far smaller than the source is a resolution loss disguised as
    # a layout change. RAISE long_edge to fix it rather than refuse the
    # candidate: how large the sources are is a property of THIS ITEM, and
    # a 640px-source domain and a 2048px-source domain need different
    # sheets for the same `cols`. Refusing made the optimizer's choice of
    # `cols` depend on data it cannot see, and cost 4 candidates on
    # G1 alone.
    #
    # Same rule as every other guard here: repair locally when the repair
    # is well-defined, and only refuse when it is not. The cap at
    # MAX_SIDE means a sheet can still be too small if cols is extreme --
    # that IS the optimizer's error and is still refused below.
    # An absolute tile floor, deliberately NOT a fraction of the source.
    #
    # I replaced this with `min_tile >= 0.6 * src_long` on the theory that
    # what costs accuracy is the FRACTION of each source's pixels that
    # survives a composition -- VSI's contact sheet destroyed 86% of the
    # imagery and its gate fell 0.466 -> 0.378, which looked conclusive.
    #
    # It is not. Across VSI's 11 gate configs the correlation between pixels
    # retained and gate score is -0.06 (+0.08 excluding the abstention
    # collapse at e021e4f6). A candidate retaining 129% of baseline's pixels
    # scored 0.305; one retaining 43% scored 0.378. Resolution loss is real
    # and worth repairing when extreme, but it does not explain why the
    # visual candidates were rejected, so the floor stays absolute rather
    # than being tuned on a correlation that is not there.
    min_tile = int(ctx.get("min_tile_px") or 220)
    if tw < min_tile and tw < src_long * 0.5:
        want = int(min(MAX_SIDE, min(src_long, min_tile) * cols
                       + gap * (cols + 1)))
        if want > le:
            le = want
            cap = max(0, int(le * 0.022)) if label else 0
            tw = max(32, (le - gap * (cols + 1)) // cols)
            th = max(32, int(tw * 0.78))
    if tw < 64 and tw < src_long * 0.25:
        raise OperatorRejected(
            f"grid(cols={cols}) cannot fit within {MAX_SIDE}px: each tile "
            f"would be ~{tw}px from {src_long}px sources. Use fewer "
            "columns.")

    tiles = [_fit_into(im.convert("RGB"), tw, th) for im in imgs]
    cw = max(t.width for t in tiles)
    ch = max(t.height for t in tiles) + cap
    W = gap + cols * (cw + gap)
    H = gap + rows * (ch + gap)
    if max(W, H) > MAX_SIDE:
        # SHRINK THE TILES, do not fail the unit. How many source units an
        # item has is a property of THAT ITEM: `grid(views, cols=2)` fits
        # an 8-page document and overflows a 40-page one. Raising here
        # killed 11 of 64 mmlongdoc units and voided the round as
        # incomparable (64 vs 53) after 1417 seconds -- the same scope
        # error as the per-item inertness raise: an item property failing
        # the candidate.
        for _ in range(6):
            if max(W, H) <= MAX_SIDE:
                break
            sc = (MAX_SIDE / float(max(W, H))) * 0.98
            tiles = [t.resize((max(1, int(t.width * sc)),
                               max(1, int(t.height * sc)))) for t in tiles]
            cw = max(t.width for t in tiles)
            ch = max(t.height for t in tiles) + cap
            W = gap + cols * (cw + gap)
            H = gap + rows * (ch + gap)
    sheet = Image.new("RGB", (W, H), bg)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        x = gap + c * (cw + gap)
        y = gap + r * (ch + gap)
        if cap:
            _draw_caption(sheet, (x + 2, y), _label_text(label, k, ctx),
                          max(9, int(cap * 0.85)))
        sheet.paste(t, (x + (cw - t.width) // 2, y + cap))
    return [sheet]


def L_stack_v(imgs, ctx, gap=8, label="", long_edge=None):
    """Vertical stack: context above detail, or a caption column."""
    if not imgs:
        raise OperatorRejected("stack_v received no images")
    le = int(long_edge or ctx.get("long_edge") or 1024)
    gap = max(0, int(gap))
    scaled = [_fit_into(im.convert("RGB"), le, le) for im in imgs]
    W = max(i.width for i in scaled)
    cap = max(0, int(le * 0.022)) if label else 0
    H = sum(i.height + cap for i in scaled) + gap * (len(scaled) + 1)
    if max(W, H) > MAX_SIDE:
        # Shrink to fit rather than fail the unit; see the note in L_grid.
        for _ in range(6):
            if max(W, H) <= MAX_SIDE:
                break
            sc = (MAX_SIDE / float(max(W, H))) * 0.98
            scaled = [i.resize((max(1, int(i.width * sc)),
                                max(1, int(i.height * sc)))) for i in scaled]
            W = max(i.width for i in scaled)
            H = sum(i.height + cap for i in scaled) + gap * (len(scaled) + 1)
    out = Image.new("RGB", (W, H), (245, 245, 245))
    y = gap
    for k, im in enumerate(scaled):
        if cap:
            _draw_caption(out, (4, y), _label_text(label, k, ctx),
                          max(9, int(cap * 0.85)))
        out.paste(im, ((W - im.width) // 2, y + cap))
        y += im.height + cap + gap
    return [out]


def L_beside(imgs, ctx, gap=10, label="", long_edge=None, divider=True):
    """Place images side by side with an optional divider.

    A paired figure presented as ONE composed image is a genuinely different
    visual context from two separate images, which is what HallusionBench's
    paired design needs.
    """
    if not imgs:
        raise OperatorRejected("beside received no images")
    le = int(long_edge or ctx.get("long_edge") or 1024)
    gap = max(0, int(gap))
    per = max(64, (le - gap * (len(imgs) + 1)) // max(1, len(imgs)))
    scaled = [_fit_into(im.convert("RGB"), per, le) for im in imgs]
    cap = max(0, int(le * 0.022)) if label else 0
    W = sum(i.width for i in scaled) + gap * (len(scaled) + 1)
    H = max(i.height for i in scaled) + cap
    if max(W, H) > MAX_SIDE:
        # Shrink to fit rather than fail the unit; see the note in L_grid.
        for _ in range(6):
            if max(W, H) <= MAX_SIDE:
                break
            sc = (MAX_SIDE / float(max(W, H))) * 0.98
            scaled = [i.resize((max(1, int(i.width * sc)),
                                max(1, int(i.height * sc)))) for i in scaled]
            W = sum(i.width for i in scaled) + gap * (len(scaled) + 1)
            H = max(i.height for i in scaled) + cap
    out = Image.new("RGB", (W, H), (245, 245, 245))
    x = gap
    for k, im in enumerate(scaled):
        if cap:
            _draw_caption(out, (x + 2, 0), _label_text(label or "letter",
                                                       k, ctx),
                          max(9, int(cap * 0.85)))
        out.paste(im, (x, cap))
        if divider and k < len(scaled) - 1:
            d = ImageDraw.Draw(out)
            xd = x + im.width + gap // 2
            d.line([xd, 0, xd, H], fill=(120, 120, 120), width=2)
        x += im.width + gap
    return [out]


def L_overlay(imgs, ctx, boxes=None, colour=None, width=3):
    """Draw fractional boxes on the first image, e.g. crop provenance."""
    if not imgs:
        raise OperatorRejected("overlay received no images")
    base = imgs[0].convert("RGB").copy()
    col = tuple(colour) if isinstance(colour, (list, tuple)) else (255, 0, 0)
    regs = boxes if isinstance(boxes, (list, tuple)) else \
        (ctx.get("crop_regions") or [])
    if not regs:
        # DEGRADE, do not fail the unit. Whether the executor zoomed is a
        # property of THIS ITEM; on a single-image domain it usually has
        # not, so `crops` is empty and `overlay(views, boxes=crops)` has
        # nothing to draw. Returning the base image unchanged lets the
        # enclosing composition still render -- the same fix already
        # applied to grid/stack_v/pick, missed here.
        return [base]
    d = ImageDraw.Draw(base)
    w, h = base.size
    for i, bx in enumerate(list(regs)[:200]):
        if not (isinstance(bx, (list, tuple)) and len(bx) == 4):
            continue
        x0, y0, x1, y1 = [float(v) for v in bx]
        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=col,
                    width=int(width))
        _draw_caption(base, (x0 * w + 3, max(0, y0 * h - 16)), str(i + 1),
                      max(10, int(min(w, h) * 0.025)))
    return [base] + list(imgs[1:])


def L_pick(imgs, ctx, *idx):
    """Select views by index, so an expression can choose what to compose."""
    want = []
    for i in idx:
        if isinstance(i, (list, tuple)):
            want.extend(int(v) for v in i)
        else:
            want.append(int(i))
    out = [imgs[i] for i in want if 0 <= i < len(imgs)]
    if not out:
        # `pick(views, 1)` is out of range on a single-image item, and how
        # many images an item has is a property of THAT ITEM. Clamping to
        # the first image keeps the composition alive; raising killed the
        # candidate on every single-image sample (observed on
        # blink_hard_h1). This is the same degrade-don't-fail rule already
        # applied to grid / stack_v / overlay / empty crops.
        out = imgs[:1]
    return out


def L_pad(imgs, ctx, top=0, bottom=0, left=0, right=0, colour=None):
    """Add a margin, giving captions or notes room outside the content."""
    col = tuple(colour) if isinstance(colour, (list, tuple)) else (255,) * 3
    out = []
    for im in imgs:
        im = im.convert("RGB")
        nw = im.width + int(left) + int(right)
        nh = im.height + int(top) + int(bottom)
        if max(nw, nh) > MAX_SIDE:
            # SHRINK THE CONTENT to make room, do not fail the unit. How
            # tall an image already is is a property of THIS ITEM: the same
            # `pad(x, top=10)` fits a 600px page and overflows a 4090px
            # composite. Refusing made the optimizer's padding choice
            # depend on data it cannot see -- the same scope error already
            # fixed for grid, stack_v, beside, overlay, pick and crop.
            room = MAX_SIDE - max(int(left) + int(right),
                                  int(top) + int(bottom))
            if room < 64:
                raise OperatorRejected(
                    f"pad({top},{bottom},{left},{right}) leaves no room "
                    f"inside {MAX_SIDE}px")
            sc = room / float(max(im.width, im.height))
            im = im.resize((max(1, int(im.width * sc)),
                            max(1, int(im.height * sc))))
            nw = im.width + int(left) + int(right)
            nh = im.height + int(top) + int(bottom)
        c = Image.new("RGB", (nw, nh), col)
        c.paste(im, (int(left), int(top)))
        out.append(c)
    return out


def L_resize(imgs, ctx, long_edge=None):
    le = int(long_edge or ctx.get("long_edge") or 1024)
    return [_fit_into(im.convert("RGB"), le, le) for im in imgs]


def L_crop(imgs, ctx, x0=0.0, y0=0.0, x1=1.0, y1=1.0, region=None,
           box=None):
    """Crop every image to a normalised rectangle.

    The optimizer asked for this on DUDE: tile each page's header and
    footer bands beside the full page, which is a real document-reading
    strategy (titles and totals live at the edges). The layout language
    could compose and place views but never SELECT A REGION of one, so
    the whole candidate was refused for a missing primitive rather than a
    mistake.
    """
    # `region=(x0,y0,x1,y1)` and `box=[...]` are the spellings the
    # optimizer reaches for -- a rectangle is one argument in its head,
    # not four. Accepting both costs nothing and avoids refusing a
    # candidate over argument shape (observed on mpdocvqa_H2).
    rect = region if region is not None else box
    if rect is not None:
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            raise OperatorRejected(
                "crop: region/box must be (x0, y0, x1, y1)")
        x0, y0, x1, y1 = rect
    x0, y0, x1, y1 = (float(x0), float(y0), float(x1), float(y1))
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise OperatorRejected(
            f"crop: need 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got "
            f"({x0}, {y0}, {x1}, {y1})")
    out = []
    for im in imgs:
        w, h = im.size
        box = (int(x0 * w), int(y0 * h),
               max(int(x0 * w) + 1, int(x1 * w)),
               max(int(y0 * h) + 1, int(y1 * h)))
        out.append(im.convert("RGB").crop(box))
    return out


LAYOUT_FUNCS = {
    "crop": L_crop,
    "grid": L_grid, "stack_v": L_stack_v, "beside": L_beside,
    "overlay": L_overlay, "pick": L_pick, "pad": L_pad, "resize": L_resize,
}


def eval_layout_expr(expr: str, views: list, ctx: dict) -> list:
    """Evaluate a layout expression. `views` and `crops` are the free names."""
    if not isinstance(expr, str) or not expr.strip():
        raise OperatorRejected("empty layout expression")
    if len(expr) > MAX_EXPR_LEN:
        raise OperatorRejected(f"layout expression over {MAX_EXPR_LEN} chars")
    # LET-BINDINGS, same as the pixel language. A multi-panel layout names
    # its panels before assembling them --
    #     main = pick(views, 0)
    #     ctx  = grid(pick(views,0,1,2,3), cols=2)
    #     stack_v(overlay(main, boxes=crops), ctx)
    # -- and `mode="eval"` rejects that outright. The pixel language got
    # bindings; the layout language did not, so the same capability
    # existed in one half of the interface and not the other. Observed on
    # nextgqa_H2 and mpdocvqa_H2: three candidates lost to it.
    text = expr.strip().rstrip(";")
    let_stmts = []
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        try:
            mod = ast.parse(text, mode="exec")
        except SyntaxError as e:
            raise OperatorRejected(
                f"layout expression does not parse: {e}") from e
        if not mod.body or not isinstance(mod.body[-1], ast.Expr):
            raise OperatorRejected(
                "a layout expression must END in an expression")
        for st in mod.body[:-1]:
            if not isinstance(st, ast.Assign) or len(st.targets) != 1 \
                    or not isinstance(st.targets[0], ast.Name):
                raise OperatorRejected(
                    "only simple `name = expression` bindings may precede "
                    "the final layout expression")
            if st.targets[0].id in ("views", "crops"):
                raise OperatorRejected(
                    f"{st.targets[0].id!r} is a source list and may not be "
                    "rebound")
        let_stmts = mod.body[:-1]
        tree = ast.Expression(body=mod.body[-1].value)

    names = {
        "views": list(views),
        "crops": list(ctx.get("crop_views") or []),
        "times": "times", "index": "index", "letter": "letter",
        # Ways of writing "no label". The optimizer reaches for `none`,
        # `None` or `off`, and rejecting them discarded the WHOLE joint
        # proposal over a keyword value -- observed on blink_ev7 rounds 1
        # and 2, where both attempts died on `unknown name 'none'` and the
        # round fell back to a theta-only edit. A visual proposal must not
        # be turned into a text-only one by a vocabulary gap.
        "none": "", "None": "", "null": "", "off": "", "no": "",
        # Booleans, for the same reason: `divider=False` is natural.
        "True": True, "False": False, "true": True, "false": False,
        # The shared scalar namespace, bound here for the same reason:
        # chartqa round 1 lost BOTH attempts to `unknown name 'height'`,
        # a name the PIXEL language binds and this one did not. Asking for
        # the source height while composing a layout is at least as natural
        # as asking for it while transforming pixels. See views/scalars.py.
        **shared_scalars(
            n=len(views) or int(ctx.get("n_media") or 0),
            budget=int(ctx.get("n_media") or len(views) or 0),
            ctx=ctx,
            width=max((im.width for im in views), default=0),
            height=max((im.height for im in views), default=0)),
    }

    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in names:
                return names[n.id]
            raise OperatorRejected(
                f"unknown name {n.id!r}; available: {sorted(names)}")
        if isinstance(n, ast.List):
            return [ev(e) for e in n.elts]
        if isinstance(n, ast.Tuple):
            return tuple(ev(e) for e in n.elts)
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                return -ev(n.operand)
            if isinstance(n.op, ast.UAdd):
                return ev(n.operand)
            if isinstance(n.op, ast.Not):
                return not ev(n.operand)
        # Arithmetic, comparison and conditionals. A layout is naturally
        # written in terms of how many views there are --
        # `grid(views, cols=2 if len(views) > 3 else 1)` -- and refusing
        # these discarded the whole joint proposal over ordinary algebra.
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            op = type(n.op)
            if op is ast.Add:
                if isinstance(a, list) or isinstance(b, list):
                    la = a if isinstance(a, list) else [a]
                    lb = b if isinstance(b, list) else [b]
                    return la + lb
                return a + b
            for t, f in ((ast.Sub, lambda x, y: x - y),
                         (ast.Mult, lambda x, y: x * y),
                         (ast.Div, lambda x, y: x / (y or 1e-9)),
                         (ast.FloorDiv, lambda x, y: x // (y or 1e-9)),
                         (ast.Mod, lambda x, y: x % (y or 1)),
                         (ast.Pow, lambda x, y: x ** y)):
                if op is t:
                    return f(a, b)
            raise OperatorRejected(f"unsupported operator {op.__name__}")
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            a, b = ev(n.left), ev(n.comparators[0])
            for t, f in ((ast.Gt, lambda x, y: x > y),
                         (ast.GtE, lambda x, y: x >= y),
                         (ast.Lt, lambda x, y: x < y),
                         (ast.LtE, lambda x, y: x <= y),
                         (ast.Eq, lambda x, y: x == y),
                         (ast.NotEq, lambda x, y: x != y)):
                if type(n.ops[0]) is t:
                    return f(a, b)
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            if isinstance(n.op, ast.And):
                return all(vals)
            return any(vals)
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise OperatorRejected("only direct calls are allowed")
            # Scalar helpers. A layout expression is naturally written in
            # terms of how many views there are -- `grid(views,
            # cols=min(4, len(views)))` -- and refusing `len` discarded the
            # whole joint proposal over arithmetic.
            if n.func.id in ("len", "min", "max", "int", "round", "abs",
                             "maximum", "minimum"):
                argv = [ev(a) for a in n.args]
                if n.func.id == "len":
                    x = argv[0]
                    return len(x) if isinstance(x, list) else 1
                return {"min": min, "max": max, "int": int,
                        "round": round, "abs": abs,
                        "maximum": max, "minimum": min}[n.func.id](*argv)
            fn = LAYOUT_FUNCS.get(n.func.id)
            if fn is None:
                raise OperatorRejected(
                    f"unknown layout function {n.func.id!r}; "
                    f"available: {sorted(LAYOUT_FUNCS)}")
            args = [ev(a) for a in n.args]
            kw = {k.arg: ev(k.value) for k in n.keywords}
            imgs = args[0] if args else list(views)
            if not isinstance(imgs, list):
                imgs = [imgs]
            # FLATTEN. `stack_v([grid(...), grid(...)], gap=10)` is the
            # commonest way to write a two-row composition, and each
            # element is itself a LIST of images returned by the inner
            # call. Filtering for `.convert` without flattening dropped
            # every element and reported "produced no images" -- measured
            # on realworldqa_g3 and blink_hard_g3, where it took the whole
            # joint proposal with it.
            flat = []
            for it in imgs:
                if isinstance(it, (list, tuple)):
                    flat.extend(x for x in it if hasattr(x, "convert"))
                elif hasattr(it, "convert"):
                    flat.append(it)
            imgs = flat
            if not imgs:
                # `crops` is EMPTY unless the executor actually zoomed on
                # that item, so `grid(crops, ...)` is not a typo -- it is a
                # composition that cannot run on this trajectory. Saying so
                # lets the optimizer write a guard instead of re-proposing
                # the same dead expression. Without this the refusal read
                # like a syntax error and the round was spent re-trying.
                n_crops = len(ctx.get("crop_views") or [])
                if args and n_crops == 0:
                    # DEGRADE, do not fail the unit. Whether the executor
                    # zoomed is a property of THIS ITEM: on a document
                    # domain it usually does not, so `crops` is [] and a
                    # nested call like
                    #   stack_v(grid(views,cols=2), grid(crops,cols=2))
                    # killed the candidate on every such item. Observed on
                    # mmlongdoc_p13 / realworldqa_p13 round 1, where every
                    # attempt died this way.
                    #
                    # Returning an empty list lets the ENCLOSING combinator
                    # drop this operand (empty image lists are skipped
                    # there) and compose the half that does exist.
                    return []
                raise OperatorRejected(
                    f"{n.func.id}: first argument must be images "
                    "(views, crops, or another layout call)")
            # Any FURTHER positional argument that is itself a list of images
            # is a nested composition, e.g.
            #   stack_v(pick(views,0), grid(crops, cols=2))
            # Those must be concatenated into the image list, not passed
            # through as scalar parameters -- doing the latter made every
            # nested expression fail with a TypeError deep inside a
            # combinator, which reads as a harness fault rather than as the
            # expression being fine.
            rest = []
            for a in args[1:]:
                # An EMPTY image list is a composition that degrades, not an
                # error: `stack_v(views, crops)` is a reasonable thing to
                # write, and on an item where the executor never zoomed
                # `crops` is simply []. Treating it as a scalar parameter
                # sent an empty list into a combinator's int() and failed
                # the whole candidate with a TypeError -- measured on
                # chartqa_p11 round 2, where every attempt died this way on
                # a single-image domain. Dropping it keeps the rest of the
                # composition alive.
                if isinstance(a, list) and not a:
                    continue
                if isinstance(a, list) and a and hasattr(a[0], "convert"):
                    imgs = imgs + a
                elif hasattr(a, "convert"):
                    imgs = imgs + [a]
                else:
                    rest.append(a)
            try:
                return fn(imgs, ctx, *rest, **kw)
            except TypeError as e:
                # CALL-SHAPE errors are the expression's, not the
                # harness's: `stack_v(views, 4, gap=8)` passes 4 into gap
                # positionally AND by keyword. Converting to
                # OperatorRejected routes them into the candidate-refusal
                # path; without it the raw TypeError killed the whole arm
                # (slidevqa_q4b r1, "multiple values for argument 'gap'").
                # The filter is deliberate: a TypeError from INSIDE the
                # function body is a framework bug and must propagate.
                msg = str(e)
                if any(s in msg for s in (
                        "multiple values for argument",
                        "positional argument",
                        "unexpected keyword argument",
                        "required positional argument")):
                    raise OperatorRejected(
                        f"{n.func.id}: bad call shape: {msg}") from e
                raise
        if isinstance(n, ast.ListComp) and len(n.generators) >= 1:
            # `[resize(v, 0.5) for v in views]` is the natural way to write
            # a per-view transform, and refusing it discarded the whole
            # joint proposal. Same class as the Subscript gap below.
            # NESTED generators too. The optimizer wrote
            #   [c for v in views for c in [crop(v,...), crop(v,...)]]
            # -- one band per page, flattened. Supporting only a single
            # generator refused it for a syntax rule, not a mistake.
            for g in n.generators:
                # `if` filters allowed here too, for the same reason they
                # are allowed in the selection language: "compose these
                # views, skipping the ones I do not want" adds no capability
                # the language lacks, and a capability present in one of the
                # three expression languages but missing in another is the
                # asymmetry that has cost this study several rounds.
                if not isinstance(g.target, ast.Name):
                    raise OperatorRejected(
                        "layout comprehension: plain loop variables only")
            saved = {g.target.id: names.get(g.target.id)
                     for g in n.generators}
            out_c = []

            def _walk(depth):
                if depth == len(n.generators):
                    val = ev(n.elt)
                    out_c.extend(val if isinstance(val, list) else [val])
                    return
                g = n.generators[depth]
                seq = ev(g.iter)
                if not isinstance(seq, (list, tuple)):
                    raise OperatorRejected(
                        "layout comprehension must iterate a list of views")
                for item in seq:
                    names[g.target.id] = item
                    # every `if` on this generator must hold, matching
                    # Python's own comprehension semantics
                    if g.ifs and not all(bool(ev(c)) for c in g.ifs):
                        continue
                    _walk(depth + 1)

            _walk(0)
            for k, v in saved.items():
                if v is None:
                    names.pop(k, None)
                else:
                    names[k] = v
            return [i for i in out_c if hasattr(i, "convert")]
        if isinstance(n, ast.Starred):
            # `stack_v(*views)` means the same as `stack_v(views)` here,
            # because a combinator already takes a list of images.
            val = ev(n.value)
            return val if isinstance(val, list) else [val]
        if isinstance(n, ast.Subscript):
            # `views[0]` is the idiom any Python writer reaches for, and
            # rejecting it discarded whole joint proposals: observed on
            # blink_ev5 and mmlongdoc_ev5 rounds 1, where every first
            # attempt died on "stack_v: first argument must be images"
            # because the optimizer had indexed the list. The visual half
            # of the round was then lost to a syntax rule, and the fallback
            # was a theta-only edit -- which is how a rejected VISUAL
            # proposal turns into evidence for text-only evolution.
            seq = ev(n.value)
            if not isinstance(seq, list):
                raise OperatorRejected(
                    "only `views` / `crops` / a layout call may be indexed")
            idx = n.slice
            if isinstance(idx, ast.Slice):
                lo = ev(idx.lower) if idx.lower is not None else None
                hi = ev(idx.upper) if idx.upper is not None else None
                st = ev(idx.step) if idx.step is not None else None
                out = seq[slice(
                    int(lo) if lo is not None else None,
                    int(hi) if hi is not None else None,
                    int(st) if st is not None else None)]
                if not out:
                    raise OperatorRejected(
                        f"slice selected no views (list has {len(seq)})")
                return out
            i = ev(idx)
            try:
                return seq[int(i)]
            except (TypeError, ValueError, IndexError) as e:
                raise OperatorRejected(
                    f"index {i!r} out of range for {len(seq)} views") from e
        raise OperatorRejected(
            f"unsupported syntax in layout expression: {type(n).__name__}")

    for st in let_stmts:
        names[st.targets[0].id] = ev(st.value)
    out = ev(tree.body)
    if not isinstance(out, list):
        out = [out]
    out = [o for o in out if hasattr(o, "convert")]
    if not out:
        raise OperatorRejected("layout expression produced no images")
    return out


def layout_help() -> str:
    return (
        "    grid(imgs, cols=, gap=, label=times|index|letter, long_edge=)\n"
        "    stack_v(imgs, gap=, label=)          vertical, context over detail\n"
        "    beside(imgs, gap=, label=, divider=) side by side, for pairs\n"
        "    overlay(imgs, boxes=crops)           draw provenance boxes\n"
        "    pick(imgs, 0, 2)                     choose which views\n"
        "    pad(imgs, top=, bottom=, ...)        margin for captions\n"
        "    resize(imgs, long_edge=)\n"
        "  free names: `views` (all sent views), `crops` (zoom views only)\n"
        f"  scalars: {', '.join(SHARED_SCALAR_NAMES)}")
