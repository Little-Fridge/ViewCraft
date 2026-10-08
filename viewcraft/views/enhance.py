"""Enhancement views: transforms that change pixel VALUES, not geometry.

The view mechanism as first operationalised could only move the *window* over
the source (whole image / crop / grid tile / annotated overview). Nine of the
fifteen phi/psi fields were resolution-and-crop knobs, so on benchmarks whose
images already fit inside the view budget (ChartQA ~800px, MMStar 512px) every
reachable config emitted the identical pixels and the search space was inert by
construction. That inertness was an artefact of the operationalisation, not a
property of the data: a page can be perfectly in-budget and still be hard to
read because the ink is low-contrast, the scan is soft, or the lighting is
uneven. Changing pixel values addresses exactly that, and needs no extra source
resolution at all.

This module supplies that second family. Every function here re-reads the
original file, optionally crops a `Region`, fits to the pixel budget with
render.py's `_fit`, and only then rewrites pixel values. Each emits a
`RenderedView` whose `ViewRecord.created_by` names the transform *and every
parameter*, e.g. ``enhance.unsharp(radius=2,percent=150)``, so an auditor
holding only the record can regenerate the exact bytes.

Ordering note (fit first, then enhance)
---------------------------------------
`radius`, `block_px` and friends are measured in pixels. If we enhanced at
source resolution and downscaled afterwards, the same config would mean a
different physical neighbourhood on every image, and a binarised page would be
re-greyed by the resampling filter on the way down. Fitting first makes the
parameters refer to *emitted* pixels, which is the resolution the model
actually reads, and keeps a binary image binary.

Invalid parameters are a hard error
-----------------------------------
A config that claims a transform and emits identical pixels is the specific
defect class this codebase exists to catch (see the retirement note on
`route_by_question_kind` in views/config.py). So a parameter that could only
ever be a no-op -- ``gamma=1.0``, ``percent=0``, ``radius=0`` -- raises
`ViewConfigError` rather than quietly passing the image through. An image that
happens to be unchanged because it was *already* in the target form (binarising
an already-binary scan) is a property of the input, not a lie in the config,
and is allowed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ..schema import Region, ViewRecord, ViewKind, new_view_id
# Reuse rather than reimplement: the resize/encode discipline (and the
# upscale flag that says "interpolation added no detail") must be identical
# to the geometric renderer, or two view families would disagree about what a
# `long_edge_px` budget means.
from .render import RenderedView, ViewConfigError, _encode, _fit

__all__ = [
    "OPS",
    "autocontrast_view",
    "gamma_view",
    "unsharp_view",
    "binarise_view",
    "grayscale_view",
    "enhance_chain",
    "apply_ops",
    "format_chain",
    "diff_fraction",
]

# `ViewKind` is not extended with an ENHANCED member. The kind field answers
# "which pixels of the source does this image cover?" -- it is consumed by the
# coordinate round-trip (`view_point_to_source`), by eviction policy
# (`keep_overview_drop_crops`) and by the overview annotator, all of which are
# purely geometric. An enhanced whole image is still, geometrically, a FULL
# view; an enhanced crop is still a CROP, and must remain evictable and
# back-mappable as one. Introducing a third kind would silently break every
# `kind == "crop"` branch. WHAT was done to the pixel values is orthogonal to
# geometry and is carried, fully and reproducibly, by `created_by`. (It would
# also mean editing schema.py, which is shared.)
_KIND_FOR_WHOLE = ViewKind.FULL.value
_KIND_FOR_REGION = ViewKind.CROP.value

_FULL_REGION = Region(0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# parameter formatting: the record must be reproducible from the record alone


def _fmt_num(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return repr(round(v, 6))
    return str(v)


def _fmt_op(name: str, params: dict[str, Any]) -> str:
    inner = ",".join(f"{k}={_fmt_num(params[k])}" for k in sorted(params))
    return f"enhance.{name}({inner})"


def format_chain(ops: Sequence[tuple[str, dict]]) -> str:
    """Canonical `created_by` for a chain, naming every op in applied order.

    Ops are joined with '+' in application order (order matters: autocontrast
    then unsharp is not the same image as unsharp then autocontrast), while
    each op's own parameters are sorted so the same config always yields the
    same string.
    """
    return "+".join(_fmt_op(n, p) for n, p in ops)


# ---------------------------------------------------------------------------
# parameter validation


def _need_number(name: str, op: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ViewConfigError(
            f"enhance.{op}: {name} must be a number, got {value!r}")
    v = float(value)
    if not np.isfinite(v):
        raise ViewConfigError(f"enhance.{op}: {name} must be finite, got {v}")
    return v


def _reject_unknown(op: str, params: dict, allowed: Iterable[str]) -> None:
    unknown = set(params) - set(allowed)
    if unknown:
        # Same rule as PhiConfig.from_dict: an unread key is a config that
        # claims a behaviour the runtime never performs.
        raise ViewConfigError(
            f"enhance.{op}: unsupported parameter(s) {sorted(unknown)}")


# ---------------------------------------------------------------------------
# the pixel transforms. each takes/returns an RGB PIL image and is pure.


def _op_autocontrast(img: Image.Image, *, cutoff: float = 1.0) -> Image.Image:
    """Per-channel histogram stretch, dropping `cutoff` percent from each tail.

    Per-channel (rather than luminance-preserving) is deliberate: faded scans
    and photographed pages usually have a colour cast, and stretching each band
    independently removes the cast at the same time as it recovers contrast.
    """
    c = _need_number("cutoff", "autocontrast", cutoff)
    if not (0.0 <= c < 50.0):
        raise ViewConfigError(
            f"enhance.autocontrast: cutoff must be in [0,50), got {c}")
    return ImageOps.autocontrast(img.convert("RGB"), cutoff=c)


def _op_gamma(img: Image.Image, *, gamma: float = 0.5) -> Image.Image:
    """out = 255 * (in/255) ** gamma. gamma<1 brightens, gamma>1 darkens.

    Applied through a 256-entry LUT, so it is exactly reproducible and
    independent of image size.
    """
    g = _need_number("gamma", "gamma", gamma)
    if g <= 0.0:
        raise ViewConfigError(f"enhance.gamma: gamma must be > 0, got {g}")
    if abs(g - 1.0) < 1e-9:
        # identity by construction -- the exact "claims a transform, emits the
        # same pixels" failure this module refuses to ship.
        raise ViewConfigError(
            "enhance.gamma: gamma=1.0 is the identity transform; a view that "
            "claims an enhancement must change pixel values")
    ramp = np.arange(256, dtype=np.float64) / 255.0
    lut = np.clip(np.rint((ramp ** g) * 255.0), 0, 255).astype(np.uint8)
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return Image.fromarray(lut[arr], mode="RGB")


def _op_unsharp(img: Image.Image, *, radius: float = 2.0,
                percent: float = 150.0, threshold: int = 0) -> Image.Image:
    """Unsharp mask: add `percent`% of (image - gaussian_blur(image, radius)).

    This is the transform for soft scans and JPEG-mushed small type: it does
    not invent detail, it restores the local contrast at glyph edges that the
    capture chain low-passed away.
    """
    r = _need_number("radius", "unsharp", radius)
    p = _need_number("percent", "unsharp", percent)
    t = _need_number("threshold", "unsharp", threshold)
    if r <= 0.0:
        raise ViewConfigError(f"enhance.unsharp: radius must be > 0, got {r}")
    if p <= 0.0:
        raise ViewConfigError(
            f"enhance.unsharp: percent must be > 0 (percent=0 is a no-op), got {p}")
    if not (0 <= t <= 255):
        raise ViewConfigError(
            f"enhance.unsharp: threshold must be in [0,255], got {t}")
    return img.convert("RGB").filter(
        ImageFilter.UnsharpMask(radius=r, percent=int(round(p)),
                                threshold=int(round(t))))


def _box_mean(gray: np.ndarray, block_px: int) -> np.ndarray:
    """Exact mean over a block_px x block_px window, edge-replicated.

    Computed with a summed-area table so cost is O(pixels) and independent of
    the window size -- a 101px window is as cheap as a 5px one, which matters
    because the useful block size for a page scan is large.
    """
    rad = block_px // 2
    pad = np.pad(gray.astype(np.float64), rad, mode="edge")
    # integral image with a zero row/col so every window is one 4-term lookup
    integ = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1), dtype=np.float64)
    np.cumsum(np.cumsum(pad, axis=0), axis=1, out=integ[1:, 1:])
    h, w = gray.shape
    k = 2 * rad + 1
    total = (integ[k:k + h, k:k + w] - integ[0:h, k:k + w]
             - integ[k:k + h, 0:w] + integ[0:h, 0:w])
    return total / float(k * k)


def _op_binarise(img: Image.Image, *, block_px: int = 31,
                 offset: float = 10.0) -> Image.Image:
    """ADAPTIVE local-mean thresholding (Bradley/Wellner style).

    Implementation: pure numpy, via a summed-area table (`_box_mean`). scipy is deliberately
    not a dependency, so `scipy.ndimage.uniform_filter` was not an option; the integral image gives the same exact box mean in O(pixels) with
    no dependency, and unlike a separable convolution it costs the same at
    block_px=101 as at block_px=5.

    Why adaptive and not a global threshold: a global cut is decided by the
    whole-page histogram, so on a photographed page with a lighting gradient --
    or a chart with a dark panel beside a white one -- one side of the image is
    flooded to solid black and the other bleached to solid white, taking the
    text with it. The local rule compares each pixel only to the mean of its
    own `block_px` neighbourhood, so the decision follows the illumination.
    `offset` is the classic bias that stops flat paper (where every pixel is
    within noise of its local mean) from turning into salt-and-pepper.

    block_px must be odd and >= 3, and is clamped down to the image if the
    emitted view is smaller than the window.
    """
    if isinstance(block_px, bool) or not isinstance(block_px, (int, np.integer)):
        raise ViewConfigError(
            f"enhance.binarise: block_px must be an int, got {block_px!r}")
    b = int(block_px)
    off = _need_number("offset", "binarise", offset)
    if b < 3:
        raise ViewConfigError(
            f"enhance.binarise: block_px must be >= 3, got {b}")
    if b % 2 == 0:
        raise ViewConfigError(
            f"enhance.binarise: block_px must be odd (window is centred), got {b}")
    if not (-128.0 <= off <= 128.0):
        raise ViewConfigError(
            f"enhance.binarise: offset must be in [-128,128], got {off}")

    gray = np.asarray(img.convert("L"), dtype=np.float64)
    h, w = gray.shape
    eff = min(b, (min(h, w) // 2) * 2 + 1)   # keep odd, keep <= image
    eff = max(3, eff)
    mean = _box_mean(gray, eff)
    out = np.where(gray > (mean - off), np.uint8(255), np.uint8(0))
    return Image.fromarray(out.astype(np.uint8), mode="L").convert("RGB")


def _op_grayscale(img: Image.Image) -> Image.Image:
    """Luminance only (ITU-R 601-2 via PIL's "L").

    Cheap, and it is the honest control for the colour-carrying transforms: if
    a gain shows up under grayscale too, the gain was never about colour.
    """
    return img.convert("L").convert("RGB")


#: name -> (callable, allowed parameter names, defaults)
OPS: dict[str, tuple[Callable[..., Image.Image], dict[str, Any]]] = {
    "autocontrast": (_op_autocontrast, {"cutoff": 1.0}),
    "gamma": (_op_gamma, {"gamma": 0.5}),
    "unsharp": (_op_unsharp, {"radius": 2.0, "percent": 150.0, "threshold": 0}),
    "binarise": (_op_binarise, {"block_px": 31, "offset": 10.0}),
    "grayscale": (_op_grayscale, {}),
}


def _resolve(ops: Sequence[tuple[str, dict | None]]) -> list[tuple[str, dict]]:
    """Validate op names, reject unknown params, fill defaults.

    Returns the fully-specified op list, which is what gets written into
    `created_by` -- defaults are materialised so the record never depends on
    what the defaults happened to be on the day it was produced.
    """
    if not ops:
        raise ViewConfigError("enhance: empty op chain emits the plain view; "
                              "pass at least one transform")
    resolved: list[tuple[str, dict]] = []
    for entry in ops:
        try:
            name, params = entry
        except (TypeError, ValueError):
            raise ViewConfigError(
                f"enhance: each op must be a (name, params) pair, got {entry!r}")
        if name not in OPS:
            raise ViewConfigError(
                f"enhance: unknown transform {name!r}; known: {sorted(OPS)}")
        _fn, defaults = OPS[name]
        params = dict(params or {})
        _reject_unknown(name, params, defaults)
        full = dict(defaults)
        full.update(params)
        resolved.append((name, full))
    return resolved


def apply_ops(img: Image.Image,
              ops: Sequence[tuple[str, dict | None]]) -> Image.Image:
    """Apply a resolved-or-raw op chain to a PIL image, in order.

    Exposed so callers (and the smoke test) can compose transforms on pixels
    they already hold, without a file round-trip.
    """
    out = img.convert("RGB")
    for name, params in _resolve(ops):
        fn, _defaults = OPS[name]
        out = fn(out, **params)
    return out


# ---------------------------------------------------------------------------
# the view entry points


def enhance_chain(source_path: str | Path, media_id: str,
                  ops: Sequence[tuple[str, dict | None]],
                  long_edge_px: int,
                  region: Region | None = None,
                  parent_view_id: str | None = None,
                  margin: float = 0.0,
                  allow_upscale: bool = False,
                  created_by: str | None = None,
                  page_index: int | None = None,
                  seconds: float | None = None) -> RenderedView:
    """Crop (optionally), fit to budget, then apply `ops` in order.

    `ops` is a list of ``(name, params)``; the record's `created_by` names the
    full chain in order, so ``[("autocontrast", {"cutoff": 1.0}),
    ("unsharp", {"radius": 2, "percent": 150})]`` records as
    ``enhance.autocontrast(cutoff=1)+enhance.unsharp(percent=150,radius=2,threshold=0)``.

    With `region=None` the whole image is enhanced and the view is a FULL;
    with a region it is a CROP, geometrically identical to `render_crop`, so
    coordinate round-tripping and eviction behave exactly as before.
    """
    resolved = _resolve(ops)
    if long_edge_px <= 0:
        raise ViewConfigError(f"enhance: long_edge_px must be > 0, got {long_edge_px}")

    box = region if region is not None else _FULL_REGION
    if region is not None and margin:
        box = box.expand(margin)
    whole = region is None

    with Image.open(source_path) as im:
        im.load()
        src_w, src_h = im.size
        if whole:
            base = im.convert("RGB")
        else:
            x0, y0, x1, y1 = box.to_pixels(src_w, src_h)
            x1 = max(x1, x0 + 1)
            y1 = max(y1, y0 + 1)
            base = im.crop((x0, y0, x1, y1)).convert("RGB")
        fitted, scale, up = _fit(base, long_edge_px, allow_upscale)
        # values are rewritten AFTER the fit, so radius/block_px are in
        # emitted pixels and a binarised view survives to the wire as binary.
        out = apply_ops(fitted, resolved)
        ow, oh = out.size
        png = _encode(out)

    rec = ViewRecord(
        view_id=new_view_id("enh"), source_media_id=media_id,
        kind=_KIND_FOR_WHOLE if whole else _KIND_FOR_REGION,
        long_edge_px=long_edge_px,
        out_w=ow, out_h=oh, src_w=src_w, src_h=src_h,
        parent_view_id=parent_view_id, region=box, scale=scale, upscaled=up,
        page_index=page_index, seconds=seconds,
        bytes_len=len(png),
        created_by=created_by or format_chain(resolved),
    )
    return RenderedView(rec, png)


def _single(name: str, params: dict, source_path, media_id, long_edge_px,
            region, parent_view_id, margin, allow_upscale,
            page_index, seconds) -> RenderedView:
    return enhance_chain(
        source_path, media_id, [(name, params)], long_edge_px,
        region=region, parent_view_id=parent_view_id, margin=margin,
        allow_upscale=allow_upscale, page_index=page_index, seconds=seconds)


def autocontrast_view(source_path: str | Path, media_id: str, long_edge_px: int,
                      region: Region | None = None, *, cutoff: float = 1.0,
                      parent_view_id: str | None = None, margin: float = 0.0,
                      allow_upscale: bool = False,
                      page_index: int | None = None,
                      seconds: float | None = None) -> RenderedView:
    """Per-channel histogram stretch. See `_op_autocontrast`."""
    return _single("autocontrast", {"cutoff": cutoff}, source_path, media_id,
                   long_edge_px, region, parent_view_id, margin, allow_upscale,
                   page_index, seconds)


def gamma_view(source_path: str | Path, media_id: str, long_edge_px: int,
               region: Region | None = None, *, gamma: float = 0.5,
               parent_view_id: str | None = None, margin: float = 0.0,
               allow_upscale: bool = False,
               page_index: int | None = None,
               seconds: float | None = None) -> RenderedView:
    """Gamma correction; gamma<1 brightens shadows, gamma>1 darkens."""
    return _single("gamma", {"gamma": gamma}, source_path, media_id,
                   long_edge_px, region, parent_view_id, margin, allow_upscale,
                   page_index, seconds)


def unsharp_view(source_path: str | Path, media_id: str, long_edge_px: int,
                 region: Region | None = None, *, radius: float = 2.0,
                 percent: float = 150.0, threshold: int = 0,
                 parent_view_id: str | None = None, margin: float = 0.0,
                 allow_upscale: bool = False,
                 page_index: int | None = None,
                 seconds: float | None = None) -> RenderedView:
    """Unsharp mask on the emitted pixels."""
    return _single("unsharp",
                   {"radius": radius, "percent": percent, "threshold": threshold},
                   source_path, media_id, long_edge_px, region, parent_view_id,
                   margin, allow_upscale, page_index, seconds)


def binarise_view(source_path: str | Path, media_id: str, long_edge_px: int,
                  region: Region | None = None, *, block_px: int = 31,
                  offset: float = 10.0,
                  parent_view_id: str | None = None, margin: float = 0.0,
                  allow_upscale: bool = False,
                  page_index: int | None = None,
                  seconds: float | None = None) -> RenderedView:
    """Adaptive local-mean binarisation. See `_op_binarise`."""
    return _single("binarise", {"block_px": block_px, "offset": offset},
                   source_path, media_id, long_edge_px, region, parent_view_id,
                   margin, allow_upscale, page_index, seconds)


def grayscale_view(source_path: str | Path, media_id: str, long_edge_px: int,
                   region: Region | None = None,
                   parent_view_id: str | None = None, margin: float = 0.0,
                   allow_upscale: bool = False,
                   page_index: int | None = None,
                   seconds: float | None = None) -> RenderedView:
    """Luminance only."""
    return _single("grayscale", {}, source_path, media_id, long_edge_px,
                   region, parent_view_id, margin, allow_upscale,
                   page_index, seconds)


# ---------------------------------------------------------------------------
# measurement helper: "did this actually change anything?"


def diff_fraction(a: bytes | Image.Image, b: bytes | Image.Image) -> float:
    """Fraction of pixels differing between two emitted views.

    A pixel counts as differing if any channel differs by >= 1 level. This is
    the quantity that decides whether a transform is real or dead code, so it
    lives next to the transforms rather than in a test file.
    """
    def arr(x):
        if isinstance(x, (bytes, bytearray)):
            import io
            with Image.open(io.BytesIO(x)) as im:
                im.load()
                return np.asarray(im.convert("RGB"), dtype=np.int16)
        return np.asarray(x.convert("RGB"), dtype=np.int16)

    pa, pb = arr(a), arr(b)
    if pa.shape != pb.shape:
        raise ValueError(f"shape mismatch {pa.shape} vs {pb.shape}")
    return float(np.any(pa != pb, axis=-1).mean())
