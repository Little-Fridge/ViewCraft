"""View rendering: turn a phi/psi config plus a source image into real pixels.

Every emitted image gets a ViewRecord carrying the exact transform, so a
coordinate can be mapped back to the source and an auditor can confirm the
model saw what the config claims. Cropping re-reads the original file, so
detail genuinely returns; upscaling is flagged because interpolation adds none.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from ..schema import Region, ViewRecord, ViewKind, new_view_id

# Pillow's LANCZOS moved namespace across versions.
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


class ViewConfigError(RuntimeError):
    """Raised when a phi/psi field is unknown or unsupported at runtime.

    Silently ignoring an unknown key would let a config claim a behaviour the
    runtime never performs, so unknown fields are a hard error.
    """


@dataclass
class RenderedView:
    """An emitted image plus its audit record."""
    record: ViewRecord
    png_bytes: bytes

    @property
    def data_url(self) -> str:
        import base64
        return "data:image/png;base64," + \
            base64.b64encode(self.png_bytes).decode("ascii")


def _encode(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _fit(img: Image.Image, long_edge_px: int,
         allow_upscale: bool = False) -> tuple[Image.Image, float, bool]:
    """Scale so the long edge equals long_edge_px. Returns (img, scale, upscaled)."""
    w, h = img.size
    longest = max(w, h)
    scale = long_edge_px / longest
    if scale > 1.0 and not allow_upscale:
        return img, 1.0, False
    if abs(scale - 1.0) < 1e-9:
        return img, 1.0, False
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new, RESAMPLE), scale, scale > 1.0


def render_full(source_path: str | Path, media_id: str, long_edge_px: int,
                created_by: str = "phi.full",
                page_index: int | None = None,
                seconds: float | None = None) -> RenderedView:
    """Whole image downscaled to the budget — the ordinary baseline view."""
    with Image.open(source_path) as im:
        im.load()
        src_w, src_h = im.size
        out, scale, up = _fit(im, long_edge_px)
        ow, oh = out.size
        png = _encode(out)

    rec = ViewRecord(
        view_id=new_view_id("full"), source_media_id=media_id,
        kind=ViewKind.FULL.value, long_edge_px=long_edge_px,
        out_w=ow, out_h=oh, src_w=src_w, src_h=src_h,
        region=Region(0.0, 0.0, 1.0, 1.0), scale=scale, upscaled=up,
        page_index=page_index, seconds=seconds,
        bytes_len=len(png), created_by=created_by,
    )
    return RenderedView(rec, png)


def render_crop(source_path: str | Path, media_id: str, region: Region,
                long_edge_px: int, parent_view_id: str | None = None,
                margin: float = 0.0, allow_upscale: bool = False,
                created_by: str = "phi.crop",
                page_index: int | None = None,
                seconds: float | None = None) -> RenderedView:
    """Crop a normalized region from the ORIGINAL file, then fit to budget.

    Because the source is re-read at full resolution, a small target recovers
    real pixels rather than interpolated ones.
    """
    box = region.expand(margin) if margin else region
    with Image.open(source_path) as im:
        im.load()
        src_w, src_h = im.size
        px = box.to_pixels(src_w, src_h)
        # guarantee at least one pixel in each axis
        x0, y0, x1, y1 = px
        x1 = max(x1, x0 + 1)
        y1 = max(y1, y0 + 1)
        crop = im.crop((x0, y0, x1, y1))
        cw, ch = crop.size
        out, scale, up = _fit(crop, long_edge_px, allow_upscale)
        ow, oh = out.size
        png = _encode(out)

    rec = ViewRecord(
        view_id=new_view_id("crop"), source_media_id=media_id,
        kind=ViewKind.CROP.value, long_edge_px=long_edge_px,
        out_w=ow, out_h=oh, src_w=src_w, src_h=src_h,
        parent_view_id=parent_view_id, region=box, scale=scale, upscaled=up,
        page_index=page_index, seconds=seconds,
        bytes_len=len(png), created_by=created_by,
    )
    return RenderedView(rec, png)


def render_grid(source_path: str | Path, media_id: str, rows: int, cols: int,
                long_edge_px: int, index: int,
                overlap: float = 0.0, created_by: str = "phi.grid",
                ) -> RenderedView:
    """One tile of an r x c grid over the source, at the budget's long edge.

    Tiling raises effective resolution per emitted image without exceeding the
    per-image budget: each tile covers 1/(r*c) of the frame.
    """
    if not (0 <= index < rows * cols):
        raise ViewConfigError(f"grid index {index} out of range for {rows}x{cols}")
    r, c = divmod(index, cols)
    w, h = 1.0 / cols, 1.0 / rows
    x0, y0 = c * w, r * h
    reg = Region(max(0.0, x0 - overlap), max(0.0, y0 - overlap),
                 min(1.0, x0 + w + overlap), min(1.0, y0 + h + overlap))
    view = render_crop(source_path, media_id, reg, long_edge_px,
                       created_by=created_by)
    view.record.kind = ViewKind.CROP.value
    return view


def render_overview_with_boxes(source_path: str | Path, media_id: str,
                               regions: list[Region], long_edge_px: int,
                               labels: list[str] | None = None,
                               created_by: str = "psi.overview") -> RenderedView:
    """Downscaled whole image with regions outlined.

    This is the layout device that keeps a crop's spatial context: the model
    can see WHERE the detail views came from, which a bare crop destroys.
    """
    with Image.open(source_path) as im:
        im.load()
        src_w, src_h = im.size
        out, scale, up = _fit(im.convert("RGB"), long_edge_px)
        ow, oh = out.size
        d = ImageDraw.Draw(out)
        for i, reg in enumerate(regions):
            x0, y0, x1, y1 = reg.to_pixels(ow, oh)
            width = max(2, ow // 320)
            d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=width)
            tag = (labels[i] if labels and i < len(labels) else str(i + 1))
            ty = max(0, y0 - 14)
            d.text((x0 + 2, ty), tag, fill=(255, 0, 0))
        png = _encode(out)

    rec = ViewRecord(
        view_id=new_view_id("ovw"), source_media_id=media_id,
        kind=ViewKind.OVERVIEW.value, long_edge_px=long_edge_px,
        out_w=ow, out_h=oh, src_w=src_w, src_h=src_h,
        region=Region(0.0, 0.0, 1.0, 1.0), scale=scale, upscaled=up,
        bytes_len=len(png), created_by=created_by,
    )
    return RenderedView(rec, png)


def apply_view_ops(view: "RenderedView", ops: list, ctx: dict) -> "RenderedView":
    """Run a model-authored operator program over ONE emitted view.

    The ViewRecord is updated so `created_by` names the program and the
    output dimensions stay truthful. A program leaving the pixels
    byte-identical raises: "config changed but pixels did not" is the defect
    class this codebase treats as a bug, never as a null result.
    """
    if not ops:
        return view
    from .operators import OperatorRejected
    from .steps import run_program

    before = Image.open(io.BytesIO(view.png_bytes)).convert("RGB")
    out = run_program(ops, [before], ctx)
    if len(out) != 1:
        raise OperatorRejected(
            "a phi view_ops program must yield exactly one image per view, "
            f"got {len(out)}")
    after = out[0]
    if after.size == before.size and after.tobytes() == before.tobytes():
        # PER-ITEM no-op: return the view untouched rather than failing the
        # unit. Inertness is a CANDIDATE-level judgement, checked against
        # recorded traces before the candidate costs an evaluation; raising
        # here kills one unit and silently shrinks the gate set, which makes
        # the whole round incomparable. Measured twice: mmlongdoc_ev1 r1
        # (99 vs 96 items, 2752s lost) and blink_ev4 r1 (96 vs 95).
        return view

    png = _encode(after)
    rec = view.record
    rec.out_w, rec.out_h = after.size
    rec.bytes_len = len(png)
    names = "+".join(str(o.get("op")) for o in ops)
    rec.created_by = f"{rec.created_by}|phi.ops({names})"
    return RenderedView(rec, png)


def apply_context_ops(views: list, ops: list, ctx: dict) -> list:
    """Run a model-authored program over the ASSEMBLED visual context.

    This is where a reduction lives: eight sampled frames can become one
    labelled contact sheet, or a paired figure a difference map. A composite
    has no single source region, so `source_media_id` lists the contributors
    and `region` is the composite's own full frame. That convention is stated
    here rather than faked as a region of one input.
    """
    if not ops or not views:
        return views
    from .operators import OperatorRejected
    from .steps import run_program

    imgs = [Image.open(io.BytesIO(v.png_bytes)).convert("RGB") for v in views]
    out = run_program(ops, imgs, ctx)
    if len(out) == len(imgs) and all(
            a.size == b.size and a.tobytes() == b.tobytes()
            for a, b in zip(out, imgs)):
        # Same reasoning as apply_view_ops: a per-item no-op is a property
        # of this item, not of the candidate, and must not shrink the gate.
        return views

    names = "+".join(str(o.get("op")) for o in ops)
    srcs = ",".join(sorted({v.record.source_media_id for v in views}))[:400]
    made = []
    for im in out:
        png = _encode(im)
        rec = ViewRecord(
            view_id=new_view_id("ctx"), source_media_id=srcs,
            kind=(ViewKind.OVERVIEW.value if len(out) < len(imgs)
                  else views[0].record.kind),
            long_edge_px=max(im.size), out_w=im.size[0], out_h=im.size[1],
            src_w=views[0].record.src_w, src_h=views[0].record.src_h,
            region=Region(0.0, 0.0, 1.0, 1.0), scale=1.0,
            bytes_len=len(png), created_by=f"psi.ops({names})")
        made.append(RenderedView(rec, png))
    return made


def source_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


# ---------------------------------------------------------------------------
# coordinate round-trip, used by the deterministic acceptance tests


def view_point_to_source(rec: ViewRecord, vx: float, vy: float) -> tuple[float, float]:
    """Map a point in *view* pixel coords back to source pixel coords."""
    reg = rec.region or Region(0.0, 0.0, 1.0, 1.0)
    sx0 = reg.x0 * rec.src_w
    sy0 = reg.y0 * rec.src_h
    span_w = reg.width * rec.src_w
    span_h = reg.height * rec.src_h
    return (sx0 + (vx / rec.out_w) * span_w,
            sy0 + (vy / rec.out_h) * span_h)


def source_point_to_view(rec: ViewRecord, sx: float, sy: float) -> tuple[float, float]:
    """Inverse of view_point_to_source."""
    reg = rec.region or Region(0.0, 0.0, 1.0, 1.0)
    sx0 = reg.x0 * rec.src_w
    sy0 = reg.y0 * rec.src_h
    span_w = reg.width * rec.src_w
    span_h = reg.height * rec.src_h
    return ((sx - sx0) / span_w * rec.out_w,
            (sy - sy0) / span_h * rec.out_h)
