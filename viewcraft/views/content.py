"""Cheap per-unit content features, so selection can be CONTENT-AWARE.

WHY. `select_expr` was pure index arithmetic over (n, budget, n_frames): it
could not see a single pixel, so "pick the frame where the action happens" or
"pick the page containing the table" was inexpressible, and select_expr stayed
empty in every accepted config. For a 32-frame video or a 42-page document the
choice of WHICH units is the dominant visual decision, so this blindness closed
off the biggest lever.

The features here are NEUTRAL, measured from the data (never from any score):
  motion     mean |diff| to the previous unit (video: temporal change; 0 for p0)
  edges      mean gradient magnitude (text/detail density proxy)
  colorful   colour saturation
  bright     mean luminance
They are computed on small thumbnails, so the cost is negligible next to a
model call. Each is a plain list of length n, bound into the selection env;
the new primitives topk/bottomk/above/order_by turn them into index sets. Which
feature, what k, and how to blend with uniform coverage are all the optimizer's
to write -- these are primitives for USING a content signal, not named schemes.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _thumb(media_path: str, size: int = 96) -> np.ndarray | None:
    from PIL import Image
    try:
        with Image.open(media_path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            return np.asarray(im, dtype=np.float32)
    except Exception:                                       # noqa: BLE001
        return None


def content_features(media_list: list[str], thumb: int = 96) -> dict[str, list]:
    """Per-unit neutral content features for a pool of source media.

    Returns lists aligned with media_list order. A unit that fails to load gets
    the pool median for scalar features and 0 for motion, so one bad frame
    never makes a whole candidate invalid (the recurring scope defect).
    """
    arrs = [_thumb(m, thumb) for m in media_list]
    n = len(arrs)

    edges, colorful, bright, motion = [], [], [], []
    prev = None
    for a in arrs:
        if a is None:
            edges.append(None); colorful.append(None); bright.append(None)
            motion.append(0.0)
            continue
        g = a.mean(axis=2)
        gx = np.abs(np.diff(g, axis=1)).mean() if g.shape[1] > 1 else 0.0
        gy = np.abs(np.diff(g, axis=0)).mean() if g.shape[0] > 1 else 0.0
        edges.append(float(gx + gy))
        mx = a.max(axis=2); mn = a.min(axis=2)
        colorful.append(float((mx - mn).mean()))
        bright.append(float(g.mean()))
        # Motion is TEMPORAL change between CONSECUTIVE UNITS. On a video the
        # frames share a size and the subtraction is elementwise. On
        # multi-IMAGE pools (mmiu forensic pairs, mixed-resolution sets) the
        # neighbours can differ in size, and the raw subtraction raised
        # "operands could not be broadcast" -- which killed 72 mmiu gate
        # units as needs_review and voided the candidate round (2026-09-19).
        # Resample the neighbour to THIS unit's size before subtracting:
        # same estimator, defined for any pool.
        if prev is not None:
            p = (prev if prev.shape == a.shape else
                 np.array(Image.fromarray(
                     prev.astype(np.uint8)).resize(
                     (a.shape[1], a.shape[0]), Image.BILINEAR),
                     dtype=np.float32))
            motion.append(float(np.abs(a - p).mean()))
        else:
            motion.append(0.0)
        prev = a

    def fill(vals):
        good = [v for v in vals if v is not None]
        med = float(np.median(good)) if good else 0.0
        return [med if v is None else v for v in vals]

    return {"motion": motion, "edges": fill(edges),
            "colorful": fill(colorful), "bright": fill(bright)}
