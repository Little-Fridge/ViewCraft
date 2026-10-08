"""Materialize parquet-embedded images onto disk, once, content-addressed.

The view renderer re-reads the ORIGINAL file whenever it makes a crop: that is
the whole reason a zoom recovers real detail instead of interpolated pixels.
An image living inside a parquet row cannot be re-read that way, so every
parquet-backed domain writes its images out once and then refers to them by
path.

The bytes are written VERBATIM, not re-encoded. Re-encoding through PIL would
silently change the pixels the benchmark ships (JPEG requantisation), so a
measured accuracy would belong to our re-encode rather than to the dataset.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..paths import media_root

MEDIA_ROOT = media_root()

# Magic-number sniffing, because the parquet row often carries no usable name.
_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
]


def _ext_for(blob: bytes, hint: str = "") -> str:
    for magic, ext in _MAGIC:
        if blob.startswith(magic):
            return ext
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    h = Path(hint or "").suffix.lower()
    return h if h in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
                      ".tif", ".tiff"} else ".png"


def write_media(domain: str, blob: bytes, hint: str = "") -> str:
    """Write one image blob under data/_media/<domain>/ and return its path.

    Content-addressed: the same bytes always land on the same path, so a
    re-run is idempotent and two rows sharing an image (ChartQA asks several
    questions per chart) share one file and one `group_key`.
    """
    if not blob:
        raise ValueError(f"{domain}: empty image blob")
    digest = hashlib.sha256(blob).hexdigest()
    ext = _ext_for(blob, hint)
    d = MEDIA_ROOT / domain / digest[:2]
    p = d / f"{digest}{ext}"
    if p.exists() and p.stat().st_size == len(blob):
        return str(p)
    d.mkdir(parents=True, exist_ok=True)
    # Atomic publish: a partially-written file would be opened by a concurrent
    # worker and decoded as a truncated image.
    tmp = p.with_name(f".{digest}.{os.getpid()}.tmp")
    tmp.write_bytes(blob)
    tmp.replace(p)
    return str(p)


def media_digest(path: str) -> str:
    """The content hash embedded in a path written by write_media."""
    return Path(path).stem
