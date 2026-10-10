"""Prepare user documents and clips as addressable visual evidence."""
from __future__ import annotations

import math
from pathlib import Path


def parse_pages(spec: str | None, count: int) -> list[int]:
    if not spec:
        if count > 64:
            raise ValueError(f'This PDF has {count} pages. Select up to 64 with --pages (for example 1-8,20-24).')
        return list(range(count))
    selected = set()
    try:
        for part in spec.split(','):
            bounds = [int(v.strip()) for v in part.split('-')]
            if len(bounds) not in (1, 2):
                raise ValueError
            first, last = bounds[0], bounds[-1]
            if not 1 <= first <= last <= count:
                raise ValueError
            selected.update(range(first - 1, last))
    except ValueError as exc:
        raise ValueError(f'Invalid --pages {spec!r}; use 1-based pages within 1-{count}, such as 1-3,7.') from exc
    if len(selected) > 64:
        raise ValueError('Select at most 64 PDF pages per document with --pages.')
    return sorted(selected)


def validate_times(times: list[float], count: int) -> None:
    if len(times) != count or any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError('Frame timestamps must be finite, nonnegative and match the image count.')
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Frame timestamps must be strictly increasing.')


def prepare_pdf(paths: list[Path], output: Path, *, pages: str | None = None):
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ValueError('PDF input needs pypdfium2. Install: pip install -r requirements-media.txt') from exc
    output.mkdir(parents=True, exist_ok=True)
    images, labels = [], []
    for document_index, path in enumerate(paths, 1):
        try:
            document = pdfium.PdfDocument(str(path))
        except Exception as exc:
            raise ValueError(f'Cannot open PDF {path.name}; check that it is valid and unlocked.') from exc
        try:
            count = len(document)
            for index in parse_pages(pages, count):
                page = document[index]
                bitmap = None
                try:
                    width, height = page.get_size()
                    bitmap = page.render(scale=2560 / max(width, height))
                    target = output / f'doc-{document_index:02d}-page-{index + 1:04d}.png'
                    bitmap.to_pil().convert('RGB').save(target)
                    images.append(str(target.resolve()))
                    labels.append(f'document {document_index}: {path.name}, file page {index + 1} of {count}')
                finally:
                    if bitmap is not None:
                        bitmap.close()
                    page.close()
        finally:
            document.close()
    if not images:
        raise ValueError('No PDF pages were rendered. Check the document and --pages selection.')
    return images, labels


def prepare_video(path: Path, output: Path, *, start=0., end=None, interval=1., max_frames=120):
    if not math.isfinite(start) or start < 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError('--start must be nonnegative and --frame-interval must be positive and finite.')
    if end is not None and (not math.isfinite(end) or end <= start):
        raise ValueError('--end must be finite and later than --start.')
    if max_frames < 1:
        raise ValueError('--max-frames must be positive.')
    try:
        import av
    except ImportError as exc:
        raise ValueError('Video input needs PyAV. Install: pip install -r requirements-media.txt') from exc
    output.mkdir(parents=True, exist_ok=True)
    images, times, labels = [], [], []
    next_time = start
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f'{path.name} has no video track.')
            stream = container.streams.video[0]
            origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else None
            for frame in container.decode(stream):
                if frame.time is None:
                    raise ValueError('The video has no reliable frame timestamps. Supply timestamped images instead.')
                if origin is None:
                    origin = float(frame.time)
                seconds = float(frame.time) - origin
                if end is not None and seconds > end + 1e-8:
                    break
                if seconds + 1e-8 < next_time:
                    continue
                if len(images) >= max_frames:
                    raise ValueError('Video exceeds --max-frames. Narrow --start/--end or increase --frame-interval/--max-frames.')
                target = output / f'frame-{len(images):04d}-{seconds:.3f}s.png'
                image = frame.to_image()
                image.thumbnail((2560, 2560))
                image.save(target)
                images.append(str(target.resolve()))
                times.append(seconds)
                labels.append(f'{path.name}, time {seconds:.6f} s from clip start')
                next_time = start + (math.floor((seconds - start) / interval + 1e-8) + 1) * interval
    except av.error.FFmpegError as exc:
        raise ValueError(f'Cannot decode {path.name}: {exc}') from exc
    if not images:
        raise ValueError('No video frames in the requested time window. Check --start and --end.')
    validate_times(times, len(images))
    return images, times, labels
