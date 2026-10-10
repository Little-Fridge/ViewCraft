#!/usr/bin/env python3
"""Run a ViewCraft task skill on images, PDFs or a video clip."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skill', required=True, help='Skill library ID or skill directory')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/models.yaml')
    media = ap.add_mutually_exclusive_group()
    media.add_argument('--images', nargs='+', type=Path)
    media.add_argument('--pdf', nargs='+', type=Path, help='PDF files in source/version order')
    media.add_argument('--video', type=Path)
    ap.add_argument('--pages', help='PDF file pages, 1-based: 1-4,8 (applied to each PDF)')
    ap.add_argument('--frame-times', nargs='+', type=float)
    ap.add_argument('--frame-interval', type=float, default=1., help='Video sample spacing in seconds')
    ap.add_argument('--start', type=float, default=0., help='Video window start in seconds')
    ap.add_argument('--end', type=float, help='Video window end in seconds')
    ap.add_argument('--max-frames', type=int, default=120)
    ap.add_argument('--question')
    ap.add_argument('--options', nargs='*')
    ap.add_argument('--output', type=Path, help='New output directory')
    ap.add_argument('--record', action='store_true', help='Opt in to model-response traces and visual replay')
    ap.add_argument('--check', action='store_true', help='Validate and prepare inputs without calling a model')
    args = ap.parse_args()
    if not (args.images or args.pdf or args.video) or not args.question:
        ap.error('Supply --images and --question, or use --pdf / --video with --question.')
    if args.pages and not args.pdf:
        ap.error('--pages requires --pdf.')
    if args.frame_times is not None and not args.images:
        ap.error('--frame-times requires --images.')
    if not args.video and (args.start != 0 or args.end is not None or args.frame_interval != 1 or args.max_frames != 120):
        ap.error('--start, --end, --frame-interval and --max-frames require --video.')
    os.environ['VIEWCRAFT_CONFIG'] = str(args.config.resolve())
    import yaml
    from PIL import Image
    from viewcraft.evolution.chain import Chain
    from viewcraft.evolution.propose import Candidate
    from viewcraft.models import ModelClient
    from viewcraft.paths import runs_root
    from viewcraft.replay import RecordingClient, render_replay
    from viewcraft.runtime import run_static_item
    from viewcraft.schema import PublicItem, sha
    from viewcraft.skill_inputs import prepare_pdf, prepare_video, validate_times
    from viewcraft.views.config import Coordinates

    directory = ROOT / 'skill-library' / args.skill
    if not directory.is_dir():
        directory = Path(args.skill).resolve()
    meta = yaml.safe_load((directory / 'skill.yaml').read_text())
    coords = Coordinates.read_snapshot(directory)
    Chain._assert_expressions_parse(None, Candidate('joint', coords, 'skill-library', 'skill-library'))
    budget = meta['budget']
    if any(type(budget.get(k)) is not int or budget[k] < 1 for k in ('observations', 'max_images_per_request')):
        raise ValueError('Skill observation and image budgets must be positive integers.')
    out = args.output or runs_root() / 'skills' / meta['id'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not args.check and out.exists():
        raise ValueError(f'Output already exists: {out}. Choose a new --output.')
    # Validation uses temporary media; actual runs retain their source pages for replay.
    with TemporaryDirectory(prefix='viewcraft-input-') as scratch:
        media_dir = Path(scratch) if args.check else out / 'inputs'
        times = []
        if args.pdf:
            images, labels = prepare_pdf(args.pdf, media_dir, pages=args.pages)
        elif args.video:
            images, times, labels = prepare_video(args.video, media_dir, start=args.start, end=args.end,
                                                  interval=args.frame_interval, max_frames=args.max_frames)
        else:
            images = [str(p.resolve()) for p in args.images]
            labels = [f'image {i + 1}: {p.name}' for i, p in enumerate(args.images)]
            for path in images:
                with Image.open(path) as image:
                    image.verify()
            if args.frame_times is not None:
                times = args.frame_times
                validate_times(times, len(images))
                labels = [f'{label}, time {t:g} s' for label, t in zip(labels, times)]
        if args.check:
            print(f"CHECK OK: {meta['id']}; {len(images)} inputs")
            return
        client = ModelClient('executor')
        client.validate()
        out.mkdir(parents=True, exist_ok=True)
        if args.record:
            client = RecordingClient(client, out)
        sources = '\n'.join(f'Source unit {i}: {label}' for i, label in enumerate(labels))
        question = args.question + '\n\nSource map (zero-based units for view requests):\n' + sources
        question += '\nOnly inspected sources support the answer; the initial view may show a subset of this pool.'
        item = PublicItem('skill/' + meta['id'], 'custom', question, tuple(images), tuple(args.options or []),
                          meta={'frame_times_s': times, 'source_labels': labels})
        trajectory = run_static_item(item, coords, client, budget, sha({'budget':budget,'skill':coords.config_hash}),
                                     arm='viewcraft', condition='normal', split='inference',
                                     answer_style=meta.get('answer_style', 'free_text'))
        result = {'skill':meta['id'], 'answer':trajectory.answer_text,
                  'status':trajectory.status, 'error':trajectory.error}
        (out/'answer.md').write_text(trajectory.answer_text + '\n')
        (out/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        if args.record:
            serialized = trajectory.to_dict()
            replacements = {path:label for path, label in zip(images, labels)}
            for view in serialized['views']:
                view['source_media_id'] = replacements.get(view['source_media_id'], Path(view['source_media_id']).name)
            (out/'trajectory.json').write_text(json.dumps(serialized, indent=2, ensure_ascii=False))
            render_replay(out, meta['title'], args.question, result)
        print(trajectory.answer_text)
        print('\nSaved:', out / 'answer.md')
        if trajectory.error or not trajectory.exec_result or not trajectory.exec_result.passed:
            raise RuntimeError(trajectory.error or 'Execution contract failed; see result.json.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
