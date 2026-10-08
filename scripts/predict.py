#!/usr/bin/env python3
"""Use an initial or evolved skill on a new question, without optimization."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, required=True, help="Snapshot directory with all three skill files")
    parser.add_argument("--images", nargs="+", type=Path, required=True, help="Image files, or ordered frames/pages")
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="*", default=[])
    parser.add_argument("--frame-times", nargs="+", type=float)
    parser.add_argument("--answer-style", choices=("short_text", "letter", "yes_no", "point"), default="short_text")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/models.yaml")
    parser.add_argument("--observations", type=int, default=6)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Optional trajectory JSON")
    args = parser.parse_args()
    if args.observations < 1 or args.max_images < 1:
        parser.error("Budgets must be positive")
    if args.frame_times and len(args.frame_times) != len(args.images):
        parser.error("--frame-times must have one timestamp per image")
    os.environ["VIEWCRAFT_CONFIG"] = str(args.config.resolve())

    from PIL import Image
    from viewcraft.models import ModelClient
    from viewcraft.runtime import run_static_item
    from viewcraft.schema import PublicItem, sha
    from viewcraft.views.config import Coordinates

    coords = Coordinates.read_snapshot(args.skill)
    images = tuple(str(p.resolve()) for p in args.images)
    for path in images:
        with Image.open(path) as im:
            im.verify()
    client = ModelClient("executor")
    client.validate()
    budget = {"observations": args.observations, "max_images_per_request": args.max_images}
    item = PublicItem("custom/predict", "custom", args.question, images, tuple(args.options),
                      meta={"frame_times_s": args.frame_times} if args.frame_times else {})
    trajectory = run_static_item(item, coords, client, budget, sha(budget),
                                 arm="viewcraft", condition="normal", split="inference",
                                 answer_style=args.answer_style)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(trajectory.to_dict(), indent=2, ensure_ascii=False))
    if trajectory.error or not trajectory.answer_text:
        raise RuntimeError(trajectory.error or f"No answer returned (status={trajectory.status})")
    print(trajectory.answer_text)
    print(f"status={trajectory.status}; steps={trajectory.observation_steps}; "
          f"images={trajectory.n_images_sent}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
