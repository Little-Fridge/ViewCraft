#!/usr/bin/env python3
"""Run a reusable skill on your own images or timestamped frame pool."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compare_answer(prediction, expected):
    answer = str(expected["answer"])
    match = expected.get("match", "exact")
    if match == "numeric":
        try:
            passed = abs(float(prediction.strip()) - float(answer)) <= expected.get("tolerance", 1e-6)
        except ValueError:
            passed = False
    elif match == "point":
        from viewcraft.evaluation.scorers import parse_point
        point = parse_point(prediction)
        region = expected["region"]
        passed = point is not None and region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]
    elif match == "contains":
        passed = answer.strip().casefold() in prediction.strip().casefold()
    else:
        passed = answer.strip().casefold() == prediction.strip().casefold()
    return {"passed": bool(passed), "reference": answer, "match": match}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skill", required=True, help="Skill library ID or skill directory")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/models.yaml")
    ap.add_argument("--images", nargs="+", type=Path)
    ap.add_argument("--question")
    ap.add_argument("--options", nargs="*")
    ap.add_argument("--frame-times", nargs="+", type=float)
    ap.add_argument("--output", type=Path, help="New output directory")
    ap.add_argument("--record", action="store_true", help="Save actual model views, replies and standalone HTML; disable response cache")
    ap.add_argument("--check", action="store_true", help="Check the skill and inputs without calling a model")
    args = ap.parse_args()
    os.environ["VIEWCRAFT_CONFIG"] = str(args.config.resolve())
    import yaml
    from PIL import Image
    from viewcraft.evolution.chain import Chain
    from viewcraft.evolution.propose import Candidate
    from viewcraft.models import ModelClient
    from viewcraft.paths import runs_root
    from viewcraft.replay import RecordingClient, render_replay
    from viewcraft.runtime import run_static_item
    from viewcraft.schema import PublicItem, sha
    from viewcraft.views.config import Coordinates

    directory = ROOT / "skill-library" / args.skill
    if not directory.is_dir():
        directory = Path(args.skill).resolve()
    meta = yaml.safe_load((directory / "skill.yaml").read_text())
    coords = Coordinates.read_snapshot(directory)
    Chain._assert_expressions_parse(None, Candidate("joint", coords, "gallery", "gallery"))
    bundled = args.images is None and args.question is None
    if bool(args.images) != bool(args.question):
        ap.error("For your own case supply both --images and --question")
    if bundled:
        if not (directory / "case/request.json").exists():
            ap.error("This skill needs your own input. Supply --images and --question; see skill-library/README.md.")
        request = json.loads((directory / "case/request.json").read_text())
        image_paths = [str((directory / "case" / p).resolve()) for p in request["images"]]
    else:
        request = {"question": args.question, "options": args.options or [], "frame_times_s": args.frame_times or []}
        image_paths = [str(p.resolve()) for p in args.images]
    for path in image_paths:
        with Image.open(path) as im:
            im.verify()
    times = request.get("frame_times_s", [])
    if times and (len(times) != len(image_paths) or times != sorted(times)):
        raise ValueError("Frame timestamps must be sorted and match the image count")
    budget = meta["budget"]
    if any(type(budget.get(k)) is not int or budget[k] < 1 for k in ("observations", "max_images_per_request")):
        raise ValueError("Gallery budgets must be positive integers")
    if args.check:
        print(f"CHECK OK: {meta['id']}; {len(image_paths)} inputs; {coords.config_hash[:12]}")
        return
    client = ModelClient("executor")
    client.validate()
    out = args.output or runs_root() / "gallery" / meta["id"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if out.exists():
        raise ValueError(f"Output already exists: {out}. Choose a new --output to preserve the recorded run.")
    out.mkdir(parents=True)
    base_client = client
    if args.record:
        client = RecordingClient(client, out)
    item = PublicItem("gallery/" + meta["id"], "custom", request["question"], tuple(image_paths),
                      tuple(request.get("options", [])), meta={"frame_times_s": times} if times else {})
    trajectory = run_static_item(item, coords, client, budget, sha({"budget": budget, "skill": coords.config_hash}),
                                 arm="viewcraft", condition="normal", split="inference",
                                 answer_style=meta.get("answer_style", "short_text"))
    serialized = trajectory.to_dict()
    # Keep gallery artifacts portable; private machine paths have no user value.
    replacements = {path: f"input/{Path(path).name}" for path in image_paths}
    for view in serialized["views"]:
        view["source_media_id"] = replacements.get(view["source_media_id"], Path(view["source_media_id"]).name)
    result = {"skill": meta["id"], "skill_hash": coords.config_hash,
              "model": base_client.model_id, "answer": trajectory.answer_text,
              "status": trajectory.status, "error": trajectory.error,
              "observation_steps": trajectory.observation_steps,
              "recorded_at": datetime.now(timezone.utc).isoformat(),
              "input_tokens": trajectory.input_tokens, "output_tokens": trajectory.output_tokens,
              "case_check": None}
    if bundled:
        expected = json.loads((directory / "case/expected.json").read_text())
        result["case_check"] = compare_answer(trajectory.answer_text, expected)
    (out / "trajectory.json").write_text(json.dumps(serialized, indent=2, ensure_ascii=False))
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.record:
        render_replay(out, meta["title"], request["question"], result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Output:", out)
    if trajectory.error or not trajectory.exec_result or not trajectory.exec_result.passed:
        raise RuntimeError(trajectory.error or "Execution contract failed")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
