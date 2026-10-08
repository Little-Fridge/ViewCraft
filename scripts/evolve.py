#!/usr/bin/env python3
"""Evolve or evaluate a user task. Run --check before making model calls."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/models.yaml")
    parser.add_argument("--run-id", default="custom-demo")
    parser.add_argument("--stage", choices=("evolve", "test"), default="evolve")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--check", action="store_true", help="Validate data, skill, and splits without an API call")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("--run-id must be a single directory name")
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.environ["VIEWCRAFT_CONFIG"] = str(args.config.resolve())

    from viewcraft.custom import file_hash, load_task
    from viewcraft.adapters import registry
    from viewcraft.evolution.chain import Chain
    from viewcraft.jobs.ledger import Ledger
    from viewcraft.models import CFG, ModelClient
    from viewcraft.paths import runs_root
    from viewcraft.schema import sha

    task, samples, initial, spec, manifest = load_task(args.task)
    registry.register(spec)
    CFG["run"] = {"seed": task["seed"]}
    CFG["evolution"] = dict(task["evolution"])
    budget = task["budget"]
    print("Validated task:", json.dumps({"items": manifest["split_counts"],
          "groups": manifest["split_groups"], "scorer": spec.scorer_name,
          "initial_skill": initial.config_hash[:12], "budget": budget}, indent=2))
    if args.check:
        print("CHECK OK. No model calls made. Next: configure models and omit --check.")
        return

    executor = ModelClient("executor")
    executor.validate()
    optimizer = ModelClient("optimizer")
    if args.stage == "evolve":
        optimizer.validate()
    if os.getenv("VIEWCRAFT_SKIP_GROUNDING") == "1":
        raise ValueError("Unset VIEWCRAFT_SKIP_GROUNDING; the paired gate requires evidence support")

    run_dir = runs_root() / args.run_id
    manifest["models"] = {role: {"spec": client.spec, "model_id": client.model_id,
                                 "endpoint": client.endpoints[client.transport]}
                          for role, client in (("executor", executor), ("optimizer", optimizer))}
    manifest["throughput"] = CFG.get("throughput", {})
    manifest["code"] = {str(p.relative_to(ROOT)): file_hash(p)
                        for folder in (ROOT / "viewcraft", ROOT / "scripts")
                        for p in sorted(folder.rglob("*.py"))}
    identity = sha(manifest)
    manifest["identity"] = identity
    manifest["item_ids"] = {s.public.item_id: s.private.meta["source_id"] for s in samples}
    saved = run_dir / "configs/task_manifest.json"
    if saved.exists():
        if json.loads(saved.read_text())["identity"] != identity:
            raise ValueError("This run-id belongs to different data, code, models, or settings. Choose a new --run-id.")
    elif args.stage == "test":
        raise FileNotFoundError(f"No evolution run at {run_dir}; run --stage evolve first")
    elif run_dir.exists():
        raise ValueError("Run directory exists without a custom-task manifest; choose another --run-id")
    else:
        for sub in ("configs", "metrics", "arms", "audit"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        initial.write_snapshot(run_dir / "initial_skill")
    ledger = Ledger(run_dir / "ledger.sqlite")
    ledger.reclaim_expired()
    if args.stage == "evolve":
        chain = Chain("custom", "viewcraft", run_dir, samples, budget, identity,
                      executor, optimizer, ledger, initial,
                      train_batch_size=task["evolution"]["train_batch_size"],
                      rounds=task["evolution"]["rounds"], seed=task["seed"],
                      concurrency=args.workers)
        result = chain.run()
        (run_dir / "metrics/evolution_summary.json").write_text(json.dumps(result, indent=2))
        print("Final skill:", run_dir / "arms/viewcraft/snapshots/final")
        print("Next: repeat this command with --stage test, or load this directory with scripts/predict.py.")
    else:
        from run_domain import run_test
        run_test(SimpleNamespace(run_id=args.run_id, test_cap=0, arms=["viewcraft"],
                                arm_parallel=1, per_arm_concurrency=args.workers),
                 "custom", samples, spec, budget, identity, ledger)
        print("Test results:", run_dir / "metrics/test_rows.jsonl")
    ledger.export_jsonl(run_dir / "jobs.jsonl", run_dir / "audit.jsonl")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
