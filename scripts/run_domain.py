#!/usr/bin/env python3
"""Run one domain: an evolution chain of four rounds, then a test pass.

Domain-agnostic: everything specific to a benchmark comes from
viewcraft.adapters.registry, so this driver is the same code for every
dataset and an arm cannot accidentally get a different protocol on one domain.

Arms run in parallel with each other; within an arm, rounds are serial (a
candidate depends on the previous accepted snapshot). Test conditions are NOT
submitted here - they run only after every arm's final state is fixed, via
--stage test, so no test score can flow back into evolution.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Relocatable project root: this file lives at <project>/scripts/run_domain.py,
# so the project dir is two levels up regardless of where the tree is checked
# out. New-mechanism experiments can run from an isolated copy of the tree so
# they never share a runs/ ledger with an archived earlier mechanism.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viewcraft import splits                                    # noqa: E402
from viewcraft.splits_cap import cap_by_groups                 # noqa: E402
from viewcraft.adapters import registry                         # noqa: E402
from viewcraft.evolution.chain import Chain                      # noqa: E402
from viewcraft.initial import initial_coordinates                # noqa: E402
from viewcraft.jobs.ledger import Ledger                         # noqa: E402
from viewcraft.paths import runs_root
from viewcraft.models import CONFIG_PATH, CFG, ModelClient                    # noqa: E402
from viewcraft.runtime import run_static_item                    # noqa: E402
from viewcraft.schema import Trajectory, sha                     # noqa: E402
from viewcraft.views.config import Coordinates                   # noqa: E402

# The method runs as a single arm. The name is kept as a directory label
# so run layouts stay comparable with recorded runs.
ARMS = ["viewcraft"]


def code_version() -> str:
    """Fingerprint of the view languages this process actually loaded.

    WHY. A long chain imports `viewcraft.views.*` at launch and holds that
    code for hours. Fixing a bug on disk does NOT reach a chain already
    running. On 2026-09-13 the `pad` size guard was changed to shrink
    content instead of refusing at 08:31; mmlongdoc's chain, launched at
    06:01, kept raising `pad would be 224x4120` until 12:03, lost 19 of 99
    gate units, and round 2 was voided as incomparable (99 vs 80).

    The 19 losses were not the bug -- the bug was already fixed. They were
    STALE CODE, and nothing on disk or in the ledger said so. Recording the
    fingerprint means a later reader can tell which runs predate a fix
    instead of inferring it from file mtimes, and `stale_chains()` can warn
    while the run is still cheap to restart.
    """
    import hashlib
    h = hashlib.sha256()
    for name in ("views/exprs.py", "views/layout.py", "views/selection.py",
                 "views/scalars.py", "runtime.py", "evolution/chain.py",
                 "evolution/propose.py"):
        p = Path(__file__).resolve().parent.parent / "viewcraft" / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def warn_if_stale(run_dir: Path, version: str) -> None:
    """Say plainly when a resumed run's units were produced by older code."""
    marker = run_dir / "configs" / "code_version.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        seen = [l.strip() for l in marker.read_text().splitlines() if l.strip()]
        if seen and seen[-1] != version:
            print(f"!! code changed since this run last started: "
                  f"{seen[-1]} -> {version}. Units already in the ledger were "
                  f"produced by the OLDER code; a round mixing both is not a "
                  f"clean comparison.")
    with marker.open("a") as f:
        f.write(version + "\n")


def protocol_hash(domain: str, budget: dict, model_id: str,
                  scorer_name: str) -> str:
    """Identity of the measurement protocol.

    The scorer name is part of it: the same sample scored by a different
    metric is a different measurement, and two such results must never share
    a unit key and collide in the ledger.

    Deliberately NOT including `code_version()`: a bug fix would then change
    every unit key and discard thousands of valid units, which is a far
    larger harm than the staleness it would catch. Staleness is reported by
    `warn_if_stale` instead.
    """
    return sha({"domain": domain, "budget": budget, "executor": model_id,
                "runtime": "static_v2_multimedia", "scoring": scorer_name})


def load_samples(domain: str, run_dir: Path, args=None):
    spec = registry.get(domain)
    samples = spec.load(seed=CFG["run"]["seed"])
    if not samples:
        raise SystemExit(f"{domain}: loader returned no samples")

    plan_path = run_dir / "configs" / f"splits_{domain}.json"
    if plan_path.exists():
        plan = splits.load_plan(plan_path)
    else:
        spec_splits = CFG["splits"].get(domain) or CFG["splits"]["large_default"]
        # PILOT SCALE (--scale). Divides every split by the given factor so a
        # full protocol run fits in the time available, then the paper
        # reports the pilot honestly and the full-scale numbers replace it
        # later.
        #
        # This is a COST decision, not a threshold change: the acceptance
        # rule is untouched, and a smaller gate makes eps = 1/n LARGER, so
        # the bar to accept a candidate goes UP, not down. It cannot
        # manufacture a positive result. Splits stay disjoint at the same
        # grouping unit, so no image/document/scene spans two splits.
        scale = float(getattr(args, "scale", 1.0) or 1.0)
        if scale > 1.0:
            scaled = {k: v for k, v in spec_splits.items()
                      if not isinstance(v, (int, float))}
            for k, v in spec_splits.items():
                if isinstance(v, int):
                    # floor at 8 groups: below that a split is too small to
                    # carry the domain's question-type mix at all.
                    scaled[k] = max(8, int(round(v / scale)))
                elif isinstance(v, float):
                    scaled[k] = v / scale
            print(f"PILOT SCALE 1/{scale:g}: {spec_splits} -> {scaled}")
            spec_splits = scaled
        plan = splits.build_plan(domain, [s.group_key for s in samples],
                                 spec_splits, CFG["run"]["seed"])
        splits.save_plan(plan, plan_path)
    splits.verify_disjoint(plan)
    for s in samples:
        object.__setattr__(s, "split", plan.split_of(s.group_key) or "")

    grp = getattr(args, "type_group", None)
    if grp:
        from viewcraft.type_groups import group_of_sample
        before = len(samples)
        samples = [s for s in samples if group_of_sample(s) == grp]
        if not samples:
            raise SystemExit(
                f"{domain}: no samples in requirement group {grp!r}")
        kept = {}
        for s in samples:
            kept[s.split] = kept.get(s.split, 0) + 1
        print(f"TYPE GROUP {grp!r}: {before} -> {len(samples)} samples "
              f"{kept}")
    return samples, plan, spec


def run_evolution(args, domain, samples, spec, budget, phash, ledger):
    executor = ModelClient("executor")
    optimizer = ModelClient("optimizer")
    run_dir = runs_root() / args.run_id

    initial = (Coordinates.read_snapshot(Path(args.initial_skill))
               if args.initial_skill else initial_coordinates(domain))
    if CFG["evolution"].get("warmstart"):
        from viewcraft.evolution.warmstart import warmstart_coordinates
        initial = warmstart_coordinates(
            domain, samples, spec, executor, optimizer, budget, phash,
            run_dir, initial, concurrency=args.per_arm_concurrency)

    def one_arm(arm: str) -> dict:
        # deep copy: the module-level initial tables taught us that a shared
        # config object lets one run's edit mutate another's start.
        coords = copy.deepcopy(initial)
        return Chain(domain, arm, run_dir, samples, budget, phash,
                     executor, optimizer, ledger, coords,
                     train_batch_size=CFG["evolution"]["train_batch_size"],
                     rounds=(args.rounds or CFG["evolution"]["rounds"]),
                     seed=CFG["run"]["seed"],
                     concurrency=args.per_arm_concurrency).run()

    results = {}
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.arm_parallel) as ex:
        futs = {ex.submit(one_arm, a): a for a in args.arms}
        for fut in as_completed(futs):
            arm = futs[fut]
            try:
                results[arm] = fut.result()
                print(f"\n=== {domain}/{arm} finished "
                      f"({time.monotonic()-t0:.0f}s elapsed) ===", flush=True)
            except Exception as e:                            # noqa: BLE001
                import traceback
                traceback.print_exc()
                results[arm] = {"arm": arm, "error":
                                f"{type(e).__name__}: {str(e)[:300]}"}
                ledger.audit(args.run_id, f"{domain}/{arm}", "arm_failed",
                             {"error": str(e)[:400]})

    out = run_dir / "metrics" / "evolution_summary.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    if any("error" in r for r in results.values()):
        raise RuntimeError("Evolution failed; see evolution_summary.json")
    return results


def run_test(args, domain, samples, spec, budget, phash, ledger):
    """Evaluate every arm's FINAL snapshot on the held-out test split."""
    executor = ModelClient("executor")
    run_dir = runs_root() / args.run_id
    test = [s for s in samples if s.split == "test"]
    # SAME group->question cap the gate gets, from the SAME function.
    # The gate learned this lesson ("gate: 80" = 466 VSI questions) and the
    # fix went into EvolutionChain only; the test stage has its own sample
    # selection and so inherited the original defect -- VSI's "test: 64
    # groups" expanded to 1367 questions, 21x the plan, pushing the run to
    # 6.6h. `--test-cap` existed but was opt-in, which means it is off
    # exactly when someone forgets, i.e. when it matters.
    cap = args.test_cap or int(
        CFG["evolution"].get("test_question_cap", 0) or 0)
    test = cap_by_groups(
        test, cap, label="test",
        freeze_path=run_dir / "configs" / "test_capped_items.json")
    if not test:
        raise ValueError("Test split is empty; check dataset and split sizes.")
    print(f"test split: {len(test)} items")

    finals = {}
    for arm in args.arms:
        snap = run_dir / "arms" / arm / "snapshots" / "final"
        if snap.exists():
            finals[arm] = Coordinates.read_snapshot(snap)
        else:
            raise FileNotFoundError(
                f"Final skill missing: {snap}. Run --stage evolve first; "
                "use scripts/predict.py to execute an explicit initial skill.")

    def one(arm, coords, s):
        traj = run_static_item(s.public, coords, executor, budget, phash,
                               arm=arm, condition="normal", split="test",
                               answer_style=spec.answer_style)
        traj.score = spec.scorer(traj.answer_text, s.private, s.public)
        # Mixed-metric domains append the per-item metric, so an MRA and an
        # accuracy are never pooled into one headline number.
        metric = (s.private.meta or {}).get("metric")
        traj.scored_by = (f"{spec.scorer_name}:{metric}" if metric
                          else spec.scorer_name)
        return arm, s, traj

    def one_claimed(arm, coords, s, key, meta):
        """Claim at the moment of execution, then run.

        Returns None when another worker already holds or finished the unit,
        so the lease is held only while work is actually in flight.
        """
        if ledger.is_done(key) or not ledger.claim(key, meta):
            return None
        return one(arm, coords, s)

    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.arm_parallel *
                            args.per_arm_concurrency) as ex:
        futs = []
        # INTERLEAVE BY ITEM, NOT BY ARM.
        #
        # `for arm: for item:` queues every item of arm 1, then every item of
        # arm 2, and so on. The pool drains that queue in order, so at any
        # moment before completion the early arms are finished and the late
        # ones have zero results -- measured on nextgqa (242 / 80 / 0 / 0) and
        # mvbench (213 / 189 / 0). `viewcraft` sorts last in `finals`, so the
        # arm the paper is actually about was always the one with no data.
        #
        # Interleaving means an interrupted or still-running stage yields
        # roughly EQUAL n per arm, which is the only state in which the
        # arms are comparable at all. It also makes the partial table
        # honest rather than systematically biased against one arm.
        for s in test:
            for arm, coords in finals.items():
                key = Trajectory(
                    item_id=s.public.item_id, domain=domain, split="test",
                    arm=arm, condition="normal",
                    config_hash=coords.config_hash,
                    protocol_hash=phash).unit_key()
                meta = dict(domain=domain, split="test",
                            item_id=s.public.item_id, arm=arm,
                            condition="normal",
                            config_hash=coords.config_hash,
                            protocol_hash=phash)
                if ledger.is_done(key):
                    continue
                # CLAIM INSIDE THE WORKER, NOT HERE.
                #
                # Claiming in this loop leases every unit up front: chartqa's
                # test stage took all 980 leases in a few seconds and then
                # fed them to 4 workers. 245 viewcraft units sat leased and
                # untouched for 87 minutes behind a rate limiter, with a
                # 6-hour TTL and a LIVE owner -- so neither the TTL nor the
                # dead-owner check could reclaim them, and no other process
                # could help. The run serialises behind one straggler.
                #
                # A lease should mean "someone is working on this now", not
                # "someone intends to eventually". Claiming at the moment of
                # execution makes that true by construction, and leaves
                # unclaimed work visible to any other worker.
                futs.append((ex.submit(one_claimed, arm, coords, s, key,
                                       meta), key))
        for fut, key in futs:
            try:
                got = fut.result()
                if got is None:        # another worker had it
                    continue
                arm, s, traj = got
            except Exception as e:                            # noqa: BLE001
                ledger.mark(key, "failed", str(e)[:300])
                failures.append(str(e)[:300])
                continue
            if traj.error or not traj.exec_result or not traj.exec_result.passed:
                ledger.mark(key, "failed", traj.error or "execution contract failed")
                failures.append(traj.error or "execution contract failed")
                continue
            d = (run_dir / "arms" / arm / "trajectories" /
                 traj.config_hash[:12] / "test")
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{s.public.item_id.replace('/', '__')}.json"
            p.write_text(json.dumps(traj.to_dict(), ensure_ascii=False,
                                    indent=2, default=str))
            ledger.commit_done(key, traj.score, traj.status,
                               traj.request_ids, str(p))
            rows.append({"domain": domain, "arm": arm,
                         "item_id": s.public.item_id,
                         "category": category_of(s),
                         "scored_by": traj.scored_by,
                         "score": traj.score, "status": traj.status,
                         "steps": traj.observation_steps,
                         "images": traj.n_images_sent,
                         "sent_pixels": traj.sent_pixels,
                         "malformed_zooms": traj.malformed_zooms,
                         "input_tokens": traj.input_tokens,
                         "output_tokens": traj.output_tokens})

    p = run_dir / "metrics" / "test_rows.jsonl"
    with p.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"appended {len(rows)} test rows to {p}")
    if failures:
        raise RuntimeError(f"{len(failures)} test item(s) failed; results are incomplete. First error: {failures[0]}")
    return rows


def category_of(s) -> str:
    m = s.public.meta or {}
    for k in ("category", "question_type", "subset", "type"):
        if m.get(k):
            return str(m[k])
    return "all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=registry.all_names())
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--initial-skill", help="Directory containing SKILL.md, VIEW.yaml, LAYOUT.yaml")
    ap.add_argument("--stage", choices=["evolve", "test"], default="evolve")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="run directory labels; defaults to viewcraft")
    # PER-REQUIREMENT-GROUP EVOLUTION.
    #
    # Restricts every split to one question-type group, so the chain
    # searches a config for questions that want the same thing from the
    # imagery. See viewcraft/type_groups.py for the held-out measurement
    # that motivates it and for why the grouping is two clusters rather
    # than one-per-type.
    #
    # Applied to ALL runs or none: giving one run a finer grouping than
    # another would make a comparison between them a measurement of the
    # grouping instead of the method.
    ap.add_argument("--type-group", default=None,
                    choices=["metric", "relational"],
                    help="evolve/test within one requirement group only")
    ap.add_argument("--arm-parallel", type=int, default=4)
    ap.add_argument("--per-arm-concurrency", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="divide every split by this factor (pilot scale). "
                         "A smaller gate makes eps=1/n larger, i.e. raises "
                         "the acceptance bar; it cannot inflate a result.")
    ap.add_argument("--rounds", type=int, default=0,
                    help="override evolution.rounds for this run only. "
                         "Longer searches find more; applied identically "
                         "to every arm in the run.")
    ap.add_argument("--pilot", type=int, default=0,
                    help="pilot mode: cap gate and train to N groups each. "
                         "Results are for mechanism validation only and must "
                         "be labelled pilot, never reported as final.")
    ap.add_argument("--test-cap", type=int, default=0,
                    help="cap the test split; applied identically to all arms")
    args = ap.parse_args()

    if args.arms is None:
        args.arms = ARMS
    ModelClient("executor").validate()
    if args.stage == "evolve":
        ModelClient("optimizer").validate()

    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        ap.error("--run-id must be a single directory name")
    domain = args.domain
    run_dir = runs_root() / args.run_id
    for sub in ("configs", "metrics", "arms", "audit"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    budget = CFG["budgets"][domain]
    executor_id = CFG["models"]["executor"]["resolved_id"]
    samples, plan, spec = load_samples(domain, run_dir, args)
    if args.pilot:
        # Deterministic subset by item_id so every arm sees the identical
        # pilot set; the cap is declared up front, never chosen after
        # seeing which items an arm got right.
        keep = {"train": args.pilot, "gate": args.pilot}
        seen = {k: 0 for k in keep}
        out = []
        for s_ in sorted(samples, key=lambda x: x.public.item_id):
            sp = s_.split
            if sp in keep:
                if seen[sp] >= keep[sp]:
                    continue
                seen[sp] += 1
            out.append(s_)
        samples = out
        print(f"PILOT: train={seen.get('train',0)} gate={seen.get('gate',0)} "
              f"(mechanism validation only, not a reportable result)")
    phash = protocol_hash(domain, budget, executor_id, spec.scorer_name)
    ledger = Ledger(run_dir / "ledger.sqlite")
    ledger.reclaim_expired()

    print(f"run_id={args.run_id} domain={domain} stage={args.stage}")
    print(f"executor={executor_id} protocol_hash={phash[:12]}")
    _ver = code_version()
    print(f"code_version={_ver}")
    warn_if_stale(run_dir, _ver)
    print(f"samples={len(samples)} scorer={spec.scorer_name} "
          f"answer_style={spec.answer_style}")
    print(f"split counts={plan.counts} plan_hash={plan.plan_hash[:12]}")
    print(f"arms={args.arms}", flush=True)

    (run_dir / "configs" / "run_meta.json").write_text(json.dumps({
        "run_id": args.run_id, "domain": domain,
        "executor_model_id": executor_id,
        "optimizer_model_id": CFG["models"]["optimizer"]["resolved_id"],
        "protocol_hash": phash, "split_plan_hash": plan.plan_hash,
        "split_counts": plan.counts, "seed": CFG["run"]["seed"],
        "budget": budget, "arms": args.arms,
        "scorer": spec.scorer_name, "answer_style": spec.answer_style,
        "n_samples": len(samples),
        "rounds": args.rounds or CFG["evolution"]["rounds"],
        "train_batch_size": CFG["evolution"]["train_batch_size"],
        "domain_notes": spec.notes,
        # Throughput settings AS THIS PROCESS LOADED THEM. CFG is read once at
        # import, so editing configs/experiment.yaml while runs are in flight
        # changes nothing for them -- a burst fix applied at 07:17 left the
        # 06:34 processes still bursting at the old value and producing more
        # 429s, which looked like the fix had failed. Recording the effective
        # values means a run's own manifest says what it actually used.
        "effective_throughput": {
            "limiter_burst": CFG["throughput"]["limiter_burst"],
            "provider_rpm": CFG["throughput"]["provider_rpm"],
            "backoff_base_s": CFG["throughput"].get("backoff_base_s"),
        },
        "config_mtime": CONFIG_PATH.stat().st_mtime,
        "process_started_at": time.time(),
    }, indent=2, default=str))

    if args.stage == "evolve":
        run_evolution(args, domain, samples, spec, budget, phash, ledger)
    else:
        run_test(args, domain, samples, spec, budget, phash, ledger)

    print("\nledger counts:", ledger.counts())
    ledger.export_jsonl(run_dir / "jobs.jsonl", run_dir / "audit.jsonl")


if __name__ == "__main__":
    sys.exit(main() or 0)
