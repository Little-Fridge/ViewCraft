#!/usr/bin/env python3
"""One-glance health check across every running domain.

Answers the three questions that matter while a batch is in flight:
  * is work still moving, or has everything gone quiet?
  * is any unit FAILING, and with what cause?  (a failed unit becomes a
    missing gate item, which silently invalidates a paired comparison)
  * are the arms actually diverging, or is every candidate being rejected?

Read-only. Opens each ledger with mode=ro so it can never block or corrupt a
run that is mid-write.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from _paths import ARCHIVE, RUNS
ROOT = ARCHIVE


def ledger_states(db: str) -> dict:
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        states = {r[0]: r[1] for r in c.execute(
            "SELECT state, COUNT(*) FROM units GROUP BY state")}
        recent = c.execute(
            "SELECT MAX(updated_at) FROM units").fetchone()[0] or 0
        fails = c.execute(
            "SELECT status, COUNT(*) FROM units WHERE state IN "
            "('failed','needs_review') GROUP BY status LIMIT 5").fetchall()
        return {"states": states, "last_update": recent,
                "failures": [(s or "")[:110] for s, _ in fails]}
    except sqlite3.Error as e:
        return {"error": str(e)[:80]}


def main() -> int:
    now = time.time()
    print(f"=== ViewCraft status  {time.strftime('%H:%M:%S')} ===\n")

    live = os.popen("ps ax -o args | grep run_domain.py | grep -v grep"
                    ).read().strip().splitlines()
    running = set()
    for line in live:
        for tok in line.split():
            pass
        if "--domain" in line:
            parts = line.split()
            running.add(parts[parts.index("--domain") + 1])
    print(f"processes alive: {len(live)}  domains={sorted(running) or '-'}")
    # Throughput ceiling, made visible. A 429 is retried silently, so
    # without this the logs show zero throttling while the adaptive limiter
    # sinks and every run slows for no apparent reason.
    try:
        sys.path.insert(0, str(ROOT))
        from viewcraft.models import limiter_for, throttle_events
        eff = limiter_for("google").effective_rate_per_min()
        n10 = throttle_events(600)
        print(f"google limiter: {eff:.0f} rpm effective   "
              f"429s in last 10min: {n10}")
    except Exception as e:                                   # noqa: BLE001
        print(f"limiter status unavailable: {type(e).__name__}")
    print()

    print(f"{'run':34s} {'done':>6s} {'lease':>6s} {'fail':>5s} "
          f"{'idle':>7s}")
    print("-" * 64)
    stalled, failing, abandoned = [], [], []
    for db in sorted(glob.glob(str(RUNS / "*/ledger.sqlite"))):
        run = Path(db).parent.name
        if "SUPERSEDED" in run or "INVALID" in run:
            continue
        info = ledger_states(db)
        if "error" in info:
            print(f"{run:34s} {info['error']}")
            continue
        st = info["states"]
        idle = now - (info["last_update"] or now)
        done = st.get("done", 0)
        print(f"{run:34s} {done:6d} {st.get('leased', 0):6d} "
              f"{st.get('failed', 0) + st.get('needs_review', 0):5d} "
              f"{idle/60:6.1f}m")
        if info["failures"]:
            failing.append((run, info["failures"]))
        # A run with outstanding leases and no write for 20 minutes is stuck,
        # not slow: the longest legitimate single unit is a handful of model
        # calls. But a run whose process was deliberately stopped is
        # ABANDONED, not stalled -- reporting those as stalls buried the one
        # real signal under six false ones after the operator-architecture
        # switch retired the w9/w10 runs.
        if st.get("leased") and idle > 1200:
            if run in running:
                stalled.append((run, idle / 60))
            else:
                abandoned.append((run, idle / 60))

    print()
    for run in sorted(RUNS.glob("*/arms/*/history.json")):
        if "SUPERSEDED" in str(run) or "INVALID" in str(run):
            continue
        try:
            h = json.loads(run.read_text())
        except json.JSONDecodeError:
            continue
        if not h:
            continue
        arm = run.parent.name
        rid = run.parents[2].name
        acc = sum(1 for r in h if r["decision"]["accepted"])
        last = h[-1]
        diff = last["diff"][:52]
        print(f"{rid:22s} {arm:11s} rounds={len(h)} accepted={acc} "
              f"| last: {last['coordinate']}: {diff} "
              f"gate {last['gate_current']:.3f}->"
              f"{last['gate_candidate']:.3f} "
              f"{'ACCEPT' if last['decision']['accepted'] else 'reject'}")

    if failing:
        print("\n!! FAILING UNITS (each one is a missing gate item):")
        for run, fs in failing:
            for f in fs:
                print(f"   {run}: {f}")
    if stalled:
        print("\n!! STALLED (process ALIVE, leases held, no write >20min):")
        for run, mins in stalled:
            print(f"   {run}: idle {mins:.0f}m")
    if abandoned:
        # Deliberately stopped runs, not faults. Listed so their leases are
        # visible without drowning a real stall.
        print("\n(abandoned: process gone, leases left behind -- "
              f"{', '.join(r for r, _ in abandoned)})")
    if not failing and not stalled:
        print("\nno failures, no stalls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
