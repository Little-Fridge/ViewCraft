#!/usr/bin/env python3
"""One-screen status of every running chain.

Written after two monitoring ticks were wasted on wrong readings:

  * liveness was read off `ledger.sqlite`'s mtime, but SQLite in WAL mode
    writes to `ledger.sqlite-wal` and only folds back on checkpoint -- a
    healthy chain looked 45 minutes idle.
  * round history was printed with guessed key names (`round`, `gate_score`)
    and silently rendered every entry as `r?:?.`

Both are the same mistake: reading a derived artifact instead of the real
one. This script reads the real ones, so a tick is one command.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from _paths import RUNS
ROOT = Path(__file__).resolve().parent.parent
ARMS = ("text_only", "view_only", "viewcraft")


def freshness(db: str) -> float:
    """Seconds since this ledger was last written, WAL included."""
    return time.time() - max(os.path.getmtime(p) for p in
                             (db, db + "-wal", db + "-shm") if os.path.exists(p))


def history(path: Path) -> str:
    if not path.exists():
        return "-"
    try:
        rounds = json.loads(path.read_text())
    except json.JSONDecodeError:          # mid-write
        return "(writing)"
    if not isinstance(rounds, list) or not rounds:
        return "-"
    out = []
    for e in rounds[-3:]:
        i = e.get("round_index", "?")
        cur, cand = e.get("gate_current"), e.get("gate_candidate")
        fmt = (lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "----")
        decision = e.get("decision", {})
        accepted = decision.get("accepted", False) if isinstance(decision, dict) else str(decision).startswith("accept")
        mark = "A" if accepted else "."
        out.append(f"r{i}:{fmt(cur)}>{fmt(cand)}{mark}")
    return " ".join(out)


def baseline_by_config(db: str) -> str:
    """Gate score per CONFIG, never per (config, arm).

    `unit_key` deliberately excludes `arm`: an arm is provenance, not part
    of what is measured, so three arms holding the same config share ONE
    recorded result (that is the "each condition runs exactly once" rule).
    In the ledger the `arm` column records only which chain happened to
    claim the unit first, so the 96 gate items of a shared starting config
    get PARTITIONED across the three arms.

    Grouping by (config, arm) therefore shows three disjoint subsets of one
    measurement and reads as a contradiction -- blink_hard's single 0.649
    baseline appeared as 0.567 / 0.743 / 0.621, a 0.176 spread that looked
    exactly like the "identical configs scoring differently" bug class and
    cost a tick to chase down. Group by config alone.
    """
    # RESTRICT TO THE FROZEN GATE SET.
    #
    # A ledger can hold gate rows for items that are no longer in the gate:
    # when `cap_by_groups` was changed to cover question categories, VSI's
    # resumed rounds measured 77 items the original gate never contained,
    # and those rows landed on the SAME config hashes as rounds 1-2. Pooling
    # them reads baseline as 0.466 instead of its real 0.418 -- a +0.048
    # shift from items that are not part of the measurement.
    #
    # `configs/gate_capped_items.json` is the authority on what the gate is.
    # Where it exists, every gate aggregate must be filtered through it.
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    frozen = None
    fp = Path(db).parent / "configs" / "gate_capped_items.json"
    if fp.exists():
        try:
            frozen = set(json.loads(fp.read_text()))
        except (json.JSONDecodeError, OSError):
            frozen = None

    per = {}
    for cfg, item, sc in con.execute(
            "select config_hash, item_id, score from units "
            "where split='gate' and score is not null"):
        if frozen is not None and item not in frozen:
            continue
        per.setdefault(cfg, []).append(sc)
    # Print n with every score and let the reader judge, rather than
    # dropping small cells silently: a hidden cell is indistinguishable from
    # a cell that does not exist, which is the mistake fig4 made twice.
    top = sorted(per.items(), key=lambda kv: -len(kv[1]))[:3]
    return "  ".join(f"{cfg[:6]}:{sum(v)/len(v):.3f}(n={len(v)})"
                     for cfg, v in top if v)


def main(pattern: str = "*_K1") -> int:
    dbs = sorted(glob.glob(str(RUNS / pattern / "ledger.sqlite")))
    if not dbs:
        print(f"no runs matching {pattern}")
        return 1
    live = {}
    for line in os.popen("ps -eo pid=,pcpu=,args=").read().splitlines():
        if "scripts/run_domain" in line and "grep" not in line:
            parts = line.split()
            for tok in parts:
                if tok.endswith(("_K1", "_K2")):
                    live[tok] = float(parts[1])

    print(f"{'run':16s} {'cpu%':>6s} {'idle':>7s} {'ok':>6s} {'fail':>5s}"
          f"  rounds (text_only | view_only | viewcraft)")
    total_ok = 0
    for db in dbs:
        run = Path(db).parent.name
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        ok = con.execute(
            "select count(*) from units where status='ok'").fetchone()[0]
        bad = con.execute(
            "select count(*) from units where status not in ('ok','pending')"
        ).fetchone()[0]
        total_ok += ok
        hs = " | ".join(history(Path(db).parent / "arms" / a / "history.json")
                        for a in ARMS)
        cpu = live.get(run)
        cpu_s = f"{cpu:6.0f}" if cpu is not None else "  DEAD"
        print(f"{run:16s} {cpu_s} {freshness(db):6.0f}s {ok:6d} {bad:5d}  {hs}")
        bl = baseline_by_config(db)
        if bl:
            print(f"{'':16s} gate by config: {bl}")
    print(f"\ntotal ok units: {total_ok}")

    # Anything the chains refused, newest first -- the signal that a language
    # gap is still converting visual proposals into nothing.
    #
    # This query once read `where kind like '%reject%'`, but the audit table's
    # columns are (id, ts, run_id, scope, event, detail). SQLite raised
    # "no such column: kind", the except swallowed it, and the monitor printed
    # "rejections: none" for hours while rejections were in fact accumulating.
    # A monitor that reports all-clear when its own query is broken is worse
    # than no monitor, so the column list is asserted up front and any error
    # is printed rather than caught.
    seen = {}
    for db in dbs:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cols = {r[1] for r in con.execute("pragma table_info(audit)")}
        missing = {"event", "detail"} - cols
        if missing:
            raise SystemExit(
                f"{db}: audit table lacks {sorted(missing)}; monitor query is "
                f"stale (columns are {sorted(cols)})")
        for event, detail in con.execute(
                "select event, detail from audit where event != 'round_decided'"
                " order by id desc limit 20"):
            key = f"{event}: {str(detail)[:100]}"
            seen[key] = seen.get(key, 0) + 1
    if seen:
        print("\nrejections (newest 20/run):")
        for k, v in sorted(seen.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {v:3d}  {k}")
    else:
        print("\nrejections: none")

    # THROTTLES AS A RATE OVER A WINDOW, PLUS THE LIMITER'S ACTUAL STATE.
    #
    # This used to print the last three raw lines of the append-only 429 log.
    # Raw lines are cumulative, so a count taken over "the last 30 minutes"
    # was read as current pressure when it was really the residue of eight
    # concurrent chains that had since exited: 192 stale 429s led to a
    # diagnosis of "the adaptive limiter has locked itself down" while the
    # factor was in fact a healthy 1.0 and the live rate was zero.
    #
    # Same mistake as reading the ledger's mtime instead of its WAL, and as
    # grouping a shared config by arm: a cumulative artefact mistaken for a
    # present-tense reading. So print the WINDOWED rate next to the factor
    # the limiter is actually enforcing.
    thr = RUNS / "_limiter" / "throttle_counts.txt"
    if thr.exists():
        now = time.time()
        lines = [l for l in thr.read_text().splitlines()[-20000:] if l.strip()]
        def since(w):
            return sum(1 for l in lines
                       if now - float(l.split(",")[0]) < w)
        print(f"\n429s: last 60s={since(60)}  5min={since(300)}  "
              f"30min={since(1800)}  (total logged {len(lines)})")
    for b in sorted((RUNS / "_limiter").glob("*.bucket")):
        try:
            last, tokens, factor = b.read_text().strip().split(",")
        except (ValueError, OSError):
            continue
        age = time.time() - float(last)
        print(f"  limiter {b.stem:8s} factor={float(factor):.3f} "
              f"tokens={float(tokens):.2f} (touched {age:.0f}s ago)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "*_K1"))
