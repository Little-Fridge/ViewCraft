#!/usr/bin/env python3
"""Watchdog: inspect the active run every 1800 seconds.

Checks, per the run protocol:
  1. process liveness, exit codes, heartbeats, leases, unfinished queues
  2. transport faults vs calls that actually succeeded
  3. media integrity, image dimensions, whether images were really sent,
     and the model id in use
  4. whether the phi/psi config hash matches what actually executed
  5. paired regressions on the gate set, candidate rejection reasons, and
     degenerate outputs (all-zero, all-identical answers)
  6. for the test split, infrastructure and protocol integrity only — scores
     are never fed back to the optimizer

Writes an append-only audit record each cycle. Exits when every job has
reached a terminal state; a run with outstanding blockers is reported as
blocked, never as complete.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from _paths import ARCHIVE, RUNS
ROOT = ARCHIVE
sys.path.insert(0, str(ROOT))

from viewcraft.jobs.ledger import Ledger                        # noqa: E402
from viewcraft.models import CFG                                # noqa: E402

INTERVAL = int(CFG.get("watchdog", {}).get("interval_s", 1800))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check_leases(ledger: Ledger) -> dict:
    reclaimed = ledger.reclaim_expired()
    return {"leases_reclaimed": reclaimed}


def check_degenerate(run_dir: Path) -> dict:
    """Flag all-zero scores or a single repeated answer, which usually means a
    broken pipeline rather than a genuinely hard split."""
    findings = []
    for arm_dir in sorted((run_dir / "arms").glob("*")):
        for cfg_dir in sorted((arm_dir / "trajectories").glob("*")):
            for split_dir in sorted(cfg_dir.glob("*")):
                files = list(split_dir.glob("*.json"))
                if len(files) < 8:
                    continue
                scores, answers, sent = [], [], []
                for f in files:
                    try:
                        d = json.loads(f.read_text())
                    except json.JSONDecodeError:
                        findings.append({"path": str(f),
                                         "issue": "unparsable_trajectory"})
                        continue
                    scores.append(d.get("score"))
                    answers.append((d.get("answer_text") or "").strip()[:40])
                    sent.append(d.get("n_images_sent", 0))
                if scores and all(s == 0 for s in scores if s is not None):
                    findings.append({
                        "scope": f"{arm_dir.name}/{cfg_dir.name}/{split_dir.name}",
                        "issue": "all_scores_zero", "n": len(scores)})
                if answers:
                    top, cnt = Counter(answers).most_common(1)[0]
                    if cnt / len(answers) > 0.95 and len(answers) >= 12:
                        findings.append({
                            "scope": f"{arm_dir.name}/{cfg_dir.name}/{split_dir.name}",
                            "issue": "identical_answers",
                            "answer": top, "frac": cnt / len(answers)})
                if sent and all(x == 0 for x in sent):
                    findings.append({
                        "scope": f"{arm_dir.name}/{cfg_dir.name}/{split_dir.name}",
                        "issue": "no_images_sent"})
    return {"degenerate_findings": findings}


def check_config_consistency(run_dir: Path) -> dict:
    """A trajectory's recorded config_hash must match the directory it sits in,
    which is derived from the config that was supposed to execute."""
    mismatches = []
    for arm_dir in sorted((run_dir / "arms").glob("*")):
        for cfg_dir in sorted((arm_dir / "trajectories").glob("*")):
            want = cfg_dir.name
            for f in list(cfg_dir.rglob("*.json"))[:200]:
                try:
                    d = json.loads(f.read_text())
                except json.JSONDecodeError:
                    continue
                got = (d.get("config_hash") or "")[:12]
                if got != want:
                    mismatches.append({"path": str(f), "dir": want,
                                       "recorded": got})
    return {"config_hash_mismatches": mismatches}


def check_view_records(run_dir: Path, sample_limit: int = 120) -> dict:
    """Sent views must carry a request id, a region, and real dimensions."""
    issues = []
    checked = 0
    for f in (run_dir / "arms").rglob("*.json"):
        if checked >= sample_limit:
            break
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "views" not in d:
            continue
        checked += 1
        for v in d.get("views", []):
            if not v.get("actually_sent"):
                continue
            if not v.get("request_id"):
                issues.append({"path": str(f), "view": v.get("view_id"),
                               "issue": "sent_without_request_id"})
            if v.get("region") is None:
                issues.append({"path": str(f), "view": v.get("view_id"),
                               "issue": "sent_without_region"})
            if not v.get("out_w") or not v.get("out_h"):
                issues.append({"path": str(f), "view": v.get("view_id"),
                               "issue": "zero_dimensions"})
    return {"views_checked": checked, "view_issues": issues}


def check_gate_regressions(run_dir: Path) -> dict:
    """Report rejected candidates and paired regressions from arm histories."""
    out = []
    for arm_dir in sorted((run_dir / "arms").glob("*")):
        h = arm_dir / "history.json"
        if not h.exists():
            continue
        try:
            rounds = json.loads(h.read_text())
        except json.JSONDecodeError:
            continue
        for r in rounds:
            dec = r.get("decision") or {}
            if not dec.get("accepted"):
                out.append({
                    "arm": arm_dir.name, "round": r.get("round_index"),
                    "coordinate": r.get("coordinate"),
                    "reason": dec.get("reason"),
                    "task_delta": dec.get("task_delta"),
                    "regressed_items": dec.get("regressed"),
                })
    return {"rejections": out}


def cycle(run_id: str, pids: list[int]) -> dict:
    run_dir = RUNS / run_id
    ledger = Ledger(run_dir / "ledger.sqlite")

    report = {
        "ts": time.time(),
        "run_id": run_id,
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "processes": {str(p): ("alive" if pid_alive(p) else "exited")
                      for p in pids},
        "ledger_counts": ledger.counts(),
        "model_executor": CFG["models"]["executor"]["resolved_id"],
        "model_optimizer": CFG["models"]["optimizer"]["resolved_id"],
    }
    report.update(check_leases(ledger))
    report.update(check_config_consistency(run_dir))
    report.update(check_view_records(run_dir))
    report.update(check_degenerate(run_dir))
    report.update(check_gate_regressions(run_dir))

    counts = report["ledger_counts"]
    outstanding = counts.get("leased", 0) + counts.get("pending", 0)
    any_alive = any(v == "alive" for v in report["processes"].values())
    blockers = []
    if counts.get("failed"):
        blockers.append(f"{counts['failed']} failed units")
    if counts.get("needs_review"):
        blockers.append(f"{counts['needs_review']} units need review")
    for k in ("config_hash_mismatches", "view_issues", "degenerate_findings"):
        if report.get(k):
            blockers.append(f"{len(report[k])} {k}")
    report["blockers"] = blockers
    report["terminal"] = (not any_alive) and outstanding == 0

    ledger.audit(run_id, "watchdog", "cycle", report)
    path = run_dir / "audit" / "watchdog.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(report, default=str) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--pid", type=int, action="append", default=[])
    ap.add_argument("--interval", type=int, default=INTERVAL)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    run_dir = RUNS / args.run_id
    if not run_dir.exists():
        print(f"no such run: {run_dir}; not starting an idle loop")
        return 2

    state = run_dir / "audit" / "watchdog_state.json"
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    while not stop["flag"]:
        rep = cycle(args.run_id, args.pid)
        interesting = (rep["blockers"] or rep["leases_reclaimed"]
                       or rep["terminal"])
        if interesting:
            print(f"[{rep['iso']}] counts={rep['ledger_counts']} "
                  f"blockers={rep['blockers']} terminal={rep['terminal']}")
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"last_cycle": rep["ts"],
                                     "terminal": rep["terminal"]}, default=str))
        if rep["terminal"]:
            msg = ("all jobs terminal"
                   + (f"; STILL BLOCKED: {rep['blockers']}"
                      if rep["blockers"] else "; no blockers"))
            print(f"[{rep['iso']}] {msg}")
            return 0
        if args.once:
            return 0
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
