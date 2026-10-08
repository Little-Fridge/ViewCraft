#!/usr/bin/env python3
"""Collect every finished run into one tidy table, then plot.

Reads trajectories off disk rather than any in-memory state, so the numbers
in a figure can always be traced back to the individual recorded runs. Nothing
here recomputes a score: it reads the score that was written at execution
time, because re-scoring later with different code is how a result quietly
stops matching its own trajectory files.

Two rules this enforces rather than trusts:
  * an arm's number is reported with its denominator, and a missing unit is
    counted as missing rather than dropped from the mean
  * metrics are grouped by `scored_by`; an MRA and an accuracy are never
    averaged into one headline number
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from _paths import ARCHIVE, RUNS
ROOT = ARCHIVE
OUT = ROOT / "results"

ARM_ORDER = ["baseline", "text_only", "view_only", "viewcraft"]

# scorer_name -> domain, straight from the registry. `scorer_name` is the
# ledger's metric identity and is unique per domain by construction (that is
# why `blink` and `blink_hard` carry different ones despite sharing an
# adapter and a data_root).
def _build_scorer_map() -> dict:
    sys.path.insert(0, str(ROOT))
    from viewcraft.adapters import registry as _reg
    out = {}
    for nm in _reg.all_names():
        try:
            out[_reg.get(nm).scorer_name] = nm
        except Exception:                                    # noqa: BLE001
            continue
    return out


_SCORER_TO_DOMAIN = _build_scorer_map()


ARM_LABEL = {"baseline": "baseline", "text_only": "text only (θ)",
             "view_only": "view only (φ,ψ)", "viewcraft": "ViewCraft (full)"}


def _domain_of(d: dict) -> str:
    """The domain a trajectory really belongs to.

    Looked up in the REGISTRY by scorer_name, not parsed out of the string:
    a first attempt stripped suffixes like "_accuracy_v1" and mapped
    `mmlongdoc_generalized_accuracy_v1` to "mmlongdoc_generalized" and
    `vsi_mra_and_accuracy_v1` to "vsi_mra_and". Guessing at a naming
    convention reintroduces the same class of error this function exists to
    fix; the registry already states the mapping.
    """
    scored = str(d.get("scored_by") or "").split(":")[0]
    return _SCORER_TO_DOMAIN.get(scored) or str(d.get("domain") or "")


def load_trajectories(run_dir: Path) -> list[dict]:
    rows = []
    for p in (run_dir / "arms").rglob("trajectories/*/*/*.json"):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        rows.append({
            "run": run_dir.name,
            # PREFER THE SCORER'S DOMAIN OVER THE TRAJECTORY'S OWN FIELD.
            #
            # Adapters set `domain` from a module constant, so two specs
            # sharing one adapter both wrote the same value: every
            # blink_hard trajectory recorded `domain="blink"` because
            # `adapters/blink.py` has `DOMAIN = "blink"`. `scorer_name` is
            # the ledger's metric identity and was correct throughout
            # ("blink_hard_accuracy_v1"), so no SCORE was ever mixed -- but
            # grouping figures by the `domain` field split one run across
            # two labels and lost its ViewCraft bar.
            #
            # registry.DomainSpec.load now stamps the spec name at load
            # time; this recovers the right label for the 1412 trajectories
            # already written under the old code, so they need no re-run.
            "domain": _domain_of(d),
            "arm": d.get("arm"),
            "split": d.get("split"),
            "condition": d.get("condition"),
            "config_hash": d.get("config_hash", "")[:12],
            "item_id": d.get("item_id"),
            "score": d.get("score"),
            "scored_by": d.get("scored_by", ""),
            "status": d.get("status"),
            "steps": d.get("observation_steps", 0),
            "images": d.get("n_images_sent", 0),
            "sent_pixels": d.get("sent_pixels", 0),
            "malformed_zooms": d.get("malformed_zooms", 0),
            "in_tok": d.get("input_tokens", 0),
            "out_tok": d.get("output_tokens", 0),
            "answer": (d.get("answer_text") or "")[:120],
            "grounding": (d.get("grounding") or {}).get("score"),
            "unknown_rate": (d.get("grounding") or {}).get("unknown_rate"),
            "n_crops": sum(1 for v in (d.get("views") or [])
                           if v.get("kind") == "crop"
                           and v.get("actually_sent")),
        })
    return rows


def load_history(run_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for p in (run_dir / "arms").glob("*/history.json"):
        try:
            out[p.parent.name] = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
    return out


def accepted_config(run_dir: Path, arm: str) -> str | None:
    """The config hash the arm actually ENDED on, read from its snapshot.

    This must come from the arm's own recorded state, never be inferred from
    the data. Choosing "the config with the most units" picked a REJECTED
    candidate: hallusionbench/viewcraft accepted nothing, so its state is the
    round-00 snapshot, but a rejected candidate had been evaluated over the
    full 138-item gate set while the initial config kept only 56 units after
    cross-arm sharing. The collector then reported 0.848 -- a configuration
    the arm had explicitly refused -- as that arm's score, and that number
    drove the "view_only beats viewcraft everywhere" reading.

    `final` is written at the end of every chain; `round00` is the
    initialization, which is the correct answer for an arm that accepted
    nothing.
    """
    snaps = run_dir / "arms" / arm / "snapshots"
    if not snaps.is_dir():
        return None
    # `final` exists only once the chain has finished. Mid-run, falling
    # straight back to round00 reported every arm at its INITIAL config, so
    # arms that had already accepted a candidate were indistinguishable from
    # baseline. A `roundNN` snapshot is written ONLY on acceptance, so the
    # highest-numbered one is the arm's current accepted state.
    order = []
    for d in snaps.iterdir():
        if not (d / "CONFIG_HASH").exists():
            continue
        if d.name == "final":
            rank = 10 ** 6
        elif d.name.startswith("round"):
            try:
                rank = int(d.name[5:])
            except ValueError:
                continue
        else:
            continue
        order.append((rank, d))
    if not order:
        return None
    h = (max(order)[1] / "CONFIG_HASH").read_text().strip()
    return h[:12] if h else None


# An absolute floor on what may be called a comparable cell, independent of
# any denominator derived from the data. The smallest split this study runs is
# ~100 questions after the group->question cap, so a cell under 30 items is a
# remnant, not a measurement.
MIN_COMPARABLE_N = 30


def summarize(rows: list[dict], split: str, min_n: int = 0,
              accepted: dict[tuple, str] | None = None) -> dict:
    """Per (domain, arm) mean with its denominator, grouped by metric.

    `min_n` drops cells too small to mean anything. A mid-run collection
    produced `ocrbench baseline n=1 score=0.000` sitting in the same table as
    `n=149 score=0.899`, which invites reading a single in-flight unit as an
    arm's performance. The threshold is on COVERAGE, not on score, so it can
    never select for a result: it is computed as a fraction of the largest
    cell in the same (domain, metric) group, which is the only honest
    reference for "how many units this condition should have".

    THE SELF-JUSTIFYING DENOMINATOR. That reference is derived from the
    OBSERVED rows, so a domain whose only surviving rows are leftovers
    declares itself complete: `blink` had 7 stale test rows, `expected`
    became 7, and 7/7 passed the 0.9 coverage bar. fig4 then drew a 7-item
    bar beside dude's 341-item bar at equal weight and equal visual
    authority. A completeness test that takes its target from the same data
    it is testing can only ever pass, so an ABSOLUTE floor is required as
    well -- see `MIN_COMPARABLE_N`.
    """
    # Bucket by CONFIG HASH too. On the gate split an arm holds several
    # config hashes -- the accepted state plus every REJECTED candidate that
    # was evaluated against it. Pooling them produced an "arm score" that is
    # a weighted blend of configs the arm explicitly refused, and an
    # n_expected of 278 against a 140-item gate set. A score must belong to
    # exactly one configuration.
    buckets = defaultdict(list)
    for r in rows:
        if r["split"] != split or r["score"] is None:
            continue
        buckets[(r["domain"], r["arm"], r["scored_by"],
                 r.get("config_hash", ""))].append(r)

    # Keep the arm's ACCEPTED config where the snapshot tells us which it is;
    # otherwise fall back to the widest-coverage one, ties broken by hash for
    # determinism. Never report a config the arm refused.
    accepted = accepted or {}

    # AN ARM THAT ACCEPTED NOTHING SHARES baseline's CONFIG, AND ITS ROWS ARE
    # FILED UNDER WHICHEVER ARM CLAIMED EACH UNIT FIRST.
    #
    # `unit_key` excludes `arm` by design (one execution per measurement), so
    # on DUDE all 341 test units of the shared initial config are recorded
    # under `baseline`. Selecting rows by arm name then found NOTHING for
    # view_only and viewcraft -- both of which had complete results -- and
    # fig4 showed dude and mmlongdoc as a lone baseline bar, which reads as
    # "the visual arms were never measured" when they were measured and
    # scored identically to baseline by construction.
    #
    # So synthesise the missing cells from the config the arm actually ended
    # on. This adds no new measurement: it points a second label at the one
    # shared result, which is exactly what "the arm stayed at its start"
    # means.
    by_cfg: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r["split"] == split and r["score"] is not None:
            by_cfg[(r["domain"], r["scored_by"],
                    r.get("config_hash", ""))].append(r)
    present = {(r["domain"], r["arm"], r["scored_by"],
                r.get("config_hash", "")) for r in rows
               if r["split"] == split and r["score"] is not None}
    for key, cfg in list(accepted.items()):
        dom, arm = key[-2], key[-1]
        for (d2, metric, c2), rs2 in by_cfg.items():
            if d2 != dom or c2 != cfg:
                continue
            if (dom, arm, metric, cfg) in present:
                continue
            buckets[(dom, arm, metric, cfg)] = [
                {**r, "arm": arm} for r in rs2]

    best: dict[tuple, tuple] = {}
    for key, rs in buckets.items():
        dom, arm, metric, cfg = key
        k = (dom, arm, metric)
        # Look up by (run, domain, arm) first, falling back to the
        # (domain, arm) form. Keying ONLY by run broke every caller that
        # does not track runs -- including the guard test that pins "never
        # report a config the arm rejected", which silently fell through to
        # the widest-bucket branch and reported a refused candidate. Both
        # shapes must resolve, because the run dimension exists to
        # disambiguate duplicates, not to become newly mandatory.
        want = (accepted.get((rs[0].get("run", ""), dom, arm))
                or accepted.get((dom, arm)))
        if want:
            if cfg == want:
                # Key on the CONFIG too. Two runs of one domain each match
                # their own accepted config, and with a (dom, arm, metric)
                # key the second overwrote the first -- a 5-unit pilot
                # replacing a 161-unit result. `collapsed` below then picks
                # between them by coverage instead of by iteration order.
                best[(k, cfg)] = key
            continue
        cur = best.get((k, None))
        if cur is None or (len(rs), cfg) > (len(buckets[cur]), cur[3]):
            best[(k, None)] = key
    # Collapse to (domain, arm, metric), keeping the WIDEST cell when two
    # runs of one domain both survive selection. Without this the dict write
    # order decides: blink_hard_u1's 5-unit pilot silently replaced
    # blink_hard_K1's 161-unit result for the same (domain, arm, metric).
    # Coverage, not iteration order, should pick the row.
    collapsed: dict[tuple, list] = {}
    for k in best.values():
        short = (k[0], k[1], k[2])
        if short not in collapsed or len(buckets[k]) > len(collapsed[short]):
            collapsed[short] = buckets[k]
    buckets = collapsed

    # Coverage reference: the largest cell per (domain, metric) over ALL
    # config hashes, not just the selected ones.
    #
    # Arms that accepted nothing END on the shared initial config, and because
    # the unit key excludes the arm, those units are recorded under whichever
    # arm happened to execute each one first. So an arm's OWN row count for
    # the initial config can be a handful even though the config was measured
    # on the whole gate set. Taking the reference from the selected cells made
    # three complete arms look 2/136 "PARTIAL".
    #
    # The honest denominator is the number of DISTINCT ITEMS measured for that
    # config, pooled across arms - one execution per item, shared by
    # construction.
    per_config_items: dict[tuple, set] = defaultdict(set)
    for r in rows:
        if r["split"] != split or r["score"] is None:
            continue
        per_config_items[(r["domain"], r["scored_by"],
                          r.get("config_hash", ""))].add(r["item_id"])
    expected: dict[tuple, int] = {}
    for (dom, metric, _cfg), items in per_config_items.items():
        k = (dom, metric)
        expected[k] = max(expected.get(k, 0), len(items))

    out = {}
    for (dom, arm, metric), rs in sorted(buckets.items()):
        cfg = rs[0].get("config_hash", "") if rs else ""
        # Units for this config, pooled across arms: with the arm excluded
        # from the unit key, one recorded execution serves every arm holding
        # that config, so this IS this arm's coverage.
        shared = [r for r in rows
                  if r["split"] == split and r["score"] is not None
                  and r["domain"] == dom and r["scored_by"] == metric
                  and r.get("config_hash", "") == cfg]
        by_item = {}
        for r in shared:
            by_item.setdefault(r["item_id"], r)
        rs = list(by_item.values())
        n = len(rs)
        want = expected[(dom, metric)]
        complete = n >= max(min_n, MIN_COMPARABLE_N, 0.9 * want)
        out[f"{dom}|{arm}|{metric}"] = {
            "domain": dom, "arm": arm, "metric": metric, "n": n,
            "n_expected": want,
            # An arm covering far fewer units than its peers is not
            # comparable to them; flagged rather than silently averaged.
            "coverage_complete": bool(complete),
            "score": sum(x["score"] for x in rs) / n,
            "mean_steps": sum(x["steps"] for x in rs) / n,
            "mean_images": sum(x["images"] for x in rs) / n,
            "mean_crops": sum(x["n_crops"] for x in rs) / n,
            "zoom_rate": sum(1 for x in rs if x["n_crops"] > 0) / n,
            "mean_pixels": sum(x["sent_pixels"] for x in rs) / n,
            "malformed_rate": sum(1 for x in rs
                                  if x["malformed_zooms"]) / n,
            "truncated_rate": sum(1 for x in rs
                                  if x["status"] == "truncated") / n,
            "empty_answer_rate": sum(1 for x in rs
                                     if not x["answer"].strip()) / n,
            "mean_in_tok": sum(x["in_tok"] for x in rs) / n,
            "mean_out_tok": sum(x["out_tok"] for x in rs) / n,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run ids; default = every directory under runs/")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    run_dirs = ([RUNS / r for r in args.runs] if args.runs else
                sorted(d for d in RUNS.iterdir()
                       if d.is_dir() and not d.name.startswith("_")))

    all_rows, histories = [], {}
    for d in run_dirs:
        if not (d / "arms").exists():
            continue
        rows = load_trajectories(d)
        all_rows.extend(rows)
        h = load_history(d)
        if h:
            histories[d.name] = h
        print(f"{d.name:28s} {len(rows):6d} trajectories")

    # tidy long-form table: one row per executed unit
    tidy = OUT / "trajectories.jsonl"
    with tidy.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {tidy} ({len(all_rows)} rows)")

    # (domain, arm) -> the config hash each arm ended on, from its snapshot.
    accepted: dict[tuple, str] = {}
    for d in run_dirs:
        armdir = d / "arms"
        if not armdir.exists():
            continue
        doms = {r["domain"] for r in all_rows if r["run"] == d.name}
        for arm_path in sorted(armdir.iterdir()):
            if not arm_path.is_dir():
                continue
            h = accepted_config(d, arm_path.name)
            if not h:
                continue
            for dom in doms:
                # Key by RUN as well. Two runs of the same domain both write
                # `accepted[(domain, arm)]`, so the last one iterated wins:
                # `blink_hard_u1` (an old single-arm pilot) overwrote
                # blink_hard/viewcraft with a config that does not exist in
                # `blink_hard_K1`, and K1's 161 complete viewcraft units were
                # then discarded as "not the accepted config". The figure
                # lost the ViewCraft bar for the domain with the paper's
                # result in it.
                accepted[(d.name, dom, arm_path.name)] = h

    for split in ("gate", "test", "train"):
        s = summarize(all_rows, split, accepted=accepted)
        if not s:
            continue
        p = OUT / f"summary_{split}.json"
        p.write_text(json.dumps(s, indent=2))
        print(f"\n--- {split} ---")
        print(f"{'domain':16s} {'arm':12s} {'metric':26s} "
              f"{'n':>4s}/{'exp':<4s} {'score':>7s} {'zoom%':>6s} "
              f"{'imgs':>5s}  note")
        for v in s.values():
            note = "" if v["coverage_complete"] else "PARTIAL - not comparable"
            print(f"{v['domain']:16s} {v['arm']:12s} {v['metric']:26s} "
                  f"{v['n']:4d}/{v['n_expected']:<4d} {v['score']:7.3f} "
                  f"{v['zoom_rate']*100:5.1f}% {v['mean_images']:5.2f}  {note}")

    (OUT / "histories.json").write_text(json.dumps(histories, indent=2))
    print(f"\nwrote {OUT / 'histories.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
