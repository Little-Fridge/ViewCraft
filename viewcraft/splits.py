"""Frozen data splits.

Splits are decided by source, question type and media integrity only — never
by any measured score. Seed 20260911 is used once here for sampling and never
for repeated runs. Grouping is by media unit (image / doc / scene / video) so
the same media can never appear in two splits.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .schema import sha

SEED = 20260911


@dataclass(frozen=True)
class SplitPlan:
    """Mapping from group key -> split name, plus the counts actually achieved."""
    domain: str
    mode: str
    assignment: dict[str, str]
    counts: dict[str, int]
    requested: dict[str, int]
    seed: int = SEED

    @property
    def plan_hash(self) -> str:
        return sha({"domain": self.domain, "mode": self.mode,
                    "assignment": self.assignment, "seed": self.seed})

    def split_of(self, group_key: str) -> str | None:
        return self.assignment.get(group_key)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain, "mode": self.mode, "seed": self.seed,
            "requested": self.requested, "counts_achieved": self.counts,
            "plan_hash": self.plan_hash, "assignment": self.assignment,
        }


def assign_counts(groups: list[str], requested: dict[str, int],
                  seed: int = SEED) -> dict[str, str]:
    """Assign whole groups to splits by requested absolute counts.

    Deterministic given (groups, requested, seed). If the pool is smaller than
    requested, later splits shrink and the real denominators are reported by
    the caller rather than silently padded.
    """
    ordered = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    out: dict[str, str] = {}
    i = 0
    for name in ("diagnostic", "train", "gate", "test"):
        want = requested.get(name, 0)
        take = ordered[i:i + want]
        for g in take:
            out[g] = name
        i += len(take)
    return out


def assign_fractions(groups: list[str], fracs: dict[str, float],
                     seed: int = SEED) -> dict[str, str]:
    """Assign whole groups by fraction using the largest-remainder method."""
    ordered = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)

    names = ["diagnostic", "train", "gate", "test"]
    exact = {k: fracs[k] * n for k in names}
    base = {k: int(exact[k]) for k in names}
    remainder = n - sum(base.values())
    # largest remainder first, ties broken by fixed name order
    order = sorted(names, key=lambda k: (-(exact[k] - base[k]), names.index(k)))
    for k in order[:remainder]:
        base[k] += 1

    out: dict[str, str] = {}
    i = 0
    for k in names:
        for g in ordered[i:i + base[k]]:
            out[g] = k
        i += base[k]
    return out


def build_plan(domain: str, groups: list[str], spec: dict,
               seed: int = SEED) -> SplitPlan:
    """Build a split plan from the config's `splits` entry for a domain."""
    mode = spec.get("mode", "by_count")
    uniq = sorted(set(groups))

    if any(k.endswith("_frac") for k in spec):
        fracs = {k.replace("_frac", ""): v for k, v in spec.items()
                 if k.endswith("_frac")}
        assignment = assign_fractions(uniq, fracs, seed)
        requested = {k: round(v * len(uniq)) for k, v in fracs.items()}
    else:
        requested = {k: spec[k] for k in
                     ("diagnostic", "train", "gate", "test") if k in spec}
        assignment = assign_counts(uniq, requested, seed)

    counts: dict[str, int] = {}
    for s in assignment.values():
        counts[s] = counts.get(s, 0) + 1

    return SplitPlan(domain=domain, mode=mode, assignment=assignment,
                     counts=counts, requested=requested, seed=seed)


def save_plan(plan: SplitPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True))


def load_plan(path: Path) -> SplitPlan:
    d = json.loads(Path(path).read_text())
    return SplitPlan(domain=d["domain"], mode=d["mode"],
                     assignment=d["assignment"], counts=d["counts_achieved"],
                     requested=d["requested"], seed=d["seed"])


def verify_disjoint(plan: SplitPlan) -> None:
    """A group belongs to exactly one split, by construction of the dict.

    Kept as an explicit check so the acceptance test has something to call.
    """
    seen: dict[str, str] = {}
    for group, split in plan.assignment.items():
        if group in seen and seen[group] != split:
            raise AssertionError(f"group {group} in two splits")
        seen[group] = split
