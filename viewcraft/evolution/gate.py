"""Acceptance gate: paired comparison of current vs candidate.

Both sides use already-computed single-run results on the same frozen gate
set. Nothing here bootstraps, averages seeds, or re-verifies a condition - the
gate is a development rule, not a significance test.

Rule for every arm:
  * C_exec must pass and resource limits must hold
  * the two sides must cover the same items
  * task score must not drop
  * if task score rises, accept
  * if task score is exactly level, accept only if C_ground rises
  * otherwise reject

text_only carries no phi/psi change and view_only carries no theta change, but
both are judged by the same rule, so an arm never benefits from a laxer gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArmResults:
    """Aggregate of one arm's single-run results over the gate set."""
    task_score: float                 # normalized to [0, 1]
    ground_score: float | None        # None for no_grounding
    n_items: int
    exec_pass: bool
    resource_ok: bool
    unknown_rate: float = 0.0
    per_item_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    task_delta: float
    ground_delta: float | None
    epsilon: float
    repaired: int = 0
    regressed: int = 0

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted, "reason": self.reason,
            "task_delta": self.task_delta, "ground_delta": self.ground_delta,
            "epsilon": self.epsilon, "repaired": self.repaired,
            "regressed": self.regressed,
        }


def paired_counts(current: dict[str, float],
                  candidate: dict[str, float]) -> tuple[int, int]:
    """Items the candidate fixes and items it breaks, over shared item ids."""
    shared = set(current) & set(candidate)
    repaired = sum(1 for k in shared
                   if candidate[k] > current[k])
    regressed = sum(1 for k in shared
                    if candidate[k] < current[k])
    return repaired, regressed


def decide(arm: str, current: ArmResults,
           candidate: ArmResults) -> GateDecision:
    """Apply the arm's acceptance rule."""
    n = max(1, candidate.n_items)
    eps = 1.0 / n
    task_delta = candidate.task_score - current.task_score
    repaired, regressed = paired_counts(current.per_item_scores,
                                        candidate.per_item_scores)

    def reject(reason, gd=None):
        return GateDecision(False, reason, task_delta, gd, eps,
                            repaired, regressed)

    # The comparison is only meaningful on the same items. An unequal set means
    # some units failed or were skipped, and a score difference would then be
    # a sampling artefact rather than an effect of the candidate.
    cur_ids = set(current.per_item_scores)
    cand_ids = set(candidate.per_item_scores)
    if cur_ids != cand_ids:
        missing = len(cur_ids ^ cand_ids)
        return reject(
            f"gate sets differ by {missing} item(s): cannot compare "
            f"(current n={len(cur_ids)}, candidate n={len(cand_ids)})")

    # shared structural / resource constraints
    if not candidate.exec_pass:
        return reject("C_exec failed")
    if not candidate.resource_ok:
        return reject("resource or action limits violated")

    if arm in ("no_grounding", "reward_only_gate"):
        raise ValueError(
            f"arm {arm!r} disables the evidence-support side of the gate, "
            "which Equation 3 requires on both sides of a comparison")

    # the standard rule needs grounding on both sides
    if current.ground_score is None or candidate.ground_score is None:
        return reject("C_ground missing; cannot apply the acceptance rule")

    ground_delta = candidate.ground_score - current.ground_score

    if task_delta < 0:
        return reject("task score dropped", ground_delta)

    if task_delta > 0:
        # TASK IMPROVED -> ACCEPT. This is the standard skill-evolution
        # rule: keep the candidate whose downstream score went up.
        #
        # An earlier version vetoed a rising task score when C_ground fell
        # by more than epsilon. That veto cost a real improvement --
        # blink_ev4 round 2 raised task 0.812 -> 0.833 on a joint
        # layout+contract+skill candidate and was refused because C_ground
        # "fell" 0.5278, a drop the verifier produced by looking at four
        # separate images where the executor had seen one composed sheet.
        # Even with that replay bug fixed, a grounding veto on a rising
        # task score means an evidence-support heuristic can overrule the
        # benchmark's own metric, which is not a trade this study should
        # make. C_ground is still recorded on every round and reported; it
        # simply no longer blocks an accepted task gain.
        return GateDecision(True, "task score improved",
                            task_delta, ground_delta, eps, repaired,
                            regressed)

    # exactly level on task score: C_ground breaks the tie, in its favour.
    if ground_delta > 0:
        return GateDecision(True, "task level, C_ground improved",
                            task_delta, ground_delta, eps, repaired,
                            regressed)
    return reject("no change: task level and C_ground did not improve",
                  ground_delta)


def projection_eligible(decision: GateDecision, submissions: int,
                        submission_limit: int, evaluations: int,
                        evaluation_limit: int) -> bool:
    """Paper §3.4: a task loss, non-degraded support, and remaining budget."""
    return (not decision.accepted and decision.task_delta < 0
            and decision.ground_delta is not None and decision.ground_delta >= 0
            and submissions < submission_limit and evaluations < evaluation_limit)
