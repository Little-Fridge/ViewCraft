"""Regression tests for the viewcraft visual-projection fallback.

The invariant: when a joint candidate is rejected, the framework re-gates the
SAME visual change with theta reverted, so a good visual move is not vetoed by
a bad coupled text. viewcraft is then joint-first but never worse than
view_only in expectation. The fallback must revert theta, keep phi/psi, and
offer nothing when the candidate was theta-only.
"""
from __future__ import annotations

from viewcraft.evolution.propose import (CandidateRejected, describe_change,
                                         enforce_arm_scope)
from viewcraft.views.config import (Coordinates, PhiConfig, PsiConfig,
                                    SkillConfig)


def _cur():
    return Coordinates(theta=SkillConfig("current skill"),
                       phi=PhiConfig(pixel_expr=""),
                       psi=PsiConfig())


def test_projection_reverts_theta_keeps_visual():
    cur = _cur()
    cand = Coordinates(theta=SkillConfig("a BAD rewritten skill"),
                       phi=PhiConfig(pixel_expr="red + 0.4*(red-gaussian(red,1))"),
                       psi=PsiConfig(layout_expr="grid(views, cols=2)"))
    proj = enforce_arm_scope("view_only", cur, cand)
    assert proj.theta.text == "current skill"          # theta reverted
    assert "gaussian" in proj.phi.pixel_expr           # visual kept
    assert proj.psi.layout_expr == "grid(views, cols=2)"
    # the projection carries ONLY visual changes
    diff = describe_change(cur, proj)
    assert "theta" not in diff and "phi.pixel_expr" in diff


def test_theta_only_candidate_offers_no_projection():
    cur = _cur()
    theta_only = Coordinates(theta=SkillConfig("only text changed"),
                             phi=cur.phi, psi=cur.psi)
    try:
        enforce_arm_scope("view_only", cur, theta_only)
        raised = False
    except CandidateRejected:
        raised = True
    assert raised, "a theta-only candidate must offer no visual projection"


def test_fallback_flag_defaults_on():
    from viewcraft.models import CFG
    assert bool((CFG.get("evolution") or {}).get(
        "viewcraft_visual_fallback", True))
