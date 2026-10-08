"""Regression tests for the strategy induction layer (pure-logic parts).

The reflection step calls an LLM and is exercised in integration; here we pin
the book's data invariants: persistence round-trips, the diversity driver picks
the highest-value UNRESOLVED strategy, and outcome write-back moves a strategy
through its lifecycle without ever touching a measurement.
"""
from __future__ import annotations

from viewcraft.evolution.strategy_book import (Strategy, StrategyBook,
                                               directive_block, record_outcome)


def _book():
    return StrategyBook(strategies=[
        Strategy(failure_mode="target too small to read",
                 mechanism="zoom-native crops", prevalence=0.4),
        Strategy(failure_mode="lost track while counting",
                 mechanism="mark-as-you-count on the canvas", prevalence=0.6),
    ])


def test_next_target_picks_highest_prevalence_untried():
    b = _book()
    tgt = b.next_target()
    assert "counting" in tgt.failure_mode        # 0.6 > 0.4


def test_next_target_skips_resolved():
    b = _book()
    b.by_key("lost track while counting").status = "helped"
    tgt = b.next_target()
    assert "too small" in tgt.failure_mode       # the helped one is skipped


def test_directive_names_unresolved_and_forbids_done():
    b = _book()
    b.by_key("target too small to read").status = "hurt"
    blk = directive_block(b)
    assert "STRATEGY MAP" in blk
    assert "lost track while counting" in blk     # aimed at the unresolved one
    assert "Do NOT re-propose" in blk             # past failures are barred


def test_record_outcome_marks_helped_and_hurt():
    b = _book()
    record_outcome(b, "phi.canvas_expr ... mark each counted dot",
                   "ring the counted dots to track them", True, 1)
    assert b.by_key("lost track while counting").status == "helped"
    record_outcome(b, "phi.crop_long_edge_px for small text",
                   "read the tiny label", False, 2)
    assert b.by_key("target too small to read").status == "hurt"


def test_roundtrip_persistence(tmp_path):
    b = _book()
    p = tmp_path / "book.json"
    b.save(p)
    b2 = StrategyBook.load(p)
    assert len(b2.strategies) == 2
    assert b2.by_key("lost track while counting").prevalence == 0.6


def test_empty_book_has_no_directive():
    assert directive_block(StrategyBook()) == ""
    assert StrategyBook().next_target() is None
