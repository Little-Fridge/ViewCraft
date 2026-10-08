"""BabyVision TRANSFER adapter, gpt-6-synthesised v2 sets (bvgs_*).

Same protocol as bvsynth.py -- evolve on synthetic items whose gold is known
by construction, test the frozen config on the real BabyVision subset -- but
reading the v2 generator output (babygen/out_v2/<sub>/, ~200 items per
subtype, written by gpt-6-astra against real examples via
scripts/bvgs_author.py) instead of the original 24-item sets.

Differences from bvsynth.py are deliberately minimal: the source directory,
the subtype table (adds find_shadow), and nothing else. Scoring is imported
from bvsynth so the two generations of the experiment are strictly
comparable.

Splits are FORCED by a pre-written plan (scripts/prep_bvgs.py): synthetic
groups land in train/gate/diagnostic, real groups in test.
"""
from __future__ import annotations

import re
from pathlib import Path

from .bvsynth import REAL_ROOT  # noqa: F401
from .bvsynth import _norm_mapping, _norm_numbers, _norm_text
from . import bvsynth as _v1

DOMAIN = "bvgsynth"

BABYGEN = Path(__file__).resolve().parents[2] / "babygen" / "out_v2"

SUBTYPES = {
    "maze": {"syn_dir": "maze", "real_subtype": "Maze"},
    "count_clusters": {"syn_dir": "count_clusters",
                       "real_subtype": "Count Clusters"},
    "find_different": {"syn_dir": "find_different",
                       "real_subtype": "Find the different"},
    "connect_lines": {"syn_dir": "connect_lines",
                      "real_subtype": "Connect the lines"},
    "find_shadow": {"syn_dir": "find_shadow",
                    "real_subtype": "Find the shadow"},
}


def score(answer_text: str, private, public) -> float:
    """bvsynth.score plus one measured extraction repair.

    Calibration (bvgs_calib_connect_lines.json, item connect_lines_v2_0039)
    showed a correct reply 'A-1, B-2, C-3' scored 0 against gold '1,2,3':
    _norm_numbers parses the mapping separators as minus signs, so the reply
    reads as (-1,-2,-3). Golds in these subtypes are counts / grid coords /
    option ordinals -- never negative -- so when the strict comparison
    fails, retry with letters stripped from the reply before extracting
    numbers. This changes ONLY the bvgs_* metrics (scorer_name
    bvgs_*_acc_v1); the bv_* runs keep bvsynth.score byte-for-byte.
    """
    got = _v1.score(answer_text, private, public)
    if got == 1.0:
        return 1.0
    meta = private.meta or {}
    if meta.get("origin") == "real":
        if meta.get("choice_letter"):
            return 0.0
        want = _norm_numbers(meta.get("blank_ans", ""))
    else:
        gold = str(private.answer).strip()
        if re.fullmatch(r"[A-Za-z]", gold) or \
                re.search(r"\d\s*[-=:]\s*[A-Za-z]", gold):
            return 0.0
        want = _norm_numbers(gold)
    if not want or any(v < 0 for v in want):
        return 0.0
    stripped = re.sub(r"[A-Za-z]", " ", str(answer_text) or "")
    return 1.0 if _norm_numbers(stripped) == want else 0.0


def load(root: str | Path, seed: int = 20260911,
         subtype: str = "maze") -> list:
    """bvsynth.load with the v2 paths and table; identical item semantics."""
    old_table, old_root = _v1.SUBTYPES, _v1.BABYGEN
    old_domain = _v1.DOMAIN
    try:
        _v1.SUBTYPES = SUBTYPES
        _v1.BABYGEN = BABYGEN
        _v1.DOMAIN = DOMAIN
        return _v1.load(root, seed=seed, subtype=subtype)
    finally:
        _v1.SUBTYPES, _v1.BABYGEN = old_table, old_root
        _v1.DOMAIN = old_domain
