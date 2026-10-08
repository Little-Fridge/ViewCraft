"""Filesystem roots for the analysis scripts, resolved instead of hardcoded.

Every script here reads `ROOT / "runs"` through ARCHIVE rather than importing
`viewcraft.paths`, on purpose: some scripts re-read trajectories recorded by
another checkout of this code, and importing `viewcraft` first would pin
`sys.modules["viewcraft"]` to THIS copy before the script can choose which
library should load. The duplication of a dozen lines is cheaper than that
failure, which would be invisible and would change published numbers.

Overrides, for running anywhere else:

    VIEWCRAFT_ROOT          project dir holding `data/`
    VIEWCRAFT_DATA_ROOT     the benchmark data directly
    VIEWCRAFT_ARCHIVE_ROOT  a checkout whose `runs/` should be analysed

Without overrides, PROJECT, ARCHIVE and DATA all resolve inside this repo.
"""
from __future__ import annotations

import os
from pathlib import Path

# This file is <repo>/scripts/_paths.py.
REPO = Path(__file__).resolve().parents[1]

def _project_root() -> Path:
    env = os.getenv("VIEWCRAFT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return REPO


def _archive_root() -> Path:
    env = os.getenv("VIEWCRAFT_ARCHIVE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return REPO


def _data_root() -> Path:
    env = os.getenv("VIEWCRAFT_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "data"


PROJECT = _project_root()
ARCHIVE = _archive_root()
DATA = _data_root()
RUNS = Path(os.environ.get("VIEWCRAFT_RUNS_ROOT", ARCHIVE / "runs")).expanduser().resolve()
