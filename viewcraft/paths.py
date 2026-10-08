"""The one place a filesystem root is decided.

Nine modules used to each carry their own copy of a hardcoded data root.
That was invisible until the tree had to run somewhere else: a reader cloning
the repo hits a `FileNotFoundError` from inside an adapter, with no indication
that the path is configurable at all.

Resolution order, most specific first:

  1. `VIEWCRAFT_DATA_ROOT` -- points directly at the benchmark data directory
  2. `VIEWCRAFT_ROOT` -- points at the project root; data is `<root>/data`
  3. `<repo>/data` -- the in-tree default a fresh clone gets

Nothing here creates directories. A missing data root is a real error and must
surface where it happens, not be papered over with an empty tree.
"""
from __future__ import annotations

import os
from pathlib import Path

# The repository itself: this file is <repo>/viewcraft/paths.py.
REPO_ROOT = Path(__file__).resolve().parents[1]

def project_root() -> Path:
    """The directory that holds `data/` (and, historically, the code)."""
    env = os.getenv("VIEWCRAFT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT


def data_root() -> Path:
    """Root of the benchmark data. Large files live here, never in the repo."""
    env = os.getenv("VIEWCRAFT_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "data"


def media_root() -> Path:
    """Where parquet-embedded images are materialized, content-addressed.

    A sibling of the datasets rather than a subdirectory of the repo: the
    files are derived data, can reach tens of GB, and must not be committed.
    """
    env = os.getenv("VIEWCRAFT_MEDIA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return data_root() / "_media"


def runs_root() -> Path:
    """Where run directories are written."""
    env = os.getenv("VIEWCRAFT_RUNS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "runs"


def describe() -> dict[str, str]:
    """Resolved roots, for a run manifest or a startup banner."""
    return {"repo_root": str(REPO_ROOT), "project_root": str(project_root()),
            "data_root": str(data_root()), "media_root": str(media_root()),
            "runs_root": str(runs_root())}
