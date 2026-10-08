"""Credential resolution for ViewCraft.

Credentials are read from the environment only. Nothing is defaulted,
written to config, or logged. Import this rather than reading os.environ
directly so there is exactly one place that touches secrets.
"""
from __future__ import annotations

import os


class MissingCredentials(RuntimeError):
    pass


# The AIHUB_* names are this project's own convention. The Taiji runtime
# injects the SAME credential pair into every shell under its platform names,
# so those are accepted as a fallback: requiring the operator to re-export
# values that are already present led to a run failing at startup for no
# reason. Order matters - the project-specific name wins, so an explicit
# override in the launching shell is still respected.
_APP_ID_VARS = ("AIHUB_APP_ID", "APP_ID")
_API_KEY_VARS = ("AIHUB_API_KEY", "APP_KEY")


def _first_set(names: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Return (value, which_var) for the first name that has a value."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v, n
    return None, None


def auth_pair() -> tuple[str, str]:
    """Return (app_id, api_key) from the environment.

    Raises MissingCredentials with an actionable message if either is unset,
    so a misconfigured run fails loudly at startup instead of emitting
    hundreds of 401s.
    """
    app_id, _ = _first_set(_APP_ID_VARS)
    api_key, _ = _first_set(_API_KEY_VARS)
    missing = []
    if not app_id:
        missing.append(" or ".join(_APP_ID_VARS))
    if not api_key:
        missing.append(" or ".join(_API_KEY_VARS))
    if missing:
        raise MissingCredentials(
            f"unset environment variable(s): {', '.join(missing)}. "
            "Export them in the shell that launches the run.")
    return app_id, api_key


def source_vars() -> dict[str, str]:
    """Which env var each credential came from. Names only, never values.

    Recorded in the run manifest so an auditor can tell which credential
    path a run used without the secret ever being written down.
    """
    _, id_var = _first_set(_APP_ID_VARS)
    _, key_var = _first_set(_API_KEY_VARS)
    return {"app_id_from": id_var or "<unset>",
            "api_key_from": key_var or "<unset>"}


def redact(text: str) -> str:
    """Strip credential material from a string before it reaches a log."""
    out = text
    for var in set(_APP_ID_VARS + _API_KEY_VARS + ("AIHUB_V2_KEY", "OPENAI_API_KEY")) | {k for k in os.environ if k.endswith(("_API_KEY", "_TOKEN"))}:
        val = os.getenv(var)
        if val:
            out = out.replace(val, f"[{var}]")
    return out
