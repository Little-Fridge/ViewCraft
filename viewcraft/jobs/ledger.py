"""Job ledger: one valid completion per experiment unit key.

A lease is claimed atomically before any model call, the response is persisted,
and only then is completion committed. A crash between call and commit leaves
the unit marked `uncertain` so it can be reconciled from the request record
rather than blindly re-run and cherry-picked.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
  unit_key      TEXT PRIMARY KEY,
  domain        TEXT NOT NULL,
  split         TEXT NOT NULL,
  item_id       TEXT NOT NULL,
  arm           TEXT NOT NULL,
  condition     TEXT NOT NULL,
  config_hash   TEXT NOT NULL,
  protocol_hash TEXT NOT NULL,
  env_seed      INTEGER,
  state         TEXT NOT NULL,          -- leased|done|failed|uncertain|invalid
  owner         TEXT,
  lease_expires REAL,
  score         REAL,
  status        TEXT,
  request_ids   TEXT,
  result_path   TEXT,
  invalid_reason TEXT,
  created_at    REAL NOT NULL,
  updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_units_state ON units(state);
CREATE INDEX IF NOT EXISTS idx_units_scope
  ON units(domain, split, arm, condition, config_hash);

CREATE TABLE IF NOT EXISTS audit (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        REAL NOT NULL,
  run_id    TEXT,
  scope     TEXT,
  event     TEXT NOT NULL,
  detail    TEXT
);
"""


class Ledger:
    """SQLite-backed unit ledger. Safe across threads and processes."""

    # A lease must outlast the SLOWEST legitimate unit by a wide margin.
    # Measured 2026-09-12: with the adaptive limiter holding ~64 rpm shared
    # across six domains, a queued video unit (8 frames, up to 24 observation
    # steps) waited 50-75 minutes between claim and completion. At the old
    # 3600s lease, 158 of 750 vsi leases had ALREADY EXPIRED while their work
    # was still in flight -- so the unit became re-claimable, a second worker
    # could start the same measurement, and the one-valid-result-per-unit rule
    # was only being held by luck. Widened to 6 hours: long enough that a
    # queued unit never expires, short enough that a genuinely dead process
    # still releases its work within one monitoring window.
    DEFAULT_LEASE_SECONDS = 21600.0

    def __init__(self, path: Path,
                 lease_seconds: float = DEFAULT_LEASE_SECONDS):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = lease_seconds
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=60.0,
                                   isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=60000")
            self._local.conn = conn
        yield conn

    # -- leases ----------------------------------------------------------

    @staticmethod
    def _owner_alive(owner: str | None) -> bool:
        """Is the process that holds this lease still running?

        A LEASE IS A PROMISE TO FINISH, and a dead process cannot keep it.
        The TTL alone is not enough: it is 6 hours, so a process killed at
        07:00 kept 3 units hostage until 13:00. Measured on K1 -- four of
        seven chains sat in `_await_contended` at <4% CPU for 12+ minutes
        waiting on PIDs that no longer existed, while the run had only
        ~8 hours of budget left.

        Checked with signal 0, which tests existence without touching the
        process. A PID recycled onto a different program is the only false
        positive, and that merely restores today's behaviour of waiting
        for the TTL.
        """
        if not owner:
            return False
        try:
            pid = int(str(owner).split(":", 1)[0])
        except (TypeError, ValueError):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def claim(self, unit_key: str, meta: dict) -> bool:
        """Atomically claim a unit. False if it is done or actively leased.

        "Actively leased" means the lease is unexpired AND its owner is
        alive; a lease held by a dead process is reclaimable immediately.
        """
        now = time.time()
        owner = f"{os.getpid()}:{threading.get_ident()}"
        with self._conn() as c:
            try:
                c.execute("BEGIN IMMEDIATE")
                row = c.execute(
                    "SELECT state, lease_expires, owner FROM units "
                    "WHERE unit_key=?", (unit_key,)).fetchone()
                if row is not None:
                    state, expires, holder = row
                    if state in ("done", "invalid"):
                        c.execute("COMMIT")
                        return False
                    if state == "leased" and (expires or 0) > now \
                            and self._owner_alive(holder):
                        c.execute("COMMIT")
                        return False
                    c.execute(
                        "UPDATE units SET state='leased', owner=?, "
                        "lease_expires=?, updated_at=? WHERE unit_key=?",
                        (owner, now + self.lease_seconds, now, unit_key))
                else:
                    c.execute(
                        "INSERT INTO units (unit_key, domain, split, item_id, "
                        "arm, condition, config_hash, protocol_hash, env_seed, "
                        "state, owner, lease_expires, created_at, updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?, 'leased', ?, ?, ?, ?)",
                        (unit_key, meta["domain"], meta["split"],
                         meta["item_id"], meta["arm"], meta["condition"],
                         meta["config_hash"], meta["protocol_hash"],
                         meta.get("env_seed"), owner,
                         now + self.lease_seconds, now, now))
                c.execute("COMMIT")
                return True
            except sqlite3.Error:
                c.execute("ROLLBACK")
                raise

    def touch(self, unit_key: str) -> None:
        """Extend an active lease. Called while a unit is still being worked.

        Belt and braces alongside the longer default: a unit that genuinely
        takes hours keeps its claim, while a crashed owner stops touching and
        its lease lapses on schedule.
        """
        with self._conn() as c:
            c.execute(
                "UPDATE units SET lease_expires=?, updated_at=? "
                "WHERE unit_key=? AND state='leased'",
                (time.time() + self.lease_seconds, time.time(), unit_key))

    def commit_done(self, unit_key: str, score: float | None, status: str,
                    request_ids: list[str], result_path: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE units SET state='done', score=?, status=?, "
                "request_ids=?, result_path=?, updated_at=? WHERE unit_key=?",
                (score, status, json.dumps(request_ids), result_path,
                 time.time(), unit_key))

    def mark(self, unit_key: str, state: str, detail: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE units SET state=?, status=?, updated_at=? "
                "WHERE unit_key=?", (state, detail[:400], time.time(),
                                     unit_key))

    def invalidate(self, unit_keys: list[str], reason: str) -> int:
        """Mark units invalid, keeping the original record and the reason."""
        with self._conn() as c:
            n = 0
            for k in unit_keys:
                c.execute(
                    "UPDATE units SET state='invalid', invalid_reason=?, "
                    "updated_at=? WHERE unit_key=?",
                    (reason[:400], time.time(), k))
                n += c.total_changes
            return n

    # -- queries ---------------------------------------------------------

    def get(self, unit_key: str) -> dict | None:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM units WHERE unit_key=?",
                            (unit_key,)).fetchone()
            return dict(row) if row else None

    def is_done(self, unit_key: str) -> bool:
        u = self.get(unit_key)
        return bool(u and u["state"] == "done")

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT state, COUNT(*) FROM units GROUP BY state").fetchall()
        return {s: n for s, n in rows}

    def scope_results(self, domain: str, split: str, arm: str, condition: str,
                      config_hash: str) -> dict[str, float]:
        """item_id -> score for one fully-specified condition."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT item_id, score FROM units WHERE domain=? AND split=? "
                "AND arm=? AND condition=? AND config_hash=? AND state='done'",
                (domain, split, arm, condition, config_hash)).fetchall()
        return {i: s for i, s in rows if s is not None}

    def reclaim_expired(self) -> int:
        """Release leases whose owner died OR whose lease expired.

        This docstring used to say "whose owner died" while the SQL only
        checked `lease_expires < now`. Those are different conditions, and
        the gap burns quota: two test processes killed at 13:20 held 4049
        leases on a 6-hour TTL, so their replacements sat idle until 19:00
        waiting on units whose owner had not existed for hours.

        `claim()` already refuses to respect a dead owner's lease via
        `_owner_alive`. This entry point did not -- the recurring
        "invariant enforced at 1 of N entry points" class. Liveness is the
        primary test; the TTL remains the fallback for an owner on another
        host, whose pid means nothing locally.
        """
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "UPDATE units SET state='pending', owner=NULL "
                "WHERE state='leased' AND lease_expires < ?", (now,))
            freed = cur.rowcount
            rows = c.execute(
                "SELECT DISTINCT owner FROM units WHERE state='leased' "
                "AND owner IS NOT NULL").fetchall()
            for owner in [r[0] for r in rows if not self._owner_alive(r[0])]:
                cur = c.execute(
                    "UPDATE units SET state='pending', owner=NULL "
                    "WHERE state='leased' AND owner=?", (owner,))
                freed += cur.rowcount
            return freed

    # -- audit -----------------------------------------------------------

    def audit(self, run_id: str, scope: str, event: str,
              detail: dict | str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO audit (ts, run_id, scope, event, detail) "
                "VALUES (?,?,?,?,?)",
                (time.time(), run_id, scope, event,
                 json.dumps(detail, default=str) if not isinstance(detail, str)
                 else detail))

    def audit_tail(self, n: int = 50) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM audit ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    def export_jsonl(self, units_path: Path, audit_path: Path) -> None:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            with Path(units_path).open("w") as f:
                for row in c.execute("SELECT * FROM units ORDER BY created_at"):
                    f.write(json.dumps(dict(row), default=str) + "\n")
            with Path(audit_path).open("w") as f:
                for row in c.execute("SELECT * FROM audit ORDER BY id"):
                    f.write(json.dumps(dict(row), default=str) + "\n")
