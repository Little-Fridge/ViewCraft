"""Model access layer for ViewCraft.

Responsibilities:
  * resolve executor / optimizer model IDs from the frozen config
  * speak both AI Hub transports (standard router + passthrough)
  * global shared rate limiting (not per-worker)
  * content-hash response caching, so identical requests across arms reuse
    one inference instead of re-calling
  * retry on transport faults only (timeout / 429 / 5xx), never on a
    successfully-returned wrong answer

Credentials come from viewcraft.credentials, which reads the environment and
never logs or persists them.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml

from .credentials import auth_pair, redact
from .paths import runs_root

ROOT = Path(__file__).resolve().parents[1]
# Env override so parallel self-evolution lines (gpt-5.5 line, Qwen-vLLM
# line) can run from the same tree with different model profiles; the
# committed default stays the recorded experiment.
CONFIG_PATH = Path(os.environ.get(
    "VIEWCRAFT_CONFIG", ROOT / "configs" / "experiment.yaml"))


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("models"), dict):
        raise ValueError("Model config requires a models mapping")
    for role, spec in cfg["models"].items():
        if any(k in spec for k in ("api_key", "token", "authorization")):
            raise ValueError(f"Use models.{role}.api_key_env; keep credentials out of YAML")
        override = os.getenv(f"VIEWCRAFT_{role.upper()}_MODEL")
        if override:
            spec["resolved_id"] = override
    return cfg


CFG = load_config()

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ModelUnavailable(RuntimeError):
    """Raised when the requested model cannot be reached at all."""


class ModelFormatError(RuntimeError):
    """Raised when a response cannot be parsed under the registered protocol."""


class RateLimiter:
    """Token-bucket limiter shared process-wide across all workers."""

    def __init__(self, rate_per_min: float | None, burst: int = 8):
        self.rate = (rate_per_min / 60.0) if rate_per_min else None
        self.burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rate is None:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.burst,
                                   self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            # sleep OUTSIDE the lock: holding it would serialize every waiter
            # behind one sleeping thread and starve the bucket refill check
            time.sleep(min(wait, 0.5))


class CrossProcessRateLimiter:
    """Token bucket shared by every PROCESS on this machine, via a lock file.

    The in-process limiter is correct for one process and useless for six.
    Running six domains concurrently, each with its own RateLimiter set to
    the provider's full rpm, sent 6x the intended rate at one provider and
    produced a 429 flood that failed 98 units across four runs - units the
    gate then reported as an incomplete set.

    The bucket state lives in one small file per provider under a fixed
    directory, guarded by fcntl.flock. Every acquire takes the lock, refills
    by elapsed time, and either consumes a token or sleeps. Contention is
    tiny compared with a multi-second model call.

    A stale or unreadable state file is treated as a full bucket rather than
    an error: failing closed here would halt an experiment because of a
    scratch file, and the worst case is one extra burst.

    ADAPTIVE. The provider's real ceiling is not a constant. Measured on
    2026-09-12, the achieved rate fell 159 -> 86 -> 51 units/min at a
    configured 195 rpm; after cutting to 120 rpm it still fell
    64 -> 34 -> 26. A fixed rate cannot track a moving ceiling, and the
    overshoot is not free: it converts throughput into 429 waits and
    eventually into LOST GATE UNITS (ocrbench viewcraft r2 had a candidate
    refused for "gate sets differ by 1 item" because one unit was throttled
    out). So the bucket carries a multiplicative factor in its state file:
    `penalise()` halves it when the provider returns 429, and successful
    calls restore it slowly.

    This is transport control only. It changes latency and nothing else - no
    protocol, metric, threshold, split or score depends on it.
    """

    # Never drop below this fraction of the configured rate: a permanent
    # collapse would look like progress stalling for no visible reason.
    MIN_FACTOR = 0.15
    # Recovery is deliberately slower than the cut. Climbing back as fast as
    # we backed off just re-triggers the throttle in a loop.
    # Recovery must be slow relative to how OFTEN the cut fires, not merely
    # slower than one cut. At +0.02 per success with ~24 successes/minute the
    # factor climbed ~0.48/min, so a halving was undone within seconds:
    # measured, the limiter oscillated 52% -> 100% while taking 59 throttles
    # in ten minutes, and never settled at a rate the provider sustains.
    # 0.002 recovers ~0.05/min at the same success rate, slow enough for the
    # equilibrium to sit where the throttles stop.
    RECOVER_PER_OK = 0.002

    def __init__(self, name: str, rate_per_min: float | None,
                 burst: int = 8, root: Path | None = None):
        self.name = name
        self.rate = (rate_per_min / 60.0) if rate_per_min else None
        self.burst = float(burst)
        self.root = Path(root or (runs_root() / "_limiter"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{name}.bucket"
        self._local = threading.Lock()

    def _rw(self, fn):
        """Run `fn(now, last, tokens, factor)` under the file lock and persist.

        `fn` returns (last, tokens, factor, retval). Serializing this
        process's threads first keeps N threads from all contending on the
        file lock for a bucket only one can win.
        """
        import fcntl
        with self._local:
            with open(self.path, "a+b") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    raw = f.read().decode("ascii", "ignore").strip()
                    now = time.time()
                    try:
                        parts = raw.split(",")
                        last, tokens = float(parts[0]), float(parts[1])
                        factor = float(parts[2]) if len(parts) > 2 else 1.0
                    except (ValueError, IndexError):
                        last, tokens, factor = now, self.burst, 1.0
                    factor = min(1.0, max(self.MIN_FACTOR, factor))
                    last, tokens, factor, ret = fn(now, last, tokens, factor)
                    f.seek(0)
                    f.truncate()
                    f.write(f"{last},{tokens},{factor}".encode("ascii"))
                    f.flush()
                    return ret
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def penalise(self) -> float:
        """Halve the effective rate after a 429. Returns the new factor."""
        if self.rate is None:
            return 1.0

        def fn(now, last, tokens, factor):
            new = max(self.MIN_FACTOR, factor * 0.5)
            # spend the bucket too, so the cut takes effect immediately
            return now, 0.0, new, new
        return self._rw(fn)

    def reward(self) -> None:
        """Nudge the effective rate back up after a success."""
        if self.rate is None:
            return

        def fn(now, last, tokens, factor):
            return (last, tokens,
                    min(1.0, factor + self.RECOVER_PER_OK), None)
        self._rw(fn)

    def effective_rate_per_min(self) -> float:
        """Current allowed rate, for reporting."""
        if self.rate is None:
            return 0.0

        def fn(now, last, tokens, factor):
            return last, tokens, factor, factor
        return self.rate * 60.0 * self._rw(fn)

    def acquire(self) -> None:
        if self.rate is None:
            return
        while True:
            def fn(now, last, tokens, factor):
                rate = self.rate * factor
                # A clock jump backwards must not mint tokens.
                elapsed = max(0.0, now - last)
                tokens = min(self.burst, tokens + elapsed * rate)
                if tokens >= 1.0:
                    return now, tokens - 1.0, factor, 0.0
                return now, tokens, factor, (1.0 - tokens) / rate

            wait = self._rw(fn)
            if wait <= 0.0:
                return
            # Jitter so released waiters do not re-collide in lockstep.
            time.sleep(min(wait, 0.5) * (0.6 + random.random() * 0.6))


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITER_LOCK = threading.Lock()

# Quota scope was measured, not assumed (scripts/probe_quota_scope.py,
# 2026-09-12): a 25-request burst down google/ and anthropic/ simultaneously
# sustained 215 and 167 ok/min respectively with zero 429s. The 60 rpm ceiling
# seen earlier came from the doubao passthrough route, not from an
# account-wide cap. So the bucket is per PROVIDER: keying it per provider lets
# executor and optimizer traffic proceed independently, while still preventing
# one provider's workers from flooding that provider.
DEFAULT_PROVIDER_RPM = 120


def limiter_for(queue: str) -> RateLimiter:
    """Return the shared limiter for a provider bucket.

    Cross-process by default: several domains are run as separate processes
    at once, and a per-process bucket then lets each one use the provider's
    whole allowance simultaneously. Set VIEWCRAFT_LIMITER=process to get the
    old in-process behaviour (used by the unit tests, which must not touch a
    shared file).
    """
    with _LIMITER_LOCK:
        if queue not in _LIMITERS:
            tp = CFG.get("throughput") or {}
            per = (tp.get("provider_rpm") or {})
            rpm = per.get(queue, tp.get("default_provider_rpm",
                                        DEFAULT_PROVIDER_RPM))
            burst = int(tp.get("limiter_burst", 8))
            if os.getenv("VIEWCRAFT_LIMITER") == "process":
                _LIMITERS[queue] = RateLimiter(rpm, burst=burst)
            else:
                _LIMITERS[queue] = CrossProcessRateLimiter(
                    queue, rpm, burst=burst)
        return _LIMITERS[queue]


def image_to_data_url(src: str | Path | bytes, mime: str = "image/png") -> str:
    """Encode a local file or raw bytes as a data URL. http(s)/data pass through."""
    if isinstance(src, (str, Path)):
        s = str(src)
        if s.startswith(("http://", "https://", "data:")):
            return s
        p = Path(s)
        guessed = mimetypes.guess_type(p.name)[0]
        if guessed and guessed.startswith("image/"):
            mime = guessed
        raw = p.read_bytes()
    else:
        raw = src
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def parse(cls, u: dict | None) -> "Usage":
        u = u or {}
        pd = u.get("prompt_tokens_details") or {}
        cd = u.get("completion_tokens_details") or {}
        return cls(
            input_tokens=u.get("prompt_tokens") or 0,
            output_tokens=u.get("completion_tokens") or 0,
            total_tokens=u.get("total_tokens") or 0,
            cached_tokens=pd.get("cached_tokens") or 0,
            reasoning_tokens=cd.get("reasoning_tokens") or 0,
        )


@dataclass
class ModelResponse:
    content: str
    model_reported: str
    request_id: str | None
    usage: Usage
    raw: dict = field(repr=False, default_factory=dict)
    from_cache: bool = False
    content_hash: str = ""
    n_images: int = 0
    sent_pixels: int = 0
    wall_time_s: float = 0.0


class ResponseCache:
    """Content-hash keyed cache on disk.

    The key covers the full request payload plus the resolved model id, so two
    different arms that construct an identical request reuse one inference
    rather than paying for a duplicate call.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def key(payload: dict, model_id: str) -> str:
        blob = json.dumps({"payload": payload, "model": model_id},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with self._lock:
            tmp.write_text(json.dumps(value, ensure_ascii=False))
            tmp.replace(p)


class ModelClient:
    """Client for one configured role ('executor' or 'optimizer')."""

    def __init__(self, role: str, cfg: dict | None = None,
                 cache_dir: Path | None = None):
        cfg = cfg or CFG
        if role not in cfg["models"]:
            raise KeyError(f"unknown model role {role!r}")
        self.role = role
        self.spec = cfg["models"][role]
        self.model_id = os.getenv(f"VIEWCRAFT_{role.upper()}_MODEL", self.spec["resolved_id"])
        self.transport = self.spec["transport"]
        self.provider = self.spec.get("provider")
        self.tok_field = self.spec.get("max_tokens_field", "max_tokens")
        # The endpoint is overridable from the environment so the committed
        # config can keep naming the internal gateway the recorded runs used
        # while a reader elsewhere points at their own OpenAI-compatible one.
        # Env beats config for the same reason credentials do: it is the layer
        # that belongs to the operator, not to the experiment.
        self.endpoints = dict(cfg["endpoints"])
        _override = os.getenv("VIEWCRAFT_ENDPOINT")
        if _override:
            self.endpoints["standard"] = _override
            self.endpoints["passthrough"] = _override
        role_endpoint = os.getenv(f"VIEWCRAFT_{role.upper()}_ENDPOINT")
        if role_endpoint:
            self.endpoints[self.transport] = role_endpoint
        self.throughput = cfg.get("throughput", {})
        self.cache = ResponseCache(cache_dir or (Path(os.getenv("VIEWCRAFT_CACHE_ROOT", runs_root() / "_cache")) / role))
        # bucket by provider, since that is where the platform enforces rate
        self.limiter = limiter_for(self.provider or role)

    def _url_and_headers(self) -> tuple[str, dict]:
        if self.transport == "passthrough":
            app_id, api_key = auth_pair()
            params = "&".join([
                f"provider={self.provider}",
                f"model={self.model_id}",
                "timeout=300",
                "max_retry=1",
            ])
            return self.endpoints["passthrough"], {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {app_id}:{api_key}?{params}",
            }
        headers = {"Content-Type": "application/json"}
        if self.spec.get("auth", "bearer") != "none":
            headers["Authorization"] = f"Bearer {self._bearer()}"
        return self.endpoints["standard"], headers

    def _bearer(self) -> str:
        from .credentials import MissingCredentials
        env = self.spec.get("api_key_env", "OPENAI_API_KEY")
        token = os.environ.get(env, "")
        if token:
            return token
        if self.spec.get("auth") == "aihub":
            app_id, api_key = auth_pair()
            return f"{app_id}:{api_key}"
        raise MissingCredentials(
            f"Set {env} for {self.role}, or set models.{self.role}.auth: none "
            "for a local server that does not require authentication.")

    def validate(self) -> None:
        """Check operator settings without issuing a request."""
        if not self.model_id or self.model_id.startswith("your-"):
            raise ValueError(f"Configure models.{self.role}.resolved_id or "
                             f"VIEWCRAFT_{self.role.upper()}_MODEL")
        if self.transport not in ("standard", "passthrough"):
            raise ValueError(f"Unsupported transport: {self.transport}")
        url, _ = self._url_and_headers()
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid endpoint for {self.role}")

    def build_payload(self, messages: list[dict], max_tokens: int = 4096,
                      temperature: float | None = None,
                      extra: dict | None = None) -> dict:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            self.tok_field: max_tokens,
            "stream": False,
        }
        # Decoding is part of the frozen protocol: fall back to the configured
        # temperature rather than the server default, so every arm samples
        # identically and a gate comparison reflects the config, not the draw.
        if temperature is None:
            temperature = self.spec.get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        # Explicit reasoning-effort control (gpt-5.5 line, 2026-09-19): the
        # server default burns 40-140s per vision call in hidden reasoning;
        # a measured A/B on a counting item showed effort=low answers in
        # 6.3s and stays CORRECT (28/28) where the default also passes.
        # Constraint measured on this gateway: an explicit effort forces the
        # default temperature (temp=1), so profiles that set an effort must
        # leave temperature unset -- the two knobs are coupled server-side.
        eff = self.spec.get("reasoning_effort")
        if eff:
            payload["reasoning_effort"] = eff
        if self.spec.get("reasoning") == "disabled" and self.provider == "doubao":
            payload["thinking"] = {"type": "disabled"}
        if self.spec.get("reasoning") == "disabled" and self.provider == "ali":
            payload["enable_thinking"] = False
        if extra:
            payload.update(extra)
        return payload

    def call(self, messages: list[dict], max_tokens: int = 4096,
             temperature: float | None = None, extra: dict | None = None,
             max_attempts: int = 6, use_cache: bool | None = None) -> ModelResponse:
        payload = self.build_payload(messages, max_tokens, temperature, extra)
        if use_cache is None:
            use_cache = self.role != "optimizer"
        # A model alias served by another endpoint is a different model.
        cache_payload = {"request": payload,
                         "endpoint": self.endpoints[self.transport],
                         "transport": self.transport}
        ckey = ResponseCache.key(cache_payload, self.model_id)

        if use_cache:
            hit = self.cache.get(ckey)
            if hit is not None:
                return ModelResponse(
                    content=hit["content"],
                    model_reported=hit.get("model_reported", ""),
                    request_id=hit.get("request_id"),
                    usage=Usage(**hit.get("usage", {})),
                    raw=hit.get("raw", {}),
                    from_cache=True,
                    content_hash=ckey,
                    n_images=hit.get("n_images", 0),
                    sent_pixels=hit.get("sent_pixels", 0),
                )

        url, headers = self._url_and_headers()
        backoff = float(self.throughput.get("backoff_base_s", 2.0))
        backoff_max = float(self.throughput.get("backoff_max_s", 120.0))
        # Throttling is bounded by wall clock, separately from the fault
        # budget, so a provider saying "slow down" can never exhaust the
        # retries reserved for real transport errors.
        throttle_budget = float(self.throughput.get("throttle_budget_s", 900.0))
        throttle_deadline = None
        throttle_waits = 0
        last_err = None

        attempt = 0
        while attempt < max_attempts:
            self.limiter.acquire()
            t0 = time.monotonic()
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=float(self.spec.get("timeout_s", 330)))
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                attempt += 1
                time.sleep(min(backoff * (2 ** attempt) * (0.5 + random.random()),
                               backoff_max))
                continue

            dt = time.monotonic() - t0

            if r.status_code in RETRYABLE_STATUS:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                # A 429 is the provider asking us to slow down. It is NOT a
                # fault, so it must not consume the fault-retry budget: a
                # sustained throttle burned all 18 attempts and turned a
                # "wait a moment" into a permanently missing gate unit, which
                # the paired gate then refuses to compare. Measured across
                # several waves, that is how ~100 units were lost.
                #
                # So throttling is bounded by WALL CLOCK, not by attempt
                # count: keep waiting as long as the provider keeps saying
                # "later", up to throttle_budget_s, and do not advance
                # `attempt`. A genuine outage still terminates, because the
                # deadline is finite and every other status keeps its own
                # bounded retry count.
                if r.status_code == 429:
                    # Tell the shared bucket to slow down for EVERY process,
                    # not just this one: the ceiling is account-wide.
                    if hasattr(self.limiter, "penalise"):
                        self.limiter.penalise()
                    # Count throttles. Retrying a 429 silently made the logs
                    # show ZERO throttling while the adaptive limiter was
                    # visibly sinking (79.5 -> 54.6 rpm in 45s), so the one
                    # signal that explains the throughput ceiling was
                    # invisible. Cheap counter, printed by the status tool.
                    try:
                        _bump_throttle_count(self.provider or self.role)
                    except Exception:                        # noqa: BLE001
                        pass
                    if throttle_deadline is None:
                        throttle_deadline = time.monotonic() + throttle_budget
                    if time.monotonic() >= throttle_deadline:
                        raise ModelUnavailable(redact(
                            f"{self.role}/{self.model_id} throttled for more "
                            f"than {throttle_budget:.0f}s: {last_err}"))
                    ra = r.headers.get("retry-after")
                    try:
                        delay = float(ra) if ra else 0.0
                    except ValueError:
                        delay = 0.0
                    delay = max(delay, min(backoff * (1.6 ** throttle_waits),
                                           backoff_max))
                    throttle_waits += 1
                    time.sleep(delay * (0.7 + random.random() * 0.6))
                    continue
                attempt += 1
                time.sleep(min(backoff * (2 ** attempt) * (0.5 + random.random()),
                               backoff_max))
                continue

            if r.status_code != 200:
                raise ModelUnavailable(redact(
                    f"{self.role}/{self.model_id} HTTP {r.status_code}: "
                    f"{r.text[:300]}"))

            try:
                data = r.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
            except (ValueError, KeyError, IndexError) as e:
                raise ModelFormatError(redact(
                    f"unparseable response from {self.model_id}: "
                    f"{type(e).__name__} body={r.text[:300]}"))

            if hasattr(self.limiter, "reward"):
                self.limiter.reward()

            usage = Usage.parse(data.get("usage"))
            resp = ModelResponse(
                content=content,
                model_reported=data.get("model", ""),
                request_id=data.get("id") or r.headers.get("x-request-id"),
                usage=usage,
                raw=data,
                content_hash=ckey,
                n_images=count_images(messages),
                wall_time_s=dt,
            )
            if use_cache:
                self.cache.put(ckey, {
                    "content": resp.content,
                    "model_reported": resp.model_reported,
                    "request_id": resp.request_id,
                    "usage": resp.usage.__dict__,
                    "n_images": resp.n_images,
                    "raw": {"id": data.get("id"), "model": data.get("model")},
                })
            return resp

        raise ModelUnavailable(redact(
            f"{self.role}/{self.model_id} exhausted {max_attempts} attempts; "
            f"last error: {last_err}"))


_THROTTLE_PATH = runs_root() / "_limiter" / "throttle_counts.txt"


def _bump_throttle_count(bucket: str) -> None:
    """Append one line per 429. Read by scripts/status.py.

    A file append rather than an in-memory counter, because the runs are
    separate processes and the interesting number is the total across them.
    """
    _THROTTLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_THROTTLE_PATH, "a") as f:
        f.write(f"{time.time():.0f},{bucket}\n")


def throttle_events(window_s: float = 600.0) -> int:
    """How many 429s the whole machine saw in the last `window_s`."""
    if not _THROTTLE_PATH.exists():
        return 0
    cut = time.time() - window_s
    n = 0
    try:
        for line in _THROTTLE_PATH.read_text().splitlines()[-20000:]:
            try:
                if float(line.split(",")[0]) >= cut:
                    n += 1
            except (ValueError, IndexError):
                continue
    except OSError:
        return 0
    return n


def count_images(messages: Iterable[dict]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for part in c if part.get("type") == "image_url")
    return n


def user_message(text: str, images: list[str] | None = None,
                 detail: str = "high") -> dict:
    """Build one user turn. Images precede text, matching the house convention."""
    if not images:
        return {"role": "user", "content": text}
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": u, "detail": detail}}
        for u in images
    ]
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def system_message(text: str) -> dict:
    return {"role": "system", "content": text}
