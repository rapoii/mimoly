#!/usr/bin/env python3
"""mimoly_reliability — self-contained reliability primitives for the mimoly proxy.

Harvested (and re-implemented in plain asyncio/stdlib, zero new dependencies)
from the GitHub survey of web2api bridges and LLM gateways:

  * Error taxonomy            (Bifrost / Portkey / new-api)
  * Time-budgeted retry+full jitter (Portkey / AWS full-jitter / RFC 9110)
  * Per-account circuit breaker + least-inflight pool (litellm / gpt-load / Godde3s)
  * Admission control -> fast 503 (litellm max_in_flight)
  * Exact-match TTL cache, non-streaming only (litellm / new-api)

This file deliberately has NO imports from mimoly.py so it can be unit-tested
standalone (see test_reliability.py) and reused without circular imports.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------
ERROR_NONE = None
ERROR_AUTH = "auth"            # 401/403 — session dead, rotate key, do NOT retry same key
ERROR_RATE_LIMIT = "rate_limit"  # 429 — retry/rotate after backoff
ERROR_SERVER = "server_error"    # 5xx — retry same key with backoff
ERROR_NETWORK = "network"        # connect/read timeout — retry same key
ERROR_CLIENT = "client_error"    # 4xx (not 401/403/429) — fatal, do NOT retry
ERROR_GATEWAY = "gateway_error"  # internal mimoly fault — abort chain, do NOT retry

_RETRYABLE = {ERROR_RATE_LIMIT, ERROR_SERVER, ERROR_NETWORK}
_ROTATE = {ERROR_AUTH, ERROR_RATE_LIMIT}


def classify_status(status_code: int) -> Optional[str]:
    """Map an upstream HTTP status to the shared error taxonomy."""
    if status_code is None:
        return ERROR_NETWORK
    if 200 <= status_code < 300:
        return ERROR_NONE
    if status_code in (401, 403):
        return ERROR_AUTH
    if status_code == 429:
        return ERROR_RATE_LIMIT
    if 500 <= status_code < 600:
        return ERROR_SERVER
    if 400 <= status_code < 500:
        return ERROR_CLIENT
    return ERROR_SERVER


def classify_exception(exc: BaseException) -> str:
    """Map a raised exception to the taxonomy (timeouts/conn errors -> network)."""
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return ERROR_NETWORK
    if "connect" in name or "network" in name or "readerror" in name:
        return ERROR_NETWORK
    return ERROR_GATEWAY


def is_retryable(error_class: Optional[str]) -> bool:
    """True if the SAME key may be retried after backoff."""
    return error_class in _RETRYABLE


def should_rotate_key(error_class: Optional[str]) -> bool:
    """True if the failure is key/account specific and the next attempt must use another key."""
    return error_class in _ROTATE


# ---------------------------------------------------------------------------
# Time-budgeted retry with full jitter
# ---------------------------------------------------------------------------
class RetryBudget:
    """Bounded attempts + exponential backoff with FULL jitter + wall-clock deadline.

    Retry is only legal *before the first streamed byte*; callers must not use
    this once content has been emitted (see the 'started guard' in the survey).
    """

    def __init__(self, max_attempts: int = 3, base_delay: float = 0.5,
                 max_delay: float = 8.0, deadline: float = 20.0,
                 min_delay: float = 0.0,
                 rng: Optional[random.Random] = None):
        self.max_attempts = max(1, int(max_attempts))
        self.base_delay = max(0.0, float(base_delay))
        self.max_delay = max(0.0, float(max_delay))
        self.deadline = max(0.0, float(deadline))
        self.min_delay = max(0.0, float(min_delay))
        self._rng = rng or random.Random()
        self._started_at: Optional[float] = None

    def start(self) -> None:
        self._started_at = time.monotonic()

    def elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    def expired(self) -> bool:
        return self.deadline <= 0.0 or self.elapsed() >= self.deadline

    def should_retry(self, attempt: int) -> bool:
        """True if another attempt is allowed (attempt is 0-indexed)."""
        if attempt + 1 >= self.max_attempts:
            return False
        return not self.expired()

    def next_delay(self, attempt: int, error_class: Optional[str] = None) -> float:
        """Full-jitter backoff: uniform(floor, min(base * 2**attempt, max_delay))."""
        if self.expired():
            return 0.0
        ceiling = min(self.base_delay * (2 ** max(0, attempt)), self.max_delay)
        floor = self.min_delay
        if error_class == ERROR_RATE_LIMIT:
            # 429 rate limits need a real pause for upstream tokens to replenish
            floor = max(floor, 1.5)
            ceiling = max(ceiling, floor + 1.0)
        if ceiling <= 0.0:
            return 0.0
        return self._rng.uniform(floor, max(floor, ceiling))

    async def sleep(self, attempt: int, error_class: Optional[str] = None) -> None:
        d = self.next_delay(attempt, error_class=error_class)
        if d > 0:
            await asyncio.sleep(d)


# ---------------------------------------------------------------------------
# Account pool + per-account circuit breaker
# ---------------------------------------------------------------------------
class PooledAccount:
    """A single upstream credential (Xiaomi session) with health bookkeeping."""

    __slots__ = ("id", "cookies", "failures", "in_flight", "cooldown_until",
                 "last_error", "uses", "successes")

    def __init__(self, account_id: str, cookies: Dict[str, str]):
        self.id = account_id
        self.cookies = cookies
        self.failures = 0
        self.in_flight = 0
        self.cooldown_until = 0.0
        self.last_error: Optional[str] = None
        self.uses = 0
        self.successes = 0

    def healthy(self) -> bool:
        return self.cooldown_until <= time.monotonic()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "healthy": self.healthy(),
            "failures": self.failures,
            "in_flight": self.in_flight,
            "cooldown_remaining": max(0.0, round(self.cooldown_until - time.monotonic(), 1)),
            "last_error": self.last_error,
            "uses": self.uses,
            "successes": self.successes,
        }


class AccountPool:
    """Least-inflight selection over N credentials, with a per-account breaker.

    * ``acquire`` returns the healthy account with the fewest in-flight requests
      (deterministic tie-break by fewest total uses -> round-robin-ish spread).
    * ``release(success=False, error_class=...)`` accumulates failures and, at
      ``max_failures``, opens the breaker with exponential cooldown
      (``cooldown_base * 2**(failures-max_failures)`` capped at ``cooldown_max``).
    * An ``auth`` failure parks the account immediately (dead session).
    * A success resets the failure counter (self-healing, like Godde3s/glm-free-api).
    """

    def __init__(self, accounts: List[Dict[str, Any]], max_failures: int = 3,
                 cooldown_base: float = 30.0, cooldown_max: float = 1800.0):
        self.accounts: List[PooledAccount] = []
        for i, a in enumerate(accounts or []):
            acc_id = str(a.get("id") or a.get("userId") or f"acct{i+1}")
            cookies = a.get("cookies") or {k: v for k, v in a.items() if k != "id"}
            self.accounts.append(PooledAccount(acc_id, cookies))
        self.max_failures = max(1, int(max_failures))
        self.cooldown_base = max(1.0, float(cooldown_base))
        self.cooldown_max = max(self.cooldown_base, float(cooldown_max))

    # -- selection ---------------------------------------------------------
    def _available(self) -> List[PooledAccount]:
        return [a for a in self.accounts if a.healthy()]

    def healthy_count(self) -> int:
        return len(self._available())

    def acquire(self) -> Optional[PooledAccount]:
        """Pick the healthiest least-busy account, or None if all are cooling down."""
        avail = self._available()
        if not avail:
            return None
        # least in-flight first, then fewest lifetime uses (spreads cold start)
        chosen = min(avail, key=lambda a: (a.in_flight, a.uses, a.failures))
        chosen.in_flight += 1
        chosen.uses += 1
        return chosen

    # -- feedback ----------------------------------------------------------
    def release(self, account: PooledAccount, success: bool,
                error_class: Optional[str] = None) -> None:
        if account is None:
            return
        if account.in_flight > 0:
            account.in_flight -= 1

        if success:
            account.failures = 0
            account.successes += 1
            account.cooldown_until = 0.0
            account.last_error = None
            return

        account.last_error = error_class
        if error_class == ERROR_AUTH:
            # Dead session: park hard, no point counting up slowly.
            account.failures = self.max_failures
            account.cooldown_until = time.monotonic() + self.cooldown_max
            return

        account.failures += 1
        if account.failures >= self.max_failures:
            over = account.failures - self.max_failures
            cooldown = min(self.cooldown_base * (2 ** over), self.cooldown_max)
            account.cooldown_until = time.monotonic() + cooldown

    def snapshot(self) -> List[Dict[str, Any]]:
        return [a.snapshot() for a in self.accounts]


# ---------------------------------------------------------------------------
# Admission control (fast-fail overload)
# ---------------------------------------------------------------------------
class AdmissionController:
    """Cap concurrent in-flight requests; excess is rejected fast (-> HTTP 503).

    Prevents unbounded queueing from turning an upstream slowdown into a
    self-inflicted outage (litellm ``max_in_flight_requests_per_worker``).
    """

    def __init__(self, max_in_flight: int = 0):
        self.max_in_flight = max(0, int(max_in_flight))
        self._in_flight = 0
        self._lock = threading.Lock()
        self.rejected = 0
        self.accepted = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def acquire(self) -> bool:
        with self._lock:
            if self.max_in_flight and self._in_flight >= self.max_in_flight:
                self.rejected += 1
                return False
            self._in_flight += 1
            self.accepted += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1


# ---------------------------------------------------------------------------
# Per-key sliding-window rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Sliding-window rate limiter keyed by an arbitrary string (IP, tenant, ...).

    A deque of hit timestamps per key; entries older than ``window`` are dropped
    on each check. ``limit=0`` disables limiting. Cheap (no locks held during
    computation beyond a tiny critical section) and allocation-light.
    """

    def __init__(self, limit: int = 0, window: float = 60.0, max_keys: int = 10000):
        self.limit = max(0, int(limit))
        self.window = max(0.0, float(window))
        self.max_keys = max(1, int(max_keys))
        self._hits: "OrderedDict[str, deque]" = OrderedDict()
        self._lock = threading.Lock()
        self.rejected = 0
        self.allowed = 0

    def allow(self, key: str) -> bool:
        """Record a hit for ``key`` and return True if it is within the limit."""
        if self.limit == 0 or self.window <= 0:
            self.allowed += 1
            return True
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            dq = self._hits.get(key)
            if dq is None:
                dq = deque()
                self._hits[key] = dq
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= self.limit:
                self.rejected += 1
                self._hits.move_to_end(key)
                return False
            dq.append(now)
            self._hits.move_to_end(key)
            while len(self._hits) > self.max_keys:
                self._hits.popitem(last=False)
            self.allowed += 1
            return True

    def retry_after(self, key: str) -> float:
        """Seconds until the oldest hit for ``key`` leaves the window (0 if free)."""
        if self.limit == 0 or self.window <= 0:
            return 0.0
        with self._lock:
            dq = self._hits.get(key)
            if not dq:
                return 0.0
            oldest = dq[0]
        remaining = (oldest + self.window) - time.monotonic()
        return max(0.0, remaining)

    def stats(self) -> Dict[str, Any]:
        return {
            "limit": self.limit,
            "window": self.window,
            "tracked_keys": len(self._hits),
            "allowed": self.allowed,
            "rejected": self.rejected,
        }


# ---------------------------------------------------------------------------
# Exact-match TTL cache (non-streaming only)
# ---------------------------------------------------------------------------
# Request keys that must NOT affect the cache key: they are either volatile or
# already normalized away by the caller.
_VOLATILE_KEYS = {"user", "stream", "stream_options", "request_id", "metadata"}


class TTLCache:
    """Small bounded LRU + TTL cache for exact-match non-streaming responses."""

    def __init__(self, maxsize: int = 256, ttl: float = 300.0):
        self.maxsize = max(1, int(maxsize))
        self.ttl = float(ttl)
        self._store: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def make_key(self, body: Dict[str, Any]) -> str:
        """Stable hash of the semantically-relevant request fields."""
        relevant = {k: v for k, v in (body or {}).items() if k not in _VOLATILE_KEYS}
        canonical = json.dumps(relevant, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str):
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self.misses += 1
                return None
            value, expires_at = item
            if expires_at <= time.monotonic():
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic() + self.ttl)
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "maxsize": self.maxsize,
            "ttl": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
