#!/usr/bin/env python3
"""TDD unit tests for mimoly reliability primitives (Tier 1 + Tier 2).

Covers the techniques harvested from the GitHub survey:
  * error taxonomy (retryable vs auth vs client)         -> classify_status / classify_upstream_error
  * time-budgeted retry with full jitter                  -> RetryBudget
  * per-account circuit breaker + least-inflight pool     -> AccountPool
  * admission control (fast 503 on overload)              -> AdmissionController
  * exact-match TTL cache (non-streaming only)            -> TTLCache

Run:  .venv/Scripts/python.exe test_reliability.py
"""
import importlib.util
import json
import sys
import time

MODULE = "mimoly_reliability.py"

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def load_module():
    spec = importlib.util.spec_from_file_location("mimoly_reliability_mod", MODULE)
    m = importlib.util.module_from_spec(spec)
    sys.modules["mimoly_reliability_mod"] = m
    spec.loader.exec_module(m)
    return m


def main():
    try:
        m = load_module()
    except FileNotFoundError:
        print(f"  FAIL  module {MODULE} not found (TDD RED — implement it)")
        return 1

    # ------------------------------------------------------------------
    # [unit] error taxonomy
    # ------------------------------------------------------------------
    print("[unit] error taxonomy")
    cs = getattr(m, "classify_status", None)
    check("classify_status exists", cs is not None)
    if cs is not None:
        check("429 -> rate_limit", cs(429) == m.ERROR_RATE_LIMIT, f"got {cs(429)}")
        check("401 -> auth", cs(401) == m.ERROR_AUTH, f"got {cs(401)}")
        check("403 -> auth", cs(403) == m.ERROR_AUTH, f"got {cs(403)}")
        check("500 -> server_error", cs(500) == m.ERROR_SERVER, f"got {cs(500)}")
        check("503 -> server_error", cs(503) == m.ERROR_SERVER, f"got {cs(503)}")
        check("400 -> client_error", cs(400) == m.ERROR_CLIENT, f"got {cs(400)}")
        check("200 -> none/None", cs(200) in (None, "ok"), f"got {cs(200)}")

    # retryability + rotation semantics
    check("is_retryable exists", hasattr(m, "is_retryable"))
    if hasattr(m, "is_retryable"):
        check("rate_limit retryable", m.is_retryable(m.ERROR_RATE_LIMIT))
        check("network retryable", m.is_retryable(m.ERROR_NETWORK))
        check("server_error retryable", m.is_retryable(m.ERROR_SERVER))
        check("auth NOT retryable on same key", not m.is_retryable(m.ERROR_AUTH))
        check("client_error NOT retryable", not m.is_retryable(m.ERROR_CLIENT))

    check("should_rotate_key exists", hasattr(m, "should_rotate_key"))
    if hasattr(m, "should_rotate_key"):
        check("auth rotates key", m.should_rotate_key(m.ERROR_AUTH))
        check("rate_limit rotates key", m.should_rotate_key(m.ERROR_RATE_LIMIT))
        check("server_error does NOT rotate", not m.should_rotate_key(m.ERROR_SERVER))
        check("network does NOT rotate", not m.should_rotate_key(m.ERROR_NETWORK))

    # ------------------------------------------------------------------
    # [unit] retry budget (time-bounded, full jitter)
    # ------------------------------------------------------------------
    print("[unit] retry budget")
    RB = getattr(m, "RetryBudget", None)
    check("RetryBudget exists", RB is not None)
    if RB is not None:
        import random
        b = RB(max_attempts=3, base_delay=0.5, max_delay=8.0, deadline=20.0, rng=random.Random(1))
        b.start()
        check("first retry allowed", b.should_retry(0))
        check("third attempt (attempt=2) not retried", not b.should_retry(2))
        d0 = b.next_delay(0)
        check("delay within [0, base]", 0.0 <= d0 <= 0.5 + 1e-9, f"got {d0}")
        # jitter: with a seeded rng two budgets give deterministic but bounded values
        vals = set()
        for _ in range(20):
            bb = RB(rng=random.Random())
            bb.start()
            vals.add(round(bb.next_delay(0), 6))
        check("full jitter produces varying delays", len(vals) > 1, f"got {vals}")
        # expired budget -> no retry
        exp = RB(deadline=0.0)
        exp.start()
        check("expired budget refuses retry", not exp.should_retry(0))
        check("expired budget delay is 0", exp.next_delay(0) == 0.0)

    # ------------------------------------------------------------------
    # [unit] account pool + circuit breaker
    # ------------------------------------------------------------------
    print("[unit] account pool & circuit breaker")
    AP = getattr(m, "AccountPool", None)
    check("AccountPool exists", AP is not None)
    if AP is not None:
        accts = [
            {"id": "a1", "cookies": {"userId": "1"}},
            {"id": "a2", "cookies": {"userId": "2"}},
            {"id": "a3", "cookies": {"userId": "3"}},
        ]
        pool = AP(accts, max_failures=3, cooldown_base=30.0, cooldown_max=1800.0)
        check("pool loads 3 accounts", len(pool.accounts) == 3)
        check("healthy_count starts at 3", pool.healthy_count() == 3)

        a = pool.acquire()
        check("acquire returns an account", a is not None and a.id in ("a1", "a2", "a3"))
        # least-inflight: second acquire should pick a different account
        b2 = pool.acquire()
        check("least-inflight spreads load", b2 is not None and b2.id != a.id, f"got {a.id} then {b2.id}")

        # success resets failures
        pool.release(a, success=True)
        check("success keeps account healthy", pool.healthy_count() == 3)

        # 401 immediately parks the key (auth = dead)
        c = pool.acquire()
        pool.release(c, success=False, error_class=m.ERROR_AUTH)
        check("auth failure parks key", c.id not in [x.id for x in pool._available()] or pool.healthy_count() < 3,
              f"healthy={pool.healthy_count()}")
        check("auth failure sets cooldown", c.cooldown_until > time.monotonic())
        # parked key must never be handed out again
        picked = []
        for _ in range(3):
            x = pool.acquire()
            if x is not None:
                picked.append(x.id)
                pool.release(x, success=True)
        check("parked key not acquired again", c.id not in picked, f"got {picked}")

        # server errors accumulate then trip the breaker at max_failures
        pool2 = AP([{"id": "b1", "cookies": {}}], max_failures=3, cooldown_base=10.0, cooldown_max=100.0)
        acc = pool2.acquire()
        for _ in range(2):
            pool2.release(acc, success=False, error_class=m.ERROR_SERVER)
            acc = pool2.acquire()
            check("breaker still closed below threshold", acc is not None and acc.cooldown_until == 0.0)
        pool2.release(acc, success=False, error_class=m.ERROR_SERVER)
        check("breaker opens after max_failures", pool2.acquire() is None, "expected pool exhausted")
        check("cooldown is exponential-capped", acc.cooldown_until - time.monotonic() <= 100.0)

        # snapshot for observability
        snap = pool2.snapshot()
        check("snapshot is a list of dicts", isinstance(snap, list) and all(isinstance(s, dict) for s in snap))
        check("snapshot exposes health fields", snap and {"id", "healthy", "failures", "in_flight"} <= set(snap[0].keys()))

    # ------------------------------------------------------------------
    # [unit] admission control
    # ------------------------------------------------------------------
    print("[unit] admission control")
    AC = getattr(m, "AdmissionController", None)
    check("AdmissionController exists", AC is not None)
    if AC is not None:
        ac = AC(max_in_flight=2)
        check("first acquire ok", ac.acquire() is True)
        check("second acquire ok", ac.acquire() is True)
        check("third acquire rejected (fast 503)", ac.acquire() is False)
        check("rejected counter increments", ac.rejected == 1)
        ac.release()
        check("release frees a slot", ac.acquire() is True)
        ac.release(); ac.release()
        check("in_flight returns to 0", ac.in_flight == 0)
        ac.release()
        check("release never goes negative", ac.in_flight == 0)

    # ------------------------------------------------------------------
    # [unit] exact-match TTL cache
    # ------------------------------------------------------------------
    print("[unit] exact TTL cache")
    TC = getattr(m, "TTLCache", None)
    check("TTLCache exists", TC is not None)
    if TC is not None:
        c = TC(maxsize=4, ttl=100.0)
        key = c.make_key({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False})
        check("make_key deterministic", key == c.make_key({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}),
              "stream flag must not affect key")
        c.set(key, {"ok": 1})
        check("cache hit", c.get(key) == {"ok": 1})
        c.set("k2", 2); c.set("k3", 3); c.set("k4", 4); c.set("k5", 5)
        check("maxsize enforced", len(c) <= 4, f"len={len(c)}")
        check("make_key ignores volatile keys", c.make_key({"model": "m", "user": "u1"}) == c.make_key({"model": "m", "user": "u2"}))
        exp = TC(ttl=0.0)
        exp.set("x", 1)
        check("expired entry misses", exp.get("x") is None)

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
