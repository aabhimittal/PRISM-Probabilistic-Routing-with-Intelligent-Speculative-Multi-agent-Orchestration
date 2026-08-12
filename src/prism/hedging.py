"""Latency hedging — speculation on the *time* axis.

PRISM's core mechanism speculates because of *quality* uncertainty: "I don't
know which agent will answer best." This module addresses the other uncertainty
that dominates production LLM systems: *latency* uncertainty. Real agent calls
have heavy-tailed latency — the p50 is fine, the p99 is a disaster, and a
pipeline's latency is gated by its slowest stage.

The borrowed idea this time is Dean & Barroso's **hedged requests** ("The Tail
at Scale", CACM 2013): send a request to one replica; if it hasn't answered
within ~the p95 latency, send a duplicate to another replica and take whichever
answers first. The tail collapses because you only pay the duplicate cost on
the slow tail, not on every request.

PRISM generalises the replica to a *different agent*: when the router commits a
single arm (the greedy path) and that arm blows through its learned latency
fence, the runner-up arm is launched as a hedge and the first successful result
wins. It is lazy speculation — the second branch materialises only when the
first one stalls. The two mechanisms compose into one economics:

    quality uncertainty high  → speculate eagerly (parallel branches up front)
    quality uncertainty low   → commit one arm, but hedge its latency tail

A late-arriving hedged/primary result is not wasted either — its score is still
fed to the bandit when it eventually lands (the same "squashed work still
teaches" principle).
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class LatencyTracker:
    """Learns each (stage, agent)'s latency profile online from real calls.

    ``fence(stage, agent)`` returns the hedge trigger point: the empirical
    ``quantile`` of a sliding window of recent latencies.

    Why a windowed quantile rather than mean + k·σ? Because production agent
    latency is **bimodal**, not Gaussian: a fast mode and a fat tail. A σ-based
    fence self-defeats there — the tail inflates both the mean and σ, hoisting
    the fence *above* the stragglers it exists to catch (we hit exactly this
    building the tests). The empirical quantile of a bounded window is exact,
    O(window) memory, adapts to drift by construction, and is the statistic the
    hedged-request literature actually specifies.
    """

    quantile: float = 0.95      # hedge when the arm is past this recent quantile
    window: int = 50            # sliding window per (stage, agent)
    min_fence: float = 0.010    # never hedge before 10ms — avoids thrash
    min_observations: int = 5   # no fence until the profile has some shape
    _windows: dict[tuple[str, str], deque] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe(self, stage: str, agent: str, latency: float) -> None:
        if latency <= 0 or not math.isfinite(latency):
            return
        with self._lock:
            self._windows.setdefault(
                (stage, agent), deque(maxlen=self.window)).append(latency)

    def fence(self, stage: str, agent: str) -> float | None:
        """The latency beyond which this arm is considered 'in its tail'.
        None until we have enough observations to say anything."""
        with self._lock:
            win = self._windows.get((stage, agent))
            if win is None or len(win) < self.min_observations:
                return None
            ordered = sorted(win)
            idx = min(int(self.quantile * len(ordered)), len(ordered) - 1)
            return max(ordered[idx], self.min_fence)


@dataclass
class HedgePolicy:
    """Decides whether a committed single arm gets a latency hedge armed.

    Deliberately conservative by default: hedging duplicates cost on the tail,
    so it must clear the same kind of economic bar speculation does — enough
    stakes, a viable backup, and a learned latency profile to aim at.
    """

    tracker: LatencyTracker = field(default_factory=LatencyTracker)
    stakes_floor: float = 0.3      # don't hedge tasks that barely matter
    backup_floor: float = 0.05     # backup needs at least this P(best)
    fallback_fence: float | None = None  # used before the profile warms up

    def plan(self, stage: str, primary: str, p_best: dict[str, float],
             stakes: float) -> tuple[str, float] | None:
        """Return (backup_agent, fire_after_seconds) or None for no hedge."""
        if stakes < self.stakes_floor or len(p_best) < 2:
            return None
        fence = self.tracker.fence(stage, primary)
        if fence is None:
            fence = self.fallback_fence
        if fence is None:
            return None
        backups = sorted(
            ((n, p) for n, p in p_best.items() if n != primary),
            key=lambda kv: kv[1], reverse=True,
        )
        name, p = backups[0]
        if p < self.backup_floor:
            return None
        return name, fence
