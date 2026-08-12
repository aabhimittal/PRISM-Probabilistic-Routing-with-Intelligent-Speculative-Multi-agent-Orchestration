"""Drift detection and uncertainty resurrection.

The quiet lie in every bandit deployment is **stationarity** — the assumption
that an arm's quality today predicts its quality tomorrow. Real agents drift:
the provider silently swaps the model, a retrieval index goes stale, an API
starts rate-limiting. A converged bandit is maximally *confident* exactly when
drift makes that confidence maximally *wrong*, and with tight beliefs it can
take hundreds of observations to un-learn a dead regime.

PRISM's answer is one of its most on-thesis mechanisms: **uncertainty
resurrection**. A Page–Hinkley change-point detector watches each arm's reward
stream. When it detects a downward shift, we don't slowly drag the posterior —
we *collapse the belief's evidence* (soft-reset to a wide Beta), which:

  1. widens the arm's posterior → Thompson sampling explores again,
  2. flattens P(best) at that stage → the **speculation policy reignites**,
     hedging across branches exactly as it did during cold-start,
  3. lets a handful of fresh observations dominate the posterior → the router
     re-converges on the new regime in tens, not hundreds, of tasks.

Drift is the cold-start problem happening again, and PRISM already has a
machine for cold-start: speculation. The detector just presses that button.

Why Page–Hinkley? It's the classic sequential test for a shift in mean: O(1)
memory, O(1) update, two interpretable knobs (``delta`` — how much wobble to
tolerate, ``threshold`` — how much cumulative evidence of a shift to demand),
and it's one-sided — we only alarm on *degradation*. An agent that silently
*improves* is found naturally by Thompson exploration; it needs no alarm.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from .uncertainty import Belief


@dataclass
class PageHinkley:
    """One-sided Page–Hinkley test for a downward shift in a reward stream.

    Maintains the running mean and the cumulative sum
    ``s_t = Σ (x_i − mean_i + delta)``. In a stable regime ``s_t`` random-walks
    upward (each term ≈ +delta); after a downward shift each term goes negative
    and ``s_t`` falls away from its running maximum ``m_t``. We alarm when
    ``m_t − s_t > threshold``.
    """

    delta: float = 0.01       # tolerated per-observation wobble
    threshold: float = 1.2    # cumulative shift mass required to alarm
    min_samples: int = 15     # no alarms before the mean estimate stabilises

    n: int = 0
    mean: float = 0.0
    cum: float = 0.0
    cum_max: float = 0.0
    # Short memory of the newest rewards. At alarm time the *stream* mean is
    # dominated by the dead regime; the recent window is our only estimate of
    # the NEW regime, and it's what the belief reset should anchor on.
    recent: deque = field(default_factory=lambda: deque(maxlen=10))

    def observe(self, x: float) -> bool:
        self.n += 1
        # incremental mean BEFORE adding deviation, per the standard recurrence
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean + self.delta
        self.cum_max = max(self.cum_max, self.cum)
        self.recent.append(x)
        if self.n < self.min_samples:
            return False
        return (self.cum_max - self.cum) > self.threshold

    @property
    def recent_mean(self) -> float:
        return sum(self.recent) / len(self.recent) if self.recent else self.mean

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.cum_max = 0.0
        self.recent.clear()


@dataclass
class DriftEvent:
    stage: str
    agent: str
    old_belief: Belief
    new_belief: Belief
    observed_mean: float      # detector's recent mean at alarm time

    def describe(self) -> str:
        return (f"DRIFT detected for ({self.stage}, {self.agent}): posterior "
                f"{self.old_belief.mean:.2f}±{self.old_belief.std:.2f} "
                f"(n={self.old_belief.evidence:.0f}) no longer matches reward "
                f"stream (recent mean {self.observed_mean:.2f}) — belief reset to "
                f"{self.new_belief.mean:.2f}±{self.new_belief.std:.2f} "
                f"(n={self.new_belief.evidence:.0f}); speculation will reignite")


@dataclass
class DriftMonitor:
    """Per-(stage, agent) Page–Hinkley detectors + the soft-reset policy.

    ``reset_evidence`` controls how humble the resurrected belief is: the reset
    keeps a *discounted memory* of the old mean — pulled 50% toward the recent
    (post-shift) observed mean — but backs it with only ``reset_evidence``
    pseudo-observations, so fresh evidence dominates immediately.
    """

    delta: float = 0.01
    threshold: float = 1.2
    min_samples: int = 15
    reset_evidence: float = 6.0
    _detectors: dict[tuple[str, str], PageHinkley] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    events: list[DriftEvent] = field(default_factory=list)

    def observe(self, stage: str, agent: str, reward: float,
                belief: Belief) -> Belief | None:
        """Feed one reward. Returns a *replacement* belief if drift fired,
        else None. Called by the router under its own update lock."""
        with self._lock:
            det = self._detectors.setdefault(
                (stage, agent),
                PageHinkley(delta=self.delta, threshold=self.threshold,
                            min_samples=self.min_samples),
            )
            if not det.observe(reward):
                return None

            # Alarm: resurrect uncertainty. Anchor on the detector's RECENT
            # window mean — the whole-stream mean is dominated by the dead
            # regime and would reset the belief optimistically high; the recent
            # window is our only estimate of the new regime.
            recent = det.recent_mean
            anchor = min(max(recent, 0.05), 0.95)
            n = self.reset_evidence
            new = Belief(alpha=max(anchor * n, 0.5), beta=max((1 - anchor) * n, 0.5))
            event = DriftEvent(stage=stage, agent=agent, old_belief=belief,
                               new_belief=new, observed_mean=recent)
            self.events.append(event)
            det.reset()
            return new

    def drain_events(self) -> list[DriftEvent]:
        """Return and clear pending events (the orchestrator folds them into
        the causal trace of the run in which they fired)."""
        with self._lock:
            out, self.events = self.events, []
            return out
