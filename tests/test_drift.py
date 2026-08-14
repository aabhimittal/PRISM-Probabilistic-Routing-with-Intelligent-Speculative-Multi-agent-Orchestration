"""v0.2 industrial-hardening tests: drift detection, uncertainty resurrection,
and bounded-evidence (exponential forgetting) routing."""

from __future__ import annotations

import random

import pytest

from prism.agents import SimulatedAgent
from prism.drift import DriftMonitor, PageHinkley
from prism.graph import TaskGraph
from prism.orchestrator import Orchestrator
from prism.routing import BanditRouter
from prism.uncertainty import Belief


# --- PageHinkley unit --------------------------------------------------------

def test_page_hinkley_no_alarm_on_stable_stream():
    ph = PageHinkley()
    rng = random.Random(0)
    for _ in range(200):
        x = 0.8 + rng.uniform(-0.02, 0.02)
        assert ph.observe(x) is False


def test_page_hinkley_alarms_soon_after_regime_change():
    ph = PageHinkley()
    rng = random.Random(1)
    for _ in range(50):
        assert ph.observe(0.8 + rng.uniform(-0.02, 0.02)) is False

    alarm_at = None
    for i in range(1, 41):
        if ph.observe(0.3 + rng.uniform(-0.02, 0.02)):
            alarm_at = i
            break
    assert alarm_at is not None
    # A 0.8 → 0.3 collapse should be caught quickly (well within 40 obs).
    assert alarm_at < 40


def test_page_hinkley_no_alarm_before_min_samples():
    ph = PageHinkley(min_samples=30)
    for _ in range(10):
        assert ph.observe(0.9) is False
    # A brutal shift, but n stays below min_samples: still no alarm.
    for _ in range(15):
        assert ph.observe(0.0) is False
    assert ph.n == 25


def test_page_hinkley_reset_clears_state():
    ph = PageHinkley()
    for _ in range(20):
        ph.observe(0.7)
    ph.reset()
    assert ph.n == 0
    assert ph.mean == pytest.approx(0.0)
    assert ph.cum == pytest.approx(0.0)
    assert ph.cum_max == pytest.approx(0.0)


# --- DriftMonitor ------------------------------------------------------------

def test_drift_monitor_stable_then_shift_resets_belief():
    mon = DriftMonitor()
    old = Belief(80.0, 20.0)   # a converged, confident posterior

    # Stable regime: no replacement belief, no events.
    for _ in range(40):
        assert mon.observe("s", "a", 0.8, old) is None
    assert mon.events == []

    # Regime collapse: within a handful of observations the monitor returns a
    # resurrected (wide, low-evidence) replacement belief.
    replacement = None
    steps = 0
    for _ in range(30):
        steps += 1
        replacement = mon.observe("s", "a", 0.25, old)
        if replacement is not None:
            break
    assert replacement is not None
    assert steps < 20

    # Soft reset: evidence collapses to ~reset_evidence and the posterior is
    # much wider than the stale converged one.
    assert replacement.evidence == pytest.approx(mon.reset_evidence, rel=0.01)
    assert replacement.std > old.std

    # The event was recorded, describes itself, and drains exactly once.
    assert len(mon.events) == 1
    ev = mon.events[0]
    assert ev.stage == "s" and ev.agent == "a"
    text = ev.describe()
    assert "DRIFT detected" in text and "reset" in text
    drained = mon.drain_events()
    assert len(drained) == 1
    assert mon.events == []
    assert mon.drain_events() == []


# --- router + drift integration ---------------------------------------------

def test_router_drift_collapses_evidence_on_alarm():
    mon = DriftMonitor()
    router = BanditRouter(rng=random.Random(0), drift=mon)
    router.register("s", SimulatedAgent("a").seed(0))

    for _ in range(60):
        router.update("s", "a", 0.85)
    evidence_before = router.belief("s", "a").evidence
    assert evidence_before > 55   # 60 updates on a Beta(1,1) prior

    for _ in range(30):
        router.update("s", "a", 0.2)
        if router.belief("s", "a").evidence < 20:
            break

    evidence_after = router.belief("s", "a").evidence
    # Uncertainty resurrection: evidence collapsed to near reset_evidence,
    # a fraction of the pre-alarm mass.
    assert evidence_after < 20
    assert evidence_after < evidence_before / 3
    assert len(mon.events) >= 1


def test_max_evidence_caps_growth_but_tracks_mean():
    capped = BanditRouter(rng=random.Random(0), max_evidence=50.0)
    capped.register("s", SimulatedAgent("a").seed(0))
    for _ in range(300):
        capped.update("s", "a", 0.8)
    b = capped.belief("s", "a")
    assert b.evidence <= 52.0            # cap 50 + at most one update's worth
    assert b.mean == pytest.approx(0.8, abs=0.05)

    uncapped = BanditRouter(rng=random.Random(0))
    uncapped.register("s", SimulatedAgent("a").seed(0))
    for _ in range(300):
        uncapped.update("s", "a", 0.8)
    # Without the cap evidence keeps every observation: 2 (prior) + 300.
    assert uncapped.belief("s", "a").evidence == pytest.approx(302.0)
    assert uncapped.belief("s", "a").mean == pytest.approx(0.8, abs=0.05)


# --- end-to-end recovery -----------------------------------------------------

def _runs_until_switch(with_drift: bool) -> int:
    """Warm up on a 0.85 star vs 0.75 backup, degrade the star to 0.3, and
    count runs until greedy routing flips to the backup."""
    star = SimulatedAgent("star", {"s": (0.85, 0.10)}).seed(11)
    backup = SimulatedAgent("backup", {"s": (0.75, 0.10)}).seed(12)
    kwargs = {}
    if with_drift:
        kwargs = dict(drift=DriftMonitor(), max_evidence=60.0)
    orch = Orchestrator.build(
        graph=TaskGraph.linear("s"),
        agents_by_stage={"s": [star, backup]},
        seed=5,
        **kwargs,
    )

    for _ in range(80):
        orch.run("warmup", stakes=0.7)
    assert orch.router.greedy_select("s") == "star"

    # The provider silently degrades the star agent.
    star.competence["s"] = (0.30, 0.10)

    for i in range(1, 121):
        orch.run("post-shift", stakes=0.7)
        if orch.router.greedy_select("s") == "backup":
            return i
    return 999


def test_drift_recovery_switches_fast_and_faster_than_without():
    with_drift = _runs_until_switch(with_drift=True)
    without_drift = _runs_until_switch(with_drift=False)

    # With uncertainty resurrection + bounded evidence, the router flips to the
    # backup within a few tens of runs.
    assert with_drift <= 40
    # And it beats the plain (stationary-assumption) bandit outright.
    assert with_drift < without_drift
