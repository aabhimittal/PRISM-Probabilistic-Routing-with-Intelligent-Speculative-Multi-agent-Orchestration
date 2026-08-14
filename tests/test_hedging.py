"""v0.2 industrial-hardening tests: latency tracking, hedge planning, and the
hedged execution path (Dean & Barroso hedged requests applied to agents).

Timing-based assertions use margins of at least 4x the relevant sleeps so a
loaded CI machine cannot flip them.
"""

from __future__ import annotations

import random
import threading
import time

import pytest

from prism.agents import SimulatedAgent, SimulatedScorer
from prism.graph import TaskGraph
from prism.hedging import HedgePolicy, LatencyTracker
from prism.orchestrator import Orchestrator
from prism.speculation import SpeculationPolicy, SpeculativeExecutor
from prism.types import AgentOutput, Task
from prism.uncertainty import Belief


class _SleepAgent:
    def __init__(self, name: str, seconds: float, cost: float = 1.0,
                 quality: float = 0.8) -> None:
        self.name = name
        self.seconds = seconds
        self.cost = cost
        self.quality = quality

    def run(self, task: Task) -> AgentOutput:
        time.sleep(self.seconds)
        return AgentOutput(output=f"{self.name}-out", self_report=self.quality,
                           latent_quality=self.quality, cost=self.cost,
                           latency=self.seconds)


class _FailAgent:
    def __init__(self, name: str, cost: float = 1.0) -> None:
        self.name = name
        self.cost = cost

    def run(self, task: Task) -> AgentOutput:
        raise RuntimeError(f"{self.name} exploded")


def _executor(timeout=None) -> SpeculativeExecutor:
    return SpeculativeExecutor(
        scorer=SimulatedScorer(reliability=0.9, rng=random.Random(5)),
        timeout=timeout,
    )


# --- LatencyTracker ----------------------------------------------------------

def test_fence_is_none_before_min_observations():
    tr = LatencyTracker(min_observations=5)
    for lat in (0.02, 0.03, 0.02, 0.04):
        tr.observe("s", "a", lat)
    # Invalid latencies must not count toward the minimum either.
    tr.observe("s", "a", 0.0)
    tr.observe("s", "a", -1.0)
    tr.observe("s", "a", float("inf"))
    assert tr.fence("s", "a") is None
    # A fifth valid observation unlocks the fence.
    tr.observe("s", "a", 0.03)
    assert tr.fence("s", "a") is not None


def test_fence_matches_expected_empirical_quantile():
    values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    shuffled = list(values)
    random.Random(3).shuffle(shuffled)

    tr90 = LatencyTracker(quantile=0.9, window=50, min_fence=0.001,
                          min_observations=5)
    tr50 = LatencyTracker(quantile=0.5, window=50, min_fence=0.001,
                          min_observations=5)
    for v in shuffled:
        tr90.observe("s", "a", v)
        tr50.observe("s", "a", v)

    # idx = min(int(q * n), n - 1) over the sorted window.
    assert tr90.fence("s", "a") == pytest.approx(values[9])   # int(0.9*10) = 9
    assert tr50.fence("s", "a") == pytest.approx(values[5])   # int(0.5*10) = 5


def test_bimodal_window_fence_sits_near_the_fast_mode():
    # 46 fast calls (~20ms) and 4 tail disasters (300ms): the design point.
    # A mean+k*sigma fence would be hoisted by the tail; the empirical 0.9
    # quantile stays with the fast mode, below 100ms.
    tr = LatencyTracker(quantile=0.9, window=50, min_fence=0.001,
                        min_observations=5)
    rng = random.Random(7)
    lats = [0.02 + rng.uniform(-0.002, 0.002) for _ in range(46)]
    lats += [0.3] * 4
    rng.shuffle(lats)
    for v in lats:
        tr.observe("s", "a", v)

    fence = tr.fence("s", "a")
    assert fence is not None
    assert fence < 0.1
    assert fence == pytest.approx(0.02, abs=0.01)


def test_min_fence_floor_respected():
    tr = LatencyTracker(quantile=0.95, min_fence=0.05, min_observations=3)
    for _ in range(6):
        tr.observe("s", "a", 0.001)
    assert tr.fence("s", "a") == pytest.approx(0.05)


# --- HedgePolicy.plan --------------------------------------------------------

def test_plan_none_when_stakes_below_floor():
    pol = HedgePolicy(fallback_fence=0.1, stakes_floor=0.3)
    assert pol.plan("s", "a", {"a": 0.6, "b": 0.4}, stakes=0.1) is None


def test_plan_none_with_single_candidate():
    pol = HedgePolicy(fallback_fence=0.1)
    assert pol.plan("s", "a", {"a": 1.0}, stakes=0.9) is None


def test_plan_none_without_fence_or_fallback():
    pol = HedgePolicy()   # cold tracker, no fallback fence
    assert pol.plan("s", "a", {"a": 0.6, "b": 0.4}, stakes=0.9) is None


def test_plan_returns_best_backup_with_learned_fence():
    tr = LatencyTracker(quantile=0.5, min_observations=3)
    for _ in range(6):
        tr.observe("s", "a", 0.08)
    pol = HedgePolicy(tracker=tr)
    plan = pol.plan("s", "a", {"a": 0.5, "b": 0.2, "c": 0.3}, stakes=0.9)
    assert plan is not None
    backup, fence = plan
    assert backup == "c"                      # best NON-primary arm
    assert fence == pytest.approx(0.08)


def test_plan_uses_fallback_fence_before_profile_warms():
    pol = HedgePolicy(fallback_fence=0.07)
    plan = pol.plan("s", "a", {"a": 0.7, "b": 0.3}, stakes=0.9)
    assert plan == ("b", pytest.approx(0.07))


def test_plan_respects_backup_floor():
    pol = HedgePolicy(fallback_fence=0.1, backup_floor=0.4)
    assert pol.plan("s", "a", {"a": 0.9, "b": 0.1}, stakes=0.9) is None


# --- run_stage_hedged with real sleeps ---------------------------------------

def test_hedge_fires_and_backup_wins_the_race():
    primary = _SleepAgent("primary", seconds=0.3, cost=2.0)
    backup = _SleepAgent("backup", seconds=0.02, cost=1.0)
    priors = {"primary": Belief(5.0, 5.0), "backup": Belief(5.0, 5.0)}
    ex = _executor()

    late: list[tuple] = []
    late_seen = threading.Event()

    def on_late(name, output, error):
        late.append((name, output, error))
        late_seen.set()

    t0 = time.perf_counter()
    res = ex.run_stage_hedged(Task(task_type="t", payload="p"), primary, backup,
                              priors, hedge_after=0.05, on_late=on_late)
    elapsed = time.perf_counter() - t0

    assert res.hedged is True
    assert res.winner.agent_name == "backup"
    # ~0.07s expected; 0.25 keeps a 4x margin below the 0.3s primary sleep.
    assert elapsed < 0.25
    assert res.latency < 0.25
    # The hedge launched: both agents were paid for.
    assert res.cost == pytest.approx(primary.cost + backup.cost)
    assert "hedge" in res.note

    # The straggling primary lands later and still reports in via on_late.
    time.sleep(0.5)
    assert late_seen.wait(2.0)
    assert len(late) == 1
    name, output, error = late[0]
    assert name == "primary"
    assert output is not None and error is None
    ex.close()


def test_primary_faster_than_fence_is_not_hedged():
    primary = _SleepAgent("primary", seconds=0.01, cost=2.0)
    backup = _SleepAgent("backup", seconds=0.01, cost=1.0)
    priors = {"primary": Belief(5.0, 5.0), "backup": Belief(5.0, 5.0)}
    ex = _executor()

    res = ex.run_stage_hedged(Task(task_type="t", payload="p"), primary, backup,
                              priors, hedge_after=0.2)
    assert res.hedged is False
    assert res.winner.agent_name == "primary"
    assert res.cost == pytest.approx(primary.cost)   # backup never launched
    ex.close()


def test_primary_fails_fast_fails_over_to_backup():
    primary = _FailAgent("primary", cost=2.0)
    backup = _SleepAgent("backup", seconds=0.02, cost=1.0)
    priors = {"primary": Belief(5.0, 5.0), "backup": Belief(5.0, 5.0)}
    ex = _executor()

    res = ex.run_stage_hedged(Task(task_type="t", payload="p"), primary, backup,
                              priors, hedge_after=0.2)
    assert res.hedged is False
    assert res.winner.agent_name == "backup"
    assert "failover" in res.note
    # The failed primary call is still paid for.
    assert res.cost == pytest.approx(primary.cost + backup.cost)
    ex.close()


# --- orchestrator integration ------------------------------------------------

def test_orchestrator_hedging_collapses_the_tail():
    # Speculation off (max_branches=1): hedging is the only multi-branch
    # mechanism in play. One arm has a heavy tail (30% of calls take 0.6s);
    # the fallback fence fires the hedge at 50ms.
    jittery = SimulatedAgent("jittery", {"s": (0.75, 0.10)}, base_latency=0.01,
                             simulate_latency=True, tail=(0.3, 0.6)).seed(21)
    steady = SimulatedAgent("steady", {"s": (0.75, 0.10)}, base_latency=0.01,
                            simulate_latency=True).seed(22)
    hedge = HedgePolicy(
        tracker=LatencyTracker(min_observations=10**9),  # keep fallback active
        fallback_fence=0.05,
        stakes_floor=0.0,
        backup_floor=0.0,
    )
    orch = Orchestrator.build(
        graph=TaskGraph.linear("s"),
        agents_by_stage={"s": [jittery, steady]},
        seed=6,
        policy=SpeculationPolicy(max_branches=1, rng=random.Random(6)),
        hedge=hedge,
    )
    # Generous headroom for abandoned stragglers sleeping out their tails, so
    # queueing can never inflate a later run's measured latency.
    orch.executor.max_workers = 32

    traces = [orch.run(f"q{i}", stakes=0.8) for i in range(40)]

    assert all(t.speculations == 0 for t in traces)   # speculation truly off
    events = [e for t in traces for e in t.events]
    assert any("hedge fired" in e for e in events)

    # The whole point: no run ever pays the 0.6s tail. Expected worst case is
    # ~fence + backup (~70ms); 0.35 leaves ~5x margin over that while staying
    # well under the 0.6s tail latency.
    assert max(t.total_latency for t in traces) < 0.35

    # Stragglers eventually land and are folded into the policy.
    time.sleep(0.9)
    assert len(orch.late_events) >= 1
    assert any("late straggler" in e for e in orch.late_events)
    orch.executor.close()
