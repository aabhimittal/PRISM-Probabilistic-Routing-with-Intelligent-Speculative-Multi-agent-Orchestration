"""v0.2 industrial-hardening tests: branch failure semantics, executor
timeouts, orchestrator failover, and the circuit breaker."""

from __future__ import annotations

import random
import time

import pytest

from prism.agents import FlakyAgent, SimulatedAgent, SimulatedScorer
from prism.errors import AllBranchesFailed, StageFailure
from prism.graph import TaskGraph
from prism.orchestrator import Orchestrator
from prism.resilience import CircuitBreaker
from prism.speculation import SpeculativeExecutor
from prism.trace import CausalTrace
from prism.types import AgentOutput, Task
from prism.uncertainty import Belief


ALWAYS = (0, 10**9)  # fail_between window meaning "every call fails"


class _SleepAgent:
    """A real-sleeping agent for timeout tests."""

    def __init__(self, name: str, seconds: float, cost: float = 1.0,
                 quality: float = 0.9) -> None:
        self.name = name
        self.seconds = seconds
        self.cost = cost
        self.quality = quality

    def run(self, task: Task) -> AgentOutput:
        time.sleep(self.seconds)
        return AgentOutput(output=f"{self.name}-out", self_report=self.quality,
                           latent_quality=self.quality, cost=self.cost,
                           latency=self.seconds)


def _executor(timeout=None) -> SpeculativeExecutor:
    return SpeculativeExecutor(
        scorer=SimulatedScorer(reliability=0.9, rng=random.Random(2)),
        timeout=timeout,
    )


# --- executor failure semantics ---------------------------------------------

def test_one_failed_branch_survivors_win_and_cost_counted():
    good = SimulatedAgent("good", {"t": (0.85, 0.08)}, cost=1.0).seed(1)
    also = SimulatedAgent("also", {"t": (0.60, 0.10)}, cost=1.5).seed(1)
    bad = FlakyAgent(SimulatedAgent("bad", {"t": (0.90, 0.05)}, cost=2.0).seed(1),
                     fail_between=ALWAYS)
    agents = [good, also, bad]
    priors = {a.name: Belief(1.0, 1.0) for a in agents}
    squashed_names: list[str] = []

    res = _executor().run_stage(
        Task(task_type="t", payload="p", stakes=0.8), agents, priors,
        on_squash=lambda a, out: squashed_names.append(a.name))

    assert len(res.outcomes) == 3
    failed = [o for o in res.outcomes if o.failed]
    assert len(failed) == 1
    assert failed[0].agent_name == "bad"
    assert failed[0].error != ""
    assert "simulated failure" in failed[0].error
    assert failed[0].score == 0.0

    winners = [o for o in res.outcomes if o.is_winner]
    assert len(winners) == 1
    assert winners[0].agent_name in {"good", "also"}

    # A failed call still cost the call: cost is summed over ALL branches.
    assert res.cost == pytest.approx(good.cost + also.cost + bad.cost)

    # on_squash fires only for the surviving loser, never for the failed branch.
    assert "bad" not in squashed_names
    assert len(squashed_names) == 1
    assert squashed_names[0] in {"good", "also"}


def test_all_branches_raise_gives_all_branches_failed():
    a1 = FlakyAgent(SimulatedAgent("a1").seed(0), fail_between=ALWAYS)
    a2 = FlakyAgent(SimulatedAgent("a2").seed(0), fail_between=ALWAYS)
    priors = {"a1": Belief(1.0, 1.0), "a2": Belief(1.0, 1.0)}

    with pytest.raises(AllBranchesFailed) as ei:
        _executor().run_stage(Task(task_type="t", payload="p"), [a1, a2], priors)

    exc = ei.value
    assert exc.task_type == "t"
    assert len(exc.outcomes) == 2
    assert all(o.failed for o in exc.outcomes)
    assert all(o.error != "" for o in exc.outcomes)


def test_executor_timeout_marks_branch_failed_and_returns_fast():
    sleeper = _SleepAgent("sleeper", seconds=0.5, cost=1.0)
    fast = SimulatedAgent("fast", {"t": (0.8, 0.1)}, cost=1.0).seed(3)
    priors = {"sleeper": Belief(1.0, 1.0), "fast": Belief(1.0, 1.0)}
    ex = _executor(timeout=0.05)

    t0 = time.perf_counter()
    res = ex.run_stage(Task(task_type="t", payload="p"), [sleeper, fast], priors)
    elapsed = time.perf_counter() - t0

    # 4x+ margin: the sleeper takes 0.5s; a timely return proves the timeout cut.
    assert elapsed < 0.4

    by_name = {o.agent_name: o for o in res.outcomes}
    assert by_name["sleeper"].failed is True
    assert by_name["sleeper"].error == "timeout"
    assert by_name["sleeper"].score == 0.0
    assert res.winner.agent_name == "fast"
    assert by_name["fast"].failed is False
    # Timed-out branch still counted in cost.
    assert res.cost == pytest.approx(sleeper.cost + fast.cost)
    ex.close()


# --- orchestrator failover ---------------------------------------------------

def _failover_setup(fail_between, breaker=None):
    """One stage, one dominant-but-flaky arm, one steady backup arm."""
    flaky = FlakyAgent(SimulatedAgent("volatile", {"s": (0.90, 0.05)}).seed(2),
                       fail_between=fail_between)
    steady = SimulatedAgent("steady", {"s": (0.75, 0.10)}).seed(2)
    orch = Orchestrator.build(
        graph=TaskGraph.linear("s"),
        agents_by_stage={"s": [flaky, steady]},
        seed=4,
        priors={"s": {
            # Dominant, tight prior so the router deterministically commits to
            # the flaky arm; the backup is tight around 0.5 so Thompson never
            # out-draws Beta(200, 2).
            "volatile": Belief(200.0, 2.0),
            "steady": Belief(60.0, 60.0),
        }},
        breaker=breaker,
    )
    return orch, flaky


def test_orchestrator_failover_when_committed_branch_fails():
    orch, _ = _failover_setup(fail_between=ALWAYS)
    prior_mean = orch.router.belief("s", "volatile").mean

    trace = orch.run("q", stakes=0.7)

    # The run still succeeded via failover.
    assert trace.final_output is not None
    assert len(trace.stages) == 1
    assert any("failing over" in e for e in trace.events)

    st = trace.stages[0]
    # Failed first-round outcomes stay in the stage trace (provenance).
    assert any(o.failed and o.agent_name == "volatile" for o in st.outcomes)
    assert st.winner == "steady"

    # The bandit learned from the failure: reward 0 dropped the belief mean.
    assert orch.router.belief("s", "volatile").mean < prior_mean


def test_orchestrator_all_arms_fail_raises_stage_failure_with_trace():
    f1 = FlakyAgent(SimulatedAgent("f1", {"s": (0.8, 0.1)}).seed(0),
                    fail_between=ALWAYS)
    f2 = FlakyAgent(SimulatedAgent("f2", {"s": (0.7, 0.1)}).seed(0),
                    fail_between=ALWAYS)
    orch = Orchestrator.build(graph=TaskGraph.linear("s"),
                              agents_by_stage={"s": [f1, f2]}, seed=1)

    with pytest.raises(StageFailure) as ei:
        orch.run("q", stakes=0.8)

    exc = ei.value
    assert exc.task_type == "s"
    assert isinstance(exc.trace, CausalTrace)   # partial trace attached
    assert exc.trace.stages == []               # died at the first stage


# --- CircuitBreaker unit -----------------------------------------------------

def test_breaker_opens_after_consecutive_failures_and_blocks():
    br = CircuitBreaker(failure_threshold=3, base_cooldown=5)
    assert br.record("s", "a", True, tick=0) is None
    assert br.record("s", "a", True, tick=0) is None
    msg = br.record("s", "a", True, tick=0)
    assert msg is not None and "OPEN" in msg

    allowed, blocked = br.filter("s", ["a", "b"], tick=1)
    assert allowed == ["b"]
    assert blocked == ["a"]


def test_breaker_half_open_probe_failure_doubles_cooldown():
    br = CircuitBreaker(failure_threshold=3, base_cooldown=5)
    for _ in range(3):
        br.record("s", "a", True, tick=0)   # opens with cooldown 5 → until 5

    # Still blocked mid-cooldown.
    assert br.filter("s", ["a", "b"], tick=4)[1] == ["a"]

    # Cooldown expired: the arm passes filter again as a half-open probe.
    allowed, blocked = br.filter("s", ["a", "b"], tick=5)
    assert "a" in allowed and blocked == []

    # Failed probe re-opens with doubled cooldown (5 → 10).
    msg = br.record("s", "a", True, tick=5)
    assert msg is not None and "RE-OPENED" in msg
    assert "cooldown 10" in msg
    assert br.filter("s", ["a", "b"], tick=14)[1] == ["a"]

    # Second expiry → probe → success closes the breaker.
    allowed, _ = br.filter("s", ["a", "b"], tick=15)
    assert "a" in allowed
    msg = br.record("s", "a", False, tick=15)
    assert msg is not None and "CLOSED" in msg

    allowed, blocked = br.filter("s", ["a", "b"], tick=16)
    assert allowed == ["a", "b"] and blocked == []


def test_breaker_all_open_roster_returns_all_names():
    br = CircuitBreaker(failure_threshold=1, base_cooldown=10)
    br.record("s", "a", True, tick=0)
    br.record("s", "b", True, tick=0)
    # Both open — refusing to route would be an outage, so all pass through.
    allowed, blocked = br.filter("s", ["a", "b"], tick=1)
    assert allowed == ["a", "b"]
    assert blocked == []


def test_breaker_state_dict_round_trip():
    br = CircuitBreaker(failure_threshold=2, base_cooldown=4)
    br.record("s", "a", True, tick=1)
    br.record("s", "a", True, tick=2)      # opens
    br.record("s", "b", False, tick=2)
    br.record("other", "c", True, tick=3)

    state = br.state_dict()
    assert "s::a" in state and state["s::a"]["trips"] == 1

    fresh = CircuitBreaker(failure_threshold=2, base_cooldown=4)
    fresh.load_state_dict(state)
    assert fresh.state_dict() == state
    # Behaviour restored, not just numbers: 'a' is still blocked mid-cooldown.
    assert fresh.filter("s", ["a", "b"], tick=3)[1] == ["a"]


# --- orchestrator + breaker integration --------------------------------------

def test_orchestrator_breaker_trips_then_heals():
    # The dominant arm fails on its first 3 calls, then recovers for good.
    breaker = CircuitBreaker(failure_threshold=2, base_cooldown=3)
    orch, flaky = _failover_setup(fail_between=(0, 3), breaker=breaker)

    all_events: list[str] = []
    winners: list[str] = []
    for _ in range(20):
        trace = orch.run("q", stakes=0.7)
        all_events.extend(trace.events)
        winners.append(trace.stages[0].winner)

    # The breaker tripped at some point ...
    assert any("breaker OPEN" in e for e in all_events)
    # ... held the arm out of the candidate set while open ...
    assert any("circuit breaker holding" in e for e in all_events)
    # ... and eventually closed after a successful half-open probe.
    assert any("breaker CLOSED" in e for e in all_events)

    # After healing, the (dominant) agent is routable and winning again.
    assert "volatile" in winners[-5:]
    # And the healed agent really did serve calls beyond its outage window.
    assert flaky.calls > 3
