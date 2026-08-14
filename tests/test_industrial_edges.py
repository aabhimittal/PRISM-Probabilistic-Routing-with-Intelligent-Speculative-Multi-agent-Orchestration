"""v0.2 industrial-hardening tests: numeric garbage-tolerance, scorer failure
fallbacks, configuration errors, cycle bounds, concurrency, and pool hygiene."""

from __future__ import annotations

import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from prism.agents import FlakyAgent, SimulatedAgent, SimulatedScorer
from prism.errors import ConfigurationError
from prism.graph import TaskGraph
from prism.orchestrator import Orchestrator
from prism.routing import BanditRouter
from prism.scenarios import research_pipeline
from prism.speculation import SpeculativeExecutor
from prism.types import AgentOutput, Task
from prism.uncertainty import Belief, from_mean_std, propagate


# --- Belief garbage-tolerance ------------------------------------------------

def test_belief_updated_nan_and_inf_treated_as_zero_reward():
    b = Belief(5.0, 5.0)
    for garbage in (float("nan"), float("inf"), float("-inf")):
        after = b.updated(garbage)
        # Treated as a total failure: alpha untouched, beta gains the weight.
        assert after.alpha == pytest.approx(b.alpha)
        assert after.beta == pytest.approx(b.beta + 1.0)
        assert after.mean < b.mean
        assert math.isfinite(after.mean) and math.isfinite(after.std)


def test_belief_updated_negative_weight_is_ignored():
    b = Belief(5.0, 5.0)
    after = b.updated(0.9, weight=-2.0)
    assert after.alpha == pytest.approx(b.alpha)
    assert after.beta == pytest.approx(b.beta)
    assert after.mean == pytest.approx(b.mean)


# --- scorer failure semantics ------------------------------------------------

class _BoomScorer:
    reliability = 0.9

    def score(self, task, output):
        raise RuntimeError("judge crashed")


class _NanScorer:
    reliability = 0.9

    def score(self, task, output):
        return float("nan")


class _KnownAgent:
    """Agent whose self_report is exactly known, for fallback assertions."""

    def __init__(self, name: str, self_report: float, cost: float = 1.0) -> None:
        self.name = name
        self.cost = cost
        self._sr = self_report

    def run(self, task: Task) -> AgentOutput:
        return AgentOutput(output=f"{self.name}!", self_report=self._sr,
                           latent_quality=None, cost=self.cost, latency=0.001)


def test_scorer_exception_falls_back_to_self_report():
    ex = SpeculativeExecutor(scorer=_BoomScorer())
    strong = _KnownAgent("strong", self_report=0.8)
    weak = _KnownAgent("weak", self_report=0.4)
    priors = {"strong": Belief(1.0, 1.0), "weak": Belief(1.0, 1.0)}

    res = ex.run_stage(Task(task_type="t", payload="p"), [strong, weak], priors)

    # Both branches survive the broken judge...
    assert all(not o.failed for o in res.outcomes)
    # ...selection falls back to self_report, so the confident agent wins...
    assert res.winner.agent_name == "strong"
    assert res.winner.score == pytest.approx(0.8)
    # ...and the trace note discloses what happened.
    assert "scorer error" in res.note
    ex.close()


def test_nan_score_does_not_poison_router_belief():
    r = BanditRouter(rng=random.Random(0))
    r.register("t", SimulatedAgent("a").seed(0))

    before, after = r.update("t", "a", float("nan"))

    assert math.isfinite(after.mean) and math.isfinite(after.std)
    assert 0.0 < after.mean < 1.0
    assert after.mean < before.mean                     # NaN became reward 0
    assert after.evidence == pytest.approx(before.evidence + 1.0)


# --- configuration errors ----------------------------------------------------

def test_stage_with_no_agents_raises_configuration_error():
    orch = Orchestrator.build(graph=TaskGraph.linear("s1"),
                              agents_by_stage={}, seed=0)
    with pytest.raises(ConfigurationError):
        orch.run("q")


def test_graph_without_entry_raises_configuration_error():
    orch = Orchestrator.build(graph=TaskGraph(), agents_by_stage={}, seed=0)
    with pytest.raises(ConfigurationError):
        orch.run("q")


# --- pathological graph cycles -----------------------------------------------

def test_graph_cycle_terminates_at_max_stages():
    g = TaskGraph()
    g.add_edge("A", "B").add_edge("B", "A")
    orch = Orchestrator.build(
        graph=g,
        agents_by_stage={
            "A": [SimulatedAgent("aa", {"A": (0.7, 0.10)}).seed(0)],
            "B": [SimulatedAgent("bb", {"B": (0.7, 0.10)}).seed(0)],
        },
        seed=0,
    )
    orch.max_stages = 6

    trace = orch.run("loop", stakes=0.5)

    assert len(trace.stages) == 6
    assert [s.task_type for s in trace.stages] == ["A", "B", "A", "B", "A", "B"]
    assert trace.final_output is not None


# --- concurrency -------------------------------------------------------------

def _total_evidence(router: BanditRouter) -> float:
    return sum(a + b for arms in router.state_dict().values()
               for a, b in arms.values())


def test_concurrent_runs_are_thread_safe():
    orch = research_pipeline(seed=0)
    evidence_before = _total_evidence(orch.router)

    def job(n: int):
        return [orch.run(f"q{n}-{i}", stakes=0.7) for i in range(10)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(job, n) for n in range(8)]
        traces = [t for f in futures for t in f.result()]   # re-raises errors

    assert len(traces) == 80
    assert all(len(t.stages) == 4 for t in traces)

    # Every learned belief stayed finite and inside (0, 1).
    for arms in orch.router.snapshot().values():
        for mean, std in arms.values():
            assert 0.0 < mean < 1.0
            assert math.isfinite(std) and std >= 0.0

    # Evidence accounting: each run updates 1-3 branches at each of 4 stages,
    # one unit of evidence apiece.
    added = _total_evidence(orch.router) - evidence_before
    assert 4 * 80 - 1e-6 <= added <= 3 * 4 * 80 + 1e-6


# --- extreme numerics --------------------------------------------------------

def test_extreme_beliefs_stay_finite():
    big = Belief(1e6, 1e6)
    rng = random.Random(0)
    draw = big.sample(rng)
    assert 0.0 < draw < 1.0
    assert math.isfinite(big.entropy)
    assert math.isfinite(big.std)

    skewed = from_mean_std(0.999999, 5.0)   # absurd std gets clamped, not NaN
    assert skewed.alpha > 0.0 and skewed.beta > 0.0
    assert 0.0 < skewed.mean < 1.0
    assert math.isfinite(skewed.std)


def test_propagate_chain_of_fifty_stages_stays_valid():
    stage_belief = Belief(8.0, 2.0)
    compound = stage_belief
    for _ in range(50):
        compound = propagate(compound, stage_belief)
        assert 0.0 < compound.mean < 1.0
        assert math.isfinite(compound.std)
        assert compound.alpha > 0.0 and compound.beta > 0.0
    # Uncertainty compounded downward monotonically, never exploded.
    assert compound.mean < stage_belief.mean


# --- FlakyAgent contract -----------------------------------------------------

def test_flaky_agent_honors_fail_between_exactly():
    inner = SimulatedAgent("x", {"t": (0.7, 0.10)}).seed(0)
    flaky = FlakyAgent(inner, fail_between=(0, 3), exception=ValueError)
    task = Task(task_type="t", payload="p")

    for i in range(3):                       # calls 0, 1, 2 raise
        with pytest.raises(ValueError):
            flaky.run(task)
    out = flaky.run(task)                    # call 3 succeeds
    assert out.output is not None
    assert flaky.calls == 4


def test_flaky_agent_zero_failure_rate_never_raises():
    flaky = FlakyAgent(SimulatedAgent("y").seed(1), failure_rate=0.0, seed=1)
    task = Task(task_type="t", payload="p")
    for _ in range(50):
        assert flaky.run(task).output is not None


# --- executor pool hygiene ---------------------------------------------------

def test_pool_reuse_does_not_leak_threads():
    ex = SpeculativeExecutor(
        scorer=SimulatedScorer(reliability=0.9, rng=random.Random(3)),
        max_workers=4,
    )
    agents = [
        SimulatedAgent("p", {"t": (0.7, 0.10)}).seed(0),
        SimulatedAgent("q", {"t": (0.6, 0.10)}).seed(0),
    ]
    priors = {a.name: Belief(1.0, 1.0) for a in agents}

    for i in range(200):
        res = ex.run_stage(Task(task_type="t", payload=f"p{i}"), agents, priors)
        assert res.winner is not None

    # The persistent pool reuses its workers: thread count stays bounded.
    assert threading.active_count() < 40
    ex.close()
