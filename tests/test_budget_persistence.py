"""v0.2 industrial-hardening tests: compute budgets (graceful degradation) and
policy persistence (save/load of everything the system has learned)."""

from __future__ import annotations

import random
import threading

import pytest

from prism.agents import SimulatedAgent
from prism.budget import ComputeBudget
from prism.errors import BudgetExhausted
from prism.graph import TaskGraph
from prism.orchestrator import Orchestrator
from prism.resilience import CircuitBreaker
from prism.scenarios import STAGES, research_pipeline
from prism.speculation import SpeculationPolicy
from prism.trace import CausalTrace
from prism.types import RouteMode
from prism.uncertainty import Belief


# --- ComputeBudget unit ------------------------------------------------------

def test_budget_spend_remaining_and_affordability():
    b = ComputeBudget(total=100.0, reserve_fraction=0.25)
    assert b.remaining == pytest.approx(100.0)

    b.spend(30.0)
    assert b.spent == pytest.approx(30.0)
    assert b.remaining == pytest.approx(70.0)

    # Extra (speculative) spend only draws on the unreserved slice.
    assert b.can_afford_extra(70.0 * 0.75)
    assert not b.can_afford_extra(70.0 * 0.75 + 0.5)
    # Base (greedy) spend may dip into the reserve.
    assert b.can_afford_base(70.0)
    assert not b.can_afford_base(70.5)

    # Negative spends are ignored, and remaining never goes below zero.
    b.spend(-10.0)
    assert b.spent == pytest.approx(30.0)
    b.spend(1000.0)
    assert b.remaining == 0.0

    snap = b.snapshot()
    assert snap["total"] == pytest.approx(100.0)
    assert snap["spent"] == pytest.approx(1030.0)
    assert snap["remaining"] == 0.0


def test_budget_spend_is_thread_safe():
    b = ComputeBudget(total=1000.0)

    def worker():
        for _ in range(100):
            b.spend(1.0)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert b.spent == pytest.approx(800.0)
    assert b.remaining == pytest.approx(200.0)


# --- SpeculationPolicy budget governor ---------------------------------------

def _near_tie_inputs():
    # Wide, near-tied beliefs: textbook speculation territory.
    beliefs = {"a": Belief(8.0, 2.0), "b": Belief(7.8, 2.2)}
    costs = {"a": 1.0, "b": 1.0}
    return beliefs, costs


def test_policy_speculates_without_budget_pressure():
    pol = SpeculationPolicy(rng=random.Random(2), cost_weight=0.001)
    beliefs, costs = _near_tie_inputs()
    d = pol.decide(beliefs, costs, stakes=1.0)
    assert d.mode is RouteMode.SPECULATIVE
    assert len(d.branches) == 2


def test_policy_exhausted_budget_forces_greedy_and_says_so():
    pol = SpeculationPolicy(rng=random.Random(2), cost_weight=0.001)
    beliefs, costs = _near_tie_inputs()
    exhausted = ComputeBudget(total=10.0)
    exhausted.spend(10.0)

    d = pol.decide(beliefs, costs, stakes=1.0, budget=exhausted)
    assert d.mode is RouteMode.GREEDY
    assert len(d.branches) == 1
    assert "budget" in d.explanation


# --- orchestrator budget gate ------------------------------------------------

def test_orchestrator_ample_budget_records_spend():
    orch = research_pipeline(seed=0)
    budget = ComputeBudget(total=500.0)
    trace = orch.run("q", stakes=0.7, budget=budget)
    assert len(trace.stages) == len(STAGES)
    assert budget.spent > 0.0
    assert budget.spent == pytest.approx(trace.total_cost)
    assert budget.remaining < 500.0


def test_orchestrator_tiny_budget_raises_budget_exhausted_with_trace():
    orch = research_pipeline(seed=0)
    # Enough for one triage call (cheapest 0.5) but never for retrieve
    # (cheapest 0.3) afterwards.
    budget = ComputeBudget(total=0.6)

    with pytest.raises(BudgetExhausted) as ei:
        orch.run("q", stakes=0.7, budget=budget)

    exc = ei.value
    assert isinstance(exc.trace, CausalTrace)     # partial provenance attached
    assert len(exc.trace.stages) == 1             # triage ran, retrieve did not
    assert exc.task_type == "retrieve"
    assert 0.0 <= exc.remaining < exc.needed
    assert exc.needed == pytest.approx(0.3)       # bm25 is the cheapest arm


# --- persistence -------------------------------------------------------------

def test_save_load_round_trips_beliefs_and_edges(tmp_path):
    orch = research_pipeline(seed=0)
    for _ in range(30):
        orch.run("q", stakes=0.7)

    path = tmp_path / "policy.json"
    orch.save_policy(path)

    fresh = research_pipeline(seed=0)
    counts = fresh.load_policy(path)

    assert counts["beliefs"] == 12    # 4 stages x 3 agents
    assert counts["edges"] == 3       # linear 4-stage graph

    # Beliefs identical, bit for bit (JSON round-trips Python floats exactly).
    assert fresh.router.state_dict() == orch.router.state_dict()
    assert fresh.graph.state_dict() == orch.graph.state_dict()

    # And they are genuinely *learned* beliefs, not the uninformative prior.
    warmed = orch.router.state_dict()
    assert any(a + b > 2.5 for arms in warmed.values() for a, b in arms.values())


def _mini_orch(names_by_stage: dict[str, list[str]],
               breaker: CircuitBreaker | None = None) -> Orchestrator:
    graph = TaskGraph.linear("s1", "s2")
    agents = {
        stage: [SimulatedAgent(n, {stage: (0.7, 0.10)}).seed(7) for n in names]
        for stage, names in names_by_stage.items()
    }
    return Orchestrator.build(graph=graph, agents_by_stage=agents, seed=7,
                              breaker=breaker)


def test_load_into_smaller_roster_loads_intersection(tmp_path):
    full = _mini_orch({"s1": ["a", "b"], "s2": ["c", "d"]})
    for _ in range(10):
        full.run("q", stakes=0.6)
    path = tmp_path / "mini.json"
    full.save_policy(path)

    smaller = _mini_orch({"s1": ["a"], "s2": ["c", "d"]})   # 'b' left the roster
    counts = smaller.load_policy(path)                       # must not raise

    assert counts["beliefs"] == 3     # intersection only
    assert counts["edges"] == 1       # the single s1 -> s2 edge
    assert smaller.router.state_dict()["s1"]["a"] == \
        full.router.state_dict()["s1"]["a"]
    assert "b" not in smaller.router.state_dict()["s1"]


def test_graph_edge_beliefs_round_trip(tmp_path):
    orch = _mini_orch({"s1": ["a", "b"], "s2": ["c", "d"]})
    for _ in range(10):
        orch.run("q", stakes=0.6)
    rows = orch.graph.state_dict()
    assert len(rows) == 1
    # The edge belief actually learned something during the warm runs.
    assert rows[0]["alpha"] + rows[0]["beta"] > 2.5

    path = tmp_path / "edges.json"
    orch.save_policy(path)
    fresh = _mini_orch({"s1": ["a", "b"], "s2": ["c", "d"]})
    fresh.load_policy(path)
    assert fresh.graph.state_dict() == rows


def test_breaker_state_round_trips_via_orchestrator(tmp_path):
    a = _mini_orch({"s1": ["a", "b"], "s2": ["c", "d"]},
                   breaker=CircuitBreaker(failure_threshold=2, base_cooldown=4))
    a.breaker.record("s1", "a", True, tick=1)
    a.breaker.record("s1", "a", True, tick=2)     # opens the (s1, a) arm
    a.breaker.record("s2", "d", False, tick=2)

    path = tmp_path / "with_breaker.json"
    a.save_policy(path)

    b = _mini_orch({"s1": ["a", "b"], "s2": ["c", "d"]},
                   breaker=CircuitBreaker(failure_threshold=2, base_cooldown=4))
    counts = b.load_policy(path)

    assert counts["breaker_arms"] == 2
    assert b.breaker.state_dict() == a.breaker.state_dict()
    # Behavioural check: the restored breaker still blocks the open arm.
    assert b.breaker.filter("s1", ["a", "b"], tick=3)[1] == ["a"]
