"""Tests for prism.speculation — the greedy-vs-speculate decision and executor."""

from __future__ import annotations

import random

import pytest

from prism.agents import SimulatedAgent, SimulatedScorer
from prism.speculation import SpeculationPolicy, SpeculativeExecutor
from prism.types import RouteMode, Task
from prism.uncertainty import Belief


def _policy(**kw) -> SpeculationPolicy:
    kw.setdefault("rng", random.Random(11))
    return SpeculationPolicy(**kw)


# --- SpeculationPolicy.decide ----------------------------------------------

def test_single_candidate_is_greedy():
    pol = _policy()
    d = pol.decide({"only": Belief(5.0, 5.0)}, {"only": 1.0}, stakes=1.0)
    assert d.mode is RouteMode.GREEDY
    assert d.branches == ["only"]


def test_dominant_arm_does_not_speculate():
    pol = _policy()
    beliefs = {
        "dominant": Belief(90.0, 10.0),  # mean 0.9, tight
        "x": Belief(40.0, 60.0),         # mean 0.4, tight
        "y": Belief(40.0, 60.0),
    }
    costs = {n: 1.0 for n in beliefs}
    d = pol.decide(beliefs, costs, stakes=1.0)  # high stakes, still not worth it
    assert d.mode is RouteMode.GREEDY
    assert d.greedy == "dominant"
    assert len(d.branches) == 1


def test_near_tie_high_stakes_low_cost_speculates():
    pol = _policy(cost_weight=0.001)
    beliefs = {"a": Belief(80.0, 20.0), "b": Belief(78.0, 22.0)}  # ~0.8 near-tie
    costs = {"a": 1.0, "b": 1.0}
    d = pol.decide(beliefs, costs, stakes=1.0)
    assert d.mode is RouteMode.SPECULATIVE
    assert len(d.branches) > 1
    assert d.value_of_speculation > 0.0


def test_max_branches_caps_branch_count():
    pol = _policy(cost_weight=0.001, max_branches=2)
    beliefs = {n: Belief(80.0, 20.0) for n in ["a", "b", "c", "d", "e"]}
    costs = {n: 1.0 for n in beliefs}
    d = pol.decide(beliefs, costs, stakes=1.0)
    assert d.mode is RouteMode.SPECULATIVE
    assert len(d.branches) == 2


def test_max_branches_one_disables_speculation():
    pol = _policy(max_branches=1, cost_weight=0.001)
    beliefs = {"a": Belief(80.0, 20.0), "b": Belief(78.0, 22.0)}
    d = pol.decide(beliefs, {"a": 1.0, "b": 1.0}, stakes=1.0)
    assert d.mode is RouteMode.GREEDY
    assert len(d.branches) == 1


# --- SpeculativeExecutor.run_stage -----------------------------------------

def _agents():
    return [
        SimulatedAgent("strong", {"t": (0.85, 0.10)}).seed(0),
        SimulatedAgent("mid", {"t": (0.60, 0.15)}).seed(0),
        SimulatedAgent("weak", {"t": (0.35, 0.15)}).seed(0),
    ]


def test_run_stage_single_winner_losers_squashed_and_on_squash_called():
    agents = _agents()
    scorer = SimulatedScorer(reliability=0.9, rng=random.Random(5))
    ex = SpeculativeExecutor(scorer=scorer)
    task = Task(task_type="t", payload="hello", stakes=1.0)
    priors = {a.name: Belief(1.0, 1.0) for a in agents}

    squashed_calls = []
    result = ex.run_stage(task, agents, priors,
                          on_squash=lambda a, out: squashed_calls.append(a.name))

    assert len(result.outcomes) == 3
    winners = [o for o in result.outcomes if o.is_winner]
    assert len(winners) == 1
    assert result.winner.agent_name == winners[0].agent_name

    squashed = [o for o in result.outcomes if o.squashed]
    assert len(squashed) == 2
    assert winners[0].squashed is False
    # on_squash fired exactly once per squashed (losing) branch.
    assert sorted(squashed_calls) == sorted(o.agent_name for o in squashed)

    # cost is the sum over branches; latency is the max (parallel).
    assert result.cost == pytest.approx(sum(o.output.cost for o in result.outcomes))
    assert result.latency == pytest.approx(max(o.output.latency for o in result.outcomes))


def test_run_stage_single_branch_has_no_squash():
    agents = [_agents()[0]]
    ex = SpeculativeExecutor(scorer=SimulatedScorer(reliability=0.9, rng=random.Random(5)))
    task = Task(task_type="t", payload="hi", stakes=0.5)
    priors = {agents[0].name: Belief(1.0, 1.0)}

    calls = []
    result = ex.run_stage(task, agents, priors, on_squash=lambda a, out: calls.append(a.name))
    assert len(result.outcomes) == 1
    assert result.outcomes[0].is_winner is True
    assert result.outcomes[0].squashed is False
    assert calls == []


def test_run_stage_is_deterministic_with_seeds():
    def once():
        agents = _agents()
        ex = SpeculativeExecutor(scorer=SimulatedScorer(reliability=0.9, rng=random.Random(5)))
        task = Task(task_type="t", payload="hello", stakes=1.0)
        priors = {a.name: Belief(1.0, 1.0) for a in agents}
        return ex.run_stage(task, agents, priors).winner.agent_name

    assert once() == once()
