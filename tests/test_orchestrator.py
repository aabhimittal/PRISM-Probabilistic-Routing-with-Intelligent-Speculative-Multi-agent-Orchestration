"""End-to-end tests for prism.orchestrator via the research pipeline scenario."""

from __future__ import annotations

import json
import random

import pytest

from prism.scenarios import STAGES, ground_truth_best, research_pipeline
from prism.speculation import SpeculationPolicy
from prism.trace import CausalTrace
from prism.uncertainty import Belief


def test_single_run_produces_well_formed_trace():
    orch = research_pipeline(seed=0)
    trace = orch.run("what is PRISM?", stakes=0.7)

    assert isinstance(trace, CausalTrace)
    # One StageTrace per pipeline stage.
    assert len(trace.stages) == len(STAGES) == 4
    assert [s.task_type for s in trace.stages] == STAGES

    assert trace.final_output is not None
    assert isinstance(trace.final_belief, Belief)
    assert 0.0 < trace.final_belief.mean < 1.0
    assert trace.total_cost > 0.0

    # Exactly one committed winner per stage.
    for st in trace.stages:
        winners = [o for o in st.outcomes if o.is_winner]
        assert len(winners) == 1
        assert st.winner == winners[0].agent_name


def test_router_converges_to_ground_truth_on_most_stages():
    orch = research_pipeline(seed=0)
    for _ in range(250):
        orch.run("query", stakes=0.7)

    snap = orch.router.snapshot()
    truth = ground_truth_best()
    hits = 0
    for stage in STAGES:
        arms = snap[stage]
        learned_best = max(arms, key=lambda n: arms[n][0])  # highest mean
        if learned_best == truth[stage]:
            hits += 1
    # Robust convergence threshold: agree with the oracle on >= 3 of 4 stages.
    assert hits >= 3


def test_trace_serialization_and_explain_nonempty():
    orch = research_pipeline(seed=1)
    trace = orch.run("query", stakes=0.8)

    js = trace.to_json()
    assert isinstance(js, str) and len(js) > 0
    parsed = json.loads(js)  # must be valid JSON
    assert parsed["task_type"] == "triage"

    text = trace.explain()
    assert isinstance(text, str) and len(text) > 0
    assert "CAUSAL TRACE" in text


def test_greedy_only_policy_yields_zero_speculations():
    policy = SpeculationPolicy(max_branches=1, rng=random.Random(9))
    orch = research_pipeline(seed=0, policy=policy)
    total_specs = 0
    for _ in range(20):
        trace = orch.run("query", stakes=1.0)  # high stakes, but speculation off
        total_specs += trace.speculations
    assert total_specs == 0


def test_speculation_fires_at_least_once_over_a_run():
    # With the default policy and the engineered near-ties, PRISM should
    # speculate on at least one stage across a handful of tasks.
    orch = research_pipeline(seed=0)
    specs = sum(orch.run("query", stakes=1.0).speculations for _ in range(25))
    assert specs > 0


def test_determinism_same_seed_same_outcome():
    a = research_pipeline(seed=3).run("q", stakes=0.7)
    b = research_pipeline(seed=3).run("q", stakes=0.7)
    assert [s.winner for s in a.stages] == [s.winner for s in b.stages]
    assert a.final_belief.mean == pytest.approx(b.final_belief.mean)
