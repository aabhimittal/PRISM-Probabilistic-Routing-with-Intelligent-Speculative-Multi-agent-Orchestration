"""Tests for prism.routing.BanditRouter."""

from __future__ import annotations

import random

import pytest

from prism.agents import SimulatedAgent
from prism.routing import BanditRouter
from prism.uncertainty import Belief


def _router(seed: int = 0) -> BanditRouter:
    return BanditRouter(rng=random.Random(seed))


def test_register_candidates_and_beliefs():
    r = _router()
    good = SimulatedAgent("good")
    bad = SimulatedAgent("bad")
    r.register("triage", good)
    r.register("triage", bad, prior=Belief(2.0, 8.0))

    cands = r.candidates("triage")
    assert set(cands) == {"good", "bad"}
    assert cands["good"] is good

    beliefs = r.beliefs("triage")
    assert set(beliefs) == {"good", "bad"}
    # Default prior for 'good', explicit prior for 'bad'.
    assert beliefs["good"].mean == pytest.approx(0.5)
    assert beliefs["bad"].mean == pytest.approx(0.2)
    # beliefs()/candidates() return copies, not internal dicts.
    beliefs.pop("good")
    assert "good" in r.beliefs("triage")


def test_update_returns_before_after_and_mutates_store():
    r = _router()
    r.register("s", SimulatedAgent("a"))
    before, after = r.update("s", "a", reward=1.0)
    assert before.mean == pytest.approx(0.5)
    assert after.mean > before.mean
    # The stored belief is now the 'after' belief.
    assert r.belief("s", "a").alpha == after.alpha
    assert r.belief("s", "a").beta == after.beta


def test_thompson_and_greedy_return_registered_names():
    r = _router(3)
    for name in ["a", "b", "c"]:
        r.register("s", SimulatedAgent(name))
    registered = set(r.candidates("s"))
    for _ in range(50):
        assert r.thompson_select("s") in registered
    assert r.greedy_select("s") in registered


def test_greedy_converges_to_high_reward_arm():
    r = _router(7)
    r.register("s", SimulatedAgent("winner"))
    r.register("s", SimulatedAgent("loser1"))
    r.register("s", SimulatedAgent("loser2"))
    for _ in range(100):
        r.update("s", "winner", reward=1.0)
        r.update("s", "loser1", reward=0.05)
        r.update("s", "loser2", reward=0.05)
    assert r.greedy_select("s") == "winner"
    beliefs = r.beliefs("s")
    assert beliefs["winner"].mean > 0.9


def test_p_best_sums_to_one():
    r = _router(1)
    for name in ["a", "b", "c"]:
        r.register("s", SimulatedAgent(name))
    p = r.p_best("s", samples=1000)
    assert set(p) == {"a", "b", "c"}
    assert sum(p.values()) == pytest.approx(1.0)


def test_snapshot_shape():
    r = _router()
    r.register("s", SimulatedAgent("a"))
    snap = r.snapshot()
    mean, std = snap["s"]["a"]
    assert mean == pytest.approx(0.5)
    assert std > 0.0
