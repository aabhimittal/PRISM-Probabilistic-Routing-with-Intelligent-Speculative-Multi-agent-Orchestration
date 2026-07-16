"""Tests for prism.graph.TaskGraph / Edge."""

from __future__ import annotations

import random

import pytest

from prism.graph import Edge, TaskGraph
from prism.types import AgentOutput


def _out() -> AgentOutput:
    return AgentOutput(output="payload")


def test_linear_builds_chain_with_entry_and_terminal():
    g = TaskGraph.linear("a", "b", "c")
    assert g.entry == "a"
    assert g.is_terminal("c") is True
    assert g.is_terminal("a") is False
    assert g.is_terminal("b") is False
    # Next node along the chain.
    assert g.next_node("a", _out(), {}, random.Random(0)) == "b"
    assert g.next_node("b", _out(), {}, random.Random(0)) == "c"
    # Terminal node routes nowhere.
    assert g.next_node("c", _out(), {}, random.Random(0)) is None


def test_add_edge_picks_among_successors():
    g = TaskGraph()
    g.add_edge("a", "b").add_edge("a", "c")
    picks = {g.next_node("a", _out(), {}, random.Random(s)) for s in range(30)}
    assert picks <= {"b", "c"}
    # With equal weights both successors should be reachable across seeds.
    assert picks == {"b", "c"}


def test_predicate_false_edge_never_chosen():
    g = TaskGraph()
    g.add_edge("a", "blocked", predicate=lambda out, ctx: False)
    g.add_edge("a", "open", predicate=lambda out, ctx: True)
    for s in range(50):
        assert g.next_node("a", _out(), {}, random.Random(s)) == "open"


def test_predicate_can_read_context():
    g = TaskGraph()
    g.add_edge("a", "bug", predicate=lambda out, ctx: ctx.get("kind") == "bug")
    g.add_edge("a", "feature", predicate=lambda out, ctx: ctx.get("kind") == "feature")
    assert g.next_node("a", _out(), {"kind": "bug"}, random.Random(0)) == "bug"
    assert g.next_node("a", _out(), {"kind": "feature"}, random.Random(0)) == "feature"


def test_no_eligible_edges_returns_none():
    g = TaskGraph()
    g.add_edge("a", "x", predicate=lambda out, ctx: False)
    assert g.next_node("a", _out(), {}, random.Random(0)) is None


def test_is_terminal_for_unknown_and_marked_nodes():
    g = TaskGraph()
    g.add_edge("a", "b")
    assert g.is_terminal("b") is True   # no outgoing edges
    assert g.is_terminal("zzz") is True  # unknown node
    g.terminal("a")
    assert g.is_terminal("a") is True    # explicitly marked


def test_update_edge_moves_belief():
    g = TaskGraph()
    g.add_edge("a", "b")
    edge = g.successors("a")[0]
    assert edge.belief.mean == pytest.approx(0.5)
    g.update_edge("a", "b", reward=1.0)
    moved = g.successors("a")[0]
    assert moved.belief.mean > 0.5


def test_edge_eligible_default_true():
    e = Edge(src="a", dst="b")
    assert e.eligible(_out(), {}) is True
