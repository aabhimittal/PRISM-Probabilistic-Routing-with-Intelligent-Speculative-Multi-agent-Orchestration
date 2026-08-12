"""Probabilistic task graphs.

A conventional pipeline is a DAG: node A always flows to node B. PRISM's task
graph is a **weighted, uncertain** graph — an edge from A to B carries a
*belief* about how likely that transition is the right one, and edges can be
**conditional** on the content produced so far. Traversal is therefore a routing
problem in its own right, not a fixed schedule.

Why bother? Real agent workflows branch on content: a "triage" stage might send
a bug report to a *debugging* sub-pipeline but a feature request to a *design*
one — and it's often genuinely unsure which. Modelling transitions as beliefs
lets PRISM (a) learn which routes tend to produce good end-to-end outcomes and
(b) apply the very same speculative machinery at the *graph* level that it uses
at the *agent* level.

This module keeps the structure minimal and composable: nodes are task types,
edges optionally carry a predicate and a prior weight. The orchestrator walks it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .types import AgentOutput, Task
from .uncertainty import Belief


# A predicate decides whether an edge is eligible given the produced output and
# the running task context. Returning False removes the edge from consideration.
EdgePredicate = Callable[[AgentOutput, dict], bool]


@dataclass
class Edge:
    src: str
    dst: str
    weight: float = 1.0                      # prior propensity to take this edge
    predicate: Optional[EdgePredicate] = None
    belief: Belief = field(default_factory=lambda: Belief(1.0, 1.0))

    def eligible(self, output: AgentOutput, context: dict) -> bool:
        return self.predicate is None or self.predicate(output, context)


@dataclass
class TaskGraph:
    """A weighted, conditional graph over task types.

    Nodes are implicit (any task type referenced by an edge, plus registered
    terminals). Use :meth:`add_edge` to wire stages, :meth:`terminal` to mark
    end nodes, and :meth:`next_node` to route a completed stage to its successor.
    """

    _edges: dict[str, list[Edge]] = field(default_factory=dict)
    _terminals: set[str] = field(default_factory=set)
    entry: Optional[str] = None

    def add_edge(self, src: str, dst: str, weight: float = 1.0,
                 predicate: Optional[EdgePredicate] = None) -> "TaskGraph":
        if self.entry is None:
            self.entry = src
        self._edges.setdefault(src, []).append(
            Edge(src=src, dst=dst, weight=weight, predicate=predicate)
        )
        return self

    def terminal(self, node: str) -> "TaskGraph":
        self._terminals.add(node)
        return self

    def is_terminal(self, node: str) -> bool:
        return node in self._terminals or node not in self._edges

    def successors(self, node: str) -> list[Edge]:
        return list(self._edges.get(node, []))

    def next_node(self, node: str, output: AgentOutput, context: dict,
                  rng: random.Random) -> Optional[str]:
        """Choose the successor of ``node`` after producing ``output``.

        Eligible edges (predicate passes) are weighted by ``weight × belief.mean``
        and one is sampled. Sampling rather than argmax keeps the graph
        *probabilistic*: low-probability routes still get occasional exploration,
        and their edge beliefs get a chance to be learned. Returns ``None`` at a
        terminal node."""
        if self.is_terminal(node):
            return None
        eligible = [e for e in self._edges.get(node, []) if e.eligible(output, context)]
        if not eligible:
            return None
        if len(eligible) == 1:
            return eligible[0].dst
        weights = [max(e.weight * e.belief.mean, 1e-9) for e in eligible]
        total = sum(weights)
        r = rng.random() * total
        upto = 0.0
        for e, w in zip(eligible, weights):
            upto += w
            if r <= upto:
                return e.dst
        return eligible[-1].dst

    def update_edge(self, src: str, dst: str, reward: float) -> None:
        """Reinforce or weaken a transition based on downstream outcome quality
        — the graph learns which routes tend to pay off, just like the arms do."""
        for e in self._edges.get(src, []):
            if e.dst == dst:
                e.belief = e.belief.updated(reward)
                return

    # --- persistence --------------------------------------------------------

    def state_dict(self) -> list[dict]:
        """The learned edge beliefs as JSON-able rows. Predicates and weights
        are code/config, not learned state — they are not serialized."""
        return [
            {"src": e.src, "dst": e.dst,
             "alpha": e.belief.alpha, "beta": e.belief.beta}
            for edges in self._edges.values() for e in edges
        ]

    def load_state_dict(self, rows: list[dict]) -> int:
        """Restore edge beliefs saved by :meth:`state_dict`; unknown edges are
        skipped (roster drift tolerated). Returns edges restored."""
        loaded = 0
        for row in rows:
            for e in self._edges.get(row["src"], []):
                if e.dst == row["dst"]:
                    e.belief = Belief(float(row["alpha"]), float(row["beta"]))
                    loaded += 1
                    break
        return loaded

    @staticmethod
    def linear(*task_types: str) -> "TaskGraph":
        """Convenience: build a straight-line pipeline stage0 -> stage1 -> ...

        The common case. You still get speculation at each stage; you just don't
        need conditional branching between stages."""
        g = TaskGraph()
        for a, b in zip(task_types, task_types[1:]):
            g.add_edge(a, b)
        g.terminal(task_types[-1])
        g.entry = task_types[0]
        return g
