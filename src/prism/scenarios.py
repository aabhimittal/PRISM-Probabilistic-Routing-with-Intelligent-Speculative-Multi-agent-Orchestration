"""Ready-made demo scenarios.

A realistic 4-stage "research assistant" pipeline used by the CLI, the examples,
and the tests so everything tells one coherent story:

    triage  ->  retrieve  ->  synthesize  ->  critique

Each stage has a roster of agents with *different hidden competence profiles*:
specialists that are excellent at one stage, generalists that are okay
everywhere, and a couple of deliberately ambiguous near-ties so speculation has
a reason to fire. The point is that the *best* agent per stage is not obvious a
priori — PRISM has to discover it while occasionally hedging with speculation.
"""

from __future__ import annotations

import random
from typing import Optional

from .agents import SimulatedAgent
from .graph import TaskGraph
from .orchestrator import Orchestrator

STAGES = ["triage", "retrieve", "synthesize", "critique"]


def _agents() -> dict[str, list[SimulatedAgent]]:
    """Build the agent roster. Competence is (mean_quality, spread) per stage.

    Note the engineered tensions:
      * On `retrieve`, `dense-retriever` (0.82) and `hybrid-retriever` (0.80) are
        a near-tie — the canonical case where committing greedily is a coin flip
        and speculation earns its keep.
      * `frontier-llm` is a strong generalist (good everywhere, expensive), so
        the cost model must decide when its quality edge justifies its price.
      * `cheap-llm` is fast and cheap but mediocre — a tempting greedy pick early
        before the bandit has learned better.
    """
    return {
        "triage": [
            SimulatedAgent("router-clf", {"triage": (0.86, 0.10)}, cost=0.5),
            SimulatedAgent("cheap-llm", {"triage": (0.70, 0.16)}, cost=0.5),
            SimulatedAgent("frontier-llm", {"triage": (0.80, 0.12)}, cost=3.0),
        ],
        "retrieve": [
            SimulatedAgent("dense-retriever", {"retrieve": (0.82, 0.13)}, cost=1.0),
            SimulatedAgent("hybrid-retriever", {"retrieve": (0.80, 0.13)}, cost=1.2),
            SimulatedAgent("bm25", {"retrieve": (0.63, 0.15)}, cost=0.3),
        ],
        "synthesize": [
            SimulatedAgent("frontier-llm", {"synthesize": (0.85, 0.12)}, cost=3.0),
            SimulatedAgent("cheap-llm", {"synthesize": (0.66, 0.17)}, cost=0.5),
            SimulatedAgent("mid-llm", {"synthesize": (0.78, 0.14)}, cost=1.5),
        ],
        "critique": [
            SimulatedAgent("critic-a", {"critique": (0.79, 0.13)}, cost=1.0),
            SimulatedAgent("critic-b", {"critique": (0.79, 0.13)}, cost=1.0),
            SimulatedAgent("frontier-llm", {"critique": (0.83, 0.11)}, cost=3.0),
        ],
    }


def research_pipeline(seed: int = 0,
                      policy=None,
                      reseed_agents: bool = True) -> Orchestrator:
    """Return an :class:`Orchestrator` wired for the research pipeline."""
    graph = TaskGraph.linear(*STAGES)
    roster = _agents()
    if reseed_agents:
        # Give each agent a deterministic stream tied to the run seed so demos
        # reproduce exactly but different seeds explore different luck.
        for stage_agents in roster.values():
            for a in stage_agents:
                a.seed(seed)
    return Orchestrator.build(
        graph=graph,
        agents_by_stage=roster,  # type: ignore[arg-type]
        seed=seed,
        policy=policy,
    )


def ground_truth_best() -> dict[str, str]:
    """The agent with the highest *true* mean quality per stage. Benchmarks
    measure how often PRISM's learned routing agrees with this oracle."""
    roster = _agents()
    best = {}
    for stage, agents in roster.items():
        best[stage] = max(agents, key=lambda a: a.competence[stage][0]).name
    return best
