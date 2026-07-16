"""Core data types shared across PRISM.

These are deliberately plain dataclasses with no behaviour — they are the
vocabulary the rest of the system speaks in. Keeping them dependency-free and
serializable is what makes the causal trace (see :mod:`prism.trace`) able to
snapshot an entire run as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


@dataclass
class Task:
    """A unit of work flowing through the orchestrator.

    Attributes
    ----------
    task_type:
        The node key in the :class:`~prism.graph.TaskGraph`. Determines which
        set of candidate agents (bandit arms) are eligible.
    payload:
        Arbitrary input handed to the agent. For the built-in simulated agents
        this is just a string prompt; with a real backend it could be messages,
        tool state, retrieved context, anything.
    stakes:
        How much this task *matters*, in ``[0, 1]``. Speculation spends extra
        compute; PRISM only pays that price when the expected quality gain,
        scaled by ``stakes``, beats the cost. A cheap logging step has low
        stakes; the final user-facing answer has high stakes.
    context:
        Free-form carry-along state threaded between pipeline stages.
    """

    task_type: str
    payload: Any
    stakes: float = 0.5
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """What an agent returns.

    ``latent_quality`` is a simulation-only channel: the built-in mock agents
    report the *true* (hidden) quality of the output they just produced so the
    benchmarks can measure regret against ground truth. Real agents leave it
    ``None`` — PRISM never reads it in production paths; only the simulated
    scorer peeks at it, standing in for a real verifier/judge.
    """

    output: Any
    self_report: float = 0.5  # the agent's own confidence, if it offers one
    latent_quality: Optional[float] = None
    cost: float = 1.0
    latency: float = 0.0


class RouteMode(str, Enum):
    """Was this stage routed greedily (commit to one agent) or speculatively
    (run several branches and keep the winner)?"""

    GREEDY = "greedy"
    SPECULATIVE = "speculative"


@dataclass
class BranchOutcome:
    """The result of running one speculative branch (one agent) at a stage."""

    agent_name: str
    output: AgentOutput
    score: float              # calibrated quality score in [0, 1]
    prior_mean: float         # belief mean *before* this run (for the trace)
    posterior_mean: float     # belief mean *after* incorporating the score
    is_winner: bool = False
    squashed: bool = False    # True for speculative losers (CPU: branch squash)
