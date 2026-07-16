"""PRISM — Probabilistic Routing with Intelligent Speculative Multi-agent Orchestration.

Public API
----------
    from prism import Orchestrator, TaskGraph, SimulatedAgent, SimulatedScorer

See the ``examples/`` directory for runnable, no-API-key demos of every feature:
speculation, bandit convergence, uncertainty propagation, and causal tracing.
"""

from __future__ import annotations

from .agents import Agent, CallableAgent, Scorer, SimulatedAgent, SimulatedScorer
from .graph import Edge, TaskGraph
from .orchestrator import Orchestrator
from .routing import BanditRouter
from .scoring import CalibratedScore, calibrate, select_winner
from .speculation import (
    SpeculationDecision,
    SpeculationPolicy,
    SpeculativeExecutor,
    StageResult,
)
from .trace import CausalTrace, RoutingRationale, StageTrace
from .types import AgentOutput, BranchOutcome, RouteMode, Task
from .uncertainty import (
    Belief,
    from_mean_std,
    probability_best,
    propagate,
    selection_entropy,
)

__version__ = "0.1.0"

__all__ = [
    "Orchestrator",
    "TaskGraph",
    "Edge",
    "BanditRouter",
    "SpeculationPolicy",
    "SpeculativeExecutor",
    "SpeculationDecision",
    "StageResult",
    "Agent",
    "SimulatedAgent",
    "CallableAgent",
    "Scorer",
    "SimulatedScorer",
    "Belief",
    "from_mean_std",
    "propagate",
    "probability_best",
    "selection_entropy",
    "CausalTrace",
    "StageTrace",
    "RoutingRationale",
    "Task",
    "AgentOutput",
    "BranchOutcome",
    "RouteMode",
    "CalibratedScore",
    "calibrate",
    "select_winner",
    "__version__",
]
