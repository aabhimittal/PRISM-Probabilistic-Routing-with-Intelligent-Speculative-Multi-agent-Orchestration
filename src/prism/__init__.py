"""PRISM — Probabilistic Routing with Intelligent Speculative Multi-agent Orchestration.

Public API
----------
    from prism import Orchestrator, TaskGraph, SimulatedAgent, SimulatedScorer

See the ``examples/`` directory for runnable, no-API-key demos of every feature:
speculation, bandit convergence, uncertainty propagation, and causal tracing.
"""

from __future__ import annotations

from .agents import (
    Agent,
    CallableAgent,
    FlakyAgent,
    Scorer,
    SimulatedAgent,
    SimulatedScorer,
)
from .budget import ComputeBudget
from .drift import DriftEvent, DriftMonitor, PageHinkley
from .errors import (
    AllBranchesFailed,
    BudgetExhausted,
    ConfigurationError,
    PrismError,
    StageFailure,
)
from .graph import Edge, TaskGraph
from .hedging import HedgePolicy, LatencyTracker
from .orchestrator import Orchestrator
from .resilience import CircuitBreaker
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

__version__ = "0.2.0"

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
    "FlakyAgent",
    "Scorer",
    "SimulatedScorer",
    "CircuitBreaker",
    "DriftMonitor",
    "DriftEvent",
    "PageHinkley",
    "HedgePolicy",
    "LatencyTracker",
    "ComputeBudget",
    "PrismError",
    "ConfigurationError",
    "AllBranchesFailed",
    "StageFailure",
    "BudgetExhausted",
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
