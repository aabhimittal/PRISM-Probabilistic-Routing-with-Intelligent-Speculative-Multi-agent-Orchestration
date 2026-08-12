"""PRISM exception hierarchy.

Industrial systems fail in specific ways, and callers need to catch them
specifically. Everything raised on purpose by PRISM derives from
:class:`PrismError`, so ``except PrismError`` is the catch-all; the subclasses
carry enough state to make an informed recovery decision (which stage, which
agents, the partial trace so far).
"""

from __future__ import annotations

from typing import Any, Optional


class PrismError(Exception):
    """Base class for all deliberate PRISM errors."""


class ConfigurationError(PrismError):
    """The orchestrator was mis-wired: unknown task type, a stage with no
    registered agents, a graph with no entry node, etc. Raised eagerly with a
    message naming the exact gap, because silent misconfiguration is the most
    expensive industrial failure mode of all."""


class AllBranchesFailed(PrismError):
    """Every branch at a stage raised or timed out.

    Carries the stage name and the per-branch outcomes (with their ``failed``
    flags and, where available, the exception repr) so a supervisor can decide
    whether to retry, re-route, or surface the failure."""

    def __init__(self, task_type: str, outcomes: list[Any]) -> None:
        self.task_type = task_type
        self.outcomes = outcomes
        names = [getattr(o, "agent_name", "?") for o in outcomes]
        super().__init__(
            f"all {len(outcomes)} branch(es) failed at stage {task_type!r}: {names}"
        )


class StageFailure(PrismError):
    """A stage could not produce a committed output even after failover.

    ``trace`` holds the partial :class:`~prism.trace.CausalTrace` accumulated
    before the failure — provenance matters *most* when things go wrong, so the
    trace is never thrown away."""

    def __init__(self, task_type: str, message: str, trace: Optional[Any] = None) -> None:
        self.task_type = task_type
        self.trace = trace
        super().__init__(f"stage {task_type!r} failed: {message}")


class BudgetExhausted(PrismError):
    """The compute budget cannot cover even a single greedy branch for the next
    stage. The partial trace is attached, mirroring :class:`StageFailure`."""

    def __init__(self, task_type: str, remaining: float, needed: float,
                 trace: Optional[Any] = None) -> None:
        self.task_type = task_type
        self.remaining = remaining
        self.needed = needed
        self.trace = trace
        super().__init__(
            f"budget exhausted before stage {task_type!r}: "
            f"remaining={remaining:.2f}, cheapest branch needs {needed:.2f}"
        )
