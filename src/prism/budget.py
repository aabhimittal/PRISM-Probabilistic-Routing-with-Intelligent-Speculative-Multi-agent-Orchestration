"""Compute budgets — speculation that degrades gracefully under pressure.

Speculation is a luxury good: it converts spare compute into quality. In
production there is no such thing as spare compute — there is a budget. This
module makes the budget a first-class object the policy consults *before*
deciding to speculate, so behaviour degrades in a controlled, observable order:

    full speculation  →  narrower speculation  →  greedy only  →  BudgetExhausted

rather than the uncontrolled alternative (blow the budget early, then starve).

The ``reserve_fraction`` implements the "save something for the rest of the
pipeline" instinct: a stage may only spend *extra* (speculative) compute out of
the budget that remains after holding back a reserve for the stages still to
come. Greedy baseline cost is always allowed to draw on the reserve — finishing
the pipeline cheaply beats speculating early and failing to finish at all.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class ComputeBudget:
    """A mutable, thread-safe compute allowance shared across a run (or many).

    Units are the same arbitrary "cost units" agents declare (`agent.cost`) —
    map them to dollars, tokens, or GPU-seconds as you see fit.
    """

    total: float
    reserve_fraction: float = 0.15
    spent: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def remaining(self) -> float:
        return max(self.total - self.spent, 0.0)

    def spend(self, amount: float) -> None:
        with self._lock:
            self.spent += max(amount, 0.0)

    def can_afford_extra(self, extra: float) -> bool:
        """May a stage spend ``extra`` *speculative* compute? Only out of the
        unreserved slice — the reserve is for keeping future stages alive."""
        return extra <= self.remaining * (1.0 - self.reserve_fraction)

    def can_afford_base(self, base: float) -> bool:
        """May a stage spend its baseline (greedy) cost? Allowed to dip into
        the reserve — this is the 'finish the pipeline' allowance."""
        return base <= self.remaining

    def snapshot(self) -> dict[str, float]:
        return {"total": self.total, "spent": self.spent, "remaining": self.remaining}
