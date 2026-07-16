"""Causal tracing: every PRISM output carries its full decision provenance.

Most orchestration frameworks give you a log line. PRISM gives you a *causal
trace* — a structured, serializable record that answers, for every stage:

  * which agents were even considered, and what did we believe about each of
    them *at that instant* (mean quality + uncertainty)?
  * did we speculate? if so, *why* — what was the estimated value of information
    versus the compute cost?
  * which branches actually ran, what did each score, and why did the winner
    win over the runners-up?
  * how did the uncertainty of the answer evolve, stage by stage, and what is
    the compound confidence of the final output?

Because the whole thing is dataclasses -> ``dict`` -> JSON, you can diff two
runs, feed a trace to an LLM to explain itself, or replay a decision offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .types import BranchOutcome, RouteMode
from .uncertainty import Belief


@dataclass
class RoutingRationale:
    """The *why* behind a single routing decision — the crux of explainability.

    This is populated by the speculation policy and is intentionally verbose:
    it captures the exact numbers that tipped the greedy-vs-speculate scale so a
    human (or an auditing agent) can second-guess the orchestrator after the
    fact."""

    mode: RouteMode
    p_best: dict[str, float]          # P(arm is best) per candidate
    selection_entropy: float          # how torn the router was (nats)
    greedy_choice: str                # who greedy routing would have picked
    branches: list[str]               # who actually ran
    expected_regret_greedy: float     # E[best - greedy] under current beliefs
    value_of_speculation: float       # stakes-scaled expected quality recovered
    speculation_cost: float           # extra compute paid for the branches
    explanation: str                  # human-readable one-liner


@dataclass
class StageTrace:
    """Everything that happened at one node of the task graph."""

    task_type: str
    rationale: RoutingRationale
    outcomes: list[BranchOutcome]
    winner: str
    stage_belief_before: Belief       # winner's belief pre-update
    stage_belief_after: Belief        # winner's belief post-update
    propagated_belief: Belief         # compound belief up to and incl. this stage
    latency: float                    # wall-clock for the stage (max of branches)
    cost: float                       # total compute spent (sum over branches)
    realized_quality: float = 0.0     # winner's TRUE quality (simulation only)


@dataclass
class CausalTrace:
    """The complete provenance of one end-to-end orchestration run."""

    task_type: str
    stages: list[StageTrace] = field(default_factory=list)
    final_output: Any = None
    final_belief: Optional[Belief] = None  # compound epistemic uncertainty
    total_cost: float = 0.0
    total_latency: float = 0.0
    speculations: int = 0

    @property
    def mean_realized_quality(self) -> float:
        """Mean TRUE quality of the committed winners across stages (simulation
        only — this is the ground-truth measure benchmarks use to show that
        speculation produces genuinely better outputs, not just higher scores)."""
        vals = [s.realized_quality for s in self.stages if s.realized_quality > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def add(self, stage: StageTrace) -> None:
        self.stages.append(stage)
        self.total_cost += stage.cost
        self.total_latency += stage.latency
        if stage.rationale.mode is RouteMode.SPECULATIVE:
            self.speculations += 1
        self.final_belief = stage.propagated_belief

    # --- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    # --- human-friendly rendering ------------------------------------------

    def explain(self) -> str:
        """Render the trace as an annotated, indented decision tree — the thing
        you actually paste into a PR description or an incident review."""
        lines: list[str] = []
        conf = self.final_belief
        lines.append(f"CAUSAL TRACE  ·  pipeline={self.task_type!r}")
        lines.append(
            f"  final confidence: mean={conf.mean:.3f} ± {conf.std:.3f} "
            f"(n={conf.evidence:.1f})" if conf else "  final confidence: n/a"
        )
        lines.append(
            f"  cost={self.total_cost:.1f} units · latency={self.total_latency:.3f}s "
            f"· speculations={self.speculations}/{len(self.stages)}"
        )
        for i, st in enumerate(self.stages):
            r = st.rationale
            tag = "⚡SPEC" if r.mode is RouteMode.SPECULATIVE else "→ greedy"
            lines.append(f"  ├─ stage {i} [{st.task_type}]  {tag}")
            lines.append(f"  │    {r.explanation}")
            lines.append(
                f"  │    p_best={_fmt_probs(r.p_best)}  H={r.selection_entropy:.2f} nats"
            )
            for oc in st.outcomes:
                mark = "★ winner" if oc.is_winner else " squashed" if oc.squashed else ""
                lines.append(
                    f"  │      • {oc.agent_name:<14} score={oc.score:.3f} "
                    f"({oc.prior_mean:.2f}→{oc.posterior_mean:.2f}) {mark}"
                )
            lines.append(
                f"  │    propagated confidence → mean={st.propagated_belief.mean:.3f} "
                f"± {st.propagated_belief.std:.3f}"
            )
        lines.append(f"  └─ output: {_truncate(self.final_output)}")
        return "\n".join(lines)


def _fmt_probs(p: dict[str, float]) -> str:
    return "{" + ", ".join(f"{k}:{v:.2f}" for k, v in p.items()) + "}"


def _truncate(x: Any, n: int = 80) -> str:
    s = str(x)
    return s if len(s) <= n else s[: n - 1] + "…"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Belief):
        return {"alpha": obj.alpha, "beta": obj.beta,
                "mean": obj.mean, "std": obj.std}
    raise TypeError(f"not JSON serializable: {type(obj)}")
