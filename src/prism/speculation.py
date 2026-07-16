"""Speculative multi-branch execution — PRISM's headline mechanism.

The analogy, made literal
-------------------------
A modern CPU hits a conditional branch whose outcome isn't resolved yet. Rather
than stall, it *speculatively* executes down the predicted path (often *both*
paths), and once the condition resolves it **commits** the correct path and
**squashes** the wrong one, throwing that work away.

PRISM applies the same idea to agent routing. When the router is genuinely
unsure which agent is best for a task, committing to one is a gamble. So PRISM
speculatively runs several candidate agents **in parallel**, scores their
outputs, **commits** the winner, and **squashes** the losers.

Two things make this more than a gimmick:

1. **We only speculate when it pays.** Running k agents costs ~k× the compute.
   :class:`SpeculationPolicy` computes the *value of information* — the expected
   quality we'd recover by not committing blindly — scales it by the task's
   stakes, and speculates only when that beats the extra compute cost. This is
   an explicit economic decision, recorded in the trace.

2. **Squashed work still teaches us.** A CPU's squashed branch is pure waste.
   PRISM feeds every branch's score — winners *and* losers — back into the
   bandit. So even a "wasted" speculative branch sharpens the routing policy for
   next time. Speculation is simultaneously a hedge *and* an exploration engine.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from .agents import Agent, Scorer
from .scoring import calibrate, select_winner
from .types import AgentOutput, BranchOutcome, RouteMode, Task
from .uncertainty import Belief, probability_best, selection_entropy


# ---------------------------------------------------------------------------
# The decision: greedy or speculative, and over which branches?
# ---------------------------------------------------------------------------

@dataclass
class SpeculationDecision:
    mode: RouteMode
    branches: list[str]
    p_best: dict[str, float]
    entropy: float
    greedy: str
    expected_regret_greedy: float
    value_of_speculation: float
    cost: float
    explanation: str


@dataclass
class SpeculationPolicy:
    """Decides whether to speculate, and over how many branches.

    Parameters
    ----------
    max_branches:
        Hard cap on parallel branches (the speculation "width").
    viable_floor:
        An arm must have at least this P(best) to be worth including as a
        branch. Filters out no-hopers so we don't burn compute on them.
    cost_weight:
        Price of one unit of compute, expressed in the same units as quality
        (which lives in ``[0, 1]``). Higher -> more parsimonious, speculates
        less. This is the single knob that tunes PRISM's aggressiveness.
    scorer_efficiency:
        How reliably the scorer actually picks the best branch (0..1). We haircut
        the theoretical value of information by this, because a bad judge can't
        cash in the value of running extra branches.
    samples:
        Monte-Carlo sample count for the regret/`p_best` estimates.
    """

    max_branches: int = 3
    viable_floor: float = 0.06
    # Quality-units we are willing to trade per unit of extra compute. Quality
    # lives in [0, 1], so realistic values are small: 0.03 means "an extra agent
    # call is worth it if it buys >0.03 expected quality (scaled by stakes)."
    cost_weight: float = 0.03
    scorer_efficiency: float = 0.85
    samples: int = 512
    rng: random.Random = field(default_factory=lambda: random.Random(7))

    def _thompson_pick(self, beliefs: dict[str, Belief]) -> str:
        """One Thompson draw. Used to commit the single arm when we DON'T
        speculate — so the 'cheap' path still explores. Without this, the router
        would exploit its current favourite forever and never gather evidence on
        the others (the classic bandit cold-start trap)."""
        best, best_val = None, -1.0
        for name, b in beliefs.items():
            v = b.sample(self.rng)
            if v > best_val:
                best, best_val = name, v
        assert best is not None
        return best

    def decide(self, beliefs: dict[str, Belief], costs: dict[str, float],
               stakes: float) -> SpeculationDecision:
        names = list(beliefs)
        greedy = max(names, key=lambda n: beliefs[n].mean)  # VOI reference arm

        # Only one option -> nothing to speculate over.
        if len(names) <= 1 or self.max_branches <= 1:
            committed = self._thompson_pick(beliefs)
            return SpeculationDecision(
                mode=RouteMode.GREEDY, branches=[committed], p_best={committed: 1.0},
                entropy=0.0, greedy=greedy, expected_regret_greedy=0.0,
                value_of_speculation=0.0, cost=costs.get(committed, 1.0),
                explanation=f"single-arm route via Thompson sampling → {committed!r}"
                            + ("" if self.max_branches > 1 else " (speculation disabled)"),
            )

        p_best = probability_best(beliefs, self.rng, samples=self.samples)
        entropy = selection_entropy(p_best)

        # Candidate branch set: viable contenders, best-first, greedy always in.
        contenders = sorted(names, key=lambda n: p_best[n], reverse=True)
        branches = [n for n in contenders if p_best[n] >= self.viable_floor]
        if greedy not in branches:
            branches.append(greedy)
        branches = branches[: self.max_branches]

        # --- value of information ------------------------------------------
        # One joint Monte-Carlo pass estimates two regrets against ground-truth
        # "best arm this sample":
        #   R_greedy = E[ max_i θ_i - θ_greedy ]   (cost of committing greedily)
        #   R_spec   = E[ max_i θ_i - max_{b∈B} θ_b ]  (residual if we run B and
        #              an ideal scorer keeps the best branch)
        # VOI (quality recovered) = R_greedy - R_spec = E[ max_{b∈B} θ - θ_greedy ]
        r_greedy, voi = self._regret_and_voi(beliefs, greedy, branches)
        value = stakes * voi * self.scorer_efficiency

        # Extra compute beyond the one agent we'd have run anyway.
        base_cost = costs.get(greedy, 1.0)
        spec_cost = sum(costs.get(b, 1.0) for b in branches)
        extra_cost = (spec_cost - base_cost) * self.cost_weight

        speculate = len(branches) > 1 and value > extra_cost
        if speculate:
            mode = RouteMode.SPECULATIVE
            expl = (
                f"speculating over {len(branches)} branches: "
                f"value_of_info={value:.3f} > extra_cost={extra_cost:.3f} "
                f"(greedy P(best)={p_best[greedy]:.2f}, regret if committed="
                f"{r_greedy:.3f}, stakes={stakes:.2f})"
            )
        else:
            mode = RouteMode.GREEDY
            # Not worth speculating — but still route the single arm by Thompson
            # sampling so under-explored arms keep getting occasional evidence.
            committed = self._thompson_pick(beliefs)
            branches = [committed]
            explore_note = "" if committed == greedy else " [exploring]"
            expl = (
                f"single-arm route → {committed!r}{explore_note}: "
                f"value_of_info={value:.3f} ≤ extra_cost={extra_cost:.3f} "
                f"— not worth the compute (greedy P(best)={p_best[greedy]:.2f})"
            )

        return SpeculationDecision(
            mode=mode, branches=branches, p_best=p_best, entropy=entropy,
            greedy=greedy, expected_regret_greedy=r_greedy,
            value_of_speculation=value,
            cost=spec_cost if speculate else costs.get(branches[0], 1.0),
            explanation=expl,
        )

    def _regret_and_voi(self, beliefs: dict[str, Belief], greedy: str,
                        branches: list[str]) -> tuple[float, float]:
        names = list(beliefs)
        bset = set(branches)
        r_greedy_acc = 0.0
        voi_acc = 0.0
        for _ in range(self.samples):
            draws = {n: beliefs[n].sample(self.rng) for n in names}
            best_all = max(draws.values())
            g = draws[greedy]
            best_b = max(draws[n] for n in names if n in bset)
            r_greedy_acc += best_all - g
            voi_acc += best_b - g  # = R_greedy - R_spec for this sample
        return r_greedy_acc / self.samples, voi_acc / self.samples


# ---------------------------------------------------------------------------
# The mechanism: run branches in parallel, commit the winner, squash the rest.
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    winner: BranchOutcome
    outcomes: list[BranchOutcome]
    latency: float           # wall-clock = slowest branch (they run in parallel)
    cost: float              # compute = sum over branches


class SpeculativeExecutor:
    """Runs the chosen branch set concurrently and selects the winner.

    ``rollback`` semantics: losing branches are marked ``squashed`` and their
    outputs never flow downstream — the pipeline commits only the winner's
    output. If your agents have *external* side effects (writes, emails, tool
    calls), register a compensating action via ``on_squash`` and PRISM will
    invoke it for every squashed branch, mirroring a CPU discarding a
    mis-speculated store buffer.
    """

    def __init__(self, scorer: Scorer, max_workers: int = 8) -> None:
        self.scorer = scorer
        self.max_workers = max_workers

    def run_stage(
        self,
        task: Task,
        branch_agents: list[Agent],
        priors: dict[str, Belief],
        on_squash=None,
    ) -> StageResult:
        # 1. Fan out: execute every branch agent in parallel. In real use these
        #    are I/O-bound LLM calls, so threads give true wall-clock overlap.
        outputs: dict[str, AgentOutput] = {}
        if len(branch_agents) == 1:
            a = branch_agents[0]
            outputs[a.name] = a.run(task)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(branch_agents))) as ex:
                futures = {ex.submit(a.run, task): a.name for a in branch_agents}
                for fut in futures:
                    outputs[futures[fut]] = fut.result()

        # 2. Score + calibrate each branch against its prior (empirical Bayes).
        scored = []
        raw_scores: dict[str, float] = {}
        for a in branch_agents:
            out = outputs[a.name]
            raw = self.scorer.score(task, out)
            raw_scores[a.name] = raw
            scored.append(calibrate(a.name, raw, priors[a.name],
                                    getattr(self.scorer, "reliability", 0.85)))

        # 3. Commit the calibrated winner; squash the losers.
        best = select_winner(scored)
        outcomes: list[BranchOutcome] = []
        latency = 0.0
        cost = 0.0
        for a in branch_agents:
            out = outputs[a.name]
            latency = max(latency, out.latency)
            cost += out.cost
            is_win = a.name == best.agent_name
            cs = next(s for s in scored if s.agent_name == a.name)
            outcome = BranchOutcome(
                agent_name=a.name,
                output=out,
                score=raw_scores[a.name],
                prior_mean=cs.prior_mean,
                posterior_mean=cs.posterior_mean,
                is_winner=is_win,
                squashed=not is_win and len(branch_agents) > 1,
            )
            outcomes.append(outcome)
            if outcome.squashed and on_squash is not None:
                on_squash(a, out)  # compensate external side effects

        winner = next(o for o in outcomes if o.is_winner)
        return StageResult(winner=winner, outcomes=outcomes,
                           latency=latency, cost=cost)
