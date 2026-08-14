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
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .agents import Agent, Scorer
from .errors import AllBranchesFailed
from .scoring import calibrate, select_winner
from .types import AgentOutput, BranchOutcome, RouteMode, Task
from .uncertainty import Belief, probability_best, selection_entropy

if TYPE_CHECKING:
    from .budget import ComputeBudget


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
               stakes: float,
               budget: "Optional[ComputeBudget]" = None) -> SpeculationDecision:
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

        # --- budget governor -----------------------------------------------
        # Speculation may only spend *extra* compute the budget can spare.
        # Degrade in order: full width → narrower → greedy. This is the
        # graceful-degradation ladder, and every rung is recorded.
        budget_note = ""
        if speculate and budget is not None:
            def _raw_extra() -> float:
                return sum(costs.get(b, 1.0) for b in branches) - base_cost

            while len(branches) > 2 and not budget.can_afford_extra(_raw_extra()):
                # Drop the least-likely branch that isn't the greedy anchor.
                for i in range(len(branches) - 1, -1, -1):
                    if branches[i] != greedy:
                        del branches[i]
                        break
            if not budget.can_afford_extra(_raw_extra()):
                speculate = False
                budget_note = (f" [budget: speculation suppressed — "
                               f"remaining={budget.remaining:.1f}]")
            elif len(branches) < self.max_branches:
                budget_note = f" [budget: narrowed to {len(branches)} branches]"
            spec_cost = sum(costs.get(b, 1.0) for b in branches)

        if speculate:
            mode = RouteMode.SPECULATIVE
            expl = (
                f"speculating over {len(branches)} branches: "
                f"value_of_info={value:.3f} > extra_cost={extra_cost:.3f} "
                f"(greedy P(best)={p_best[greedy]:.2f}, regret if committed="
                f"{r_greedy:.3f}, stakes={stakes:.2f})" + budget_note
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
                + budget_note
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
    hedged: bool = False     # True when a latency hedge actually fired
    note: str = ""           # short annotation for the trace (hedge/failure info)


class SpeculativeExecutor:
    """Runs the chosen branch set concurrently and selects the winner.

    ``rollback`` semantics: losing branches are marked ``squashed`` and their
    outputs never flow downstream — the pipeline commits only the winner's
    output. If your agents have *external* side effects (writes, emails, tool
    calls), register a compensating action via ``on_squash`` and PRISM will
    invoke it for every squashed branch, mirroring a CPU discarding a
    mis-speculated store buffer. (Failed branches do NOT trigger ``on_squash`` —
    there is no committed output to compensate; agents are expected to be
    internally transactional on failure.)

    Industrial failure semantics
    ----------------------------
    * A branch that **raises** becomes a failed outcome (score 0, ``failed=True``)
      rather than crashing the stage — the winner is chosen among survivors.
    * With ``timeout`` set, branches still pending at the deadline are marked
      failed with ``error='timeout'``. Python threads cannot be killed, so the
      worker is *abandoned*, not cancelled: it may complete later and its slot
      in the pool is occupied until then. Size ``max_workers`` with headroom,
      and prefer async cancellation in latency-critical deployments (disclosed
      limitation, not hidden).
    * A **scorer** exception downgrades to the agent's ``self_report`` rather
      than failing the branch — a broken judge shouldn't erase real work, but
      an unjudged score is weak evidence, and calibration treats it as such.
    * If every branch fails, :class:`~prism.errors.AllBranchesFailed` is raised
      with the outcomes attached; the orchestrator uses it to fail over.

    The worker pool is persistent (created lazily) so that hedged/abandoned
    calls can outlive a single ``run_stage`` invocation; call :meth:`close`
    on shutdown if you care about prompt thread teardown.
    """

    def __init__(self, scorer: Scorer, max_workers: int = 8,
                 timeout: float | None = None) -> None:
        self.scorer = scorer
        self.max_workers = max_workers
        self.timeout = timeout
        self._pool: ThreadPoolExecutor | None = None

    # --- pool management ----------------------------------------------------

    def _ensure_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.max_workers,
                                            thread_name_prefix="prism")
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    @staticmethod
    def _call_agent(agent: Agent, task: Task) -> AgentOutput:
        """Run the agent and make sure the output carries a real latency (a
        wall-clock measurement fills in when the agent doesn't report one) —
        the latency tracker that powers hedging feeds on this."""
        t0 = time.perf_counter()
        out = agent.run(task)
        if out.latency <= 0.0:
            out.latency = time.perf_counter() - t0
        return out

    def _score(self, task: Task, out: AgentOutput) -> tuple[float, str]:
        """Score an output, downgrading to self_report if the scorer itself
        blows up. Returns (score, note)."""
        try:
            return self.scorer.score(task, out), ""
        except Exception as exc:  # noqa: BLE001 — judge failure must not kill the stage
            fallback = min(max(out.self_report, 0.0), 1.0)
            return fallback, f"scorer error ({exc!r}); fell back to self_report"

    # --- the main mechanism -------------------------------------------------

    def run_stage(
        self,
        task: Task,
        branch_agents: list[Agent],
        priors: dict[str, Belief],
        on_squash=None,
    ) -> StageResult:
        # 1. Fan out: execute every branch agent in parallel. In real use these
        #    are I/O-bound LLM calls, so threads give true wall-clock overlap.
        pool = self._ensure_pool()
        futures = {pool.submit(self._call_agent, a, task): a for a in branch_agents}
        done, pending = wait(list(futures), timeout=self.timeout)

        outputs: dict[str, AgentOutput] = {}
        failures: dict[str, str] = {}
        for fut in done:
            a = futures[fut]
            exc = fut.exception()
            if exc is not None:
                failures[a.name] = repr(exc)
            else:
                outputs[a.name] = fut.result()
        for fut in pending:
            a = futures[fut]
            failures[a.name] = "timeout"   # thread abandoned, slot occupied

        if not outputs:
            all_failed = [
                BranchOutcome(
                    agent_name=a.name,
                    output=AgentOutput(output=None, self_report=0.0,
                                       cost=a.cost,
                                       latency=self.timeout or 0.0),
                    score=0.0, prior_mean=priors[a.name].mean,
                    posterior_mean=priors[a.name].mean,
                    failed=True, error=failures[a.name],
                )
                for a in branch_agents
            ]
            raise AllBranchesFailed(task.task_type, all_failed)

        # 2. Score + calibrate each SURVIVING branch against its prior.
        scored = []
        raw_scores: dict[str, float] = {}
        notes: list[str] = []
        for a in branch_agents:
            if a.name not in outputs:
                continue
            out = outputs[a.name]
            raw, note = self._score(task, out)
            if note:
                notes.append(f"{a.name}: {note}")
            raw_scores[a.name] = raw
            scored.append(calibrate(a.name, raw, priors[a.name],
                                    getattr(self.scorer, "reliability", 0.85)))

        # 3. Commit the calibrated winner; squash surviving losers; record
        #    failures as zero-score outcomes (the bandit will learn from them).
        best = select_winner(scored)
        outcomes: list[BranchOutcome] = []
        latency = 0.0
        cost = 0.0
        for a in branch_agents:
            cost += a.cost   # a failed/timed-out call still cost you the call
            if a.name in outputs:
                out = outputs[a.name]
                latency = max(latency, out.latency)
                is_win = a.name == best.agent_name
                cs = next(s for s in scored if s.agent_name == a.name)
                outcome = BranchOutcome(
                    agent_name=a.name, output=out, score=raw_scores[a.name],
                    prior_mean=cs.prior_mean, posterior_mean=cs.posterior_mean,
                    is_winner=is_win,
                    squashed=not is_win and len(branch_agents) > 1,
                )
            else:
                latency = max(latency, self.timeout or 0.0)
                outcome = BranchOutcome(
                    agent_name=a.name,
                    output=AgentOutput(output=None, self_report=0.0,
                                       cost=a.cost, latency=self.timeout or 0.0),
                    score=0.0, prior_mean=priors[a.name].mean,
                    posterior_mean=priors[a.name].mean,
                    failed=True, error=failures[a.name],
                )
            outcomes.append(outcome)
            if outcome.squashed and on_squash is not None:
                on_squash(a, outcome.output)  # compensate external side effects

        winner = next(o for o in outcomes if o.is_winner)
        if failures:
            notes.append(f"failed branches: {sorted(failures)}")
        return StageResult(winner=winner, outcomes=outcomes,
                           latency=latency, cost=cost, note="; ".join(notes))

    # --- latency hedging (see prism.hedging) --------------------------------

    def run_stage_hedged(
        self,
        task: Task,
        primary: Agent,
        backup: Agent,
        priors: dict[str, Belief],
        hedge_after: float,
        on_late=None,
    ) -> StageResult:
        """Single-arm execution with a latency hedge armed.

        Runs ``primary``; if it hasn't completed within ``hedge_after`` seconds,
        launches ``backup`` and commits the **first successful** result. The
        straggler is abandoned but not wasted: when it eventually completes,
        ``on_late(agent_name, output_or_none, error_or_none)`` fires (from the
        worker thread) so its score can still update the bandit — the same
        "squashed work still teaches" principle, applied to time.
        """
        pool = self._ensure_pool()
        t0 = time.perf_counter()
        f_primary = pool.submit(self._call_agent, primary, task)
        done, _ = wait([f_primary], timeout=hedge_after)

        if done:
            exc = f_primary.exception()
            if exc is None:
                return self._single_result(task, primary, f_primary.result(),
                                           priors, t0, hedged=False)
            # Primary failed fast — run the backup as a plain failover.
            f_backup = pool.submit(self._call_agent, backup, task)
            done_b, _ = wait([f_backup], timeout=self.timeout)
            b_exc = f_backup.exception() if done_b else None
            if not done_b or b_exc is not None:
                raise AllBranchesFailed(task.task_type, [
                    self._failed_outcome(primary, repr(exc), priors),
                    self._failed_outcome(backup,
                                         repr(b_exc) if b_exc else "timeout",
                                         priors),
                ])
            res = self._single_result(task, backup, f_backup.result(),
                                      priors, t0, hedged=False)
            res.cost += primary.cost
            res.note = f"primary {primary.name!r} failed fast ({exc!r}); failover to backup"
            return res

        # Primary is in its latency tail: fire the hedge.
        f_backup = pool.submit(self._call_agent, backup, task)
        futures = {f_primary: primary, f_backup: backup}
        remaining = dict(futures)
        deadline = t0 + self.timeout if self.timeout else None

        taken_agent: Agent | None = None
        taken_out: AgentOutput | None = None
        while remaining:
            budget_left = None if deadline is None else max(deadline - time.perf_counter(), 0.0)
            done2, _ = wait(list(remaining), timeout=budget_left,
                            return_when=FIRST_COMPLETED)
            if not done2:
                break  # overall timeout: everything still pending is abandoned
            for fut in done2:
                agent = remaining.pop(fut)
                if fut.exception() is None and taken_agent is None:
                    taken_agent, taken_out = agent, fut.result()
            if taken_agent is not None:
                break

        if taken_agent is None:
            raise AllBranchesFailed(task.task_type, [
                self._failed_outcome(a, "failed or timed out in hedged race",
                                     priors)
                for a in (primary, backup)
            ])

        # Wire the straggler's eventual completion back into the learning loop.
        for fut, agent in remaining.items():
            def _late(f, name=agent.name):
                if on_late is None:
                    return
                exc = f.exception()
                on_late(name, None if exc else f.result(),
                        repr(exc) if exc else None)
            fut.add_done_callback(_late)

        res = self._single_result(task, taken_agent, taken_out, priors, t0,
                                  hedged=True)
        res.cost = primary.cost + backup.cost   # the hedge was launched: both paid
        res.note = (f"latency hedge fired at {hedge_after * 1000:.0f}ms — "
                    f"committed {taken_agent.name!r} "
                    f"({'backup' if taken_agent is backup else 'primary'} won the race)")
        return res

    def _failed_outcome(self, agent: Agent, error: str,
                        priors: dict[str, Belief]) -> BranchOutcome:
        prior = priors.get(agent.name)
        mean = prior.mean if prior is not None else 0.5
        return BranchOutcome(
            agent_name=agent.name,
            output=AgentOutput(output=None, self_report=0.0, cost=agent.cost,
                               latency=self.timeout or 0.0),
            score=0.0, prior_mean=mean, posterior_mean=mean,
            failed=True, error=error,
        )

    def _single_result(self, task: Task, agent: Agent, out: AgentOutput,
                       priors: dict[str, Belief], t0: float,
                       hedged: bool) -> StageResult:
        raw, note = self._score(task, out)
        cs = calibrate(agent.name, raw, priors[agent.name],
                       getattr(self.scorer, "reliability", 0.85))
        outcome = BranchOutcome(
            agent_name=agent.name, output=out, score=raw,
            prior_mean=cs.prior_mean, posterior_mean=cs.posterior_mean,
            is_winner=True,
        )
        return StageResult(winner=outcome, outcomes=[outcome],
                           latency=time.perf_counter() - t0, cost=agent.cost,
                           hedged=hedged, note=note)
