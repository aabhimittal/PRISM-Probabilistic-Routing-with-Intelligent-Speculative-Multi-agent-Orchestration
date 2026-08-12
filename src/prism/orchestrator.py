"""The PRISM orchestrator — where every subsystem meets.

For each task it:

  1. walks the probabilistic :class:`~prism.graph.TaskGraph` stage by stage;
  2. at each stage, asks the :class:`~prism.speculation.SpeculationPolicy`
     whether to commit to one agent (greedy) or run several (speculative),
     using the current :class:`~prism.routing.BanditRouter` beliefs;
  3. executes the chosen branches in parallel, scores + calibrates them, and
     commits the winner while squashing the losers;
  4. feeds **every** branch's score back into the bandit (winners *and* losers)
     — online learning from speculation;
  5. propagates epistemic uncertainty across stages so the final answer carries
     an honest, compound confidence;
  6. records a fully-serializable :class:`~prism.trace.CausalTrace`.

Industrial hardening (all opt-in; defaults preserve the simple behaviour):

  * **Circuit breaker** — arms that fail repeatedly are removed from the
    candidate set for a cooldown, then probed (half-open) before re-admission.
  * **Failover** — if every branch at a stage fails, the stage is retried once
    over the arms not yet tried; only then does the run raise
    :class:`~prism.errors.StageFailure` (with the partial trace attached).
  * **Latency hedging** — on the greedy path, a backup branch is launched if
    the committed arm blows through its learned latency fence ("The Tail at
    Scale" applied to agents).
  * **Drift adaptation** — the router's :class:`~prism.drift.DriftMonitor`
    soft-resets beliefs on regime change; the orchestrator folds those alarms
    into the causal trace as events.
  * **Budgets** — a :class:`~prism.budget.ComputeBudget` degrades speculation
    gracefully and stops the run cleanly when even greedy work is unaffordable.

The whole thing is deterministic given a seed *when run serially without
latency simulation*; concurrency and real timeouts trade that determinism for
robustness, which is the correct industrial trade.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .agents import Agent, Scorer, SimulatedScorer
from .budget import ComputeBudget
from .drift import DriftMonitor
from .errors import (
    AllBranchesFailed,
    BudgetExhausted,
    ConfigurationError,
    StageFailure,
)
from .graph import TaskGraph
from .hedging import HedgePolicy
from .resilience import CircuitBreaker
from .routing import BanditRouter
from .speculation import SpeculationPolicy, SpeculativeExecutor, StageResult
from .trace import CausalTrace, RoutingRationale, StageTrace
from .types import RouteMode, Task
from .uncertainty import Belief, propagate


@dataclass
class Orchestrator:
    """Top-level PRISM engine. Compose it from the parts, or use
    :meth:`Orchestrator.build` for a batteries-included default."""

    graph: TaskGraph
    router: BanditRouter
    policy: SpeculationPolicy
    executor: SpeculativeExecutor
    rng: random.Random
    max_stages: int = 32  # safety bound against pathological graph cycles
    breaker: Optional[CircuitBreaker] = None
    hedge: Optional[HedgePolicy] = None
    # Events produced by straggler (hedged) results that landed after their run
    # returned — appended from worker threads, drained by whoever cares.
    late_events: list[str] = field(default_factory=list)
    _run_counter: int = 0

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        graph: TaskGraph,
        agents_by_stage: dict[str, list[Agent]],
        scorer: Optional[Scorer] = None,
        seed: int = 0,
        policy: Optional[SpeculationPolicy] = None,
        priors: Optional[dict[str, dict[str, Belief]]] = None,
        breaker: Optional[CircuitBreaker] = None,
        hedge: Optional[HedgePolicy] = None,
        drift: Optional[DriftMonitor] = None,
        max_evidence: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> "Orchestrator":
        """Wire up a ready-to-run orchestrator.

        ``agents_by_stage`` maps each task type to the agents eligible for it
        (its bandit arms). ``priors`` optionally seeds per-(stage, agent)
        beliefs with domain knowledge. The industrial knobs (``breaker``,
        ``hedge``, ``drift``, ``max_evidence``, ``timeout``) all default to
        off, preserving the simple deterministic behaviour."""
        rng = random.Random(seed)
        router = BanditRouter(rng=random.Random(seed + 1),
                              max_evidence=max_evidence, drift=drift)
        for stage, agents in agents_by_stage.items():
            for a in agents:
                p = None
                if priors and stage in priors and a.name in priors[stage]:
                    p = priors[stage][a.name]
                router.register(stage, a, prior=p)
        scorer = scorer or SimulatedScorer(reliability=0.85,
                                           rng=random.Random(seed + 2))
        policy = policy or SpeculationPolicy(rng=random.Random(seed + 3),
                                             scorer_efficiency=getattr(
                                                 scorer, "reliability", 0.85))
        executor = SpeculativeExecutor(scorer=scorer, timeout=timeout)
        return cls(graph=graph, router=router, policy=policy,
                   executor=executor, rng=rng, breaker=breaker, hedge=hedge)

    # -- execution -----------------------------------------------------------

    def run(self, payload, stakes: float = 0.7,
            entry: Optional[str] = None,
            budget: Optional[ComputeBudget] = None) -> CausalTrace:
        """Run one task end-to-end through the graph and return its causal trace.

        The winner's output at each stage becomes the payload for the next; the
        compound belief threads through :func:`propagate`, so uncertainty
        accumulates honestly instead of being reset at each hop."""
        node = entry or self.graph.entry
        if node is None:
            raise ConfigurationError("graph has no entry node")

        self._run_counter += 1
        trace = CausalTrace(task_type=node)
        context: dict = {}
        compound: Optional[Belief] = None
        current_payload = payload

        steps = 0
        while node is not None and steps < self.max_stages:
            steps += 1

            # Budget gate: even the cheapest single branch must be affordable,
            # or the run stops cleanly with the partial trace attached.
            if budget is not None:
                candidates = self.router.candidates(node)
                if not candidates:
                    raise ConfigurationError(
                        f"no agents registered for task type {node!r}")
                cheapest = min(a.cost for a in candidates.values())
                if not budget.can_afford_base(cheapest):
                    raise BudgetExhausted(node, budget.remaining, cheapest,
                                          trace=trace)

            task = Task(task_type=node, payload=current_payload,
                        stakes=stakes, context=context)

            try:
                stage_trace, winner_output, winner_belief, events = \
                    self._run_stage(task, budget=budget)
            except StageFailure as sf:
                sf.trace = trace  # provenance survives the failure
                raise

            trace.events.extend(events)
            if budget is not None:
                budget.spend(stage_trace.cost)

            # Uncertainty propagation: the answer's confidence is gated by every
            # stage it passed through (see uncertainty.propagate for the math).
            compound = winner_belief if compound is None else propagate(compound, winner_belief)
            stage_trace.propagated_belief = compound
            trace.add(stage_trace)

            # Fold drift alarms (uncertainty resurrections) into the provenance.
            if self.router.drift is not None:
                for ev in self.router.drift.drain_events():
                    trace.events.append(ev.describe())

            # Commit the winner's output; it feeds the next stage.
            current_payload = winner_output.output
            context[f"stage_{node}"] = winner_output.output

            # Walk the (probabilistic, conditional) graph to the next node.
            nxt = self.graph.next_node(node, winner_output, context, self.rng)
            if nxt is not None:
                # Reinforce the transition by the stage's realized quality so the
                # graph learns good routes over time.
                self.graph.update_edge(node, nxt, winner_belief.mean)
            node = nxt

        trace.final_output = current_payload
        trace.final_belief = compound
        return trace

    # -- one stage, with failover --------------------------------------------

    def _run_stage(self, task: Task, budget: Optional[ComputeBudget] = None):
        stage = task.task_type
        tick = self._run_counter
        events: list[str] = []

        agents_all = self.router.candidates(stage)
        if not agents_all:
            raise ConfigurationError(
                f"no agents registered for task type {stage!r}")

        # Circuit breaker: drop OPEN arms from the candidate set (half-open
        # arms come back through as probes; a fully-open roster is overridden).
        names = list(agents_all)
        blocked: list[str] = []
        if self.breaker is not None:
            names, blocked = self.breaker.filter(stage, names, tick)
            if blocked:
                events.append(
                    f"stage {stage}: circuit breaker holding {sorted(blocked)} out "
                    f"of the candidate set")

        beliefs_all = self.router.beliefs(stage)
        beliefs = {n: beliefs_all[n] for n in names}
        costs = {n: agents_all[n].cost for n in names}

        # 1. Decide: greedy or speculative (budget-aware), and over which branches.
        decision = self.policy.decide(beliefs, costs, task.stakes, budget=budget)

        # Optional latency hedge on the greedy path: commit one arm, but arm a
        # backup that fires only if the primary blows its latency fence.
        hedge_plan = None
        if (self.hedge is not None and decision.mode is RouteMode.GREEDY
                and len(names) > 1):
            # When speculation is disabled (max_branches=1) the decision carries
            # a degenerate p_best; rank backups by belief mean instead so
            # hedging still works as the only multi-branch mechanism.
            p_for_hedge = decision.p_best
            if len(p_for_hedge) < 2:
                total = sum(beliefs[n].mean for n in names) or 1.0
                p_for_hedge = {n: beliefs[n].mean / total for n in names}
            hedge_plan = self.hedge.plan(stage, decision.branches[0],
                                         p_for_hedge, task.stakes)

        # 2. Execute, with one round of failover if every branch fails.
        tried = set(decision.branches)
        failed_outcomes: list = []
        try:
            result = self._execute(task, decision, agents_all, beliefs_all,
                                   hedge_plan, events)
            if hedge_plan is not None and result.hedged:
                tried.add(hedge_plan[0])
        except AllBranchesFailed as fail:
            failed_outcomes = list(fail.outcomes)
            if hedge_plan is not None:
                tried.add(hedge_plan[0])
            untried = [n for n in names if n not in tried]
            untried.sort(key=lambda n: beliefs_all[n].mean, reverse=True)
            if not untried:
                self._learn_from_failures(stage, failed_outcomes, tick, events)
                raise StageFailure(
                    stage, f"all branches failed ({sorted(tried)}) and no "
                           f"arms remain for failover") from fail
            fallback = untried[: max(self.policy.max_branches, 1)]
            events.append(
                f"stage {stage}: all branches failed ({sorted(tried)}); "
                f"failing over to {fallback}")
            fo_agents = [agents_all[n] for n in fallback]
            fo_priors = {n: beliefs_all[n] for n in fallback}
            try:
                result = self.executor.run_stage(task, fo_agents, fo_priors)
            except AllBranchesFailed as fail2:
                self._learn_from_failures(
                    stage, failed_outcomes + list(fail2.outcomes), tick, events)
                raise StageFailure(
                    stage, "failover branches failed too "
                           f"({sorted(tried)} then {fallback})") from fail2

        # Provenance completeness: failed first-round branches stay in the
        # stage trace alongside the failover outcomes.
        result.outcomes = failed_outcomes + result.outcomes
        if result.note:
            events.append(f"stage {stage}: {result.note}")

        # 3. Learn from EVERY branch — winners, squashed losers, and failures
        #    (reward 0) alike. This is the counterfactual signal a CPU throws
        #    away and PRISM keeps.
        winner_before = winner_after = None
        for oc in result.outcomes:
            before, after = self.router.update(stage, oc.agent_name, oc.score)
            oc.prior_mean = before.mean
            oc.posterior_mean = after.mean
            if oc.is_winner:
                winner_before, winner_after = before, after
            if self.breaker is not None:
                ev = self.breaker.record(stage, oc.agent_name, oc.failed, tick)
                if ev:
                    events.append(ev)
            if self.hedge is not None and not oc.failed:
                self.hedge.tracker.observe(stage, oc.agent_name,
                                           oc.output.latency)

        # 4. Assemble the stage trace (propagated_belief filled in by caller).
        rationale = RoutingRationale(
            mode=decision.mode,
            p_best=decision.p_best,
            selection_entropy=decision.entropy,
            greedy_choice=decision.greedy,
            branches=decision.branches,
            expected_regret_greedy=decision.expected_regret_greedy,
            value_of_speculation=decision.value_of_speculation,
            speculation_cost=decision.cost,
            explanation=decision.explanation,
        )
        stage_trace = StageTrace(
            task_type=stage,
            rationale=rationale,
            outcomes=result.outcomes,
            winner=result.winner.agent_name,
            stage_belief_before=winner_before,   # type: ignore[arg-type]
            stage_belief_after=winner_after,     # type: ignore[arg-type]
            propagated_belief=winner_after,      # placeholder; caller propagates
            latency=result.latency,
            cost=result.cost,
            realized_quality=result.winner.output.latent_quality or 0.0,
        )
        return stage_trace, result.winner.output, winner_after, events

    def _execute(self, task: Task, decision, agents_all, beliefs_all,
                 hedge_plan, events: list[str]) -> StageResult:
        if hedge_plan is not None:
            backup_name, fence = hedge_plan
            primary_name = decision.branches[0]
            priors = {primary_name: beliefs_all[primary_name],
                      backup_name: beliefs_all[backup_name]}
            return self.executor.run_stage_hedged(
                task, agents_all[primary_name], agents_all[backup_name],
                priors, hedge_after=fence,
                on_late=self._make_on_late(task, task.task_type),
            )
        branch_agents = [agents_all[n] for n in decision.branches]
        priors = {n: beliefs_all[n] for n in decision.branches}
        return self.executor.run_stage(task, branch_agents, priors)

    def _learn_from_failures(self, stage: str, outcomes, tick: int,
                             events: list[str]) -> None:
        for oc in outcomes:
            self.router.update(stage, oc.agent_name, 0.0)
            if self.breaker is not None:
                ev = self.breaker.record(stage, oc.agent_name, True, tick)
                if ev:
                    events.append(ev)

    def _make_on_late(self, task: Task, stage: str):
        """Callback for straggler results in a hedged race: the branch that lost
        the race still teaches the bandit when it eventually completes. Runs on
        a worker thread — everything it touches is lock-protected or
        append-only."""

        def _on_late(name: str, output, error: Optional[str]) -> None:
            if output is None:
                self.router.update(stage, name, 0.0)
                if self.breaker is not None:
                    self.breaker.record(stage, name, True, self._run_counter)
                self.late_events.append(
                    f"late straggler {name!r} at stage {stage} failed ({error}); "
                    f"reward 0 folded into policy")
                return
            try:
                score = self.executor.scorer.score(task, output)
            except Exception:  # noqa: BLE001
                score = min(max(output.self_report, 0.0), 1.0)
            self.router.update(stage, name, score)
            if self.breaker is not None:
                self.breaker.record(stage, name, False, self._run_counter)
            if self.hedge is not None:
                self.hedge.tracker.observe(stage, name, output.latency)
            self.late_events.append(
                f"late straggler {name!r} at stage {stage} scored {score:.3f}; "
                f"folded into policy")

        return _on_late

    # -- persistence ---------------------------------------------------------

    def save_policy(self, path) -> None:
        """Persist everything the system has *learned* — router beliefs, graph
        edge beliefs, breaker state — as one JSON document. Configuration
        (agents, priors, policy knobs) is code, not state, and is not saved."""
        data = {
            "version": 1,
            "router": self.router.state_dict(),
            "graph_edges": self.graph.state_dict(),
            "breaker": self.breaker.state_dict() if self.breaker else None,
            "run_counter": self._run_counter,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_policy(self, path) -> dict[str, int]:
        """Restore a saved policy into this (already-wired) orchestrator.
        Tolerates roster drift: only currently-registered arms/edges load.
        Returns counts of what was restored."""
        data = json.loads(Path(path).read_text())
        counts = {
            "beliefs": self.router.load_state_dict(data.get("router", {})),
            "edges": self.graph.load_state_dict(data.get("graph_edges", [])),
        }
        if data.get("breaker") and self.breaker is not None:
            self.breaker.load_state_dict(data["breaker"])
            counts["breaker_arms"] = len(data["breaker"])
        self._run_counter = int(data.get("run_counter", self._run_counter))
        return counts
