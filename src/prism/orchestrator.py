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

The whole thing is deterministic given a seed, so demos and tests are stable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .agents import Agent, Scorer, SimulatedScorer
from .graph import TaskGraph
from .routing import BanditRouter
from .speculation import SpeculationPolicy, SpeculativeExecutor
from .trace import CausalTrace, RoutingRationale, StageTrace
from .types import Task
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
    ) -> "Orchestrator":
        """Wire up a ready-to-run orchestrator.

        ``agents_by_stage`` maps each task type to the agents eligible for it
        (its bandit arms). ``priors`` optionally seeds per-(stage, agent) beliefs
        with domain knowledge."""
        rng = random.Random(seed)
        router = BanditRouter(rng=random.Random(seed + 1))
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
        executor = SpeculativeExecutor(scorer=scorer)
        return cls(graph=graph, router=router, policy=policy,
                   executor=executor, rng=rng)

    # -- execution -----------------------------------------------------------

    def run(self, payload, stakes: float = 0.7,
            entry: Optional[str] = None) -> CausalTrace:
        """Run one task end-to-end through the graph and return its causal trace.

        The winner's output at each stage becomes the payload for the next; the
        compound belief threads through :func:`propagate`, so uncertainty
        accumulates honestly instead of being reset at each hop."""
        node = entry or self.graph.entry
        assert node is not None, "graph has no entry node"

        trace = CausalTrace(task_type=node)
        context: dict = {}
        compound: Optional[Belief] = None
        current_payload = payload

        steps = 0
        while node is not None and steps < self.max_stages:
            steps += 1
            task = Task(task_type=node, payload=current_payload,
                        stakes=stakes, context=context)

            stage_trace, winner_output, winner_belief = self._run_stage(task)

            # Uncertainty propagation: the answer's confidence is gated by every
            # stage it passed through (see uncertainty.propagate for the math).
            compound = winner_belief if compound is None else propagate(compound, winner_belief)
            stage_trace.propagated_belief = compound
            trace.add(stage_trace)

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

    def _run_stage(self, task: Task):
        stage = task.task_type
        beliefs = self.router.beliefs(stage)
        agents = self.router.candidates(stage)
        costs = {name: getattr(a, "cost", 1.0) for name, a in agents.items()}

        # 1. Decide: greedy or speculative, and over which branches.
        decision = self.policy.decide(beliefs, costs, task.stakes)
        branch_agents = [agents[name] for name in decision.branches]
        priors = {name: beliefs[name] for name in decision.branches}

        # 2. Execute the branch set in parallel; commit winner, squash losers.
        result = self.executor.run_stage(task, branch_agents, priors)

        # 3. Learn from EVERY branch — winners and squashed losers alike. This is
        #    the counterfactual signal a CPU throws away and PRISM keeps.
        winner_before = winner_after = None
        for oc in result.outcomes:
            before, after = self.router.update(stage, oc.agent_name, oc.score)
            oc.prior_mean = before.mean
            oc.posterior_mean = after.mean
            if oc.is_winner:
                winner_before, winner_after = before, after

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
        return stage_trace, result.winner.output, winner_after
