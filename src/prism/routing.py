"""Online bandit routing.

The router's job: for a given task type, decide which of the candidate agents
(the bandit "arms") should handle it. PRISM uses **Thompson sampling** over a
Beta posterior per arm — draw a plausible quality for each arm from its belief,
pick the argmax. Thompson sampling is the workhorse here because it:

  * needs no hand-tuned exploration constant (unlike ε-greedy / UCB),
  * explores *exactly in proportion to uncertainty* — a barely-tried arm gets
    sampled optimistically often, a well-characterised arm rarely,
  * degrades gracefully to pure exploitation as evidence accumulates.

Every agent invocation — including speculative *losers* — feeds a reward back
via :meth:`update`, so the policy improves with every task. This is the online
adaptation the project promises: routing is never static.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .agents import Agent
from .uncertainty import Belief, probability_best, selection_entropy


@dataclass
class BanditRouter:
    """Maintains a Beta belief for every (task_type, agent) pair and routes.

    The belief table *is* the learned policy. It starts as uninformative priors
    and sharpens as the system runs. Because beliefs are immutable
    :class:`Belief` values, snapshotting them into a causal trace is free and
    safe.
    """

    rng: random.Random = field(default_factory=lambda: random.Random(0))
    prior: Belief = Belief(1.0, 1.0)
    # task_type -> agent_name -> Agent
    _arms: dict[str, dict[str, Agent]] = field(default_factory=dict)
    # task_type -> agent_name -> Belief
    _beliefs: dict[str, dict[str, Belief]] = field(default_factory=dict)

    # --- registration -------------------------------------------------------

    def register(self, task_type: str, agent: Agent,
                 prior: Belief | None = None) -> None:
        """Make ``agent`` an eligible arm for ``task_type``. An optional warm
        prior lets you inject domain knowledge ("this model is great at code")
        so the router doesn't have to learn everything from scratch."""
        self._arms.setdefault(task_type, {})[agent.name] = agent
        self._beliefs.setdefault(task_type, {})[agent.name] = prior or self.prior

    def candidates(self, task_type: str) -> dict[str, Agent]:
        return dict(self._arms.get(task_type, {}))

    def beliefs(self, task_type: str) -> dict[str, Belief]:
        return dict(self._beliefs.get(task_type, {}))

    def belief(self, task_type: str, agent_name: str) -> Belief:
        return self._beliefs[task_type][agent_name]

    # --- selection ----------------------------------------------------------

    def thompson_select(self, task_type: str) -> str:
        """One Thompson draw: sample each arm's quality, return the argmax name."""
        beliefs = self._beliefs[task_type]
        best_name, best_val = None, -1.0
        for name, b in beliefs.items():
            v = b.sample(self.rng)
            if v > best_val:
                best_name, best_val = name, v
        assert best_name is not None
        return best_name

    def greedy_select(self, task_type: str) -> str:
        """Exploit only: the arm with the highest posterior mean."""
        beliefs = self._beliefs[task_type]
        return max(beliefs, key=lambda n: beliefs[n].mean)

    def p_best(self, task_type: str, samples: int = 512) -> dict[str, float]:
        """Estimate P(arm is best) for every candidate — the raw material for
        both routing confidence and the speculation decision."""
        return probability_best(self._beliefs[task_type], self.rng, samples=samples)

    def routing_entropy(self, task_type: str, samples: int = 512) -> float:
        return selection_entropy(self.p_best(task_type, samples=samples))

    # --- learning -----------------------------------------------------------

    def update(self, task_type: str, agent_name: str, reward: float,
               weight: float = 1.0) -> tuple[Belief, Belief]:
        """Fold one observed reward into an arm's belief. Returns
        ``(before, after)`` so the caller can record the exact delta in the
        trace. Called for *every* branch that runs — winners and squashed
        losers alike — which is what makes speculation double as free
        exploration."""
        before = self._beliefs[task_type][agent_name]
        after = before.updated(reward, weight=weight)
        self._beliefs[task_type][agent_name] = after
        return before, after

    # --- introspection ------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Dump the whole learned policy as plain numbers (mean, std) for
        logging / plotting convergence."""
        return {
            tt: {n: (b.mean, b.std) for n, b in arms.items()}
            for tt, arms in self._beliefs.items()
        }
