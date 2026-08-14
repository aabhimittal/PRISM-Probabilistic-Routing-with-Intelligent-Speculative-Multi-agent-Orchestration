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
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .agents import Agent
from .uncertainty import Belief, probability_best, selection_entropy

if TYPE_CHECKING:
    from .drift import DriftMonitor


@dataclass
class BanditRouter:
    """Maintains a Beta belief for every (task_type, agent) pair and routes.

    The belief table *is* the learned policy. It starts as uninformative priors
    and sharpens as the system runs. Because beliefs are immutable
    :class:`Belief` values, snapshotting them into a causal trace is free and
    safe.

    Industrial extensions (all opt-in, default off):

    ``max_evidence``
        Caps the pseudo-count of evidence per arm by rescaling ``(α, β)`` before
        each update (mean preserved, memory bounded). A bandit with unbounded
        evidence becomes immovable — after 10,000 observations, no realistic
        stream of bad rewards can shift it in useful time. Capping evidence is
        exponential forgetting: the policy stays permanently adaptable, at the
        price of slightly wider steady-state uncertainty. Values of 100–500 are
        sensible; None disables.
    ``drift``
        A :class:`~prism.drift.DriftMonitor`. Every reward is also fed to a
        per-arm change-point detector; on an alarm the arm's belief is
        soft-reset (uncertainty resurrection) so routing re-adapts in tens of
        tasks instead of hundreds.
    Updates are serialized by an internal lock, making concurrent
    ``Orchestrator.run`` calls from multiple threads safe (though runs are then
    no longer bit-for-bit deterministic — order of interleaving is OS-scheduled).
    """

    rng: random.Random = field(default_factory=lambda: random.Random(0))
    prior: Belief = Belief(1.0, 1.0)
    max_evidence: Optional[float] = None
    drift: Optional["DriftMonitor"] = None
    # task_type -> agent_name -> Agent
    _arms: dict[str, dict[str, Agent]] = field(default_factory=dict)
    # task_type -> agent_name -> Belief
    _beliefs: dict[str, dict[str, Belief]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

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
        exploration.

        With ``max_evidence`` set, the belief is first rescaled to the cap
        (exponential forgetting); with ``drift`` set, the reward also feeds the
        change-point detector, which may replace the posterior wholesale
        (uncertainty resurrection)."""
        with self._lock:
            before = self._beliefs[task_type][agent_name]
            base = before
            if self.max_evidence is not None and base.evidence > self.max_evidence:
                scale = self.max_evidence / base.evidence
                base = Belief(alpha=max(base.alpha * scale, 1e-3),
                              beta=max(base.beta * scale, 1e-3))
            after = base.updated(reward, weight=weight)
            if self.drift is not None:
                replacement = self.drift.observe(task_type, agent_name, reward, after)
                if replacement is not None:
                    after = replacement
            self._beliefs[task_type][agent_name] = after
            return before, after

    # --- introspection ------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Dump the whole learned policy as plain numbers (mean, std) for
        logging / plotting convergence."""
        with self._lock:
            return {
                tt: {n: (b.mean, b.std) for n, b in arms.items()}
                for tt, arms in self._beliefs.items()
            }

    # --- persistence --------------------------------------------------------

    def state_dict(self) -> dict[str, dict[str, list[float]]]:
        """The learned policy as plain JSON-able numbers: ``{stage: {agent:
        [alpha, beta]}}``. This *is* everything the router has learned."""
        with self._lock:
            return {
                tt: {n: [b.alpha, b.beta] for n, b in arms.items()}
                for tt, arms in self._beliefs.items()
            }

    def load_state_dict(self, data: dict[str, dict[str, list[float]]]) -> int:
        """Restore beliefs from :meth:`state_dict` output. Only (stage, agent)
        pairs that are currently registered are restored — a saved policy from
        an older roster loads its intersection rather than exploding. Returns
        the number of beliefs restored."""
        loaded = 0
        with self._lock:
            for tt, arms in data.items():
                if tt not in self._beliefs:
                    continue
                for name, ab in arms.items():
                    if name in self._beliefs[tt]:
                        self._beliefs[tt][name] = Belief(float(ab[0]), float(ab[1]))
                        loaded += 1
        return loaded
